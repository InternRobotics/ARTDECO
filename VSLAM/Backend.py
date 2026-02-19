import time
import torch
import pypose as pp
from tqdm import tqdm
from queue import Empty
from VSLAM.ImageFrame import Mode, ImageFrame
from VSLAM.mast3r_slam.geometry import constrain_points_to_ray
from VSLAM.mast3r_slam.global_opt import FactorGraph
from VSLAM.utils_mast3r import load_retriever, load_mast3r, mast3r_match_asymmetric, mast3r_inference_mono


class Backend():
    def __init__(self, args, config,
                 dataset, H_slam, W_slam, K_slam,
                 states, keyframes,
                 model=None, device="cuda:0"):
        self.args = args
        self.config = config
        self.H_slam = H_slam
        self.W_slam = W_slam
        self.K_slam = K_slam.to(device)
        self.device = device
        self.num_GBA = args.num_GBA

        # shared memory
        self.states = states
        self.keyframes = keyframes

        # class model
        self.dataset = dataset
        if model is None:
            self.model = load_mast3r(device=self.device)
        else:
            self.model = model
        self.retrieval_database = None
        self.factor_graph = FactorGraph(config, self.model, self.keyframes, self.K_slam, self.device)

        self.embeddings = {}
        self.points_for_map = {}

    @torch.no_grad()
    def run(self):
        self.retrieval_database = load_retriever(self.args, self.config, self.model,
                                                 self.keyframes, device=self.device)

        bar = tqdm(total=0, bar_format='{desc}', position=2, leave=False)
        mode = self.states.get_mode()
        while mode is not Mode.TERMINATED:
            # Read Queue
            mode = self.states.get_mode()
            try:
                keyframe_dict = self.states.msgFromFrontend()
                keyframe_style = keyframe_dict["keyframe_style"]
                keyframe_id = keyframe_dict["keyframe_id"]
                frame_id = keyframe_dict["frame_id"]
                is_test = keyframe_dict["is_test"]
                is_important = keyframe_dict["is_important"]
                focal = keyframe_dict["focal"]
            except Empty:
                time.sleep(0.001)
                if mode == Mode.OPTIMIZING:
                    self.states.set_mode(Mode.TERMINATED)
                continue
            with torch.cuda.device(self.device):
                # Execute information
                if keyframe_style == 0:
                    frame = self.states.get_frame().to(self.device)
                    X_init, C_init, feat, pos = mast3r_inference_mono(self.model, frame)
                    frame.update_pointmap(X_init, C_init)
                    with torch.cuda.device(self.device):
                        success, lc_inds = self.relocalization(frame)
                    if success:
                        self.states.lost_number.value -= 1
                        self.states.set_mode(Mode.TRACKING)
                        n_kf = len(self.keyframes)
                        keyframe = self.keyframes[n_kf - 1].to(self.device)
                        last_keyframe = self.keyframes[n_kf - 2].to(self.device)
                        (idx_k2l, valid_match_l,
                         Xkk, Ckk, Qkk,
                         Xlk, Clk, Qlk,
                         _, _) = (mast3r_match_asymmetric(self.config, self.model, keyframe, last_keyframe,
                                                          embeddings_i=self.embeddings[n_kf - 1],
                                                          embeddings_j=self.embeddings[n_kf - 2],))
                        self.prepare_for_mapper(keyframe, last_keyframe, n_kf - 1, idx_k2l,
                                                is_test=is_test,
                                                is_important=is_important,
                                                loop_keyframe_index=lc_inds,
                                                is_slam_keyframe=True, focal=focal)
                    self.states.dequeue_backend_execute()
                elif keyframe_style == 1:
                    lc_inds, idx_k2l, keyframe, last_keyframe = self.gloabla_optimization(keyframe_id, self.num_GBA)
                    self.prepare_for_mapper(keyframe, last_keyframe, keyframe_id, idx_k2l,
                                            is_test=is_test,
                                            is_important=is_important,
                                            loop_keyframe_index=lc_inds,
                                            is_slam_keyframe=True, focal=focal)

                    self.states.dequeue_backend_execute()
                elif keyframe_style == 2:
                    original_img, info = self.dataset[frame_id]
                    T_WC = pp.Sim3(keyframe_dict["T_WC"]).to(self.device)
                    img = self.dataset.transform.to_slam(original_img, device=self.device)
                    timestamp = info["timestamp"]
                    keyframe = ImageFrame(frame_id, 0, float(timestamp), img,
                                       T_WC, K=self.K_slam if self.K_slam is not None else None)
                    last_keyframe = self.keyframes[keyframe_id].to(self.device)
                    (idx_k2l, valid_match_l,
                     Xkk, Ckk, Qkk,
                     Xlk, Clk, Qlk,
                     _, _) = (mast3r_match_asymmetric(self.config, self.model, keyframe, last_keyframe,
                                                      embeddings_j=self.embeddings[keyframe_id], ))
                    keyframe.update_pointmap(Xkk, Ckk)
                    self.prepare_for_mapper(keyframe, last_keyframe, keyframe_id, idx_k2l,
                                            is_test=is_test, is_important=is_important, focal=focal)
                    self.states.dequeue_backend_execute()

    @torch.no_grad()
    def prepare_for_mapper(self, keyframe, last_keyframe, last_keyframe_index, idx_k2l,
                           is_test = False, is_important = False, loop_keyframe_index = set(),
                           is_slam_keyframe=False, focal = None):
            if last_keyframe is None:
                X_init, C_init = keyframe.X_canon, keyframe.get_average_conf()
                point = (constrain_points_to_ray(keyframe.img.shape[-2:], X_init[None], self.K_slam).
                         squeeze(0).reshape(self.H_slam, self.W_slam, 3))
                point_valid = (keyframe.get_average_conf() > 1.5).squeeze(-1).reshape(self.H_slam, self.W_slam)
                T_CkC = None
            else:
                point, point_valid = self.compute_dense_point(keyframe, last_keyframe, idx_k2l)
                T_CkC = last_keyframe.T_WC.Inv().mul(keyframe.T_WC).cpu()
            densePoint = torch.cat([point, point_valid[..., None]], dim=-1)
            keyframe_map_dict = {
                "is_test": is_test,
                "is_important": is_important,
                "T_WC": keyframe.T_WC.data.cpu(),
                "frame_id": keyframe.frame_id,
                "densePoint": densePoint.cpu(),
                "is_slam_keyframe": is_slam_keyframe,
                # For loop closure
                "loop_keyframe_index": loop_keyframe_index,
                # For map keyframe
                "T_CkC": T_CkC,
                "last_keyframe_index": last_keyframe_index,
                "last_keyframe_frame_id": last_keyframe.frame_id if last_keyframe is not None else None,
                # camera
                "focal": focal
            }
            self.states.msg2Mapper(keyframe_map_dict)

    @torch.no_grad()
    def compute_dense_point(self, keyframe, last_keyframe, idx_k2l, valid_pixel=3):
        assert self.config["use_calib"]==True, "Only work for known calib"
        Xkk = keyframe.X_canon
        idx_k2l = idx_k2l[0]
        H, W = keyframe.img.shape[-2:]

        # get matched points on the last keyframe
        Twk = keyframe.T_WC
        Twl = last_keyframe.T_WC
        Tlk = Twl.Inv().mul(Twk)
        Xkk_calib = constrain_points_to_ray(keyframe.img.shape[-2:], Xkk[None], self.K_slam).squeeze(0)
        Xkk_calib_match = Xkk_calib[idx_k2l]
        Xkl_calib_match = Tlk.Act(Xkk_calib_match).reshape(H, W, 3)  # H W 3

        # Compute Residual
        pkl = Xkl_calib_match / Xkl_calib_match[..., 2:]
        pkl_es = (self.K_slam[None, None, ...] @ pkl[..., None]).squeeze(-1)[..., :2]  # H W 2
        uk, vk = torch.meshgrid(torch.arange(W).to(Xkk_calib.device),
                                torch.arange(H).to(Xkk_calib.device),
                                indexing="xy")
        pll = torch.stack([uk, vk], dim=-1)
        resi = torch.norm(pkl_es - pll, dim=-1).reshape(-1)
        conf_valid = torch.where(resi < valid_pixel, 1., 1 / (resi - valid_pixel + 1))

        # 找出有效的相机坐标系的点
        Xkw = Twk.Act(Xkk_calib_match)
        Tkw_se3 = pp.SE3(Twk.data[:, :7]).Inv()
        Xkk_calib_match_map = Tkw_se3.Act(Xkw)  # (H*W) 3

        # 恢复原本的相机坐标系的顺序坐标
        # 初始化输出数组
        Xkk_map_sorted = Tkw_se3.Act(Twk.Act(Xkk_calib))
        conf_valid_sorted = torch.zeros(H * W, device=conf_valid.device, dtype=conf_valid.dtype)
        # 找到有效的索引（在0到H*W-1范围内）
        valid_range = (idx_k2l >= 0) & (idx_k2l < H * W)
        # 获取有效的索引和对应的数据
        valid_idx = idx_k2l[valid_range]
        valid_Xkk = Xkk_calib_match_map[valid_range]
        valid_mask = conf_valid[valid_range]
        # 根据idx_f2k进行排序赋值
        Xkk_map_sorted[valid_idx] = valid_Xkk
        conf_valid_sorted[valid_idx] = valid_mask

        return Xkk_map_sorted.reshape(H, W, 3), valid_mask.reshape(H, W)

    @torch.no_grad()
    def gloabla_optimization(self, idx, n_consec=1):

        # 1. 计算当前帧的point_map
        keyframe = self.keyframes[idx].to(self.device)
        last_keyframe = self.keyframes[idx - 1].to(self.device) if idx > 0 else None
        idx_k2l, Xlk, Clk = None, None, None
        if last_keyframe is not None:
            (idx_k2l, valid_match_l,
             Xkk, Ckk, Qkk,
             Xlk, Clk, Qlk,
             feat_k, pos_k) = (mast3r_match_asymmetric(self.config,
                                              self.model,
                                              keyframe,
                                              last_keyframe,
                                              embeddings_i=None,
                                              embeddings_j=self.embeddings[idx-1]))
        else:
            Xkk, Ckk, feat_k, pos_k = mast3r_inference_mono(self.model, keyframe)
        self.embeddings[idx] = [feat_k, pos_k]
        keyframe.update_pointmap(Xkk, Ckk)
        self.keyframes[idx] = keyframe

        # 2. 找出与当前关键帧关联的关键帧
        ###### 1. 默认认为当前关键帧与上n_consec个关键帧关联
        kf_idx = []
        # k to previous consecutive keyframes
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        ###### 2. 从库中通过特征聚合找到与当前帧相关的k个帧，顺便加入到库中
        retrieval_inds = self.retrieval_database.update(
            self.embeddings[idx][0],
            add_after_query=True,
            k=self.config["retrieval"]["k"],
            min_thresh=self.config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        lc_inds = set(retrieval_inds)
        lc_inds.add(idx)

        # 3. 增加找到的关系到因子图
        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            self.factor_graph.add_factors(kf_idx, frame_idx, self.config["local_opt"]["min_match_frac"],
                                          self.embeddings)
        with self.states.lock:
            self.states.edges_ii[:] = self.factor_graph.ii.cpu().tolist()
            self.states.edges_jj[:] = self.factor_graph.jj.cpu().tolist()

        # 4. 全局优化
        with torch.cuda.device(self.device):
            if self.config["use_calib"]:
                self.factor_graph.solve_GN_calib()
            else:
                self.factor_graph.solve_GN_rays()

        # 5. 给建图进程计算点云
        keyframe = self.keyframes[idx].to(self.device)
        last_keyframe = self.keyframes[idx - 1].to(self.device) if idx > 0 else None
        if last_keyframe is not None:
            T_WCk = pp.Sim3(self.keyframes.T_WC[idx].to(self.device))
            T_WCl = pp.Sim3(self.keyframes.T_WC[idx-1].to(self.device))
            T_ClCk = T_WCl.Inv().mul(T_WCk)
            Xll = T_ClCk.Act(Xlk)
            last_keyframe.update_pointmap(Xll, Clk)
            self.keyframes[idx - 1] = last_keyframe

        return lc_inds, idx_k2l, keyframe, last_keyframe 

    # 通过retrieval找当前帧的重定位关键帧
    @torch.no_grad()
    def relocalization(self, frame):
        # we are adding and then removing from the keyframe, so we need to be careful.
        # The lock slows viz down but safer this way...
        with self.keyframes.lock:
            # 1. 找当前帧的相关关键帧
            kf_idx = []
            img = frame.img[None]
            img_shape = torch.tensor(img.shape[2:])[None].to(self.device)
            feat, pos, _ = self.model._encode_image(img, img_shape)
            retrieval_inds = self.retrieval_database.update(
                feat,
                add_after_query=False,
                k=self.config["retrieval"]["k"],
                min_thresh=self.config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds
            successful_loop_closure = False
            if kf_idx:
                self.keyframes.append(frame)
                # 2. 如果有匹配的关键帧，判断当前帧是否都能和这些关键帧匹配上
                # 所有找出的关键帧全都匹配上才认为成功，有一个匹配不上就认为不成功
                # 然后将当前帧的位姿初始化为最相关帧的位姿
                kf_idx = list(kf_idx)  # convert to list
                n_kf = len(self.keyframes)
                self.embeddings[n_kf-1] = [feat, pos]
                frame_idx = [n_kf - 1] * len(kf_idx)
                # print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
                if self.factor_graph.add_factors(
                        kf_idx,
                        frame_idx,
                        self.config["reloc"]["min_match_frac"],
                        is_reloc=self.config["reloc"]["strict"],
                        embeddings=self.embeddings,
                ):
                    self.retrieval_database.update(
                        feat,
                        add_after_query=True,
                        k=self.config["retrieval"]["k"],
                        min_thresh=self.config["retrieval"]["min_thresh"],
                    )
                    successful_loop_closure = True
                    self.keyframes.T_WC[n_kf - 1] = self.keyframes.T_WC[kf_idx[0]].clone()
                    print("Reloc successful")
                else:
                    self.keyframes.pop_last()
                    print("Reloc failed")
            else:
                print("Reloc failed")
            # 3. 如果成功重定位了则进行全局优化
            if successful_loop_closure:
                with torch.cuda.device(self.device):
                    if self.config["use_calib"]:
                        self.factor_graph.solve_GN_calib()
                    else:
                        self.factor_graph.solve_GN_rays()
                        self.states.set_frame(self.keyframes[-1])
            return successful_loop_closure, set(kf_idx)


