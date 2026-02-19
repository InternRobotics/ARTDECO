import pypose as pp
import torch
from VSLAM.SharedKeyframes import SharedKeyframes
from VSLAM.mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from VSLAM.utils_mast3r import mast3r_match_symmetric
import mast3r_slam_backends


class FactorGraph:
    def __init__(self, config, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.config = config
        self.cfg = config["local_opt"]
        # 与jj中相连的keyframe id
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 按照顺序排列的关键帧id， 例如 1 2 2 3 3 3 4 4 4 4之类的，与ii中相应位置构成因子图
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 在i坐标系，照j的顺序，给出i的匹配位置
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 在j坐标系，照i的顺序，给出j的匹配位置
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 在i坐标系，按照j的顺序的有效的匹配对数（match函数给出的最初的匹配mask：考虑了损失和距离的匹配mask）
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        # 在j坐标系，按照i的顺序的有效的匹配对数（match函数给出的最初的匹配mask：考虑了损失和距离的匹配mask）
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        # 在i坐标系，按照j的顺序的特征预测分数
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        # 在j坐标系，按照i的顺序的特征预测分数
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)

        self.window_size = self.cfg["window_size"]

        self.K = K

    # 向因子图中增加因子，匹配数量低于一定阈值的因子不会加进去
    def add_factors(self, ii, jj, min_match_frac, embeddings, is_reloc=False):

        # 1. 对于每个因子，分别在其坐标系进行匹配
        kf_ii = [self.frames[idx].to(self.device) for idx in ii]
        kf_jj = [self.frames[idx].to(self.device) for idx in jj]
        feat_i = torch.cat([embeddings[idx][0] for idx in ii])
        feat_j = torch.cat([embeddings[idx][0] for idx in jj])
        pos_i = torch.cat([embeddings[idx][1] for idx in ii])
        pos_j = torch.cat([embeddings[idx][1] for idx in jj])
        shape_i = [torch.tensor(kf_i.img.shape[1:])[None] for kf_i in kf_ii]
        shape_j = [torch.tensor(kf_j.img.shape[1:])[None] for kf_j in kf_jj]

        (
            idx_i2j,  # 在i坐标系，照j的顺序，给出i的匹配位置
            idx_j2i,  # 在j坐标系，照i的顺序，给出j的匹配位置
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(self.config,
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        # 2. 计算出来在i和j坐标系，各自的匹配百分比
        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)
        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj # i坐标系下的匹配百分比
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni # j坐标系下的匹配百分比

        # 3. 去除无效的因子匹配：匹配分数低的边 且 时间上（指的是keyframe的local id）不相连的边
        # 在重定位上，如果retrival找到了上一个关键帧，则上一个关键帧一定不会被去除
        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)
        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges
        # todo:如果至少有一个没匹配且处于重定位模式，则退出 这个是不是有点太严格?
        if invalid_edges.any() and is_reloc:
            return False

        # 4. 保存因子
        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    # 得到因子图中所有独一无二的帧，升序
    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    # 得到所有帧对应的点云，位姿，置信度
    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx].to(self.device) for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = pp.quat2unit(pp.Sim3(torch.stack([kf.T_WC.data for kf in kfs])))
        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    # todo:这样让计算压力加倍
    # 对于某一个因子i和j，将j到i的匹配和i到j的匹配看做两个因子，因为最初的因子是j到i，所以将i到j加到j到i的后面
    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def solve_GN_rays(self):

        # 1. 获取因子图中设计的关键帧id，位姿，点云，置信度以及因子图本身
        # 对于某一个因子i和j，将j到i的匹配和i到j的匹配看做两个因子，因为最初的因子是j到i，所以将i到j加到j到i的后面
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return
        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        # 2. 开始优化
        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]
        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # 3. 更新位姿
        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size
        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
