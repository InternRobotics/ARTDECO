import cv2
import numpy as np
import pypose as pp
import torch
from VSLAM.ImageFrame import ImageFrame
from VSLAM.mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib, backproject,
)
from VSLAM.mast3r_slam.nonlinear_optimizer import check_convergence, huber
from VSLAM.mast3r_slam.visualization_utils import save_pointcloud_ply, visualize_matches_corr
from VSLAM.utils_mast3r import mast3r_match_asymmetric, mast3r_inference_mono, inverse_normalize
from VSLAM.utils_uncertainty import local_diag_cov_from_X1, visualize_covariance_max


class CameraTracker:
    def __init__(self, args, config, min_displacement, thres_keyframe,
                 model, frames,
                 H_slam, W_slam, K_slam, device):

        self.config = config
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device
        self.H_slam = H_slam
        self.W_slam = W_slam
        self.K_slam = K_slam
        self.min_displacement = min_displacement
        self.thres_keyframe = thres_keyframe
        self.optimize_focal = args.optimize_focal
        self.covariance_filter = args.covariance_filter
        self.point_fusion_frontend = args.point_fusion_frontend

        self.last_embedding = None
        self.last_dist = 0
        self.reset_idx_f2k()

    def track_init(self, frame: ImageFrame):
        X_init, C_init,  feat, pos = mast3r_inference_mono(self.model, frame)
        frame.update_pointmap(X_init, C_init)
        self.last_embedding = [feat, pos]
        return False, True, True

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None  # 关键帧每个点云对应的当前帧的像素坐标

    # 追踪当前帧
    def track(self, frame: ImageFrame):
        if frame.frame_id == 0:
            return self.track_init(frame)

        # 1. 计算关键帧到当前帧像素的匹配关系
        keyframe = self.keyframes.last_keyframe().to(self.device)
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf, featf, posf = (
            mast3r_match_asymmetric(self.config,self.model, frame, keyframe,
                                    idx_i2j_init=self.idx_f2k, embeddings_j = self.last_embedding))
        self.idx_f2k = idx_f2k.clone()

        # 2. 用特征分数来计算上方的匹配分数
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0] # 迭代计算中损失小于一定值且点之间的距离小于一定值的mask
        Qk = torch.sqrt(Qff[idx_f2k] * Qkf) # 每一对匹配的分数，按照关键帧像素顺序

        # 3. 用点的预测分数来计算匹配分数，并且返回
        # 匹配的点对（此时xf和Xk的顺序是一一对应的，并且在各自的坐标系，按照关键帧像素顺序）
        # 初始位姿
        # 匹配的点对的置信度（cf和ck的顺序是一一对应的，按照关键帧像素顺序）
        # 深度大于阈值的像素valid_meas_k（关键帧像素顺序）
        frame.update_pointmap(Xff, Cff)
        img_size = frame.img.shape[-2:]
        # Get poses and point correspondneces and confidences
        (Xf, Xf_cov, Xk, Xk_cov, T_WCf, T_WCk, Cf, Ck,
         meas_k, valid_meas_k) = self.get_points_poses(frame, keyframe, idx_f2k, img_size)

        # 4. 根據点云分数，特征分数以及 匹配时的点损失和距离mask（valid_match_k） 获得最后的有效匹配mask
        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]
        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q # 用于优化中的匹配的对数
        valid_kf = valid_match_k & valid_Q  # 匹配的对数

        # 5. 判定匹配是否合理
        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Insufficient match {frame.frame_id}")

            # image_f = inverse_normalize(frame.img.cpu())
            # image_k = inverse_normalize(frame.img.cpu())
            #
            # points = torch.concatenate([Xff, Xkf], dim=0)
            # colors = torch.concatenate([image_f.reshape(-1, 3), image_k.reshape(-1, 3)], dim=0)
            # save_pointcloud_ply(points, colors, f"{keyframe.frame_id}_{frame.frame_id}.ply")
            #
            # x = torch.arange(0, self.W_slam, dtype=torch.float32)
            # y = torch.arange(0, self.H_slam, dtype=torch.float32)
            # grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            # pixel_k = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            #
            # u = idx_f2k % self.W_slam
            # v = idx_f2k / self.W_slam
            # pixel_f = torch.stack([u, v], dim=-1).reshape(-1, 2).cpu()
            #
            # valid_mask = valid_opt[:, 0].cpu()
            # canvas = visualize_matches_corr(image_k, image_f, pixel_k[valid_mask], pixel_f[valid_mask],
            #                                 pixel_f[valid_mask],
            #                                 torch.ones_like(pixel_k[valid_mask][:, 0]),
            #                                 filter_i=[0, 10], filter_j=[0, 10], vis_line=False)
            # cv2.imshow("stereoFrames", canvas)
            # cv2.waitKey(-1)

            return True, False, False

        # 6. 位姿推理
        try:
            # 知道位姿重投影推理
            T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                Xf, Xf_cov, Xk, Xk_cov,
                T_WCf,T_WCk,Qk,valid_opt,
                meas_k,               # 关键帧的像素坐标及其对数深度
                valid_meas_k,         # 关键帧中深度大于阈值的像素mask
                idx_f2k,
                img_size,
            )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return True, False, False
        T_WCf = pp.quat2unit(T_WCf)
        frame.T_WC = T_WCf
        if self.point_fusion_frontend:
        # Use pose to transform points to update keyframe
            Xkk = T_CkCf.Act(Xkf)
            keyframe.update_pointmap(Xkk, Ckf)
            # write back the fitered pointmap
            self.keyframes[len(self.keyframes) - 1] = keyframe

        # 7. 关键帧判断
        is_keyframe = self.check_keyframe(idx_f2k, valid_kf, valid_match_k)
        if is_keyframe:
            self.reset_idx_f2k()
            self.last_embedding = [featf, posf]
            is_keyframe_map = True
            self.last_dist = 0
        else:
            is_keyframe_map, dist = self.check_keyframe_map(idx_f2k, valid_opt)
            if is_keyframe_map:
                self.last_dist = dist

        return False, is_keyframe, is_keyframe_map



    def check_keyframe(self, idx_f2k, valid_kf, valid_match_k):
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()  # 关键帧中有效的匹配点占总数的百分比
        # 当前帧中与关键帧匹配的独一无二的点的数量占所有匹配的点的百分比，有可能会有一对多的情况
        unique_frac_f = (
                torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )
        add_new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]
        return add_new_kf


    def check_keyframe_map(self, idx_f2k, valid_opt):
        uf = idx_f2k % self.W_slam
        vf = idx_f2k // self.W_slam
        uvf = torch.stack([uf, vf], dim=-1)

        uk, vk = torch.meshgrid(torch.arange(self.W_slam).to(self.device),
                                torch.arange(self.H_slam).to(self.device),
                                indexing="xy")
        uk = uk.reshape(-1)
        vk = vk.reshape(-1)
        uvk = torch.stack([uk, vk], dim=-1)
        dist = torch.norm((uvf - uvk).float(), dim=-1)[valid_opt.squeeze(-1)]

        dist_quantile = torch.quantile(dist, self.thres_keyframe).item()
        add_new_kf = (dist_quantile - self.last_dist) > self.min_displacement

        return add_new_kf, dist_quantile

    # 按照关键帧的顺序，返回匹配的点对，位姿，匹配点对的置信度，以及深度大于阈值的像素mask
    def get_points_poses(self, frame, keyframe, idx_f2k, img_size):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC
        T_WCf = pp.quat2unit(T_WCf)
        T_WCk = pp.quat2unit(T_WCk)

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        # 根据内参重新获得相机坐标系点云，这里只用了预测点云的深度z坐标，其余xy坐标通过K的逆矩阵和像素网格获得
        Xf = constrain_points_to_ray(img_size, Xf[None], self.K_slam).squeeze(0)
        Xk = constrain_points_to_ray(img_size, Xk[None], self.K_slam).squeeze(0)

        Xf_cov = local_diag_cov_from_X1(Xf, self.H_slam, self.W_slam)
        Xk_cov = local_diag_cov_from_X1(Xk, self.H_slam, self.W_slam)
        # visualize_covariance_max(Xf_cov, self.H_slam, self.W_slam, cmap="jet")


        # Setup pixel coordinates
        uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
        uv_k = uv_k.view(-1, 2)
        meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)

        # Avoid any bad calcs in log
        valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
        meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xf_cov[idx_f2k], Xk, Xk_cov, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    # todo:这个地方的残差权重计算没搞清楚
    # 输入jacobian J， residual r， 以及权重，用高斯牛顿法计算更新量
    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    # 输入一一对应的点对（不同坐标系），初始估计位姿，按照关键帧像素顺序的特征匹配权重，以及匹配mask
    # 用LM算法计算相对位姿改变
    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        # 计算LM中每个误差项的权重
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)
        # print(sqrt_info.shape, valid.shape, Qk.shape)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.Inv().mul(T_WCf)

        # 在关键帧坐标系中，计算关键帧的所有归一化的点和模长，存储在rd_K中
        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        # 迭代优化位姿
        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            # 将Xf变换到关键帧坐标系
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            # 对关键帧坐标系中的Xf_Ck归一化，并且计算模长
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            # 损失计算为方向损失和模损失
            r = rd_k - rd_f_Ck
            # 计算损失对相似变换的Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.add(tau_ij_sim3)
            T_CkCf = pp.quat2unit(T_CkCf)
            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk.mul(T_CkCf)

        return T_WCf, T_CkCf

    # 输入一一对应的点对（不同坐标系），初始估计位姿，按照关键帧像素顺序的特征匹配权重
    # 关键帧的像素坐标及其深度， 关键帧中深度大于阈值的像素mask
    # 内参矩阵和图像大小
    # 用LM算法计算相对位姿改变
    def opt_pose_calib_sim3(self, Xf, Xf_cov, Xk, Xk_cov,
                            T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, idx_f2k, img_size):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        T_CkCf = T_WCk.Inv().mul(T_WCf)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            if self.optimize_focal:
                u = idx_f2k % self.W_slam
                v = idx_f2k // self.W_slam
                uv = torch.stack([u, v], dim=-1)

                dXox_f = -(uv[..., 0] - self.K_slam[0, 2]) / (self.K_slam[0, 0] ** 2) * Xf[..., 2]
                dXoy_f = -(uv[..., 1] - self.K_slam[1, 2]) / (self.K_slam[1, 1] ** 2) * Xf[..., 2]
                dXoz_f = torch.zeros_like(dXoy_f)
                dXf_f = torch.stack([dXox_f, dXoy_f, dXoz_f], dim=-1).unsqueeze(-1)

                Xf = backproject(uv, Xf[..., 2:3], self.K_slam)
            else:
                dXf_f = torch.zeros((Xf.shape[0], 3, 1), dtype=torch.float32).to(self.device)

            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            dXf_Ck_d_Xf = T_CkCf.matrix()[:, :3, :3].expand(Xf_Ck.shape[0], -1, -1)
            # todo：此处valid_proj要用吗
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                self.K_slam,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
                dXf_Ck_d_f= (dXf_Ck_d_Xf @ dXf_f)
            )

            # 计算协方差
            if self.covariance_filter:
                R_CkCf = T_CkCf.matrix()[:, :3, :3]
                Xfk_cov = R_CkCf @ Xf_cov @ R_CkCf.transpose(-1, -2)
                X, Y, Z = Xf_Ck.unbind(-1)
                o = torch.zeros_like(X)
                JCOV = torch.stack([self.K_slam[0,0]/Z, o, -self.K_slam[0,0]*X/(Z**2),
                                 o, self.K_slam[1,1]/Z, -self.K_slam[1,1]*Y/(Z**2),
                                 o, o, 1/Z], dim=-1).reshape(-1, 3, 3)
                pixfk_cov = JCOV @ Xfk_cov @ JCOV.transpose(-1, -2)
                pixfk_cov_det = torch.det(pixfk_cov)
                valid_cov = pixfk_cov_det < max(torch.quantile(pixfk_cov_det, 0.9), 1)
                valid_cov = valid_cov[..., None]
            else:
                valid_cov = torch.ones_like(valid_meas_k)
            # weight = torch.inverse(pixfk_cov)
            # weight_diag = torch.sqrt(torch.diagonal(weight, dim1=-2, dim2=-1))
            # weight_diag = torch.diag_embed(torch.nn.functional.normalize(weight_diag, p=2, dim=0)) * 0.01
            # U = torch.diag_embed(sqrt_info)

            # try:
            #     U = torch.linalg.cholesky(weight).transpose(-1, -2)
            #     U = torch.diag_embed(sqrt_info) + U
            # except Exception as e:
            #     weight_diag = torch.sqrt(torch.diagonal(weight, dim1=-2, dim2=-1))
            #     weight_diag = torch.nn.functional.normalize(weight_diag, p=2, dim=0)
            #     U = torch.diag_embed(sqrt_info) + torch.diag_embed(weight_diag) * 0.01
            # visualize_covariance_max(Xk_cov + Xfk_cov, self.H_slam, self.W_slam, cmap="jet")

            valid2 = valid_proj & valid_meas_k & valid_cov   # 在关键帧坐标系中，关键帧对应的点云和当前帧对应的点云深度均合理且范围也合理的mask
            sqrt_info2 = valid2 * sqrt_info

            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck[..., :3] @ dXf_Ck_dT_CkCf
            if self.optimize_focal:
                J = torch.concatenate([J,-dpzf_Ck_dXf_Ck[..., 3:]], dim=-1)

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = pp.sim3(tau_ij_sim3[..., :7]).Exp().mul(T_CkCf)
            T_CkCf = pp.quat2unit(T_CkCf)

            if self.optimize_focal:
                self.K_slam[0, 0] = self.K_slam[0, 0] + tau_ij_sim3[0, -1]
                self.K_slam[1, 1] = self.K_slam[1, 1] + tau_ij_sim3[0, -1]

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3[..., :7],
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk.mul(T_CkCf)
        return T_WCf, T_CkCf
