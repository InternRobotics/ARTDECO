import torch
import numpy as np

from safetensors.torch import load_file
from asmk import io_helpers

from VSLAM.thirdparty.Pi3.pi3.models.pi3 import Pi3
from VSLAM.thirdparty.Pi3.pi3.utils.basic import write_ply
from VSLAM.thirdparty.mast3r.mast3r.retrieval.model import how_select_local
from VSLAM.thirdparty.mast3r.mast3r.retrieval.processor import Retriever

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F

from VSLAM.utils_matching import match_pi3

def inverse_normalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    restored_img = tensor.clamp(0, 1)  # 确保值在 [0, 1] 范围内
    restored_img = restored_img.permute(1, 2, 0)
    return restored_img


def resize_image(image, target_height, target_width):
    """
    image: torch.Tensor shape [N, C, H, W]
    target_height: int H1
    target_width: int W1
    返回: torch.Tensor shape [N, C, H1, W1]
    """
    resized = F.interpolate(image,
                           size=(target_height, target_width),
                           mode='bilinear',
                           align_corners=False)
    return resized

class SimilarityGraph:
    def __init__(self):
        self.similarities = defaultdict(dict)  # {frame_id: {other_id: similarity}}

    def add_similarity(self, frame_id1, frame_id2, similarity):
        """添加两帧之间的相似度"""
        self.similarities[frame_id1][frame_id2] = similarity
        self.similarities[frame_id2][frame_id1] = similarity

    def remove_frame(self, frame_id):
        """删除指定帧及其所有相关的相似度记录"""
        # 删除以frame_id为key的所有记录
        if frame_id in self.similarities:
            del self.similarities[frame_id]

        # 删除其他帧中包含frame_id的记录
        for other_id in self.similarities:
            if frame_id in self.similarities[other_id]:
                del self.similarities[other_id][frame_id]

    def get_similar_frames_sorted_dfs(self, frame_id):
        """递归查找所有相似帧，按相似度降序返回"""
        visited = set()
        all_similarities = {}

        def dfs(current_id):
            if current_id in visited:
                return
            visited.add(current_id)

            # 收集当前帧的所有相似度
            for other_id, sim in self.similarities[current_id].items():
                if other_id not in all_similarities:
                    all_similarities[other_id] = sim
                else:
                    # 保留最高相似度
                    all_similarities[other_id] = max(all_similarities[other_id], sim)

                # 递归访问相似帧
                dfs(other_id)

        dfs(frame_id)

        # 移除查询帧本身
        all_similarities.pop(frame_id, None)

        # 按相似度降序排序返回帧ID
        return [frame_id for frame_id, _ in sorted(all_similarities.items(),
                                                   key=lambda x: x[1], reverse=True)]


    def get_similar_frames_sorted(self, frame_id):
        if frame_id not in self.similarities:
            return []

        # 按相似度降序排序返回帧ID
        return [other_id for other_id, _ in sorted(self.similarities[frame_id].items(),
                                                   key=lambda x: x[1], reverse=True)]

    def plot_similarity_heatmap(self, show_label=False):
        """生成并展示帧间相似度热力图"""
        # 获取所有帧ID
        all_frames = set()
        for frame_id in self.similarities:
            all_frames.add(frame_id)
            all_frames.update(self.similarities[frame_id].keys())

        frame_list = sorted(list(all_frames))
        n = len(frame_list)

        # 创建相似度矩阵
        max_value = 0
        similarity_matrix = np.zeros((n, n))
        for i, frame1 in enumerate(frame_list):
            for j, frame2 in enumerate(frame_list):
                if frame1 == frame2:
                    similarity_matrix[i][j] = 100.0  # 自身相似度为1
                elif frame2 in self.similarities[frame1]:
                    similarity_matrix[i][j] = self.similarities[frame1][frame2]
                    if max_value < similarity_matrix[i][j]:
                        max_value = similarity_matrix[i][j]
        for i in range(n):
            similarity_matrix[i][i] = max_value * 1.5

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix,
                    xticklabels=frame_list if show_label else False,
                    yticklabels=frame_list if show_label else False,
                    annot=False,  # 显示数值
                    cmap='YlOrRd',  # 颜色映射
                    fmt='.2f',  # 数值格式
                    cbar_kws={'label': 'Similarity'})

        plt.title('Frame Similarity Heatmap')
        plt.xlabel('Frame ID')
        plt.ylabel('Frame ID')
        plt.tight_layout()
        plt.show()

class RetrievalDatabase(Retriever):
    def __init__(self, args, config, modelname, keyframes, backbone=None, device="cuda"):
        super().__init__(modelname, backbone, device)
        self.keyframes = keyframes
        self.config = config
        self.args = args
        # 生成用于倒排查询的对象，后续可以送入特征进行查询
        self.ivf_builder = self.asmk.create_ivf_builder()

        self.min_window_number = 12
        self.max_window_number = 24
        self.accurate_loop_closure_number = 12
        self.min_match_factor = 15
        self.is_accurate_loop_closure = self.args.accurate_loop_closure

        self.kf_counter = 0   # 当前库中有几个关键帧
        self.kf_ids = []  # 关键帧的id，其从0开始 0 1 2 3 4...，为此类的loca id

        self.query_dtype = torch.float32
        self.query_device = device
        self.centroids = torch.from_numpy(self.asmk.codebook.centroids).to(
            device=self.query_device, dtype=self.query_dtype
        ) # 聚类中心
        self.sim_Graph = SimilarityGraph()

        self.model_pi3 = Pi3().to(device).eval()
        weight = load_file("./models/model.safetensors")
        self.model_pi3.load_state_dict(weight)


    # 抽取当前关键帧的视觉特征
    # Mirrors forward_local in extract_local_features from retrieval/model.py
    def prep_features(self, backbone_feat):
        retrieval_model = self.model

        # 1. 对编码特征进行后处理
        # extract_features_and_attention without the encoding!
        # 对编码器输出的特征进行白化
        backbone_feat_prewhitened = retrieval_model.prewhiten(backbone_feat)
        # 将白化后的特征放入全连接head头并加残差（实际上这一步head为单位矩阵，也没有加残差）
        proj_feat = retrieval_model.projector(backbone_feat_prewhitened) + (
            0.0 if not retrieval_model.residual else backbone_feat_prewhitened
        )
        # 对其最后一维求模 lambda x: x.norm(dim=-1)
        attention = retrieval_model.attention(proj_feat)
        # 后白化处理，单位映射
        proj_feat_whitened = retrieval_model.postwhiten(proj_feat)

        # 2. # 根据attention的分数从proj_feat_whitened中选出nfeat个分数最高的视觉特征，作为代表性特征
        # how_select_local in
        topk_features, _, _ = how_select_local(
            proj_feat_whitened, attention, retrieval_model.nfeat
        )

        return topk_features

    # 增加当前帧的特征到库中，并且查询当前帧的特征与哪些帧有关
    def update(self, feat, add_after_query, k, min_thresh=0.0):
        device = feat.device
        if device != self.query_device:
            feat = feat.to(device=self.query_device)

        # 生成图像的特征
        feat = self.prep_features(feat)
        id = self.kf_counter  # Using own counter since otherwise messes up IVF
        feat_np = feat[0].cpu().numpy()  # Assumes one frame at a time! 300*1024
        id_np = id * np.ones(feat_np.shape[0], dtype=np.int64)

        # 查询库中已经有多少关键帧
        database_size = self.ivf_builder.ivf.n_images

        # Only query if already an image
        topk_image_inds = []
        topk_codes = None  # Change this if actualy querying
        if self.kf_counter > 0:
            # 获得相关的关键帧local id 分数
            ranks, ranked_scores, topk_codes = self.query(feat_np, id_np)
            scores = np.empty_like(ranked_scores)
            scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
            scores = torch.from_numpy(scores)[0]

            # 排序，取前k个
            for i in range(database_size):
                self.sim_Graph.add_similarity(database_size, i, scores[i].item() * 100)
            # self.sim_Graph.plot_similarity_heatmap(False)
            if ((database_size < self.min_window_number) and add_after_query) or ( not self.is_accurate_loop_closure):
                topk_images = torch.topk(scores, min(k, database_size))
                valid = topk_images.values > min_thresh
                topk_image_inds = topk_images.indices[valid]
                topk_image_inds = topk_image_inds.tolist()
            else:
                topk_images = torch.topk(scores, min(k, database_size))
                valid = topk_images.values > min_thresh
                topk_image_inds_retrival = topk_images.indices[valid]
                topk_image_inds_retrival = topk_image_inds_retrival.tolist()

                if len(topk_image_inds_retrival) <= 0:
                    need_accurate_loop_closure = True
                else:
                    need_accurate_loop_closure = (database_size - min(topk_image_inds_retrival)) > self.accurate_loop_closure_number
                if not add_after_query:
                    need_accurate_loop_closure = True

                if need_accurate_loop_closure or not add_after_query:
                    topk_image_inds = self.accurate_loop_closure(database_size)
                    print(topk_image_inds_retrival, "->", topk_image_inds)
                else:
                    topk_image_inds = topk_image_inds_retrival
                # masks = torch.sigmoid(res['conf'][..., 0])[0] > 0.1
                # write_ply(res['points'][0][masks].cpu(), images.permute(0, 2, 3, 1)[masks], "test.ply")

            if not add_after_query:
                self.sim_Graph.remove_frame(database_size)

        # 将当前帧加入到库中
        if add_after_query:
            self.add_to_database(feat_np, id_np, topk_codes)

        return topk_image_inds

    def accurate_loop_closure(self, keyframe_id):
        related_indexs = self.sim_Graph.get_similar_frames_sorted(keyframe_id)
        selected_indexs = related_indexs[:self.max_window_number - 1]

        idxs_all = selected_indexs + [keyframe_id]

        images = [inverse_normalize(self.keyframes[index].img).permute(2, 0, 1) for index in idxs_all]
        images = torch.stack(images, dim=0)
        images = resize_image(images, 392, 518).to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            res = self.model_pi3(images[None].contiguous())
        points = res["points"][0]

        ii = []
        jj = []
        for i in range(len(idxs_all) - 1):
            ii.append(idxs_all[i])
            jj.append(idxs_all[-1])

        ii_indices = [idxs_all.index(x) for x in ii]
        pairs_ii = points[ii_indices]
        jj_indices = [idxs_all.index(x) for x in jj]
        pairs_jj = points[jj_indices]

        pairs = torch.stack([pairs_ii, pairs_jj], dim=1)
        idx_i2j, valid_match_j = self.process_pairs_in_chunks(pairs)

        match_percentage = valid_match_j.sum(-1) / valid_match_j[0].numel()
        loop_idxs = torch.argsort(match_percentage, descending=True)
        loop_mask = match_percentage[loop_idxs] > self.config["retrieval"]["accurate_min"]
        valid_loop_idx = loop_idxs[loop_mask]
        topk_image_inds = [ii[i] for i in valid_loop_idx.tolist()]
        # topk_image_inds = list(sorted(topk_image_inds))
        # if keyframe_id-1 in topk_image_inds:
        #     topk_image_inds.remove(keyframe_id-1)

        return topk_image_inds[:self.config["retrieval"]["k"]]

    def process_pairs_in_chunks(self, pairs, chunk_size=32):
        """分块处理pairs以减少内存消耗"""
        total_pairs = pairs.shape[0]
        all_idx_120 = []
        all_valid_match = []

        for i in range(0, total_pairs, chunk_size):
            end_idx = min(i + chunk_size, total_pairs)
            chunk_pairs = pairs[i:end_idx]

            if chunk_pairs.shape[0] == 0:
                continue

            # 处理当前chunk
            chunk_idx_120, chunk_valid_match = match_pi3(self.config, chunk_pairs[:, 0], chunk_pairs[:, 1],
                                                         None)

            all_idx_120.append(chunk_idx_120)
            all_valid_match.append(chunk_valid_match)

        # 拼接所有结果
        idx_120 = torch.cat(all_idx_120, dim=0)
        valid_match = torch.cat(all_valid_match, dim=0)

        return idx_120, valid_match

    # 查询当前视觉特征最可能属于的图片及其打分，并且返回当前视觉特征最可能属于的前五个聚类中心
    # The reason we need this function is becasue kernel and inverted file not defined when manually updating ivf_builder
    def query(self, feat, id):
        step_params = self.asmk.params.get("query_ivf")

        images2, ranks, scores, topk = self.accumulate_scores(
            self.asmk.codebook,
            self.ivf_builder.kernel,
            self.ivf_builder.ivf,
            feat,
            id,
            params=step_params,
        )

        return ranks, scores, topk

    # 将当前frame的特征添加到库中
    def add_to_database(self, feat_np, id_np, topk_codes):
        self.add_to_ivf_custom(feat_np, id_np, topk_codes)

        # Bookkeeping
        self.kf_ids.append(id_np[0])
        self.kf_counter += 1

    # 计算视觉特征与聚类中心的距离，从而计算相似度，得到前k个相似的聚类中心
    # 输入n c
    # 输出n k
    def quantize_custom(self, qvecs, params):
        # Using trick for efficient distance matrix
        # (a-b)^2
        l2_dists = (
            torch.sum(qvecs**2, dim=1)[:, None]
            + torch.sum(self.centroids**2, dim=1)[None, :]
            - 2 * (qvecs @ self.centroids.mT)
        )
        k = params["quantize"]["multiple_assignment"]
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        return topk.indices

    # 付输入的 qvecs进行查询，查询qvecs代表的图片最可能属于哪些图片
    # 返回查询关键帧id 可能属于的帧id 相关分数 以及查询关键帧对应的每个qvecs对应的可能的五个聚类中心
    def accumulate_scores(self, cdb, kern, ivf, qvecs, qimids, params):
        """Accumulate scores for every query image (qvecs, qimids) given codebook, kernel,
        inverted_file and parameters."""
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])
        acc = []
        slices = list(io_helpers.slice_unique(qimids))
        # 对每一幅图像中的特征进行查询，每次循环输入n c，得到查询结果n k
        # id index
        for imid, seq in slices:
            # 计算这些特征向量最可能属于的额前五个聚类中心
            # Calculate qvecs to centroids distance matrix (without forming diff!)
            qvecs_torch = torch.from_numpy(qvecs[seq]).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, params)
            topk_inds = topk_inds.cpu().numpy()
            # todo: 此处应该取 qvecs[seq]
            quantized = (qvecs, topk_inds)

            # 可能有多个视觉向量属于一个聚类中心，此函数规定了这些向量怎么聚合到一个或者多个代表性向量
            aggregated = kern.aggregate_image(*quantized, **params["aggregate"])

            # 根据这些特征向量及其最可能属于的聚类中心，逆序查询当前关键帧图片最可能关联哪些关键帧图片
            ranks, scores = ivf.search(
                *aggregated, **params["search"], similarity_func=similarity_func
            )
            acc.append((imid, ranks, scores, topk_inds))
            # print(imid, ranks, scores, topk_inds)

        imids_all, ranks_all, scores_all, topk_all = zip(*acc)

        return (
            np.array(imids_all),
            np.vstack(ranks_all),
            np.vstack(scores_all),
            np.vstack(topk_all),
        )

    # 添加当前关键帧到库中取
    def add_to_ivf_custom(self, vecs, imids, topk_codes=None):
        """Add descriptors and cooresponding image ids to the IVF

        :param np.ndarray vecs: 2D array of local descriptors
        :param np.ndarray imids: 1D array of image ids
        :param bool progress: step at which update progress printing (None to disable)
        """
        ivf_builder = self.ivf_builder
        step_params = self.asmk.params.get("build_ivf")

        # 得到当前图片的每个特征对应的聚类中心的前multiple_assignment个id
        if topk_codes is None:
            qvecs_torch = torch.from_numpy(vecs).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, step_params)
            topk_inds = topk_inds.cpu().numpy()
        else:
            # Reuse previously calculated! Only take top 1
            # NOTE: Assuming build params multiple assignment is less than query
            k = step_params["quantize"]["multiple_assignment"]
            topk_inds = topk_codes[:, :k]

        # 添加
        quantized = (vecs, topk_inds, imids)
        aggregated = ivf_builder.kernel.aggregate(
            *quantized, **ivf_builder.step_params["aggregate"]
        )
        ivf_builder.ivf.add(*aggregated)
