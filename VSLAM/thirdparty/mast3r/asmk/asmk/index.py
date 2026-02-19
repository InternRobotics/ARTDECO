"""Index factories for efficient search (clustering)"""

import numpy as np
import faiss


class FaissL2Index:
    """Faiss factory for indexes on cpu"""

    @staticmethod
    def _faiss_index_flat(dim):
        """Return initialized faiss.IndexFlatL2"""
        return faiss.IndexFlatL2(dim)

    def cluster(self, points, k, **index_kwargs):
        """Clustering given points into k clusters"""
        index = self._faiss_index_flat(points.shape[1], **index_kwargs)
        clus = faiss.Clustering(points.shape[1], k)
        clus.verbose = False
        clus.niter = 10
        clus.train(np.ascontiguousarray(points, dtype=np.float32), index)
        return faiss.vector_float_to_array(clus.centroids).reshape(clus.k, clus.d)

    def create_index(self, points, **index_kwargs):
        """Return faiss index with given points"""
        index = self._faiss_index_flat(points.shape[1], **index_kwargs)
        index.add(points)
        return index


class FaissGpuL2Index(FaissL2Index):
    """Faiss factory for indexes on gpu"""

    def __init__(self, gpu_id):
        super().__init__()
        self.gpu_id = gpu_id

    def _faiss_index_flat(self, dim):
        """Return initialized faiss.GpuIndexFlatL2"""
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = self.gpu_id
        return faiss.GpuIndexFlatL2(res, dim, flat_config)


def initialize_index(gpu_id):
    """Return either gpu faiss factory or cpu faiss factory (gpu_id is None)"""
    if gpu_id is not None:
        return FaissGpuL2Index(gpu_id)

    return FaissL2Index()
