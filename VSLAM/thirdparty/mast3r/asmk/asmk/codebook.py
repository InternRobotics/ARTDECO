"""Codebook implementations for quantization of features into visual words"""

import time
import warnings
import numpy as np


class Codebook:
    """Codebook of a fixed size for feature quantization"""

    def __init__(self, index_factory, *, size):
        self.params = {
            "size": size,
        }
        if isinstance(size, str) and size[-1] in "kM":
            size = int(size[:-1]) * {"k": 1024, "M": 1024**2}[size[-1]]
        self.size = size
        assert isinstance(self.size, int), self.size

        self.index_factory = index_factory
        self.search_index = None
        self.centroids = None

    #
    # Create index
    #

    def index(self, centroids=None):
        """Index either provided or stored centroids (when centroids=None). Return a dictionary
            with 'index' key where value is how long it took to index the centroids."""
        if centroids is not None:
            assert self.size == centroids.shape[0], (self.size, centroids.shape[0])
            self.centroids = centroids
        time0 = time.time()
        self.search_index = self.index_factory.create_index(self.centroids)
        return {"index_time": time.time() - time0}

    def train(self, des):
        """Cluster descriptors and index resulting cluster centers. Return a dictionary with
            'cluster', 'index' and 'train' keys where the value is how long it took to cluster,
            index or train (sum of all)."""
        time0 = time.time()
        centroids = self.index_factory.cluster(des, self.size)
        time_taken = time.time() - time0
        meta = self.index(centroids)
        return {**meta, "cluster_time": time_taken, "train_time": sum(meta.values()) + time_taken}

    #
    # Search in index
    #

    def quantize(self, des, *cols, multiple_assignment):
        """Quantize given descriptors. Additional cols can be given, and this function will make
            sure, that they still correspond to returned descriptors. Multiple assignment can be
            applied when multiple_assignment > 1."""
        _, centroid_ids = self.search_index.search(np.ascontiguousarray(des, dtype=np.float32),
                                                   multiple_assignment)

        return (des, centroid_ids) + cols

    #
    # Load and save
    #

    def state_dict(self):
        """Return state dict which is a checkpoint of current state for future recovery"""
        if self.centroids is None:
            warnings.warn("Returning state_dict when codebook is not indexed (meaningless)")

        return {
            "type": self.__class__.__name__,
            "params": self.params,
            "state": {
                "centroids": self.centroids,
            }
        }

    @classmethod
    def initialize_from_state(cls, state, index_factory):
        """Initialize from a previously stored state_dict given an index factory"""
        assert state["type"] == cls.__name__
        codebook = cls(**state["params"], index_factory=index_factory)
        if state["state"]["centroids"] is not None:
            codebook.index(state["state"]["centroids"])
        return codebook
