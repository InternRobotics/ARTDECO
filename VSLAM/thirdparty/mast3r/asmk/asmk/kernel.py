"""Kernel functionality implementation - aggregation and similarity computation"""

import numpy as np

from . import functional, hamming, io_helpers


class ASMKKernel:
    """Kernel for ASMK with the option of binarization."""

    binary_shortcuts = {"bin": True, "nobin": False}

    def __init__(self, codebook, *, binary):
        self.params = {
            "binary": binary,
        }
        self.binary = self.binary_shortcuts.get(binary, binary)
        assert self.binary in self.binary_shortcuts.values()

        self.codebook = codebook

    #
    # Aggregation
    #

    def aggregate_image(self, des, word_ids):
        """Aggregate descriptors (with corresponding visual word ids) for a single image"""
        unique_ids = np.unique(word_ids)
        ades = np.empty((unique_ids.shape[0], des.shape[1]), dtype=np.float32)

        for i, word in enumerate(unique_ids):
            ades[i] = (des[(word_ids==word).any(axis=1)] - self.codebook.centroids[word]).sum(0)

        if self.binary:
            ades = hamming.binarize_and_pack_2D(ades)
        else:
            ades = functional.normalize_vec_l2(ades)

        return ades, unique_ids

    def aggregate(self, des, word_ids, image_ids, *, progress=None, **kwargs):
        """Aggregate descriptors with corresponding visual word ids for corresponding image ids"""
        acc = []
        slices = list(io_helpers.slice_unique(image_ids))
        for imid, seq in io_helpers.progress(slices, frequency=progress, header="Aggregate"):
            ades, ids = self.aggregate_image(des[seq], word_ids[seq], **kwargs)
            acc.append((ades, ids, np.full(ids.shape[0], imid)))

        agg_des, agg_words, agg_imids = zip(*acc)
        return np.vstack(agg_des), np.hstack(agg_words), np.hstack(agg_imids)

    #
    # Similarity
    #

    def similarity(self, qvec, vecs, image_ids, *, alpha, similarity_threshold):
        """Compute similarity between given query vector and database feature vectors with their
            corresponding image ids. Alpha is the similarity exponent after the similarity
            threshold is applied."""
        # Compute similarity with vw residuals for all other images
        if self.binary:
            norm_hdist = hamming.hamming_cdist_packed(qvec.reshape(1, -1), vecs)
            sim = -2*norm_hdist.squeeze(0) + 1 # normalized hamming dist -> similarity in [-1, 1]
        else:
            sim = np.matmul(vecs, qvec)

        return functional.asmk_kernel(sim, image_ids, alpha=alpha,
                                      similarity_threshold=similarity_threshold)

    #
    # Load and save
    #

    def state_dict(self):
        """Return state dict which is a checkpoint of current state for future recovery"""
        return {
            "type": self.__class__.__name__,
            "params": self.params,
        }

    @classmethod
    def initialize_from_state(cls, state, codebook):
        """Initialize from a previously stored state_dict given a codebook"""
        assert state["type"] == cls.__name__
        return cls(**state["params"], codebook=codebook)
