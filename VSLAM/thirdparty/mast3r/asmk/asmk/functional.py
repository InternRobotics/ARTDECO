"""Mathematical functions operating on arrays"""

import numpy as np


def normalize_vec_l2(vecs):
    """Perform l2 normalization on each vector in a given matrix (axis 1)"""
    norm = np.linalg.norm(vecs, ord=2, axis=1, keepdims=True) + 1e-6
    return vecs / norm

def asmk_kernel(sim, image_ids, *, alpha, similarity_threshold):
    """Compute scores for visual words"""
    mask = (sim>=similarity_threshold)
    sim = np.power(sim[mask], alpha) # monomial kernel
    return image_ids[mask], sim
