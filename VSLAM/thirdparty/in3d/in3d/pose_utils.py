import numpy as np


def skew_sym_mat(x):
    dtype = x.dtype
    ssm = np.zeros((3, 3), dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def exp_angle_axis(angle, axis):
    dtype = axis.dtype
    # axis must be normalized
    W = skew_sym_mat(axis)
    return np.eye(3, dtype=dtype) + W * np.sin(angle) + W @ W * (1 - np.cos(angle))


def translation_matrix(t):
    T = np.eye(4, dtype=t.dtype)
    T[:3, 3] = t
    return T
