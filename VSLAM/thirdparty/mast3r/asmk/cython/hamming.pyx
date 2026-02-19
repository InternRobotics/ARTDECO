# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport cython
from libc.math cimport ceil


cdef unsigned int BIT_MASK_1 = 0x55555555
cdef unsigned int BIT_MASK_2 = 0x33333333
cdef unsigned int BIT_MASK_4 = 0x0f0f0f0f
cdef unsigned int BIT_MASK_8 = 0x00ff00ff
cdef unsigned int BIT_MASK_16 = 0x0000ffff


cdef int c_count_bits(unsigned int n) nogil:
    n = (n & BIT_MASK_1) + ((n >> 1) & BIT_MASK_1)
    n = (n & BIT_MASK_2) + ((n >> 2) & BIT_MASK_2)
    n = (n & BIT_MASK_4) + ((n >> 4) & BIT_MASK_4)
    n = (n & BIT_MASK_8) + ((n >> 8) & BIT_MASK_8)
    n = (n & BIT_MASK_16) + ((n >> 16) & BIT_MASK_16)
    return n


cdef unsigned int c_binarize_and_pack_uint32(float[::1] arr, Py_ssize_t length, int threshold) nogil:
    cdef unsigned int tmp = 0
    cdef Py_ssize_t i

    for i in range(length):
        tmp = (tmp << 1) + (arr[i] > threshold)
    return tmp


@cython.cdivision(True)
cdef float c_hamming_dist_uint32_arr(unsigned int[::1] n1, unsigned int[::1] n2, float normalization) nogil:
    cdef Py_ssize_t length = n1.shape[0]
    if normalization == 0:
        normalization = length * 32

    cdef int sum = 0
    for i in range(length):
        sum += c_count_bits(n1[i] ^ n2[i])
    return sum / normalization


#
# Python API
#

def binarize_and_pack(float[::1] arr, int threshold = 0):
    """
    binarize_and_pack(float[::1] arr, int threshold = 0)

    Binarizes given 1D numpy array by 'arr = arr > threshold' and packs its elements into bits
    in uint32 array. Returns a 1D uint32 array where each element corresponds to a started set
    of 32 bits.

    >> binarize_and_pack((np.random.rand(10) - 0.5).astype(np.float32))
    array([2206203904], dtype=uint32)
    """
    cdef Py_ssize_t dim_orig = arr.shape[0]
    cdef Py_ssize_t dim = int(ceil(dim_orig / 32.0))
    result = np.zeros(dim, dtype=np.uint32)
    cdef unsigned int[::1] result_view = result

    cdef Py_ssize_t i, offset
    cdef unsigned int tmp
    offset = 0
    for i in range(dim-1):
        result_view[i] = c_binarize_and_pack_uint32(arr[offset:], 32, threshold)
        offset += 32

    # Last iteration
    tmp = c_binarize_and_pack_uint32(arr[offset:], dim_orig - offset, threshold)
    result_view[dim-1] = tmp << (offset + 32 - dim_orig)

    return result


def binarize_and_pack_2D(float[:,::1] arr, int threshold = 0):
    """
    binarize_and_pack_2D(float[:,::1] arr, int threshold = 0)

    Binarizes given 2D numpy array by 'arr = arr > threshold' and packs its elements into bits
    in uint32 array. Returns a 2D uint32 array where each row corresponds to row in the original
    array and each element to a started set of 32 bits.

    >> binarize_and_pack_2D((np.random.rand(2, 10) - 0.5).astype(np.float32))
    array([[1786773504]
           [1509949440]], dtype=uint32)
    """
    cdef Py_ssize_t dim0 = arr.shape[0]
    cdef Py_ssize_t dim1_orig = arr.shape[1]
    cdef Py_ssize_t dim1 = int(ceil(dim1_orig / 32.0))
    result = np.zeros((dim0, dim1), dtype=np.uint32)
    cdef unsigned int[:, ::1] result_view = result

    cdef Py_ssize_t i, j, offset
    cdef unsigned int tmp
    for i in range(dim0):
        offset = 0
        for j in range(dim1-1):
            result_view[i,j] = c_binarize_and_pack_uint32(arr[i][offset:], 32, threshold)
            offset += 32

        # Last iteration
        tmp = c_binarize_and_pack_uint32(arr[i][offset:], dim1_orig - offset, threshold)
        result_view[i,dim1-1] = tmp << (offset + 32 - dim1_orig)

    return result


def hamming_dist_packed(unsigned int[::1] n1, unsigned int[::1] n2, float normalization = 0):
    """
    hamming_dist_packed(unsigned int[::1] n1, unsigned int[::1] n2, float normalization = 0)

    Computes a hamming distance between two bit arrays packed into uint32 arrays and divides
    it by normalization, if provided, otherwise by the number of bits in an array (always
    a multiplication of 32).

    >> hamming_dist_packed(np.array([3], dtype=np.uint32), np.array([1], dtype=np.uint32), 2)
    0.5
    """
    assert n1 is not None and n2 is not None
    assert n1.shape[0] == n2.shape[0]
    return c_hamming_dist_uint32_arr(n1, n2, normalization)


def hamming_cdist_packed(unsigned int[:,::1] arr1, unsigned int[:,::1] arr2, float normalization = 0):
    """
    hamming_cdist_packed(unsigned int[:,::1] arr1, unsigned int[:,::1] arr2, float normalization = 0)

    Computes a hamming distance between two sets of bit arrays packed into uint32 using
    hamming_dist_packed. Returns an array of size (arr1.shape[0], arr2.shape[0]).

    >> hamming_cdist_packed(np.array([[3], [1]], dtype=np.uint32), np.array([[1], [2]], dtype=np.uint32), 2)
    array([[0.5, 0.5],
           [0. , 1. ]], dtype=float32)
    """
    assert arr1 is not None and arr2 is not None
    assert arr1.shape[1] == arr2.shape[1]

    cdef Py_ssize_t dim0 = arr1.shape[0]
    cdef Py_ssize_t dim1 = arr2.shape[0]
    result = np.zeros((dim0, dim1), dtype=np.float32)
    cdef float[:, ::1] result_view = result

    cdef Py_ssize_t i, j
    for i in range(dim0):
        for j in range(dim1):
            result_view[i, j] = c_hamming_dist_uint32_arr(arr1[i], arr2[j], normalization)

    return result
