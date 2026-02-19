"""Unit tests of asmk.hamming"""

import unittest
from asmk import hamming
import numpy as np
from scipy.spatial.distance import cdist


class TestFunctions(unittest.TestCase):
    """Unit test of functions"""

    @staticmethod
    def _numpy_pack_uint32(arr):
        res = np.empty((arr.shape[0], int(np.ceil(arr.shape[1] / 32))), dtype=np.uint32)
        packed = np.packbits(arr, axis=1).astype(np.uint32)
        packed = np.pad(packed, ((0, 0), (0, 4 - packed.shape[1] % 4)), 'constant')
        for i in range(res.shape[1]):
            res[:,i] = (packed[:,4*i+0] << 24) + (packed[:,4*i+1] << 16) + (packed[:,4*i+2] << 8) + packed[:,4*i+3]
        return res

    def test_binarize_and_pack(self):
        for dim1 in range(1, 40):
            arr = (np.random.rand(dim1) - 0.5).astype(np.float32)
            self.assertTrue(np.allclose(self._numpy_pack_uint32(np.expand_dims(arr, 0) > 0).squeeze(),
                                        hamming.binarize_and_pack(arr)))

    def test_binarize_and_pack_2D(self):
        for dim1 in range(1, 40):
            arr = (np.random.rand(10, dim1) - 0.5).astype(np.float32)
            self.assertTrue(np.allclose(self._numpy_pack_uint32(arr > 0), hamming.binarize_and_pack_2D(arr)))

    def test_hamming_dist_packed(self):
        for dim1 in range(100, 140):
            arr1 = (np.random.rand(1, dim1) - 0.5).astype(np.float32)
            arr2 = (np.random.rand(1, dim1) - 0.5).astype(np.float32)
            res = hamming.hamming_dist_packed(hamming.binarize_and_pack_2D(arr1).squeeze(0),
                                              hamming.binarize_and_pack_2D(arr2).squeeze(0), dim1)
            if dim1 % 32 == 0:
                # Test default behaviour
                res = hamming.hamming_dist_packed(hamming.binarize_and_pack_2D(arr1).squeeze(0),
                                                  hamming.binarize_and_pack_2D(arr2).squeeze(0))
            self.assertTrue(np.allclose(res, cdist(arr1 > 0, arr2 > 0, 'hamming').squeeze()))

    def test_hamming_cdist_packed(self):
        for dim1 in range(100, 140):
            arr1 = (np.random.rand(10, dim1) - 0.5).astype(np.float32)
            arr2 = (np.random.rand(10, dim1) - 0.5).astype(np.float32)
            if dim1 % 32 == 0:
                # Test default behaviour
                res = hamming.hamming_cdist_packed(hamming.binarize_and_pack_2D(arr1),
                                                   hamming.binarize_and_pack_2D(arr2))
            res = hamming.hamming_cdist_packed(hamming.binarize_and_pack_2D(arr1),
                                               hamming.binarize_and_pack_2D(arr2), dim1)
            self.assertTrue(np.allclose(res, cdist(arr1 > 0, arr2 > 0, 'hamming')))
