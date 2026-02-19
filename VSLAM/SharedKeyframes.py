from collections import namedtuple
from queue import Empty
from typing import Optional

import pypose as pp
import torch

from VSLAM.ImageFrame import ImageFrame
import torch.multiprocessing as mp

DensePoint = namedtuple('DensePoint', ['index', 'point_map', 'point_conf'])

class SharedKeyframes:
    def __init__(self, config, manager, h, w, K_slam,
                 buffer=2048, dtype=torch.float32, device="cpu"):

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device
        self.config = config

        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        # property for keyframe
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.cam_id = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.timestamp = torch.zeros(buffer, device=device, dtype=torch.float64).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.T_WC = pp.identity_Sim3(buffer, 1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(buffer, h * w, 1,  device=device, dtype=dtype).share_memory_()
        ## point update
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        ## dense point for mapper
        self.densePoint = torch.zeros(buffer, h, w, 4, device=device, dtype=dtype).share_memory_()
        self.ready_for_map = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()

        if config["use_calib"]:
            self.set_intrinsics(K_slam)

    # def add_test_frame(self, idx, T_WC, densePoint):
    #     index = self.test_index.value
    #     self.test_idx[index] = idx
    #     self.test_T_WC[index] = T_WC.data.to(self.device)
    #     self.test_densePoint[index] = densePoint.to(self.device)
    #     self.test_index.value = index + 1

    def is_ready_for_map(self, index):
        return self.ready_for_map[index]

    def put_DensePoint(self, index, densePoint):
        densePoint.to(self.device)
        self.densePoint[index] = densePoint
        self.ready_for_map[index] = True

    def get_DensePoint(self, index):
        return self.densePoint[index]

    def __getitem__(self, idx) -> ImageFrame:
        with self.lock:
            # put all of the data into a frame
            kf = ImageFrame(
                int(self.dataset_idx[idx]),
                int(self.cam_id[idx]),
                float(self.timestamp[idx]),
                self.img[idx],
                pp.Sim3(self.T_WC[idx])
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if self.config["use_calib"]:
                kf.K = self.K
            return kf

    def __setitem__(self, idx, value: ImageFrame) -> None:

        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)
            self.dataset_idx[idx] = value.frame_id
            self.cam_id[idx] = value.cam_id
            self.timestamp[idx] = value.frame_time
            self.img[idx] = value.img.to(self.device)
            self.T_WC[idx] = value.T_WC.data.to(self.device)
            self.X[idx] = value.X_canon.to(self.device)
            self.C[idx] = value.C.to(self.device)
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: ImageFrame):
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[ImageFrame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            self.T_WC[idx] = T_WCs.data.to(self.device)

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        assert self.config["use_calib"]
        with self.lock:
            self.K[:] = K.to(self.device)

    def get_intrinsics(self):
        assert self.config["use_calib"]
        with self.lock:
            return self.K
