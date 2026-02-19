import torch
import pypose as pp
import torch.multiprocessing as mp

from VSLAM.ImageFrame import Mode, ImageFrame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cpu"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.lost_number = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.TRACKING)
        self.queue_F2B = manager.Queue(maxsize=100)
        self.queue_B2M = manager.Queue(maxsize=100)

        # 单线程模式下：需要进行重定位的帧的数量，只有这些帧都进行重定位之后才可以继续追踪
        # 非单线程模式下：不为0表示后端需要对当前帧进行重定位
        self.backend_execute = manager.Value("i", 0)
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.cam_id = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.timestamp = torch.zeros(1, device=device, dtype=dtype).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.T_WC = pp.identity_Sim3(1,device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1,device=device, dtype=dtype).share_memory_()


    # set the most recent frame that have been tracked
    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.cam_id[:] = frame.cam_id
            self.timestamp[:] = frame.frame_time
            self.img[:] = frame.img.to(self.device)
            self.T_WC[:] = frame.T_WC.data.to(self.device)
            self.X[:] = frame.X_canon.to(self.device)
            self.C[:] = frame.C.to(self.device)

    # return the most recent frame that have been tracked
    def get_frame(self):
        with self.lock:
            frame = ImageFrame(
                int(self.dataset_idx[0]),
                int(self.cam_id[0]),
                float(self.timestamp[0]),
                self.img,
                pp.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            return frame


    # 增加一个后端处理的帧
    def queue_backend_execute(self):
        with self.lock:
            self.backend_execute.value += 1

    # 减少一个后端处理的帧
    def dequeue_backend_execute(self):
        with self.lock:
            if self.backend_execute.value == 0:
                return
            self.backend_execute.value -= 1

    # 给后端发送一个要处理的关键帧
    def msg2Backend(self, msg):
        self.queue_F2B.put(msg)

    # 得到后端的消息
    def msgFromFrontend(self):
        return self.queue_F2B.get(block=False)

    # 给后端发送一个要处理的关键帧
    def msg2Mapper(self, msg):
        self.queue_B2M.put(msg)

    # 得到后端的消息
    def msgFromBackend(self):
        return self.queue_B2M.get(block=False)

    # get system status
    def get_mode(self):
        with self.lock:
            return self.mode.value

    # set system status
    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1