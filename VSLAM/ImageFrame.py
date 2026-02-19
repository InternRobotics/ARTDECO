import dataclasses
from enum import Enum
from typing import Optional
import torch
import pypose as pp

class Mode(Enum):
    TRACKING = 0
    RELOC = 1
    OPTIMIZING = 2
    TERMINATED = 3



@dataclasses.dataclass
class ImageFrame:
    frame_id: int # index
    cam_id: int
    frame_time: float
    img: torch.Tensor                                      # CHW [-1, 1]
    T_WC: pp.Sim3 = pp.identity_Sim3(1)                    # 1

    X_canon: Optional[torch.Tensor] = None                 # N 3
    C: Optional[torch.Tensor] = None                       # N 1
    N: int = 0                                             # 当前的点云是由多少个预测组成的
    N_updates: int = 0                                     # update_pointmap被調用了幾次
    K: Optional[torch.Tensor] = None                       # intrinsics 3 3

    # 更新当前帧的点云
    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        """
            X torch.Tensor N,3
            C torch.Tensor N
        """
        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            return

        self.X_canon = (((self.C * self.X_canon) + (C * X))
                        / (self.C + C))
        self.C = self.C + C
        self.N += 1
        self.N_updates += 1
        return

    # 返回当前X_canon中每个点的预测置信度
    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None

    def clear(self):
        del self.img
        del self.X_canon
        del self.C
        del self.N
        del self.N_updates
        del self.K

    def to(self, device):
        """将所有tensor和sim3变换移动到指定设备"""
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                # 处理torch.Tensor
                if isinstance(field_value, torch.Tensor):
                    setattr(self, field_name, field_value.to(device))
                # 处理其他有to方法的对象
                elif hasattr(field_value, 'to') and callable(getattr(field_value, 'to')):
                    setattr(self, field_name, field_value.to(device))
        return self

    @property
    def device(self):
        return self.img.device

