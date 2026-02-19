#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import cv2
import os
from argparse import Namespace
from pathlib import Path

from dataloaders.DatasetBasic import BaseDataset
from Reconstruct.utils import get_image_names


# 提取数字部分进行排序
def extract_number(filename):
    return int(filename.split('_')[-1].replace('.png', ''))


class SelfCapturedDataset(BaseDataset):
    """
    The main dataset class for loading images from disk in a multithreaded manner.
    It also supports loading masks and COLMAP poses if available.
    The next image can be fetched using the `getnext` method.
    """

    def __init__(self, args: Namespace):

        # load images paths and subsample image
        self.image_dir = os.path.join(args.source_path, args.images_dir)
        self.image_name_list = get_image_names(self.image_dir)
        self.image_name_list.sort()
        print(f"Found {len(self.image_name_list)} images")
        self.timestamp = []
        for i, name in enumerate(self.image_name_list):
            self.timestamp.append(float(Path(name).stem))
        self.Twc_gt = None

        super().__init__(args)

