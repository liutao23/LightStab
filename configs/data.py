# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : data.py
# @Time  : 2025/10/19 下午6:40

import numpy as np
import torch
import os
from torch.utils.data import Dataset

class MotionData(Dataset):
    def __init__(self, path, type=0, maxlength=30, datatype='all',
                 flow_file_train="train_op_all.npy",
                 kp_file_train="train_kp_all.npy",
                 flow_file_valid="valid_op_all.npy",
                 kp_file_valid="valid_kp_all.npy",
                 topk=None,
                 mmap=True):
        super().__init__()
        self.root = path
        self.length = maxlength
        self.datatype = datatype
        self.topk = topk

        if type == 0:
            flow_fp = os.path.join(self.root, flow_file_train)
            kp_fp   = os.path.join(self.root, kp_file_train)
        else:
            flow_fp = os.path.join(self.root, flow_file_valid)
            kp_fp   = os.path.join(self.root, kp_file_valid)

        mmap_mode = 'r' if mmap else None
        self.of = np.load(flow_fp, allow_pickle=True, mmap_mode=mmap_mode)
        self.kp = np.load(kp_fp,   allow_pickle=True, mmap_mode=mmap_mode)

        self.flat = False
        if isinstance(self.of, np.ndarray) and self.of.ndim == 4:  # (T,H,W,2)
            self.flat = True
            assert isinstance(self.kp, np.ndarray) and self.kp.ndim in (3, 4), \
                "kp shape must match flow layout"
            self.index = list(range(self.of.shape[0]))
        else:
            self.index = []
            num_seqs = len(self.of)
            for i in range(num_seqs):
                Ti = len(self.of[i])
                for j in range(Ti):
                    self.index.append((i, j))

        if self.flat:
            H, W = self.of[0].shape[:2]
        else:
            H, W = self.of[0][0].shape[:2]
        self.H, self.W = int(H), int(W)

    def __len__(self):
        return len(self.index)

    def _get_frame(self, idx):
        if self.flat:
            flow = self.of[idx]
            kps  = self.kp[idx]
        else:
            i, j = self.index[idx]
            flow = self.of[i][j]
            kps  = self.kp[i][j]

        if kps.shape[1] >= 4:
            kps_xy = kps[:, 2:]
        else:
            kps_xy = kps[:, :2]
        return flow, kps_xy

    def __getitem__(self, idx):
        flow, kps_xy = self._get_frame(idx)
        if self.topk is not None and kps_xy.shape[0] > self.topk:
            sel = np.random.choice(kps_xy.shape[0], self.topk, replace=False)
            kps_xy = kps_xy[sel]

        kps_int = np.clip(kps_xy, [0, 0], [self.W - 1, self.H - 1]).astype(np.int32)

        flow_chw = np.transpose(flow, (2, 0, 1)).astype(np.float32)
        origin_motion = flow_chw[:, kps_int[:, 1], kps_int[:, 0]]

        if np.random.rand() > 0.6 and origin_motion.shape[1] > 0:
            n_pick = min(10, origin_motion.shape[1])
            idxs = np.random.randint(0, origin_motion.shape[1], size=n_pick)
            origin_motion[0, idxs] *= 1.3
            origin_motion[1, idxs] *= -2.0

        kp = kps_xy.T.astype(np.float32)
        return torch.from_numpy(origin_motion), torch.from_numpy(kp)

class SmoothData(Dataset):
    def __init__(
        self,
        motion_path: str,
        kp_path: str,
        maxlength: int = 100,
        datatype: str = 'deepStab',
        grid_h: int = 480,
        grid_w: int = 640,
        split: str = 'train'
    ):
        super(SmoothData, self).__init__()

        if not os.path.isfile(motion_path) or not os.path.isfile(kp_path):
            raise FileNotFoundError(f"Missing dataset file(s): {motion_path} or {kp_path}")

        motion_np = np.load(motion_path, allow_pickle=True)
        kp_np = np.load(kp_path, allow_pickle=True)

        if split == 'train' and datatype == 'all':
            self.origin_motion = [motion_np[i] for i in range(motion_np.shape[0])]
            self.kp = [kp_np[i] for i in range(kp_np.shape[0])]
        else:
            self.origin_motion = motion_np
            self.kp = kp_np

        self.length = maxlength
        self.H = grid_h
        self.W = grid_w

    def __len__(self):
        return len(self.origin_motion)

    def __getitem__(self, idx):
        origin_motion = self.origin_motion[idx].transpose(1, 0, 2, 3)  # T, 2, H, W
        kp = self.kp[idx]  # T-1, 512, 2

        image_length = origin_motion.shape[0]
        if image_length <= self.length:
            start = 0
            length = image_length
        else:
            start = np.random.randint(0, image_length - self.length)
            length = self.length

        origin_motion = origin_motion[start:start + length, :, :, :]
        kp = kp[start:start + length - 1, :, :]

        origin_motion = np.cumsum(origin_motion, 0) - origin_motion[0]

        origin_motion = np.ascontiguousarray(origin_motion.astype(np.float32))
        kp = np.ascontiguousarray(kp.astype(np.float32))

        return torch.from_numpy(origin_motion), torch.from_numpy(kp)