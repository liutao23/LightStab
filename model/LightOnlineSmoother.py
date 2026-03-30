# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : LightOnlineSmoother.py
# @Time  : 2025/10/18 下午2:32

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import generateSmooth_online
# from model.smoothloss import loss_calculate_Spatial

_act = nn.ReLU


class LiteLS3DBlock(nn.Module):
    def __init__(self, channels: int, k_t: int = 5, d_t: int = 1,
                 use_pw: bool = False, pw_groups: int = 4):
        super().__init__()
        self.k_t, self.d_t = k_t, d_t
        self.dw_temporal = nn.Conv3d(
            channels, channels, kernel_size=(k_t, 1, 1),
            padding=(0, 0, 0), dilation=(d_t, 1, 1),
            groups=channels, bias=False
        )
        self.dw_spatial = nn.Conv3d(
            channels, channels, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), groups=channels, bias=False
        )
        self.use_pw = use_pw
        if use_pw:
            self.pw = nn.Conv3d(channels, channels, kernel_size=1,
                                groups=max(1, channels // pw_groups), bias=False)
        self.act = _act()

    def forward(self, x):  # x: [B,C,T,H,W]
        pad_t = (self.k_t - 1) * self.d_t
        x = F.pad(x, pad=(0,0, 0,0, pad_t,0))  # 仅左填充时间维，保持因果
        x = self.act(self.dw_temporal(x))
        x = self.act(self.dw_spatial(x))
        if self.use_pw:
            x = self.act(self.pw(x))
        return x


class Smoother(nn.Module):
    def __init__(self, inplanes=2, embeddingSize=64, kernel=5, dilation_t=1,
                 use_pointwise=True):
        super(Smoother, self).__init__()
        # self.loss_calculate = loss_calculate_Spatial

        self.embedding = nn.Sequential(
            nn.Linear(inplanes, embeddingSize),
            _act()
        )
        self.ls_block = LiteLS3DBlock(embeddingSize, k_t=kernel,
                                      d_t=dilation_t, use_pw=use_pointwise)

        self.decoder_x1 = nn.Linear(embeddingSize, 6)
        self.decoder_x2 = nn.Linear(embeddingSize, 6)
        self.decoder_y1 = nn.Linear(embeddingSize, 6)
        self.decoder_y2 = nn.Linear(embeddingSize, 6)
        self.scale_x = nn.Linear(embeddingSize, 1)
        self.scale_y = nn.Linear(embeddingSize, 1)
        self.sigmoid = nn.Sigmoid()

        # 在线版本
        self.generateSmooth = generateSmooth_online

    def forward(self, x):  # x: [B,2,T,H,W]
        traj = x.permute(0,2,3,4,1)
        feat = self.embedding(traj).permute(0,4,1,2,3)
        hidden = self.ls_block(feat)
        h = hidden.permute(0,2,3,4,1)

        # 过去1..6的自适应权重（因果）
        kx = self.sigmoid(self.decoder_x1(h)) * self.decoder_x2(h)
        kx = self.scale_x(h) * kx  # [B,T,H,W,6]
        ky = self.sigmoid(self.decoder_y1(h)) * self.decoder_y2(h)
        ky = self.scale_y(h) * ky  # [B,T,H,W,6]

        kernel = torch.cat([kx, ky], dim=-1).permute(0,4,1,2,3)  # [B,12,T,H,W]
        return kernel

    @torch.no_grad()
    def inference(self, x_paths, y_paths, repeat=1):
        path = np.concatenate([x_paths[...,None], y_paths[...,None]], -1)
        min_v = np.min(path, keepdims=True)
        path = path - min_v
        max_v = np.max(path, keepdims=True) + 1e-5
        path = path / max_v

        path_t = torch.from_numpy(path.astype(np.float32)).permute(0,4,3,1,2)  # [B,2,T,H,W]
        kernel_t = self.forward(path_t)  # 因果 kernel

        smooth_x, smooth_y = self.KernelSmooth(kernel_t, path_t, repeat=repeat)
        smooth_x = smooth_x.cpu().squeeze().permute(1,2,0).numpy() * max_v + min_v
        smooth_y = smooth_y.cpu().squeeze().permute(1,2,0).numpy() * max_v + min_v
        return smooth_x, smooth_y

    def KernelSmooth(self, kernel, path, repeat=1):
        # x 的 0:6 -> lag 1..6；y 的 6:12 -> lag 1..6
        smooth_x = self.generateSmooth(path[:,0:1], kernel[:,0:6], repeat=repeat)
        smooth_y = self.generateSmooth(path[:,1:2], kernel[:,6:12], repeat=repeat)
        return smooth_x, smooth_y

    def train_step(self, kpts, originPath, repeat=1):
        kernel = self.forward(originPath)
        smooth_x = self.generateSmooth(originPath[:,0:1], kernel[:,0:6], repeat=repeat)
        smooth_y = self.generateSmooth(originPath[:,1:2], kernel[:,6:12], repeat=repeat)
        smooth = torch.cat([smooth_x, smooth_y], 1)
        loss = self.loss_calculate(kpts, originPath, smooth)
        return loss

if __name__ == "__main__":

    im_raw = np.random.randn(1, 2, 4, 480, 640).astype(np.float32)  # B, 3, T, H, W (RGB)
    im_raw = torch.from_numpy(im_raw).cuda()

    kps_data = np.random.randn(512, 4, 3).astype(np.float32)
    kps_data = torch.from_numpy(kps_data).cuda()

    model = Smoother()
    model.cuda()

    kernel = model.forward(im_raw)

    print(kernel.shape)



