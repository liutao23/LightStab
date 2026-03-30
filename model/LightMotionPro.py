# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : LightMotionPro.py
# @Time  : 2025/10/16 下午7:57

import torch
import torch.nn as nn
import math
# from model.motionloss import propagationLoss
from model.utils import multiHomoEstimate, singleHomoEstimate, MedianPool2d
from configs.config import cfg

class EfficientChannelAttention(nn.Module):
    """高效通道注意力机制 (ECA-Net)"""

    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class GhostModule(nn.Module):
    """Ghost模块 (GhostNet) - 用更少的参数生成更多特征图"""

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm1d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :]

class LightweightFusionBlock(nn.Module):
    """轻量化特征融合块 (参考LSNet和Rewrite the Stars)"""

    def __init__(self, in_channels, out_channels):
        super(LightweightFusionBlock, self).__init__()

        # 使用Ghost模块减少参数
        self.ghost1 = GhostModule(in_channels, out_channels)
        self.ghost2 = GhostModule(out_channels, out_channels)

        # 高效通道注意力
        self.eca = EfficientChannelAttention(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.eca(x)
        return x + residual

class EfficientMotionPro(nn.Module):
    def __init__(self, HEIGHT=cfg.MODEL.HEIGHT, WIDTH=cfg.MODEL.WIDTH,
                 inplanes=2, embeddingSize=48, globalchoice='multi'):
        super(EfficientMotionPro, self).__init__()
        # self.loss_calculate = propagationLoss
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        # 使用更小的嵌入维度
        self.embeddingSize = embeddingSize

        # 轻量化嵌入层 (使用Ghost模块)
        self.embedding = nn.Sequential(
            GhostModule(inplanes, embeddingSize),
            nn.ReLU()
        )

        # 运动信息嵌入
        self.embedding_motion = nn.Sequential(
            GhostModule(inplanes, embeddingSize),
            nn.ReLU()
        )

        # 轻量化特征提取 (使用深度可分离卷积和注意力)
        self.distance_extractor = nn.Sequential(
            nn.Conv1d(embeddingSize, embeddingSize, 3, padding=1, groups=embeddingSize),
            nn.Conv1d(embeddingSize, embeddingSize // 2, 1),
            EfficientChannelAttention(embeddingSize // 2),
            nn.ReLU(),
            nn.Conv1d(embeddingSize // 2, 1, 1)
        )
        self.distance_reducer = nn.Sequential(
            nn.Conv1d(embeddingSize, embeddingSize // 2, 1),  # 48->24
            nn.BatchNorm1d(embeddingSize // 2),
            nn.ReLU()
        )

        # 运动特征提取
        self.motion_extractor = nn.Sequential(
            nn.Conv1d(embeddingSize, 2 * embeddingSize, 3, padding=1, groups=embeddingSize),
            nn.Conv1d(2 * embeddingSize, 2 * embeddingSize, 1),
            EfficientChannelAttention(2 * embeddingSize),
            nn.ReLU(),
            nn.Conv1d(2 * embeddingSize, embeddingSize, 3, padding=1, groups=embeddingSize),
            nn.Conv1d(embeddingSize, embeddingSize, 1),
        )

        # 特征融合模块
        self.fusion_block = LightweightFusionBlock(embeddingSize + embeddingSize // 2, embeddingSize)

        # 权重计算
        self.weighted = nn.Softmax(dim=2)

        # 轻量化解码器
        self.decoder = nn.Sequential(
            nn.Linear(embeddingSize, embeddingSize // 2),
            nn.ReLU(),
            nn.Linear(embeddingSize // 2, 2)
        )

        # 单应性估计策略
        if globalchoice == 'multi':
            self.homoEstimate = multiHomoEstimate
        elif globalchoice == 'single':
            self.homoEstimate = singleHomoEstimate

        # 中值滤波
        self.meidanPool = MedianPool2d(5, same=True)

    def forward(self, motion):
        if motion.dim() == 2:
            motion = motion.unsqueeze(0)
        B, _, N = motion.shape

        distance_info = motion[:, 0:2, :]
        motion_info = motion[:, 2:4, :]

        # 提取距离特征
        embedding_distance = self.embedding(distance_info)  # [B, 48, N]
        distance_weighted = self.weighted(self.distance_extractor(embedding_distance))  # [B, 1, N]
        embedding_distance_reduced = self.distance_reducer(embedding_distance)  # [B, 24, N]

        # 提取运动特征
        embedding_motion = self.embedding_motion(motion_info)  # [B, 48, N]
        embedding_motion = self.motion_extractor(embedding_motion)  # [B, 48, N]

        # 特征融合
        fused_features = torch.cat([embedding_motion, embedding_distance_reduced], 1)  # [B, 72, N]
        fused_features = self.fusion_block(fused_features)  # [B, 48, N]

        # 加权聚合
        fused_features = torch.sum(fused_features * distance_weighted, 2)  # [B, 48]

        # 预测输出
        out_motion = self.decoder(fused_features)  # [B, 2]
        return out_motion

    def inference(self, x_flow, y_flow, kp):
        if kp.shape[1] == 4:
            kp = kp[:, 2:]
        index = kp.long()
        origin_motion = torch.cat([x_flow, y_flow], 1)
        extracted_motion = origin_motion[0, :, index[:, 0], index[:, 1]]
        kp = kp.permute(1, 0).float()
        concat_motion = torch.cat([kp[1:2, :], kp[0:1, :], extracted_motion], 0)

        motion, gridsMotion, _ = self.homoEstimate(concat_motion, kp)
        GridMotion = (self.forward(motion) + gridsMotion.squeeze(-1)) * cfg.MODEL.FLOWC
        GridMotion = GridMotion.view(self.HEIGHT // cfg.MODEL.PIXELS, self.WIDTH // cfg.MODEL.PIXELS, 2)
        GridMotion = GridMotion.permute(2, 0, 1).unsqueeze(0)
        GridMotion = self.meidanPool(GridMotion)
        return GridMotion

    def train_step(self, motion, kp, l1=10., l2=20., l3=10.):
        motion, gridsMotion, backupMotion = self.homoEstimate(motion, kp)
        GridMotion = self.forward(motion)
        loss = self.loss_calculate(motion, GridMotion, kp, gridsMotion, backupMotion, l1, l2, l3)
        return loss, GridMotion

if __name__ == "__main__":
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并切换到 eval 模式
    model = EfficientMotionPro().to(device)
    model.eval()

    # =========================
    # 示例一：直接调用 forward(motion)
    # =========================
    # 假设我们有 N 个关键点
    N = 512
    # motion 格式为 [4, N]（也支持 [1, 4, N] 或 [B, 4, N]）
    motion = torch.randn(4, N, device=device)

    with torch.no_grad():
        pred = model.forward(motion)  # 返回 shape [2]
    print("forward 输出 shape:", pred.shape)  # torch.Size([2])
    print("predicted motion:", pred)  # e.g. tensor([0.12, -0.34])

    # =========================
    # 示例二：调用 inference(x_flow, y_flow, kp)
    # =========================
    B = 1
    H, W = cfg.MODEL.HEIGHT, cfg.MODEL.WIDTH
    PIX = cfg.MODEL.PIXELS
    TOPK = 512

    # 1. 准备光流张量 [B, 1, H, W]
    x_flow = torch.randn(B, 1, H, W, device=device)
    y_flow = torch.randn(B, 1, H, W, device=device)

    # 2. 准备关键点 kp: [TOPK, 4]，其中前两列可任意占位，后两列为 (y, x) 坐标
    kp = torch.zeros(TOPK, 4, dtype=torch.long, device=device)
    kp[:, 2] = torch.randint(0, H, (TOPK,), device=device)  # y
    kp[:, 3] = torch.randint(0, W, (TOPK,), device=device)  # x

    with torch.no_grad():
        grid_motion = model.inference(x_flow, y_flow, kp)
    # 输出的 grid_motion 形状为 [1, 2, H/PIX, W/PIX]
    print("inference 输出 shape:", grid_motion.shape)