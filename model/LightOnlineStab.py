# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : LightOnlineStab.py
# @Time  : 2025/10/19 下午7:23

import torch
import torch.nn as nn
from model.LightKeypointsDetection import KeypointDetectionSSC
from model.LightMotionEsitimation import MotionEstimation
from model.LightMotionPro import EfficientMotionPro as MotionPro
from model.LightOnlineSmoother import Smoother
from model.utils import SingleMotionPropagate, MultiMotionPropagate
from model.utils import generateSmooth_online
from configs.config import cfg

class JacobiSolver(nn.Module):
    def __init__(self):
        super(JacobiSolver, self).__init__()
        self.generateSmooth = generateSmooth_online
        self.KernelSmooth = Smoother().KernelSmooth

    def forward(self, x):
        return None

class motionPropagate(object):
    def __init__(self, inferenceMethod):
        self.inference = inferenceMethod

class SuperStab(nn.Module):
    def __init__(self, point_method=['RFdet', 'superpoint', 'aliked', 'sift', 'disk', 'dog_hardnet', 'xfeat', 'dad', None],
                 flow_method='flowformer++',
                 ensemble_weights={'RFdet': 1.0, 'xfeat': 1.0, 'superpoint': 0.8, 'aliked': 0.8,
                                  'disk': 0.8, 'dog_hardnet': 0.8, 'sift': 0.7, 'dad': 1},
                 motion_weight=None,
                 smooth_weight=None,
                 homo='multi',
                 use_ssc=False):
        super(SuperStab, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("------------模型初始化------------")

        if use_ssc:
            self.topK = 10000
        else:
            self.topK = 512
        self.motionModule = KeypointDetectionSSC(
            point_method=point_method,
            topK=self.topK,
            ssc_num=cfg.MODEL.TOPK,
            ensemble_weights=ensemble_weights,
            diversity_mode='soft',
            diversity_lambda=0.6,
            diversity_sigma=7.5,
            diversity_tau=8.0,
            use_ssc=use_ssc,  # 打开 SSC
            ssc_tolerance=0.10)

        print(f'正在使用{point_method}作为关键点检测模块')

        self.flowModule = MotionEstimation(flow_method=flow_method)
        print(f'正在使用{flow_method}作为光流估计模块')


        print('正在载入运动传播模块')
        if motion_weight is not None:
            if homo=='multi':
                print('正在使用多网格估计网络')
                self.motion_propagate = MotionPro()
            else:
                print('正在使用单网格估计网络')
                self.motion_propagate = MotionPro(globalchoice='single')
            self.motion_propagate.load_state_dict(torch.load(motion_weight, weights_only=False), strict=True)
        else:
            if homo=='multi':
                print('正在使用多网格估计')
                self.motion_propagate = motionPropagate(MultiMotionPropagate)
            else:
                print('正在使用单网格估计')
                self.motion_propagate = motionPropagate(SingleMotionPropagate)

        print('正在载入轨迹平滑模块')
        if smooth_weight is not None:
            print('正在使用深度学习轨迹平滑')
            self.smoother = Smoother()
            self.smoother.load_state_dict(torch.load(smooth_weight, weights_only=False), strict=True)
        else:
            print('正在使用自适应迭代平滑')
            self.smoother = JacobiSolver()

    @torch.no_grad()
    def inference(self, x_RGB: torch.Tensor, repeat: int = 50):
        """
        @param x_RGB: [B, T, C, H, W], 假设 B=1
        @param repeat: smoother 模块迭代次数
        @return: origin_motion [B,2,T,H,W], smoothPath [B,2,T,H,W]
        """
        # 1. 准备好 device，搬到 GPU
        device = x_RGB.device
        x = x_RGB.to(device, non_blocking=True)   # [1, T, C, H, W]
        x_seq = x.squeeze(0)                      # [T, C, H, W]

        # 2. 提取关键点
        print("detect keypoints ....")
        im_topk, kpts = self.motionModule.forward(x_seq)
        # 确保 kpts 也在 GPU
        kpts = [kp.to(device, non_blocking=True) for kp in kpts]

        # 3. 估计光流
        print("estimate motion ....")
        masked_flow = self.flowModule.inference_stab(x, im_topk)  # [T,2,H,W]
        masked_flow = masked_flow.to(device, non_blocking=True)

        # 4. 运动传播（保留列表推导式）
        print("motion propagation ....")
        origin_list = [
            self.motion_propagate.inference(
                masked_flow[i:i+1, 0:1, :, :],  # [1,1,H,W]
                masked_flow[i:i+1, 1:2, :, :],  # [1,1,H,W]
                kpts[i]                        # 已在 GPU
            )
            for i in range(len(kpts) - 1)
        ]
        # origin_list 中的每项都是 GPU 上的 [B,2,H,W]

        # 5. stack & prepend zero-motion
        origin_motion = torch.stack(origin_list, dim=2)               # [B,2,T-1,H,W]
        zeros = torch.zeros_like(origin_motion[:, :, 0:1, :, :], device=device)
        origin_motion = torch.cat([zeros, origin_motion], dim=2)      # [B,2,T,H,W]

        # 6. 累积 & 归一化
        origin_motion = origin_motion.cumsum(dim=2)
        minv = origin_motion.amin()
        origin_motion = origin_motion - minv
        maxv = origin_motion.amax() + 1e-5
        origin_motion = origin_motion / maxv

        # 7. 平滑
        print("trajectary smoothing ....")
        smoothKernel = self.smoother(origin_motion)
        smooth_list = self.smoother.KernelSmooth(smoothKernel, origin_motion, repeat)
        smoothPath = torch.cat(smooth_list, dim=1)  # 沿通道维拼接

        # 8. 恢复量纲
        origin_motion = origin_motion * maxv + minv
        smoothPath = smoothPath * maxv + minv

        return origin_motion, smoothPath