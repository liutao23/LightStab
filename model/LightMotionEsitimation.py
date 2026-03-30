# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : LightMotionEsitimation.py
# @Time  : 2025/10/16 下午6:12

import warnings
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 保持你的后端不变
import matplotlib.pyplot as plt
from torchvision import models
from loguru import logger as loguru_logger
from OffTheShelfModule.optical_module.pwcnet.PWCNet import Network as PWCNet, PWCNetestimate
from OffTheShelfModule.optical_module.liteflownet.liteflownet import liteflownet, liteflownet_estimate
from OffTheShelfModule.optical_module.NeuFlow.neuflow import NeuFlow
from OffTheShelfModule.optical_module.NeuFlow.backbone_v7 import ConvBlock
from OffTheShelfModule.optical_module.core.FlowFormer import build_flowformer
from OffTheShelfModule.optical_module.core.utils.utils import InputPadder
from OffTheShelfModule.optical_module.core.submissions import get_cfg
from OffTheShelfModule.optical_module.Memflow.inference import inference_core_skflow as inference_core
from configs.config import cfg
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
_COLORWHEEL_CACHE = None  # 缓存到进程级，避免重复构造

def visualize_optical_flow(flow):
    """
    可视化光流，将光流从 (dx, dy) 转换为颜色编码的图像并显示。

    参数:
    - flow: 形状为 (B, 2, H, W) 的光流张量，B是批次大小，2是光流分量(dx, dy)，H是高度，W是宽度。

    返回:
    - rgb: 可视化的光流图像（RGB格式）。
    """

    def make_color_wheel():
        """
        Generate color wheel according to Middlebury color code.
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
        colorwheel[col:col + YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
        colorwheel[col:col + CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
        col += BM

        # MR
        colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col + MR, 0] = 255

        return colorwheel

    def compute_color(u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u ** 2 + v ** 2)
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)
        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)
            col[notidx] *= 0.75

            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    def flow_to_image(flow):
        """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        UNKNOWN_FLOW_THRESH = 1e7
        with np.errstate(over='ignore'):
            idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)

    # 如果张量在 GPU 上，将其移到 CPU
    flow = flow.cpu().squeeze(0).numpy().transpose(1, 2, 0)  # 转换为 (H, W, 2) 形状

    # 将光流转换为颜色图像
    bgr_flow = flow_to_image(flow)  # 生成光流图像

    # 显示图像
    plt.imshow(bgr_flow)
    # plt.title("Optical Flow Visualization")
    plt.title(" ")
    plt.axis('off')
    plt.show()

    return bgr_flow


# ============================================================
class MotionEstimation(nn.Module):
    def __init__(self, flow_method='PWCNet', image_width=cfg.MODEL.WIDTH, image_height=cfg.MODEL.HEIGHT):
        super().__init__()
        self.flow_method = flow_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_width = image_width
        self.image_height = image_height

        # 这些对象在某些方法中可以复用，避免重复构造
        self._padder_flowformer = None
        self._memflow_processor = None

        self.flownet = None
        self._initialize_flow_model()
        self._neuflow_inited = False
        # 预热（warmup）让 cuDNN/编译器完成调优，避免首帧抖动
        try:
            self._warmup_once()
        except Exception:
            pass

    def _warmup_once(self):
        self.eval()
        dev = self.device
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            dummy = torch.zeros(1, 3, self.image_height, self.image_width, device=dev).contiguous(memory_format=torch.channels_last)
            _ = self.forward(dummy, dummy)

    def build_network(self, MemFlowcfg):
        name = MemFlowcfg.network
        print(name)
        if name == 'MemFlowNet_skflow':
            from OffTheShelfModule.optical_module.Memflow.core.Networks.MemFlowNet.MemFlow import MemFlowNet as network
        elif name == 'MemFlowNet_predict':
            from OffTheShelfModule.optical_module.Memflow.core.Networks.MemFlowNet.MemFlow_P import MemFlowNet as network
        else:
            raise ValueError(f"Network = {name} is not a valid name!")
        return network(MemFlowcfg[name])

    def _initialize_flow_model(self):
        """ 根据 flow_method 初始化网络（等价功能），并应用编译/融合等加速 """
        dev = self.device
        fm = self.flow_method

        if fm == 'PWCNet':
            self.flownet = PWCNet().eval().to(dev)
        elif fm == 'liteflownet':
            self.flownet = liteflownet().eval().to(dev)
        elif fm == 'RAFT_large':
            from torchvision.models.optical_flow import Raft_Large_Weights
            self.flownet = models.optical_flow.raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().to(dev)
        elif fm == 'RAFT_small':
            from torchvision.models.optical_flow import Raft_Small_Weights
            self.flownet = models.optical_flow.raft_small(weights=Raft_Small_Weights.C_T_V2, progress=False).eval().to(dev)
        elif fm == 'NeuFlow':
            self.flownet = NeuFlow().to(dev)
            self._load_neuflow_weights()
        elif fm == 'flowformer++':
            self.flownet = torch.nn.DataParallel(build_flowformer(get_cfg())).to(dev).eval()
            self._load_flowformer_weights()
        elif fm == 'Memflow':
            from OffTheShelfModule.optical_module.Memflow.configs.kitti_memflownet_t import get_cfg_
            self.flownet = self.build_network(get_cfg_()).to(dev).eval()
            loguru_logger.info("Parameter Count: %d" % self.count_parameters(self.flownet))
            ckpt = torch.load('OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_kitti.pth', map_location='cpu')
            ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
            if 'module' in list(ckpt_model.keys())[0]:
                for key in list(ckpt_model.keys()):
                    ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            self.flownet.load_state_dict(ckpt_model, strict=True)
            self.flownet.eval()
        else:
            self.flownet = None  # fallback to CPU Farneback


    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _load_neuflow_weights(self):
        """
        NeuFlow 权重加载 + Conv/BN 融合；只在首次调用时执行 init_bhwd。
        保持与现有接口一致，不更改其它方法。
        """
        checkpoint = torch.load(
            'OffTheShelfModule/optical_module/NeuFlow/neuflow_things.pth',
            map_location=self.device
        )
        self.flownet.load_state_dict(checkpoint['model'], strict=True)

        # 融合 Conv+BN（只对 NeuFlow 中的 ConvBlock 做）
        self._fuse_conv_bn_layers()

        self.flownet.eval()

        # 只初始化一次，避免多次 init 造成的额外开销/不一致
        if not getattr(self, "_neuflow_inited", False):
            # NeuFlow 的内部 AMP 建议关闭
            self.flownet.init_bhwd(
                1,  # batch 模板
                self.image_height,
                self.image_width,
                self.device,
                amp=False
            )
            self._neuflow_inited = True

    def _load_flowformer_weights(self):
        model_path = 'OffTheShelfModule/optical_module/core/weights/kitti.pth'
        self.flownet.load_state_dict(torch.load(model_path, map_location=self.device))
        self.flownet.eval()

    def _fuse_conv_bn_layers(self):
        """ Conv+BN 融合（NeuFlow 用到） """
        for m in self.flownet.modules():
            if type(m) is ConvBlock:
                m.conv1 = self.fuse_conv_and_bn(m.conv1, m.norm1)
                m.conv2 = self.fuse_conv_and_bn(m.conv2, m.norm2)
                delattr(m, 'norm1')
                delattr(m, 'norm2')
                m.forward = m.forward_fuse

    @staticmethod
    def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d()."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels, conv.out_channels,
                kernel_size=conv.kernel_size, stride=conv.stride,
                padding=conv.padding, dilation=conv.dilation,
                groups=conv.groups, bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fusedconv

    # ---------------------------
    # 推理接口（保持不变）
    # ---------------------------
    @torch.inference_mode()
    def forward(self, x, x_RGB):
        """
        x: [B, 3, H, W], x_RGB: [B, 3, H, W] ; return: [B, 2, H, W]
        """
        dev = self.device

        if self.flownet is None:
            # CPU Farneback（无法 GPU），尽量减少张量-NumPy 往返
            prev_frame_gray = cv2.cvtColor(
                (x[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
            next_frame_gray = cv2.cvtColor(
                (x_RGB[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            optical_flow = torch.from_numpy(flow.transpose(2, 0, 1)).float().unsqueeze(0).to(dev, non_blocking=True)
            return F.interpolate(optical_flow, size=(self.image_height, self.image_width),
                                 mode='bilinear', align_corners=False)

        # GPU 路径：channels_last + AMP + 非阻塞搬运
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
        x_RGB = x_RGB.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)

        amp_enable = torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=amp_enable):
            optical_flow = self._estimate_optical_flow(x, x_RGB)

        optical_flow = self._add_batch_if_needed(optical_flow)  # << 关键
        optical_flow = F.interpolate(
            optical_flow, size=(self.image_height, self.image_width),
            mode='bilinear', align_corners=False
        )
        return optical_flow

    def _add_batch_if_needed(self, flow):
        """
        将 flow 统一到 [N, 2, H, W]；如果是 [2,H,W] 或 [H,W,2] 则补 batch；
        如果已经是 [N,2,H,W] 则原样返回；如果是 [N,H,W,2] 则换轴。
        """
        if flow is None:
            raise RuntimeError("Optical flow is None")

        # 先把多余的前导 1 维(>4D)都挤掉，避免 [1,1,2,H,W] 这种情况
        while flow.dim() > 4 and flow.shape[0] == 1:
            flow = flow.squeeze(0)

        if flow.dim() == 3:
            if flow.shape[0] == 2:  # [2,H,W] -> [1,2,H,W]
                flow = flow.unsqueeze(0)
            elif flow.shape[-1] == 2:  # [H,W,2] -> [1,2,H,W]
                flow = flow.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected 3D flow shape {tuple(flow.shape)}")
        elif flow.dim() == 4:
            if flow.shape[1] != 2 and flow.shape[-1] == 2:  # [N,H,W,2] -> [N,2,H,W]
                flow = flow.permute(0, 3, 1, 2)
            elif flow.shape[1] != 2:
                raise ValueError(f"Unexpected 4D flow shape {tuple(flow.shape)}: channel dim is not 2.")
        else:
            raise ValueError(f"Flow must be 3D/4D, got {flow.dim()}D with shape {tuple(flow.shape)}")

        return flow.contiguous()


    @torch.inference_mode()
    def inference(self, x, x_RGB, im_topk):
        dev = self.device
        if self.flownet is None:
            prev_frame_gray = cv2.cvtColor(
                (x[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            next_frame_gray = cv2.cvtColor(
                (x_RGB[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            optical_flow = torch.from_numpy(flow.transpose(2, 0, 1)).float().unsqueeze(0).to(dev, non_blocking=True)
        elif self.flow_method == 'NeuFlow':
            x_ = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y_ = x_RGB.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=False):
                optical_flow = self.flownet(x_, y_)[-1]
        else:
            x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            x_RGB = x_RGB.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                optical_flow = self._estimate_optical_flow(x, x_RGB)

        optical_flow = F.interpolate(optical_flow, size=(self.image_height, self.image_width),
                                     mode='bilinear', align_corners=False)
        return optical_flow * im_topk[0, ...].to(dev, non_blocking=True)

    @torch.inference_mode()
    def inference_stab(self, x_RGB, im_topk):
        """
        x_RGB: [B, T, 3, H, W]
        返回:   [B*(T-1), 2, H_out, W_out] 的 flow_masked
        """
        dev = self.device
        B, T, C, H, W = x_RGB.shape
        H_out, W_out = self.image_height, self.image_width

        # 预分配输出流（按最终分辨率）
        flows = torch.empty(
            (B * (T - 1), 2, H_out, W_out),
            device=dev, dtype=torch.float32, memory_format=torch.channels_last
        )

        amp_enable = torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=amp_enable), torch.inference_mode():
            for i in range(T - 1):
                f = self.forward(
                    x_RGB[:, i, ...].to(dev, non_blocking=True),
                    x_RGB[:, i + 1, ...].to(dev, non_blocking=True)
                )
                flows[B * i:B * (i + 1)].copy_(f)

        # ------ 自适应构造 mask：兼容 [B,1,H,W] 、 [B*(T-1),1,H,W] 、 [T,1,H,W](B=1) 、 [B*T,1,H,W] ------
        mask_in = im_topk.to(dev, non_blocking=True)  # [N,1,hm,wm]
        if mask_in.dim() != 4 or mask_in.shape[1] != 1:
            raise ValueError(f"im_topk must be [N,1,H,W], got {tuple(mask_in.shape)}")

        # 插值到输出尺寸（保持二值/离散特性）
        if mask_in.shape[-2:] != (H_out, W_out):
            mask_in = F.interpolate(mask_in, size=(H_out, W_out), mode='nearest')

        N_mask = mask_in.shape[0]
        N_flow = flows.shape[0]  # = B*(T-1)

        if N_mask == B:
            # 情况 A：每个 batch 一张 -> 展开为每对帧一张（零拷贝视图）
            mask = mask_in.unsqueeze(1).expand(B, T - 1, 1, H_out, W_out) \
                .reshape(N_flow, 1, H_out, W_out)
        elif N_mask == N_flow:
            # 情况 B：已经是一一对应
            mask = mask_in
        elif B == 1 and N_mask == T:
            # 情况 C：B=1，按每帧给了 T 张 -> 丢掉最后一张，取前 T-1
            mask = mask_in[:T - 1, ...]
        elif N_mask == B * T:
            # 情况 D：按 (B,T) 给的 -> 先 reshape 回 [B,T,1,H,W]，再丢掉每条序列最后一张
            mask_bt = mask_in.reshape(B, T, 1, H_out, W_out)
            mask = mask_bt[:, :-1, ...].reshape(N_flow, 1, H_out, W_out)
        else:
            raise ValueError(
                f"im_topk batch mismatch: got N={N_mask}, expected one of "
                f"{B} (per-batch), {N_flow} (per-pair), {T if B == 1 else 'T if B==1'}, or {B * T} (per-frame)."
            )

        flow_masked = flows * mask.to(dtype=flows.dtype)  # 零拷贝广播相乘
        return flow_masked

    def _estimate_optical_flow(self, x, x_RGB):
        """
        x, x_RGB: [B,3,H,W] on device; return [B,2,h,w]
        """
        # 统一 0..1 归一化在 GPU 上完成
        x01 = x / 255.0
        xRGB01 = x_RGB / 255.0

        if self.flow_method == 'PWCNet':
            # PWCNetestimate 期望 [3,H,W]，保持 batch=1 语义
            return PWCNetestimate(x01[0], xRGB01[0], self.flownet).unsqueeze(0)
        elif self.flow_method == 'liteflownet':
            return liteflownet_estimate(x01[0], xRGB01[0]).unsqueeze(0)
        elif self.flow_method in ['RAFT_large', 'RAFT_small']:
            return self.flownet(x01, xRGB01)[-1]
        elif self.flow_method == 'flowformer++':
            # 复用 padder，避免每帧重建
            if self._padder_flowformer is None:
                self._padder_flowformer = InputPadder(x.shape)
            out, _ = self.flownet.forward(x, x_RGB)
            return self._padder_flowformer.unpad(out)
        elif self.flow_method == 'NeuFlow':
            x_ = x.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y_ = x_RGB.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=False):
                return self.flownet(x_, y_)[-1]

        elif self.flow_method == 'Memflow':
            from OffTheShelfModule.optical_module.Memflow.configs.kitti_memflownet_t import get_cfg_
            if self._memflow_processor is None:
                self._memflow_processor = inference_core.InferenceCore(self.flownet, config=get_cfg_())
            images = torch.stack([x, x_RGB])           # [2,B,3,H,W]
            images = images.permute(1, 0, 2, 3, 4)     # [B,2,3,H,W]
            padder = InputPadder(images.shape)
            images = padder.pad(images)
            images = 2 * (images / 255.0) - 1.0
            flow_low, optical_flow = self._memflow_processor.step(
                images, end=True, add_pe=('rope' in cfg and cfg.rope), flow_init=None
            )
            return optical_flow
        return None



# ============================================================
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Run optical flow inference with CLI args (same I/O semantics)."
    )
    parser.add_argument("--img1", type=str, default="assets/frame_00000.jpg",
                        help="Path to the first image.")
    parser.add_argument("--img2", type=str, default="assets/frame_00005.jpg",
                        help="Path to the second image.")
    parser.add_argument("--method", type=str, default="PWCNet",
                        choices=['', 'PWCNet', 'liteflownet', 'RAFT_large', 'RAFT_small', 'NeuFlow', 'flowformer++', 'Memflow'],
                        help="Optical flow backend. Empty string = Farneback (CPU).")
    parser.add_argument("--width", type=int, default=cfg.MODEL.WIDTH,
                        help="Inference width (defaults to cfg.MODEL.WIDTH).")
    parser.add_argument("--height", type=int, default=cfg.MODEL.HEIGHT,
                        help="Inference height (defaults to cfg.MODEL.HEIGHT).")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. 'cuda:0' or 'cpu'. Default: auto.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable matplotlib visualization.")
    parser.add_argument("--save", type=str, default='results/vis_flows.png',
                        help="Optional path to save the colored flow image (PNG/JPG).")
    parser.add_argument("--channels-last", action="store_true",
                        help="Use channels_last memory format for input tensors.")
    args = parser.parse_args()

    image_width = args.width
    image_height = args.height
    flow_method = args.method

    def load_and_preprocess_image(image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        image = cv2.resize(image, (image_width, image_height))

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # [3,H,W], 0..255
        return image_tensor.unsqueeze(0).pin_memory()  # [1,3,H,W]

    # 读取两张图片
    image1 = load_and_preprocess_image(args.img1)
    image2 = load_and_preprocess_image(args.img2)

    print(f"Image1 shape: {image1.shape}")
    print(f"Image2 shape: {image2.shape}")

    # 设备选择
    if args.device is not None:
        dev = torch.device(args.device)
    else:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化模型
    MotionEstimation1 = MotionEstimation(
        flow_method=flow_method,
        image_width=image_width,
        image_height=image_height
    ).to(dev)

    # 输入搬运到目标设备
    memfmt = torch.channels_last if args.channels_last else torch.contiguous_format
    with torch.no_grad():
        t0 = time.time()
        image1_dev = image1.to(device=dev, non_blocking=True).to(memory_format=memfmt)
        image2_dev = image2.to(device=dev, non_blocking=True).to(memory_format=memfmt)
        flow_masked = MotionEstimation1.forward(image1_dev, image2_dev)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        dt = time.time() - t0

    print(f"Flow tensor shape: {flow_masked.shape} | elapsed: {dt*1000:.2f} ms")

    # 可视化 & 保存
    if not args.no_viz or args.save is not None:
        rgb = visualize_optical_flow(flow_masked)  # (H,W,3) uint8
        if args.save:
            ok = cv2.imwrite(args.save, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved flow visualization to {args.save} ({'OK' if ok else 'FAIL'})")

