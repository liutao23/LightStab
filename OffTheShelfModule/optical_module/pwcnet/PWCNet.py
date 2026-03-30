#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math
import os
import sys

try:
    from OffTheShelfModule.optical_module.pwcnet import correlation  # the custom cost volume layer
except Exception:
    parentddir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.path.pardir, 'correlations'))
    sys.path.append(parentddir)
    import correlation  # you should consider upgrading python

# ==========================================================
# 环境与后端优化（不改变功能）
# ==========================================================
# 至少 PyTorch 1.3
assert (int(''.join(torch.__version__.split('.')[0:2])) >= 13)

# 固定分辨率/相近分辨率推理更快
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ==========================================================
# backwarp：缓存按 (size, device, dtype) 维度区分，避免 .cuda() 引起的搬运
# ==========================================================
_backwarp_grid_cache = {}     # key: (B,H,W,device,dtype)
_backwarp_ones_cache = {}     # key: (B,H,W,device,dtype)

def _get_backwarp_grid(B, H, W, device, dtype):
    key = (B, H, W, device, dtype)
    grid = _backwarp_grid_cache.get(key)
    if grid is None:
        # 注意：align_corners=True 时，grid [-1,1] 覆盖整个像素网格
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xs, ys], dim=1)  # (B,2,H,W)
        _backwarp_grid_cache[key] = grid
    return grid

def _get_ones(B, H, W, device, dtype):
    key = (B, H, W, device, dtype)
    ones = _backwarp_ones_cache.get(key)
    if ones is None:
        ones = torch.ones((B, 1, H, W), device=device, dtype=dtype)
        _backwarp_ones_cache[key] = ones
    return ones

def backwarp(tenInput, tenFlow):
    """
    tenInput: (B,C,H,W), tenFlow: (B,2,H,W) 像素位移，保持你原先的缩放与 align_corners=True 约定
    返回：对 tenInput 进行 flow 反向采样后的 (B,C,H,W)，带 0/1 mask
    """
    B, C, H, W = tenInput.shape
    device, dtype = tenFlow.device, tenFlow.dtype

    grid = _get_backwarp_grid(B, H, W, device, dtype)  # (B,2,H,W)
    ones = _get_ones(B, H, W, device, dtype)

    # 归一化 flow 分量到 [-1,1] 采样空间（保持你的原逻辑与分母）
    flow_norm_x = tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0)
    flow_norm_y = tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
    flow_norm = torch.cat([flow_norm_x, flow_norm_y], dim=1)  # (B,2,H,W)

    # 拼接一条 mask 通道，采样后再阈值化（等价但更快）
    tenCat = torch.cat([tenInput, ones], dim=1)  # (B,C+1,H,W)

    # grid_sample 期望 (B,H,W,2)
    samp = torch.nn.functional.grid_sample(
        input=tenCat,
        grid=(grid + flow_norm).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    # 最后一通道是 mask，经由采样后 ∈[0,1]
    tenMask = (samp[:, -1:, :, :] > 0.999).to(samp.dtype)
    return samp[:, :-1, :, :] * tenMask

# ==========================================================
# 网络定义（仅在不改变行为的前提下做拼接/内存优化）
# ==========================================================
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                act = lambda: torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 16, 3, 2, 1),
                    act(),
                    torch.nn.Conv2d(16, 16, 3, 1, 1),
                    act(),
                    torch.nn.Conv2d(16, 16, 3, 1, 1),
                    act(),
                )
                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(16, 32, 3, 2, 1),
                    act(),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    act(),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    act(),
                )
                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 3, 2, 1),
                    act(),
                    torch.nn.Conv2d(64, 64, 3, 1, 1),
                    act(),
                    torch.nn.Conv2d(64, 64, 3, 1, 1),
                    act(),
                )
                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 96, 3, 2, 1),
                    act(),
                    torch.nn.Conv2d(96, 96, 3, 1, 1),
                    act(),
                    torch.nn.Conv2d(96, 96, 3, 1, 1),
                    act(),
                )
                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(96, 128, 3, 2, 1),
                    act(),
                    torch.nn.Conv2d(128, 128, 3, 1, 1),
                    act(),
                    torch.nn.Conv2d(128, 128, 3, 1, 1),
                    act(),
                )
                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 196, 3, 2, 1),
                    act(),
                    torch.nn.Conv2d(196, 196, 3, 1, 1),
                    act(),
                    torch.nn.Conv2d(196, 196, 3, 1, 1),
                    act(),
                )

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()
                # 与原始通道计算保持一致
                intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                               81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
                intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                              81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

                if intLevel < 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(2, 2, 4, 2, 1)
                    self.netUpfeat = torch.nn.ConvTranspose2d(
                        intPrevious + 128 + 128 + 96 + 64 + 32, 2, 4, 2, 1)
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                act = lambda: torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)

                self.conv1 = torch.nn.Conv2d(intCurrent, 128, 3, 1, 1)
                self.conv2 = torch.nn.Conv2d(intCurrent + 128, 128, 3, 1, 1)
                self.conv3 = torch.nn.Conv2d(intCurrent + 128 + 128, 96, 3, 1, 1)
                self.conv4 = torch.nn.Conv2d(intCurrent + 128 + 128 + 96, 64, 3, 1, 1)
                self.conv5 = torch.nn.Conv2d(intCurrent + 128 + 128 + 96 + 64, 32, 3, 1, 1)
                self.conv6 = torch.nn.Conv2d(intCurrent + 128 + 128 + 96 + 64 + 32, 2, 3, 1, 1)

                self.act = act()

            def forward(self, tenFirst, tenSecond, objPrevious):
                if objPrevious is None:
                    tenFlow = None
                    tenFeat_up = None
                    # 保证相关层输入是 float32
                    with torch.cuda.amp.autocast(enabled=False):
                        tenVolume32 = correlation.FunctionCorrelation(
                            tenFirst=tenFirst.float(), tenSecond=tenSecond.float()
                        )
                    tenVolume = torch.nn.functional.leaky_relu(
                        tenVolume32.to(dtype=tenFirst.dtype), negative_slope=0.1, inplace=True
                    )

                    feat_list = [tenVolume]  # 延后一次性 cat
                else:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat_up = self.netUpfeat(objPrevious['tenFeat'])
                    # 先做 backwarp（它支持半精度没问题），然后相关层强制 FP32
                    tenSecond_warp = backwarp(tenSecond, tenFlow * self.fltBackwarp)

                    with torch.cuda.amp.autocast(enabled=False):
                        tenVolume32 = correlation.FunctionCorrelation(
                            tenFirst=tenFirst.float(),
                            tenSecond=tenSecond_warp.float()
                        )
                    tenVolume = torch.nn.functional.leaky_relu(
                        tenVolume32.to(dtype=tenFirst.dtype), negative_slope=0.1, inplace=True
                    )

                    feat_list = [tenVolume, tenFirst, tenFlow, tenFeat_up]

                x = torch.cat(feat_list, dim=1)

                y1 = self.act(self.conv1(x))
                x = torch.cat([y1, x], dim=1)

                y2 = self.act(self.conv2(x))
                x = torch.cat([y2, x], dim=1)

                y3 = self.act(self.conv3(x))
                x = torch.cat([y3, x], dim=1)

                y4 = self.act(self.conv4(x))
                x = torch.cat([y4, x], dim=1)

                y5 = self.act(self.conv5(x))
                x = torch.cat([y5, x], dim=1)

                tenFlow = self.conv6(x)

                return {'tenFlow': tenFlow, 'tenFeat': x}

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()
                act = lambda: torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 128, 3, 1, 1, dilation=1),
                    act(),
                    torch.nn.Conv2d(128, 128, 3, 1, 2, dilation=2),
                    act(),
                    torch.nn.Conv2d(128, 128, 3, 1, 4, dilation=4),
                    act(),
                    torch.nn.Conv2d(128, 96, 3, 1, 8, dilation=8),
                    act(),
                    torch.nn.Conv2d(96, 64, 3, 1, 16, dilation=16),
                    act(),
                    torch.nn.Conv2d(64, 32, 3, 1, 1, dilation=1),
                    act(),
                    torch.nn.Conv2d(32, 2, 3, 1, 1, dilation=1),
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)

        self.netExtractor = Extractor()
        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)
        self.netRefiner = Refiner()

        # 权重加载（更健壮的键替换）
        ckpt_path = 'OffTheShelfModule/optical_module/pwcnet/pwc.pytorch'
        state = torch.load(ckpt_path, map_location='cpu')
        state = {k.replace('module', 'net'): v for k, v in state.items()}
        self.load_state_dict(state, strict=False)

        self.eval()  # 推理模式

    def forward(self, tenFirst, tenSecond):
        tenFirst_feats = self.netExtractor(tenFirst)
        tenSecond_feats = self.netExtractor(tenSecond)

        objEstimate = self.netSix(tenFirst_feats[-1], tenSecond_feats[-1], None)
        objEstimate = self.netFiv(tenFirst_feats[-2], tenSecond_feats[-2], objEstimate)
        objEstimate = self.netFou(tenFirst_feats[-3], tenSecond_feats[-3], objEstimate)
        objEstimate = self.netThr(tenFirst_feats[-4], tenSecond_feats[-4], objEstimate)
        objEstimate = self.netTwo(tenFirst_feats[-5], tenSecond_feats[-5], objEstimate)

        return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])

# ==========================================================
# 维持原始接口/行为的前提下，加速版 PWCNetestimate
# ==========================================================
netNetwork = None  # 保持你的全局符号

def PWCNetestimate(tenFirst, tenSecond, netNetwork):
    """
    :param tenFirst: (B,C,H,W) 归一化到[0,1]
    :param tenSecond: (B,C,H,W)
    :param netNetwork: 网络实例
    :return: flow (B,2,H,W)
    """
    assert (tenFirst.shape[1] == tenSecond.shape[1])
    assert (tenFirst.shape[2] == tenSecond.shape[2])

    # ——保持你原有的“变量赋值语义”与后续用法，不改逻辑——
    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    # 维持原来使用 view(-1,3,...) 的做法与尺寸语义（不改行为）
    tenPreprocessedFirst = tenFirst.view(-1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.view(-1, 3, intHeight, intWidth)

    # 对齐到 64 倍数（不改）
    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    # 为 cudnn 更友好的内存格式
    tenPreprocessedFirst = tenPreprocessedFirst.to(memory_format=torch.channels_last)
    tenPreprocessedSecond = tenPreprocessedSecond.to(memory_format=torch.channels_last)

    # 插值保持原参数（不改）
    tenPreprocessedFirst = torch.nn.functional.interpolate(
        tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear', align_corners=False
    )
    tenPreprocessedSecond = torch.nn.functional.interpolate(
        tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear', align_corners=False
    )

    # 自动选择是否启用 AMP
    use_amp = torch.cuda.is_available()
    device = next(netNetwork.parameters()).device

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
        # 前向
        tenFlow = netNetwork(tenPreprocessedFirst.to(device, non_blocking=True),
                             tenPreprocessedSecond.to(device, non_blocking=True))
        # 与原逻辑一致：20× 缩放 + 回到“原始 intHeight/intWidth”
        tenFlow = 20.0 * torch.nn.functional.interpolate(
            input=tenFlow, size=(intHeight, intWidth),
            mode='bilinear', align_corners=False
        )

        # 保持原始缩放校正
        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    # 输出 memory_format 不强制，保持兼容
    return tenFlow

# ==========================================================
# 可选：PyTorch>=2.0 时可尝试编译优化（不改变接口；默认关闭）
# ==========================================================
try:
    if hasattr(torch, 'compile'):
        netNetwork = torch.compile(Network().to('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        netNetwork = Network().to('cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    netNetwork = Network().to('cuda' if torch.cuda.is_available() else 'cpu')
