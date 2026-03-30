import getopt
import math
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

try:
    from .correlation import correlation  # the custom cost volume layer
except Exception:
    sys.path.insert(0, './correlation')
    import correlation  # you should consider upgrading python

##########################################################
# 推理场景：不需要梯度，启用 cudnn/benchmark
##########################################################

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # 允许根据输入尺寸选择最快算法（数值不变）
# 如需进一步提速，可启用 TF32（保持 FP32 接口；可能产生极轻微数值差异）
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

##########################################################

# 解析 CLI（保持原逻辑）
for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'model=',
    'one=',
    'two=',
    'out=',
])[0]:
    if strOption == '--model' and strArg != '': args_strModel = strArg
    if strOption == '--one' and strArg != '': args_strOne = strArg
    if strOption == '--two' and strArg != '': args_strTwo = strArg
    if strOption == '--out' and strArg != '': args_strOut = strArg

##########################################################
# 高效 backwarp：按 (device, dtype, H, W) 缓存 grid
##########################################################

_backwarp_grid_cache: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}

def _get_base_grid(flow: torch.Tensor) -> torch.Tensor:
    """
    返回 [-1,1] 归一化采样网格，shape: [1, 2, H, W]，缓存按 (device, dtype, H, W)。
    """
    key = (flow.device, flow.dtype, flow.shape[2], flow.shape[3])
    g = _backwarp_grid_cache.get(key)
    if g is not None:
        return g

    H, W = flow.shape[2], flow.shape[3]
    # 直接构造目标形状，避免 repeat 的额外开销
    hor = torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=flow.dtype).view(1, 1, 1, W).expand(1, 1, H, W)
    ver = torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=flow.dtype).view(1, 1, H, 1).expand(1, 1, H, W)
    grid = torch.cat([hor, ver], 1).contiguous()  # [1, 2, H, W]
    _backwarp_grid_cache[key] = grid
    return grid

def backwarp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    # 归一化 flow 到 [-1, 1] 坐标系
    # NOTE: 与原逻辑完全一致，只是用更明确的写法，避免多余张量构造
    fx = tenFlow[:, 0:1] * (2.0 / (tenInput.shape[3] - 1.0))
    fy = tenFlow[:, 1:2] * (2.0 / (tenInput.shape[2] - 1.0))
    flow_norm = torch.cat([fx, fy], 1)

    base = _get_base_grid(tenFlow)  # [1, 2, H, W]
    grid = (base + flow_norm).permute(0, 2, 3, 1).contiguous()  # [N,H,W,2]
    return F.grid_sample(tenInput, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

##########################################################

class liteflownet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # LeakyReLU 改为 inplace=True（数值完全一致、节省内存/带宽）
                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )

            def forward(self, tenInput: torch.Tensor):
                # 显式保证 NCHW contiguous（避免隐式拷贝）
                x = tenInput.contiguous()
                one = self.netOne(x)
                two = self.netTwo(one)
                thr = self.netThr(two)
                fou = self.netFou(thr)
                fiv = self.netFiv(fou)
                six = self.netSix(fiv)
                return [one, two, thr, fou, fiv, six]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel: int):
                super().__init__()
                self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                # 保持原条件结构与通道数不变
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                else:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    )

                self.netUpflow = None if intLevel == 6 else torch.nn.ConvTranspose2d(
                    2, 2, kernel_size=4, stride=2, padding=1, bias=False, groups=2
                )

                self.netUpcorr = None if intLevel >= 4 else torch.nn.ConvTranspose2d(
                    49, 49, kernel_size=4, stride=2, padding=1, bias=False, groups=49
                )

                ksz = [0, 0, 7, 5, 5, 3, 3][intLevel]
                pad = [0, 0, 3, 2, 2, 1, 1][intLevel]
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(49, 128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(32, 2, kernel_size=ksz, stride=1, padding=pad, bias=True),
                )

            def forward(self, tenOne, tenTwo, featOne, featTwo, tenFlow):
                featOne = self.netFeat(featOne)
                featTwo = self.netFeat(featTwo)

                if tenFlow is not None and self.netUpflow is not None:
                    tenFlow = self.netUpflow(tenFlow)

                if tenFlow is not None:
                    featTwo = backwarp(featTwo, tenFlow * self.fltBackwarp)

                if self.netUpcorr is None:
                    corr = F.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=featOne, tenTwo=featTwo, intStride=1),
                        negative_slope=0.1, inplace=False
                    )
                else:
                    corr = self.netUpcorr(F.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=featOne, tenTwo=featTwo, intStride=2),
                        negative_slope=0.1, inplace=False
                    ))

                base = tenFlow if tenFlow is not None else 0.0
                return base + self.netMain(corr)

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel: int):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                else:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    )

                ksz = [0, 0, 7, 5, 5, 3, 3][intLevel]
                pad = [0, 0, 3, 2, 2, 1, 1][intLevel]
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d([0, 0, 130, 130, 194, 258, 386][intLevel], 128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(32, 2, kernel_size=ksz, stride=1, padding=pad, bias=True),
                )

            def forward(self, tenOne, tenTwo, featOne, featTwo, tenFlow):
                featOne = self.netFeat(featOne)
                featTwo = self.netFeat(featTwo)
                if tenFlow is not None:
                    featTwo = backwarp(featTwo, tenFlow * self.fltBackward)
                # 保持原 concats 与返回
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([featOne, featTwo, tenFlow], 1))

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel: int):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()
                else:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d([0, 0, 32, 64, 96, 128, 192][intLevel], 128, kernel_size=1, stride=1, padding=0, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    )

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d([0, 0, 131, 131, 131, 131, 195][intLevel], 128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )

                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(32, [0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                                        stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel], bias=True)
                    )
                else:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(32, [0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1),
                                        stride=1, padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0), bias=True),
                        torch.nn.Conv2d([0, 0, 49, 25, 25, 9, 9][intLevel], [0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]),
                                        stride=1, padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel]), bias=True),
                    )

                self.netScaleX = torch.nn.Conv2d([0, 0, 49, 25, 25, 9, 9][intLevel], 1, kernel_size=1, stride=1, padding=0, bias=True)
                self.netScaleY = torch.nn.Conv2d([0, 0, 49, 25, 25, 9, 9][intLevel], 1, kernel_size=1, stride=1, padding=0, bias=True)

            def forward(self, tenOne, tenTwo, featOne, featTwo, tenFlow):
                # 与原逻辑一致：差异、softmax 权重、unfold 融合
                tenDifference = (tenOne - backwarp(tenTwo, tenFlow * self.fltBackward)).square().sum(1, True).sqrt().detach()
                tenDist = self.netDist(self.netMain(torch.cat([tenDifference,
                                                               tenFlow - tenFlow.mean([2, 3], True),
                                                               self.netFeat(featOne)], 1)))
                tenDist = (tenDist.square().neg())
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

                tenDivisor = tenDist.sum(1, True).reciprocal()

                unfold_k = self.intUnfold
                pad = (unfold_k - 1) // 2
                ux = F.unfold(tenFlow[:, 0:1], kernel_size=unfold_k, stride=1, padding=pad).view_as(tenDist)
                uy = F.unfold(tenFlow[:, 1:2], kernel_size=unfold_k, stride=1, padding=pad).view_as(tenDist)

                tenScaleX = self.netScaleX(tenDist * ux) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * uy) * tenDivisor

                return torch.cat([tenScaleX, tenScaleY], 1)

        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([Matching(lv) for lv in [2, 3, 4, 5, 6]])
        self.netSubpixel = torch.nn.ModuleList([Subpixel(lv) for lv in [2, 3, 4, 5, 6]])
        self.netRegularization = torch.nn.ModuleList([Regularization(lv) for lv in [2, 3, 4, 5, 6]])

        # 权重加载：保持原文件名与映射
        state = torch.load('OffTheShelfModule/optical_module/liteflownet/liteflownet-default', map_location='cpu')
        self.load_state_dict({k.replace('module', 'net'): v for k, v in state.items()})

    def forward(self, tenOne: torch.Tensor, tenTwo: torch.Tensor) -> torch.Tensor:
        # 归一化（原地，数值与原实现完全一致）
        tenOne[:, 0].add_(-0.411618); tenOne[:, 1].add_(-0.434631); tenOne[:, 2].add_(-0.454253)
        tenTwo[:, 0].add_(-0.410782); tenTwo[:, 1].add_(-0.433645); tenTwo[:, 2].add_(-0.452793)

        featOne = self.netFeatures(tenOne)
        featTwo = self.netFeatures(tenTwo)

        # 构建金字塔（保持对齐与插值配置不变）
        pyrOne = [tenOne]
        pyrTwo = [tenTwo]
        for lv in [1, 2, 3, 4, 5]:
            h, w = featOne[lv].shape[2], featOne[lv].shape[3]
            pyrOne.append(F.interpolate(pyrOne[-1], size=(h, w), mode='bilinear', align_corners=False))
            pyrTwo.append(F.interpolate(pyrTwo[-1], size=(h, w), mode='bilinear', align_corners=False))

        tenFlow = None
        for idx in [-1, -2, -3, -4, -5]:  # 对应 level 6 -> 2
            tenFlow = self.netMatching[idx](pyrOne[idx], pyrTwo[idx], featOne[idx], featTwo[idx], tenFlow)
            tenFlow = self.netSubpixel[idx](pyrOne[idx], pyrTwo[idx], featOne[idx], featTwo[idx], tenFlow)
            tenFlow = self.netRegularization[idx](pyrOne[idx], pyrTwo[idx], featOne[idx], featTwo[idx], tenFlow)

        return tenFlow * 20.0

netNetwork = None

##########################################################
# 推理入口（不改输入输出/行为）
##########################################################

def liteflownet_estimate(tenOne: torch.Tensor, tenTwo: torch.Tensor) -> torch.Tensor:
    """
    tenOne, tenTwo: [1, 3, H, W] (或 [B, 3, H, W]) on current device
    return: flow [B, 2, H, W]
    """
    global netNetwork
    if netNetwork is None:
        netNetwork = liteflownet().cuda().eval()

    assert tenOne.shape[1] == tenTwo.shape[1]
    assert tenOne.shape[2] == tenTwo.shape[2]

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    # 显式确保连续 & 在同一 device
    tenPreprocessedOne = tenOne.view(-1, 3, intHeight, intWidth).contiguous()
    tenPreprocessedTwo = tenTwo.view(-1, 3, intHeight, intWidth).contiguous()

    # 更快的 32 对齐（与原逻辑等价）
    intPreprocessedWidth = ((intWidth + 31) // 32) * 32
    intPreprocessedHeight = ((intHeight + 31) // 32) * 32

    tenPreprocessedOne = F.interpolate(tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth),
                                       mode='bilinear', align_corners=False)
    tenPreprocessedTwo = F.interpolate(tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth),
                                       mode='bilinear', align_corners=False)

    tenFlow = F.interpolate(netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth),
                            mode='bilinear', align_corners=False)

    # 尺度恢复（保持完全一致）
    tenFlow[:, 0] *= (float(intWidth) / float(intPreprocessedWidth))
    tenFlow[:, 1] *= (float(intHeight) / float(intPreprocessedHeight))

    return tenFlow
