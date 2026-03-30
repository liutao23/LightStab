import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadderOld:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self.mode = mode
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == "downzero":
            self._pad = [0, pad_wd, 0, pad_ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        if self.mode == "downzero":
            return [F.pad(x, self._pad) for x in inputs]
        else:
            return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class InputPadder:
    """Pads images so that H, W are divisible by 8 (or by any divisor if you改造)
       - 支持 3D: [C,H,W]
       - 支持 4D: [N,C,H,W]
       - 支持 5D: [N,T,C,H,W]  (只在 H、W 方向 pad，不改 T)
       - 单个输入返回 Tensor；多个输入返回 List[Tensor]
    """
    def __init__(self, dims, mode='sintel', divisor=8):
        self.mode = mode

        # 末两维必须是 H, W
        H, W = dims[-2], dims[-1]
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor

        # 4 元组：只针对 H、W
        if mode == 'sintel':
            self._pad2d = [pad_w // 2, pad_w - pad_w // 2,
                           pad_h // 2, pad_h - pad_h // 2]
        elif mode == "downzero":
            self._pad2d = [0, pad_w, 0, pad_h]
        else:
            self._pad2d = [pad_w // 2, pad_w - pad_w // 2, 0, pad_h]

    def _pad_one(self, x: torch.Tensor) -> torch.Tensor:
        # 只对 H、W 做 padding；当 5D 时扩展成 6 元组，T 维不变
        if self.mode == "downzero":
            # constant 模式
            if x.dim() == 5:
                w0, w1, h0, h1 = self._pad2d
                pad_args = (w0, w1, h0, h1, 0, 0)
                return F.pad(x, pad_args, mode='constant', value=0.0)
            else:
                return F.pad(x, self._pad2d, mode='constant', value=0.0)
        else:
            # replicate 模式：PyTorch 对 5D 复制填充更稳定地用 6 元组
            if x.dim() == 5:
                w0, w1, h0, h1 = self._pad2d
                pad_args = (w0, w1, h0, h1, 0, 0)
                return F.pad(x, pad_args, mode='replicate')
            elif x.dim() in (3, 4):
                return F.pad(x, self._pad2d, mode='replicate')
            else:
                raise NotImplementedError(
                    f"Unsupported tensor dim {x.dim()} for padding; expect 3D/4D/5D."
                )

    def pad(self, *inputs):
        """单个张量 -> 返回张量；多个张量 -> 返回列表。"""
        if len(inputs) == 1:
            return self._pad_one(inputs[0])
        else:
            return [self._pad_one(x) for x in inputs]

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """按初始化时的 _pad2d 从 H、W 维裁掉填充；支持 3D/4D/5D。"""
        # 最后两个维度是 H, W
        h0, h1 = self._pad2d[2], self._pad2d[3]
        w0, w1 = self._pad2d[0], self._pad2d[1]

        H, W = x.shape[-2], x.shape[-1]
        h_start, h_end = h0, H - h1
        w_start, w_end = w0, W - w1
        return x[..., h_start:h_end, w_start:w_end]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def indexing(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    """
        TODO: directly indexing features instead of sampling
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode='nearest')

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
