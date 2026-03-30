# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : utils.py
# @Time  : 2025/10/17 下午1:46

import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.nn.modules.utils import _pair, _quadruple
from sklearn.cluster import KMeans
from configs.config import cfg

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def gauss(t, r=0, window_size=3):
    """
    @param window_size is the size of window over which gaussian to be applied
    @param t is the index of current point
    @param r is the index of point in window

    @return gaussian weights over a window size
    """
    if np.abs(r - t) > window_size:
        return 0
    else:
        return np.exp((-9 * (r - t) ** 2) / (window_size ** 2))

def generateSmooth(originPath, kernel=None, repeat=50):
    """
    originPath: (B, C, T, H, W)
    kernel:
        - None: use 7-tap gaussian with zero center (same as original)
        - or tensor shaped (B, 6, T, H, W); its 6 weights are averaged over (B,T,H,W)
          to get 6 global scalars that map to the 7-tap kernel with zero center.
    Returns: smoothed tensor with the same shape as originPath
    """
    device = originPath.device
    dtype = originPath.dtype

    B, C, T, H, W = originPath.shape
    smooth = originPath.clone()

    if kernel is None:
        w = torch.tensor([gauss(i) for i in range(-3, 4)], dtype=dtype, device=device)
        w[3] = 0
    else:
        if not (kernel.dim() == 5 and kernel.size(1) == 6):
            raise ValueError("kernel must be None or shaped (B,6,T,H,W).")
        w6 = kernel.to(dtype).mean(dim=(0, 2, 3, 4))  # (6,)
        w = torch.zeros(7, dtype=dtype, device=device)
        w[:3] = w6[:3]      # -3,-2,-1
        w[4:] = w6[3:]      # +1,+2,+3

    w_abs = w.abs()
    k = w.view(1, 1, 7)
    k_abs = w_abs.view(1, 1, 7)

    def flatten_tc(x):
        return x.permute(0, 1, 3, 4, 2).contiguous().view(-1, 1, T)

    def restore_tc(x_flat):
        return x_flat.view(B, C, H, W, T).permute(0, 1, 4, 2, 3).contiguous()

    x0 = flatten_tc(originPath)
    x = flatten_tc(smooth)

    ones = torch.ones_like(x, dtype=dtype, device=device)
    denom_dynamic = F.conv1d(ones, k_abs, padding=3)

    lambda_t = 100.0
    for _ in range(repeat):
        nbr_sum = F.conv1d(x, k, padding=3)
        x = (lambda_t * nbr_sum + x0) / (1.0 + lambda_t * denom_dynamic)

    smooth = restore_tc(x)
    return smooth

def gaussonline(t, r=0, window_size=6):
    if np.abs(r - t) > window_size:
        return 0
    else:
        return np.exp((-9 * (r - t) ** 2) / (window_size ** 2))

def generateSmooth_online(originPath, kernel6=None, repeat=1, eps=1e-12):
    """
    因果7点滑窗，级联 repeat 次（严格因果）。
    originPath: (B,1,T,H,W)
    kernel6:    (B,6,T,H,W)  对应滞后1..6的权重；None 时用固定高斯
    repeat:     >=1 的整数；相当于“多遍平滑”
    """
    assert repeat >= 1 and int(repeat) == repeat, "repeat must be integer >=1"
    out = originPath
    for _ in range(repeat):
        out = _causal7_once(out, kernel6, eps)
    return out

def _causal7_once(originPath, kernel6, eps):
    B, C, T, H, W = originPath.shape
    assert C == 1, "call per-axis: x or y"
    device, dtype = originPath.device, originPath.dtype

    # 当前帧权重 w0=1（若你以后想学第7权重，可改网络输出7通道）
    w0 = torch.ones((B, 1, T, H, W), dtype=dtype, device=device)

    if kernel6 is None:
        taps = torch.tensor([gaussonline(-i, 0, 6) for i in range(1, 7)], dtype=dtype, device=device)  # (6,)
        k6 = taps.view(1, 6, 1, 1, 1).expand(B, 6, T, H, W).contiguous()
    else:
        assert kernel6.shape == (B, 6, T, H, W), "kernel6 must be (B,6,T,H,W)"
        k6 = kernel6.to(dtype)

    out = torch.empty_like(originPath)
    for t in range(T):
        k = min(6, t)
        # 取权重切片并在可用窗口内“局部归一化”
        w_cur = [w0[:, :, t]] + [k6[:, i-1, t].unsqueeze(1) for i in range(1, k+1)]  # (B,1,H,W)×(1+k)
        w_stack = torch.stack(w_cur, dim=2)                                          # (B,1,1+k,H,W)
        w_stack = w_stack / w_stack.sum(dim=2, keepdim=True).clamp_min(eps)

        # 聚合 x_{t-i}
        x_terms = [originPath[:, :, t]] + [originPath[:, :, t-i] for i in range(1, k+1)]
        x_stack = torch.stack(x_terms, dim=2)                                        # (B,1,1+k,H,W)
        out[:, :, t] = (w_stack * x_stack).sum(dim=2)
    return out

def _as_mask_from_input(inp, thr=16, dilate=1, strict_black=True):
    """
    strict_black: True=只检测绝对黑色(0,0,0), False=使用灰度阈值
    """
    # 2D 掩码处理保持不变
    if isinstance(inp, np.ndarray) and inp.ndim == 2:
        return (inp != 0).astype(np.uint8)

    # 帧批/单帧统一处理
    frames = None
    if isinstance(inp, (list, tuple)):
        frames = list(inp)
    elif isinstance(inp, np.ndarray):
        if inp.ndim == 3 and inp.shape[2] in (1, 3):
            frames = [inp]
        elif inp.ndim == 4:  # (T,H,W[,C])
            frames = [inp[i] for i in range(inp.shape[0])]
    if frames is None or len(frames) == 0:
        return None

    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate + 1, 2 * dilate + 1)) if dilate > 0 else None

    def border_connected_black(img):
        if strict_black:
            # 方法1: 严格检测绝对黑色
            if img.ndim == 3 and img.shape[2] == 3:
                # BGR三通道都为0才是绝对黑色
                m = np.all(img == 0, axis=2).astype(np.uint8)
            elif img.ndim == 3 and img.shape[2] == 1:
                # 单通道为0
                m = (img[..., 0] == 0).astype(np.uint8)
            else:
                # 2D图像
                m = (img == 0).astype(np.uint8)
        else:
            # 原来的灰度阈值方法
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if (img.ndim == 3 and img.shape[2] == 3) else (
                img if img.ndim == 2 else img[..., 0])
            m = (g <= thr).astype(np.uint8)

        # 形态学处理
        if kd is not None:
            m = cv2.dilate(m, kd)

        # 边界连通性分析保持不变
        h, w = m.shape
        seeds = np.zeros_like(m, np.uint8)
        seeds[0, :] = m[0, :];
        seeds[-1, :] = m[-1, :]
        seeds[:, 0] |= m[:, 0];
        seeds[:, -1] |= m[:, -1]
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        prev = seeds
        max_iter = max(h, w)
        for _ in range(max_iter):
            grown = (cv2.dilate(prev, k) & m)
            if np.array_equal(grown, prev):
                break
            prev = grown
        return prev

    acc = None
    for f in frames:
        mb = border_connected_black(f)
        acc = mb if acc is None else (acc | mb)
    return acc.astype(np.uint8)

def detect_global_max_crop_tilt(inp, thr=16, dilate=1, strict_black=True):
    """
    增加strict_black参数，默认True检测绝对黑色
    """
    all_black = _as_mask_from_input(inp, thr=thr, dilate=dilate, strict_black=strict_black)
    if all_black is None:
        return 0, 0, 0, 0

    # 其余代码保持不变
    ones = (all_black == 0).astype(np.uint8)
    H, W = ones.shape
    h = np.zeros(W, np.int32)

    best_area = 0
    best = (0, 0, 0, 0)

    for i in range(H):
        row = ones[i] != 0
        h = (h + 1) * row
        st = []
        for j in range(W + 1):
            cur = h[j] if j < W else 0
            start = j
            while st and st[-1][1] > cur:
                idx, val = st.pop()
                area = val * (j - idx)
                if area > best_area:
                    best_area = area
                    y1 = i
                    y0 = i - val + 1
                    x0 = idx
                    x1 = j - 1
                    best = (y0, x0, y1, x1)
                start = idx
            st.append((start, cur))

    if best_area == 0:
        return 0, 0, 0, 0

    y0, x0, y1, x1 = best
    top = y0
    left = x0
    bottom = (H - 1) - y1
    right = (W - 1) - x1
    return int(top), int(bottom), int(left), int(right)

def singleHomoEstimate(motion, kp):
    dev = motion.device
    dtype = torch.float32

    new_kp = torch.cat([kp[1:2, :], kp[0:1, :]], 0) + motion[2:, :]
    new_points_numpy = new_kp.detach().transpose(0, 1).contiguous().to(dtype).cpu().numpy()
    old_points_numpy = torch.cat([kp[1:2, :], kp[0:1, :]], 0).detach().transpose(0, 1).contiguous().to(dtype).cpu().numpy()

    Homo, _ = cv2.findHomography(old_points_numpy, new_points_numpy, cv2.RANSAC)
    if Homo is None:
        Homo = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]], dtype=np.float32)
    else:
        Homo = Homo.astype(np.float32)

    H_t = torch.from_numpy(Homo).to(dev)

    oy = torch.from_numpy(old_points_numpy[:, 0]).to(dev, dtype)
    ox = torch.from_numpy(old_points_numpy[:, 1]).to(dev, dtype)
    denom = H_t[2, 0] * oy + H_t[2, 1] * ox + H_t[2, 2]
    px = (H_t[0, 0] * oy + H_t[0, 1] * ox + H_t[0, 2]) / denom
    py = (H_t[1, 0] * oy + H_t[1, 1] * ox + H_t[1, 2]) / denom
    new_points_projected = torch.stack([px, py], 1).to(dev).permute(1, 0)

    H = int(cfg.MODEL.HEIGHT)
    W = int(cfg.MODEL.WIDTH)
    P = int(cfg.MODEL.PIXELS)

    key = (dev, H, W, P)
    if not hasattr(singleHomoEstimate, "_cache"):
        singleHomoEstimate._cache = {}
    cache = singleHomoEstimate._cache

    if key not in cache:
        mx, my = torch.meshgrid(
            torch.arange(0, W, P, device=dev),
            torch.arange(0, H, P, device=dev)
        )
        mx = mx.float().permute(1, 0).contiguous()
        my = my.float().permute(1, 0).contiguous()
        mz = torch.ones_like(mx, device=dev)
        meshes = torch.stack([mx, my, mz], 0).contiguous()

        gx, gy = torch.meshgrid(
            torch.arange(0, W, P, device=dev),
            torch.arange(0, H, P, device=dev)
        )
        grids = torch.stack([gx, gy], 0).permute(0, 2, 1).reshape(2, -1).permute(1, 0).unsqueeze(2).float().contiguous()
        cache[key] = (meshes, grids)

    meshes, grids = cache[key]

    mp = torch.mm(H_t, meshes.view(3, -1)).view_as(meshes)
    x_motions = meshes[0] - (mp[0] / mp[2])
    y_motions = meshes[1] - (mp[1] / mp[2])
    projected_motion = torch.stack([x_motions, y_motions], 2).view(-1, 2, 1).to(dev)

    redisual_kp_motion = new_points_projected - torch.cat([kp[1:2, :], kp[0:1, :]], 0)

    motion[:2, :] = motion[:2, :] + motion[2:, :]
    motion = motion.unsqueeze(0).repeat(grids.shape[0], 1, 1)
    motion[:, :2, :] = (motion[:, :2, :] - grids) / float(W)
    origin_motion = motion[:, 2:, :] / float(cfg.MODEL.FLOWC)
    motion[:, 2:, :] = (redisual_kp_motion.unsqueeze(0) - motion[:, 2:, :]) / float(cfg.MODEL.FLOWC)

    return motion, projected_motion / float(cfg.MODEL.FLOWC), origin_motion

def HomoCalc(grids, new_grids_loc):
    _, H, W = grids.shape
    dev = grids.device
    dt = torch.float32
    g = grids.unsqueeze(0).to(dev, dt)
    ng = new_grids_loc.unsqueeze(0).to(dev, dt)
    Homo = torch.zeros(1, 3, 3, H - 1, W - 1, device=dev, dtype=dt)
    x00, y00 = g[:, 0:1, :-1, :-1], g[:, 1:2, :-1, :-1]
    x10, y10 = g[:, 0:1, 1:, :-1],  g[:, 1:2, 1:, :-1]
    x01, y01 = g[:, 0:1, :-1, 1:],  g[:, 1:2, :-1, 1:]
    x11, y11 = g[:, 0:1, 1:, 1:],   g[:, 1:2, 1:, 1:]
    x00p, y00p = ng[:, 0:1, :-1, :-1], ng[:, 1:2, :-1, :-1]
    x10p, y10p = ng[:, 0:1, 1:, :-1],  ng[:, 1:2, 1:, :-1]
    x01p, y01p = ng[:, 0:1, :-1, 1:],  ng[:, 1:2, :-1, 1:]
    x11p, y11p = ng[:, 0:1, 1:, 1:],   ng[:, 1:2, 1:, 1:]
    one = torch.ones_like(x00)
    zero = torch.zeros_like(x00)

    def _rows(x, y, xp, yp):
        r1 = torch.cat([x, y, one, zero, zero, zero, -x * xp, -y * xp], 2)
        r2 = torch.cat([zero, zero, zero, x, y, one, -x * yp, -y * yp], 2)
        return r1, r2

    r1, r2 = _rows(x00, y00, x00p, y00p)
    r3, r4 = _rows(x10, y10, x10p, y10p)
    r5, r6 = _rows(x01, y01, x01p, y01p)
    r7, r8 = _rows(x11, y11, x11p, y11p)

    A = torch.stack([r1, r3, r5, r7, r2, r4, r6, r8], 1).view(8, 8, -1).permute(2, 0, 1).contiguous()
    B_ = torch.stack([x00p, x10p, x01p, x11p, y00p, y10p, y01p, y11p], 1).view(8, -1).permute(1, 0).contiguous()

    try:
        H_rec = torch.linalg.solve(A, B_.unsqueeze(2))
        H_ = torch.cat([H_rec, torch.ones_like(H_rec[:, 0:1, :])], 1).view(H_rec.shape[0], 3, 3)
        Homo = H_.permute(1, 2, 0).view_as(Homo)
    except Exception:
        one_s = torch.ones_like(g[:, 0:1, 0, 0])
        zero_s = torch.zeros_like(g[:, 1:2, 0, 0])
        for i in range(H - 1):
            for j in range(W - 1):
                x0, y0 = g[:, 0:1, i, j], g[:, 1:2, i, j]
                x1, y1 = g[:, 0:1, i + 1, j], g[:, 1:2, i + 1, j]
                x2, y2 = g[:, 0:1, i, j + 1], g[:, 1:2, i, j + 1]
                x3, y3 = g[:, 0:1, i + 1, j + 1], g[:, 1:2, i + 1, j + 1]
                x0p, y0p = ng[:, 0:1, i, j], ng[:, 1:2, i, j]
                x1p, y1p = ng[:, 0:1, i + 1, j], ng[:, 1:2, i + 1, j]
                x2p, y2p = ng[:, 0:1, i, j + 1], ng[:, 1:2, i, j + 1]
                x3p, y3p = ng[:, 0:1, i + 1, j + 1], ng[:, 1:2, i + 1, j + 1]
                R1 = torch.cat([x0, y0, one_s, zero_s, zero_s, zero_s, -x0 * x0p, -y0 * x0p], 2)
                R2 = torch.cat([x1, y1, one_s, zero_s, zero_s, zero_s, -x1 * x1p, -y1 * x1p], 2)
                R3 = torch.cat([x2, y2, one_s, zero_s, zero_s, zero_s, -x2 * x2p, -y2 * x2p], 2)
                R4 = torch.cat([x3, y3, one_s, zero_s, zero_s, zero_s, -x3 * x3p, -y3 * x3p], 2)
                R5 = torch.cat([zero_s, zero_s, zero_s, x0, y0, one_s, -x0 * y0p, -y0 * y0p], 2)
                R6 = torch.cat([zero_s, zero_s, zero_s, x1, y1, one_s, -x1 * y1p, -y1 * y1p], 2)
                R7 = torch.cat([zero_s, zero_s, zero_s, x2, y2, one_s, -x2 * y2p, -y2 * y2p], 2)
                R8 = torch.cat([zero_s, zero_s, zero_s, x3, y3, one_s, -x3 * y3p, -y3 * y3p], 2)
                A_ij = torch.stack([R1, R2, R3, R4, R5, R6, R7, R8], 1).view(8, 8)
                B_ij = torch.stack([x0p, x1p, x2p, x3p, y0p, y1p, y2p, y3p], 1).view(8, 1)
                try:
                    H_rec = torch.linalg.solve(A_ij, B_ij)
                    Hm = torch.cat([H_rec, torch.ones_like(H_rec[:1])], 0).view(3, 3)
                except Exception:
                    Hm = torch.eye(3, device=dev, dtype=dt)
                Homo[:, :, :, i, j] = Hm
    return Homo.view(3, 3, H - 1, W - 1)

def HomoProj(homo, pts):
    px = (pts[:, 0:1] // cfg.MODEL.PIXELS).long()
    py = (pts[:, 1:2] // cfg.MODEL.PIXELS).long()
    maxW = cfg.MODEL.WIDTH // cfg.MODEL.PIXELS - 1
    maxH = cfg.MODEL.HEIGHT // cfg.MODEL.PIXELS - 1
    px = px.clamp_(0, maxW - 1)
    py = py.clamp_(0, maxH - 1)
    homo = homo.to(pts.device)
    x_dominator = pts[:, 0] * homo[0, 0, py[:, 0], px[:, 0]] + pts[:, 1] * homo[0, 1, py[:, 0], px[:, 0]] + homo[0, 2, py[:, 0], px[:, 0]]
    y_dominator = pts[:, 0] * homo[1, 0, py[:, 0], px[:, 0]] + pts[:, 1] * homo[1, 1, py[:, 0], px[:, 0]] + homo[1, 2, py[:, 0], px[:, 0]]
    noiminator = pts[:, 0] * homo[2, 0, py[:, 0], px[:, 0]] + pts[:, 1] * homo[2, 1, py[:, 0], px[:, 0]] + homo[2, 2, py[:, 0], px[:, 0]]
    new_kp_x = x_dominator / noiminator
    new_kp_y = y_dominator / noiminator
    return torch.stack([new_kp_x, new_kp_y], 1)

def MotionDistanceMeasure(motion1, motion2):
    m1 = np.sqrt(np.sum(motion1 ** 2))
    m2 = np.sqrt(np.sum(motion2 ** 2))
    dm = np.abs(m1 - m2)
    rot = lambda x: math.atan2(x[1], x[0]) / math.pi * 180
    r1 = rot(motion1)
    r2 = rot(motion2)
    dr = np.abs(r1 - r2)
    if dr > 180:
        dr = 360 - dr
    return (dm >= cfg.Threshold.MANG) or (dr >= cfg.Threshold.ROT)

def multiHomoEstimate(motion, kp):
    dev = motion.device
    new_kp = torch.cat([kp[1:2, :], kp[0:1, :]], 0) + motion[2:, :]
    new_points_numpy = new_kp.detach().transpose(0, 1).contiguous().cpu().numpy()
    old_points = torch.stack([kp[1, :], kp[0, :]], 1).to(dev)
    old_points_numpy = torch.cat([kp[1:2, :], kp[0:1, :]], 0).detach().transpose(0, 1).contiguous().cpu().numpy()
    motion_numpy = new_points_numpy - old_points_numpy

    pred_Y = KMeans(n_clusters=2, random_state=2).fit_predict(motion_numpy)
    if np.sum(pred_Y) > cfg.MODEL.TOPK / 2:
        pred_Y = 1 - pred_Y
    c1_idx = (pred_Y == 0).nonzero()[0]
    c1_old = old_points_numpy[c1_idx, :]
    c1_new = new_points_numpy[c1_idx, :]

    Homo, _ = cv2.findHomography(c1_old, c1_new, cv2.RANSAC)
    if Homo is None:
        Homo = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]], dtype=np.float32)
    dominator = (Homo[2, 0] * old_points_numpy[:, 0] + Homo[2, 1] * old_points_numpy[:, 1] + Homo[2, 2])
    proj_np = np.stack([
        (Homo[0, 0] * old_points_numpy[:, 0] + Homo[0, 1] * old_points_numpy[:, 1] + Homo[0, 2]) / dominator,
        (Homo[1, 0] * old_points_numpy[:, 0] + Homo[1, 1] * old_points_numpy[:, 1] + Homo[1, 2]) / dominator
    ], 1).astype(np.float32)
    new_points_projected = torch.from_numpy(proj_np).to(dev).permute(1, 0)

    c2_idx = (pred_Y == 1).nonzero()[0]
    attr = np.zeros_like(new_points_numpy[:, 0:1])
    c2_old = old_points_numpy[c2_idx, :]
    c2_new = new_points_numpy[c2_idx, :]
    attr[c2_idx, :] = np.expand_dims(np.ones_like(c2_idx), 1)

    c1_m = c1_new - c1_old
    c2_m = c2_new - c2_old
    c1_mean = np.mean(c1_m, 0)
    c2_mean = np.mean(c2_m, 0)
    use_two = (np.sum(pred_Y) > cfg.MODEL.THRESHOLDPOINT) and MotionDistanceMeasure(c1_mean, c2_mean)

    if not hasattr(multiHomoEstimate, "_grid_cache_np"):
        multiHomoEstimate._grid_cache_np = {}
    key = (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS)
    if key not in multiHomoEstimate._grid_cache_np:
        mx, my = np.meshgrid(np.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
                             np.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS))
        multiHomoEstimate._grid_cache_np[key] = (mx.astype(np.float32), my.astype(np.float32))
    meshes_x, meshes_y = multiHomoEstimate._grid_cache_np[key]

    if use_two:
        Homo_2, _ = cv2.findHomography(c2_old, c2_new, cv2.RANSAC)
        if Homo_2 is None:
            Homo_2 = Homo

        xd = Homo[0, 0] * meshes_x + Homo[0, 1] * meshes_y + Homo[0, 2]
        yd = Homo[1, 0] * meshes_x + Homo[1, 1] * meshes_y + Homo[1, 2]
        nd = Homo[2, 0] * meshes_x + Homo[2, 1] * meshes_y + Homo[2, 2]
        projected_1 = np.stack([xd / nd, yd / nd], 2).reshape(-1, 2)

        xd = Homo_2[0, 0] * meshes_x + Homo_2[0, 1] * meshes_y + Homo_2[0, 2]
        yd = Homo_2[1, 0] * meshes_x + Homo_2[1, 1] * meshes_y + Homo_2[1, 2]
        nd = Homo_2[2, 0] * meshes_x + Homo_2[2, 1] * meshes_y + Homo_2[2, 2]
        projected_2 = np.stack([xd / nd, yd / nd], 2).reshape(-1, 2)

        dx = np.expand_dims(new_points_numpy[:, 0], 0) - meshes_x.reshape(-1, 1)
        dy = np.expand_dims(new_points_numpy[:, 1], 0) - meshes_y.reshape(-1, 1)
        dist2 = dx * dx + dy * dy
        mask = (dist2 < (cfg.MODEL.RADIUS ** 2))
        mask_val = (mask.astype(np.float32) * attr.transpose(1, 0))
        dist = np.sum(mask_val, 1) / (np.sum(mask, 1) + 1e-9)
        blend = np.expand_dims(dist, 1)
        project_pos = (blend * projected_2 + (1 - blend) * projected_1).reshape(
            cfg.MODEL.HEIGHT // cfg.MODEL.PIXELS, cfg.MODEL.WIDTH // cfg.MODEL.PIXELS, 2
        )
        meshes_projected = torch.from_numpy(project_pos.astype(np.float32)).to(dev).permute(2, 0, 1)

        if not hasattr(multiHomoEstimate, "_grid_cache_torch"):
            multiHomoEstimate._grid_cache_torch = {}
        if key not in multiHomoEstimate._grid_cache_torch:
            gx, gy = torch.meshgrid(
                torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
                torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS)
            )
            gx = gx.float().permute(1, 0).contiguous()
            gy = gy.float().permute(1, 0).contiguous()
            multiHomoEstimate._grid_cache_torch[key] = (gx, gy)
        gx, gy = multiHomoEstimate._grid_cache_torch[key]
        gx = gx.to(dev)
        gy = gy.to(dev)
        x_motions = gx - meshes_projected[0]
        y_motions = gy - meshes_projected[1]
        homo_cal = HomoCalc(torch.stack([gx, gy], 0), meshes_projected)
        project_pts = HomoProj(homo_cal, old_points)
        new_points_projected = project_pts.to(dev).permute(1, 0)
    else:
        Homo_t = torch.from_numpy(Homo.astype(np.float32)).to(dev)
        if not hasattr(multiHomoEstimate, "_mesh3_cache"):
            multiHomoEstimate._mesh3_cache = {}
        if key not in multiHomoEstimate._mesh3_cache:
            gx, gy = torch.meshgrid(
                torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
                torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS)
            )
            gx = gx.float().permute(1, 0).contiguous()
            gy = gy.float().permute(1, 0).contiguous()
            gz = torch.ones_like(gx)
            multiHomoEstimate._mesh3_cache[key] = torch.stack([gx, gy, gz], 0).contiguous()
        meshes = multiHomoEstimate._mesh3_cache[key].to(dev)
        mp = torch.mm(Homo_t, meshes.view(3, -1)).view_as(meshes)
        x_motions = meshes[0] - (mp[0] / mp[2])
        y_motions = meshes[1] - (mp[1] / mp[2])

    grids = torch.stack(torch.meshgrid(
        torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
        torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS)
    ), 0).to(dev).permute(0, 2, 1).reshape(2, -1).permute(1, 0).unsqueeze(2).float()

    projected_motion = torch.stack([x_motions, y_motions], 2).view(-1, 2, 1).to(dev)
    redisual_kp_motion = new_points_projected - torch.cat([kp[1:2, :], kp[0:1, :]], 0)

    motion[:2, :] = motion[:2, :] + motion[2:, :]
    motion = motion.unsqueeze(0).repeat(grids.shape[0], 1, 1)
    motion[:, :2, :] = (motion[:, :2, :] - grids) / cfg.MODEL.WIDTH
    origin_motion = motion[:, 2:, :] / cfg.MODEL.FLOWC
    motion[:, 2:, :] = (redisual_kp_motion.unsqueeze(0) - motion[:, 2:, :]) / cfg.MODEL.FLOWC

    return motion, projected_motion / cfg.MODEL.FLOWC, origin_motion

def calculate_distortion_loss_inter(m0, m1, m2, rot=0):
    b, dim, time, number = m0.shape
    invP = 1.0 / float(cfg.MODEL.PIXELS)
    R = torch.tensor([[0., 1.], [-1., 0.]] if rot == 0 else [[0., -1.], [1., 0.]], device=m0.device, dtype=m0.dtype)
    v1 = (m0 - m1).mul(invP).reshape(b, dim, -1)
    v2 = (m2 - m1).mul(invP).reshape(b, dim, -1)
    rv1 = torch.matmul(R.unsqueeze(0), v1)
    loss = (rv1 - v2).pow_(2).reshape(b, dim, time, number)
    return loss.sum(dim=2).mean()

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            ph = max(self.k[0] - (ih % self.stride[0] or self.stride[0]), 0)
            pw = max(self.k[1] - (iw % self.stride[1] or self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            return (pl, pr, pt, pb)
        return self.padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.flatten(-2).median(dim=-1).values
        return x

def SingleMotionPropagate(x_flow, y_flow, pts):
    pts = pts.float()
    medfilt = MedianPool2d(same=True)

    B, _, H, W = x_flow.shape
    dev = x_flow.device
    dt = x_flow.dtype

    if not hasattr(SingleMotionPropagate, "_grid_cache"):
        SingleMotionPropagate._grid_cache = {}
    key = (H, W)
    if key not in SingleMotionPropagate._grid_cache:
        gx, gy = torch.meshgrid(torch.arange(W, device=dev), torch.arange(H, device=dev))
        gx = gx.permute(1, 0).contiguous().to(dt)
        gy = gy.permute(1, 0).contiguous().to(dt)
        grids = torch.stack([gx, gy], 0).unsqueeze(0)  # 1,2,H,W
        SingleMotionPropagate._grid_cache[key] = grids
    grids = SingleMotionPropagate._grid_cache[key]

    new_points_S = grids + torch.cat([x_flow, y_flow], 1)
    new_points = new_points_S[0, :, pts[:, 2].long(), pts[:, 3].long()].permute(1, 0)
    old_points = grids[0, :, pts[:, 2].long(), pts[:, 3].long()].permute(1, 0)

    old_np = old_points.detach().cpu().numpy()
    new_np = new_points.detach().cpu().numpy()
    Hm, _ = cv2.findHomography(old_np, new_np, cv2.RANSAC)
    if Hm is None:
        Hm = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=np.float32)
    Hm = torch.from_numpy(Hm.astype(np.float32)).to(dev)

    if not hasattr(SingleMotionPropagate, "_mesh3_cache"):
        SingleMotionPropagate._mesh3_cache = {}
    k2 = (H, W, cfg.MODEL.PIXELS)
    if k2 not in SingleMotionPropagate._mesh3_cache:
        mx, my = torch.meshgrid(torch.arange(0, W, cfg.MODEL.PIXELS, device=dev),
                                torch.arange(0, H, cfg.MODEL.PIXELS, device=dev))
        mx = mx.float().permute(1, 0).contiguous()
        my = my.float().permute(1, 0).contiguous()
        mz = torch.ones_like(mx)
        meshes = torch.stack([mx, my, mz], 0).contiguous()
        centers = torch.stack([mx.flatten(), my.flatten()], 1)  # (G,2)
        SingleMotionPropagate._mesh3_cache[k2] = (meshes, centers, mx.shape)
    meshes, centers, hw = SingleMotionPropagate._mesh3_cache[k2]

    mp = torch.mm(Hm, meshes.view(3, -1)).view_as(meshes)
    x_motions = meshes[0] - (mp[0] / (mp[2] + 1e-5))
    y_motions = meshes[1] - (mp[1] / (mp[2] + 1e-5))

    op = pts[:, 2:4].to(dev, torch.float32)
    denom = op[:, 1:2] * Hm[2, 0] + op[:, 0:1] * Hm[2, 1] + Hm[2, 2] + 1e-5
    nx = (op[:, 1:2] * Hm[0, 0] + op[:, 0:1] * Hm[0, 1] + Hm[0, 2]) / denom
    ny = (op[:, 1:2] * Hm[1, 0] + op[:, 0:1] * Hm[1, 1] + Hm[1, 2]) / denom
    new_homo = torch.cat([nx, ny], 1)

    new_flow = new_points_S[0, :, op[:, 0].long(), op[:, 1].long()].permute(1, 0)
    residual = (new_flow - new_homo).to(dt)  # (N,2)

    G = centers.size(0)
    R2 = float(cfg.MODEL.RADIUS) ** 2
    dist2 = (centers.unsqueeze(1) - op.unsqueeze(0)).pow(2).sum(-1)  # (G,N)
    mask = dist2 < R2

    res_exp = residual.unsqueeze(0).expand(G, -1, -1)  # (G,N,2)
    res_exp = res_exp.masked_fill(~mask.unsqueeze(-1), float('nan'))
    med = torch.nanmedian(res_exp, dim=1).values  # (G,2)
    med = torch.where(torch.isnan(med), torch.zeros_like(med), med)
    temp_x = med[:, 0].view(hw)
    temp_y = med[:, 1].view(hw)

    x_motions = x_motions + temp_x
    y_motions = y_motions + temp_y

    x_motion_mesh = medfilt(x_motions.unsqueeze(0).unsqueeze(0))
    y_motion_mesh = medfilt(y_motions.unsqueeze(0).unsqueeze(0))
    return torch.cat([x_motion_mesh, y_motion_mesh], 1)

def MultiMotionPropagate(x_flow, y_flow, pts):
    medfilt = MedianPool2d(same=True)
    pts = pts.float()
    B, _, H, W = x_flow.shape
    dev = x_flow.device
    dt = x_flow.dtype

    if not hasattr(MultiMotionPropagate, "_grid_cache"):
        MultiMotionPropagate._grid_cache = {}
    keyg = (H, W)
    if keyg not in MultiMotionPropagate._grid_cache:
        gx, gy = torch.meshgrid(torch.arange(W, device=dev), torch.arange(H, device=dev))
        gx = gx.permute(1, 0).contiguous().to(dt)
        gy = gy.permute(1, 0).contiguous().to(dt)
        grids = torch.stack([gx, gy], 0).unsqueeze(0)
        MultiMotionPropagate._grid_cache[keyg] = grids
    grids = MultiMotionPropagate._grid_cache[keyg]

    new_points_S = grids + torch.cat([x_flow, y_flow], 1)
    new_points = new_points_S[0, :, pts[:, 2].long(), pts[:, 3].long()].permute(1, 0)
    old_points = grids[0, :, pts[:, 2].long(), pts[:, 3].long()].permute(1, 0)

    old_np = old_points.detach().cpu().numpy()
    new_np = new_points.detach().cpu().numpy()
    motion_np = new_np - old_np

    pred_Y = KMeans(n_clusters=2, random_state=2).fit_predict(motion_np)
    if np.sum(pred_Y) > cfg.MODEL.TOPK / 2:
        pred_Y = 1 - pred_Y

    c1 = (pred_Y == 0).nonzero()[0]
    c1_old = old_np[c1, :]
    c1_new = new_np[c1, :]

    H1, _ = cv2.findHomography(c1_old, c1_new, cv2.RANSAC)
    if H1 is None:
        H1 = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=np.float32)

    denom = (H1[2, 0] * old_np[:, 0] + H1[2, 1] * old_np[:, 1] + H1[2, 2])
    proj1_np = np.stack([
        (H1[0, 0] * old_np[:, 0] + H1[0, 1] * old_np[:, 1] + H1[0, 2]) / denom,
        (H1[1, 0] * old_np[:, 0] + H1[1, 1] * old_np[:, 1] + H1[1, 2]) / denom
    ], 1)

    c2 = (pred_Y == 1).nonzero()[0]
    attr = np.zeros_like(new_np[:, 0:1])
    c2_old = old_np[c2, :]
    c2_new = new_np[c2, :]
    c1_m = c1_new - c1_old
    c2_m = c2_new - c2_old
    use_two = (np.sum(pred_Y) > cfg.MODEL.THRESHOLDPOINT) and MotionDistanceMeasure(np.mean(c1_m, 0), np.mean(c2_m, 0))

    if not hasattr(MultiMotionPropagate, "_mesh_cache_np"):
        MultiMotionPropagate._mesh_cache_np = {}
    keym = (H, W, cfg.MODEL.PIXELS)
    if keym not in MultiMotionPropagate._mesh_cache_np:
        mx, my = np.meshgrid(np.arange(0, W, cfg.MODEL.PIXELS), np.arange(0, H, cfg.MODEL.PIXELS))
        MultiMotionPropagate._mesh_cache_np[keym] = (mx.astype(np.float32), my.astype(np.float32))
    mx_np, my_np = MultiMotionPropagate._mesh_cache_np[keym]

    if use_two:
        attr[c2, :] = np.expand_dims(np.ones_like(c2), 1)
        H2, _ = cv2.findHomography(c2_old, c2_new, cv2.RANSAC)
        if H2 is None:
            H2 = H1

        xd = H1[0, 0] * mx_np + H1[0, 1] * my_np + H1[0, 2]
        yd = H1[1, 0] * mx_np + H1[1, 1] * my_np + H1[1, 2]
        nd = H1[2, 0] * mx_np + H1[2, 1] * my_np + H1[2, 2]
        proj_mesh1 = np.stack([xd / nd, yd / nd], 2).reshape(-1, 2)

        xd = H2[0, 0] * mx_np + H2[0, 1] * my_np + H2[0, 2]
        yd = H2[1, 0] * mx_np + H2[1, 1] * my_np + H2[1, 2]
        nd = H2[2, 0] * mx_np + H2[2, 1] * my_np + H2[2, 2]
        proj_mesh2 = np.stack([xd / nd, yd / nd], 2).reshape(-1, 2)

        dx = np.expand_dims(new_np[:, 0], 0) - mx_np.reshape(-1, 1)
        dy = np.expand_dims(new_np[:, 1], 0) - my_np.reshape(-1, 1)
        dist2 = dx * dx + dy * dy
        mask = (dist2 < (cfg.MODEL.RADIUS ** 2))
        w = (mask.astype(np.float32) * attr.transpose(1, 0))
        w = np.sum(w, 1) / (np.sum(mask, 1) + 1e-9)
        blend = np.expand_dims(w, 1)
        proj_pos = (blend * proj_mesh2 + (1 - blend) * proj_mesh1).reshape(
            cfg.MODEL.HEIGHT // cfg.MODEL.PIXELS, cfg.MODEL.WIDTH // cfg.MODEL.PIXELS, 2
        )
        meshes_projected = torch.from_numpy(proj_pos.astype(np.float32)).to(dev).permute(2, 0, 1)

        if not hasattr(MultiMotionPropagate, "_grid_hw_cache"):
            MultiMotionPropagate._grid_hw_cache = {}
        if keym not in MultiMotionPropagate._grid_hw_cache:
            gx, gy = torch.meshgrid(torch.arange(0, W, cfg.MODEL.PIXELS, device=dev),
                                    torch.arange(0, H, cfg.MODEL.PIXELS, device=dev))
            gx = gx.float().permute(1, 0).contiguous()
            gy = gy.float().permute(1, 0).contiguous()
            MultiMotionPropagate._grid_hw_cache[keym] = (gx, gy)
        gx, gy = MultiMotionPropagate._grid_hw_cache[keym]
        x_motions = gx - meshes_projected[0]
        y_motions = gy - meshes_projected[1]

        homo_cal = HomoCalc(torch.stack([gx, gy], 0), meshes_projected)
        project_pts = HomoProj(homo_cal, old_points)
        new_points_projected = project_pts  # (N,2) on same dev
    else:
        H1_t = torch.from_numpy(H1.astype(np.float32)).to(dev)
        if not hasattr(MultiMotionPropagate, "_mesh3_cache_t"):
            MultiMotionPropagate._mesh3_cache_t = {}
        if keym not in MultiMotionPropagate._mesh3_cache_t:
            gx, gy = torch.meshgrid(torch.arange(0, W, cfg.MODEL.PIXELS, device=dev),
                                    torch.arange(0, H, cfg.MODEL.PIXELS, device=dev))
            gx = gx.float().permute(1, 0).contiguous()
            gy = gy.float().permute(1, 0).contiguous()
            gz = torch.ones_like(gx)
            MultiMotionPropagate._mesh3_cache_t[keym] = torch.stack([gx, gy, gz], 0).contiguous()
        meshes = MultiMotionPropagate._mesh3_cache_t[keym].to(dev)
        mp = torch.mm(H1_t, meshes.view(3, -1)).view_as(meshes)
        x_motions = meshes[0] - (mp[0] / mp[2])
        y_motions = meshes[1] - (mp[1] / mp[2])
        new_points_projected = torch.from_numpy(proj1_np.astype(np.float32)).to(dev)

    op = old_points  # (N,2)
    if use_two:
        residual = -(new_points_projected - new_points)  # (N,2)
    else:
        residual = new_points - new_points_projected  # (N,2)

    if not hasattr(MultiMotionPropagate, "_centers_cache"):
        MultiMotionPropagate._centers_cache = {}
    if keym not in MultiMotionPropagate._centers_cache:
        cx = torch.arange(0, W, cfg.MODEL.PIXELS, device=dev, dtype=torch.float32)
        cy = torch.arange(0, H, cfg.MODEL.PIXELS, device=dev, dtype=torch.float32)
        CX, CY = torch.meshgrid(cx, cy)
        centers = torch.stack([CX.permute(1, 0).reshape(-1), CY.permute(1, 0).reshape(-1)], 1)
        MultiMotionPropagate._centers_cache[keym] = centers
    centers = MultiMotionPropagate._centers_cache[keym]

    G = centers.size(0)
    R2 = float(cfg.MODEL.RADIUS) ** 2
    dist2 = (centers.unsqueeze(1) - op.unsqueeze(0)).pow(2).sum(-1)  # (G,N)
    mask = dist2 < R2
    res_exp = residual.unsqueeze(0).expand(G, -1, -1)
    res_exp = res_exp.masked_fill(~mask.unsqueeze(-1), float('nan'))
    med = torch.nanmedian(res_exp, dim=1).values
    med = torch.where(torch.isnan(med), torch.zeros_like(med), med)
    temp_x = med[:, 0].view_as(x_motions)
    temp_y = med[:, 1].view_as(y_motions)

    x_motions = x_motions + temp_x
    y_motions = y_motions + temp_y

    x_motion_mesh = medfilt(x_motions.unsqueeze(0).unsqueeze(0))
    y_motion_mesh = medfilt(y_motions.unsqueeze(0).unsqueeze(0))
    return torch.cat([x_motion_mesh, y_motion_mesh], 1)

def mesh_warp_frame(frame, x_motion, y_motion):
    dev = frame.device
    dt = frame.dtype
    N, _, H, W = frame.shape

    key = (H, W, cfg.MODEL.PIXELS, dev, dt)
    if not hasattr(mesh_warp_frame, "_cache"):
        mesh_warp_frame._cache = {}
    cache = mesh_warp_frame._cache

    if key not in cache:
        gx, gy = torch.meshgrid(
            torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS, device=dev),
            torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS, device=dev)
        )
        gx = gx.permute(1, 0).to(dt).contiguous()
        gy = gy.permute(1, 0).to(dt).contiguous()
        src_grids = torch.stack([gx, gy], 0).unsqueeze(0)  # 1,2,G_H,G_W

        ox, oy = torch.meshgrid(
            torch.arange(0, cfg.MODEL.WIDTH, device=dev),
            torch.arange(0, cfg.MODEL.HEIGHT, device=dev)
        )
        ox = ox.permute(1, 0).to(dt).contiguous()
        oy = oy.permute(1, 0).to(dt).contiguous()
        origin_kp = torch.stack([ox, oy], 0)  # 2,H,W

        cache[key] = (src_grids, origin_kp)

    src_grids, origin_kp = cache[key]
    des_grids = (src_grids + torch.cat([x_motion, y_motion], 1)).contiguous()

    proj = torch.empty((N, H, W, 2), device=dev, dtype=dt)
    okp_flat = origin_kp.view(2, -1).permute(1, 0)

    for i in range(des_grids.shape[0]):
        Hm = HomoCalc(src_grids[0], des_grids[i])
        pkp = HomoProj(Hm, okp_flat).permute(1, 0).contiguous().view_as(origin_kp).permute(1, 2, 0)
        proj[i] = pkp

    proj[..., 0] = proj[..., 0] / cfg.MODEL.WIDTH * 2. - 1.
    proj[..., 1] = proj[..., 1] / cfg.MODEL.HEIGHT * 2. - 1.

    return F.grid_sample(frame, proj, align_corners=True)

def warpListImage(images, x_motion, y_motion):
    frames = np.concatenate(images, 0).astype(np.float32, copy=False)
    xm = np.transpose(x_motion, (2, 0, 1))[..., None].astype(np.float32, copy=False)
    ym = np.transpose(y_motion, (2, 0, 1))[..., None].astype(np.float32, copy=False)

    frames_t = torch.from_numpy(frames).contiguous()
    xm_t = torch.from_numpy(xm).permute(0, 3, 1, 2).contiguous()
    ym_t = torch.from_numpy(ym).permute(0, 3, 1, 2).contiguous()

    return mesh_warp_frame(frames_t, xm_t, ym_t)

def _safe_to_uint8(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _warp_one_frame(frame_chw, dx_hw, dy_hw):
    """
    frame_chw: [C,H,W]
    dx_hw, dy_hw: [G_H, G_W] 或 [G_W, G_H]
    """
    frame_list = [frame_chw[np.newaxis, ...]]

    dx_hw = np.asarray(dx_hw, dtype=np.float32)
    dy_hw = np.asarray(dy_hw, dtype=np.float32)

    candidates = [
        (dx_hw[..., np.newaxis], dy_hw[..., np.newaxis], "direct"),
        (dx_hw.T[..., np.newaxis], dy_hw.T[..., np.newaxis], "transpose"),
    ]

    last_err = None
    for dx, dy, mode in candidates:
        try:
            print(f"[warp try] mode={mode}, dx shape={dx.shape}, dy shape={dy.shape}")
            out = warpListImage(frame_list, dx, dy)
            out = _safe_to_uint8(out)
            stab_small = np.transpose(out[0], (1, 2, 0))
            print(f"[warp ok] mode={mode}")
            return stab_small
        except Exception as e:
            print(f"[warp fail] mode={mode}, err={e}")
            last_err = e

    raise RuntimeError("All warp candidates failed.") from last_err