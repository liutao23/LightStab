# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : LightKeypointsDetection.py
# @Time  : 2025/10/16 下午6:12

import torch.nn as nn
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import time
import argparse
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from configs.config import cfg
from OffTheShelfModule.point_module.rf_det_so import RFDetSO
from OffTheShelfModule.point_module.superpoint import SuperPoint
from OffTheShelfModule.point_module.sift import SIFT
from OffTheShelfModule.point_module.aliked import ALIKED
from OffTheShelfModule.point_module.dog_hardnet import  DoGHardNet
from OffTheShelfModule.point_module.disk import DISK
from OffTheShelfModule.point_module.modules.xfeat import XFeat
from OffTheShelfModule.point_module import dad

def tensor_to_pil(img_tensor: torch.Tensor):
    """
    img_tensor: [1,3,H,W] 或 [3,H,W]，值域通常是[0,1]
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().clamp(0, 1)
    img = (img * 255.0).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()  # [H,W,3]
    return Image.fromarray(img)

def draw_keypoints_on_image(pil_img: Image.Image, kpts_list, color=(255, 255, 0), radius=2):
    """
    kpts_list: List[Tensor[K,4]]，每个元素是 [b, 0, y, x]
    这里仅画第0张（与你的示例一致）
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    if len(kpts_list) == 0:
        return img

    kpts0 = kpts_list[0]  # Tensor[K,4]
    if kpts0.numel() > 0:
        ys = kpts0[:, 2].detach().cpu().numpy()
        xs = kpts0[:, 3].detach().cpu().numpy()
        for x, y in zip(xs, ys):
            x, y = int(x), int(y)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    return img

class KeypointDetectionSSC(nn.Module):
    def __init__(self, point_method='RFdet', topK=10000, ssc_num=512,
                 ensemble_weights=None,
                 diversity_mode='soft',
                 diversity_tau=8.0,
                 diversity_lambda=0.5,
                 diversity_sigma=8.0,
                 use_ssc=False,
                 ssc_tolerance=0.1):
        super(KeypointDetectionSSC, self).__init__()
        self.TOPK_ssc = ssc_num
        self.TOPK = topK
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diversity_mode = diversity_mode
        self.diversity_tau = float(diversity_tau)
        self.diversity_lambda = float(diversity_lambda)
        self.diversity_sigma = float(diversity_sigma)
        self.use_ssc = use_ssc
        self.ssc_tolerance = ssc_tolerance

        if isinstance(point_method, (list, tuple)):
            self.point_method = 'ensemble'
            self.ensemble_methods = list(point_method)
            self.ensemble_weights = ({m: 1.0 for m in self.ensemble_methods}
                                     if ensemble_weights is None
                                     else {m: float(ensemble_weights.get(m, 1.0)) for m in self.ensemble_methods})
            self.detectors = {m: self._build_detector(m, self.TOPK) for m in self.ensemble_methods}
        else:
            self.point_method = point_method
            if point_method == "RFdet":
                self.detect = RFDetSO(
                    score_com_strength=100.0, scale_com_strength=100.0,
                    nms_thresh=0.0, nms_ksize=5, topk=self.TOPK,
                    gauss_ksize=15, gauss_sigma=0.5,
                    ksize=3, padding=1, dilation=1,
                    scale_list=[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
                ).eval()
                checkpoint = torch.load('OffTheShelfModule/point_module/weight/RFDet_weights.pth',
                                        map_location=self.device)
                self.detect.load_state_dict(checkpoint, strict=False)
                self.detect = self.detect.to(self.device)
            elif point_method == 'superpoint':
                self.detect = SuperPoint(max_num_keypoints=self.TOPK).eval().to(self.device)
            elif point_method == 'aliked':
                self.detect = ALIKED(max_num_keypoints=self.TOPK).eval().to(self.device)
            elif point_method == 'sift':
                self.detect = SIFT(max_num_keypoints=self.TOPK).eval().to(self.device)
            elif point_method == 'disk':
                self.detect = DISK(max_num_keypoints=self.TOPK).eval().to(self.device)
            elif point_method == 'dog_hardnet':
                self.detect = DoGHardNet(max_num_keypoints=self.TOPK).eval().to(self.device)
            elif point_method == 'xfeat':
                self.detect = XFeat().eval().to(self.device)
            elif point_method == 'dad':
                self.detect = dad.load_DaD().eval().to(self.device)
            else:
                self.detect = dict(maxCorners=self.TOPK, qualityLevel=0.3, minDistance=7, blockSize=7)

    # =================== SSC（GPU） ===================
    # https://www.researchgate.net/publication/323388062_Efficient_adaptive_non-maximal_suppression_algorithms_for_homogeneous_spatial_keypoint_distribution
    def ssc(self, keypoints: torch.Tensor, num_ret_points: int, tolerance: float, cols: int, rows: int):
        """
        纯 GPU 近似 SSC：把候选点 rasterize 到稀疏分数图上，通过二分 kernel size 做 NMS，
        直到保留点数处于 [k_min, k_max]。保持输入/输出格式与数量（补足/截断）。
        keypoints: [N,4] [b, 0, y, x]
        """
        device = keypoints.device
        N = keypoints.size(0)
        if N < num_ret_points:
            raise ValueError(
                f"Number of input feature points ({N}) is less than the requested number ({num_ret_points})")

        # 用“输入顺序优先级”近似分数：越早出现分数越高（也可换成 detector score）
        # 为了稳定，分数用递减序列（避免相等导致 NMS 多解）
        # 你也可以把真正的 score 传进来替代这里的 rank_score
        rank = torch.arange(N, device=device, dtype=torch.float32)
        rank_score = (N - rank)  # 大者优先

        y = keypoints[:, 2].to(torch.long)
        x = keypoints[:, 3].to(torch.long)

        # 稀疏分数图：H x W
        H, W = rows, cols
        score_img = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
        score_img[0, 0, y, x] = torch.maximum(score_img[0, 0, y, x], rank_score)

        # 二分 kernel 大小（奇数），用 max-pooling 实现 NMS
        k_min = int(round(num_ret_points * (1 - tolerance)))
        k_max = int(round(num_ret_points * (1 + tolerance)))
        # 经验范围：从 1 到 max(H,W)（可以更窄：初值根据 sqrt(N/num)）
        low = 1
        high = max(1, int(max(H, W) // max(1, int((N / max(1, num_ret_points)) ** 0.5))))
        high = max(high, 1)

        def nms_count(k):
            # k 保证奇数
            if k % 2 == 0: k += 1
            # padding 让池化中心对齐
            pad = k // 2
            pooled = F.max_pool2d(score_img, kernel_size=k, stride=1, padding=pad)
            keep_map = (score_img == pooled) & (score_img > 0)
            cnt = int(keep_map.sum().item())
            return cnt, keep_map

        prev_k = -1
        selected_map = None
        while True:
            k = (low + high) // 2
            if k == prev_k or low > high:
                # 用上一次的结果
                break
            cnt, km = nms_count(k)
            if k_min <= cnt <= k_max:
                selected_map = km
                break
            elif cnt < k_min:
                # 留下太少 -> 减小 kernel
                high = k - 1
            else:
                # 留下太多 -> 增大 kernel
                low = k + 1
            prev_k = k

        if selected_map is None:
            # 兜底：用最后一次
            _, selected_map = nms_count(prev_k if prev_k > 0 else 1)

        # 从 keep_map 取回坐标：保持“输入顺序优先” -> 按 rank_score 递减/或按输入顺序筛
        keep = selected_map[0, 0, y, x]  # 直接按原 keypoints 的像素位置采样
        sel = torch.nonzero(keep, as_tuple=False).flatten()  # indices in original order

        # 精确数量：补足/截断
        M = int(sel.numel())
        if M < num_ret_points:
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[sel] = True
            remaining = torch.nonzero(~mask, as_tuple=False).flatten()
            need = num_ret_points - M
            if remaining.numel() < need:
                raise ValueError(
                    f"Available points to supplement ({int(remaining.numel())}) are insufficient to reach {num_ret_points}")
            sel = torch.cat([sel, remaining[:need]], dim=0)
        elif M > num_ret_points:
            sel = sel[:num_ret_points]

        return keypoints[sel]
    # =================== 更快的像素去重/裁剪 ===================
    def _clip_round_unique(self, xy, H, W):
        """
        输入: xy [N,2] (x,y) 张量/ndarray
        输出: LongTensor [M,2]，保持“首出现”顺序去重（与原语义一致）
        —— 全程 GPU，无 CPU 往返 ——
        """
        device = self.device
        if not isinstance(xy, torch.Tensor):
            xy = torch.as_tensor(xy, device=device)
        else:
            xy = xy.to(device)

        xy = xy.to(torch.float32)
        x = xy[:, 0].clamp_(0, W - 1).round_().to(torch.long)
        y = xy[:, 1].clamp_(0, H - 1).round_().to(torch.long)
        pix = y * W + x  # [N]

        # 取“首次出现”的索引：排序后找去重位置，再映射回原顺序
        vals, idx = torch.sort(pix)  # vals = pix 排序后, idx = 原索引
        keep = torch.ones_like(vals, dtype=torch.bool)
        keep[1:] = vals[1:] != vals[:-1]  # 与前一项不同的位置即为“该像素首次出现”
        first_idx = idx[keep]  # 首次出现的原索引
        first_idx, _ = torch.sort(first_idx, stable=True)  # 恢复为“首出现的原始顺序”

        x_unique = x[first_idx]
        y_unique = y[first_idx]
        return torch.stack([x_unique, y_unique], dim=1)

    # =================== 构建检测器 ===================
    def _build_detector(self, method, topK):
        if method == "RFdet":
            det = RFDetSO(
                score_com_strength=100.0, scale_com_strength=100.0,
                nms_thresh=0.0, nms_ksize=5, topk=topK,
                gauss_ksize=15, gauss_sigma=0.5,
                ksize=3, padding=1, dilation=1,
                scale_list=[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
            ).eval()
            ckpt = torch.load('OffTheShelfModule/point_module/weight/RFDet_weights.pth',
                              map_location=self.device)
            det.load_state_dict(ckpt, strict=False)
            if isinstance(det, nn.Module):
                det = det.to(self.device)
            return det

        elif method == 'superpoint':
            return SuperPoint(max_num_keypoints=self.TOPK).eval().to(self.device)
        elif method == 'aliked':
            return ALIKED(max_num_keypoints=self.TOPK).eval().to(self.device)
        elif method == 'sift':
            return SIFT(max_num_keypoints=self.TOPK).eval().to(self.device)
        elif method == 'disk':
            return DISK(max_num_keypoints=self.TOPK).eval().to(self.device)
        elif method == 'dog_hardnet':
            return DoGHardNet(max_num_keypoints=self.TOPK).eval().to(self.device)
        elif method == 'xfeat':
            return XFeat().eval().to(self.device)
        elif method == 'dad':
            return dad.load_DaD().eval().to(self.device)
        elif method in [None, 'shi', 'shitomasi', 'Shi-Tomasi']:
            return dict(maxCorners=self.TOPK, qualityLevel=0.3, minDistance=7, blockSize=7)
        else:
            raise ValueError(f'Unknown detector method: {method}')

    # =================== 分数筛选 ===================
    def _select_by_score(self, xy, scores, topk=None, conf_thresh=None):
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores).to(self.device)
        scores = scores.to(torch.float32)
        if xy.numel() == 0:
            return xy, scores

        if conf_thresh is not None:
            keep = scores >= float(conf_thresh)
            if keep.any():
                xy = xy[keep]
                scores = scores[keep]
            else:
                return xy[:0], scores[:0]

        if topk is not None and xy.shape[0] > topk:
            _, order = torch.topk(scores, k=topk, largest=True, sorted=True)
            xy = xy[order]
            scores = scores[order]
        return xy, scores

    # =================== 打包成 [b,0,y,x] ===================
    def _pack_kpts(self, b_idx, xy, device):
        if xy.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.long, device=device)
        y = xy[:, 1].to(torch.long)
        x = xy[:, 0].to(torch.long)
        b = torch.full_like(y, fill_value=b_idx)
        c = torch.zeros_like(y)
        return torch.stack([b, c, y, x], dim=1)

    # =================== 单张图抽取（与原行为一致） ===================
    def _extract_xy_scores_single(self, method, detector, im_1CHW):
        if isinstance(detector, nn.Module):
            try:
                det_dev = next(detector.parameters()).device
            except StopIteration:
                det_dev = self.device
        else:
            det_dev = im_1CHW.device

        if im_1CHW.device != det_dev:
            im_1CHW = im_1CHW.to(det_dev)
        device = im_1CHW.device
        H, W = im_1CHW.shape[2], im_1CHW.shape[3]

        with torch.no_grad():
            if method == 'RFdet':
                x = im_1CHW
                if x.shape[1] == 3:
                    x = torch.mean(x, dim=1, keepdim=True)
                im_rawsc, _, _ = detector(x)
                im_score = detector.process(im_rawsc)[0]
                if im_score.dim() == 4 and im_score.size(-1) == 1:
                    im_score = im_score.squeeze(-1)  # [1,H,W]
                s2d = im_score[0]  # [H,W]
                k = min(self.TOPK, H * W)
                vals, inds = torch.topk(s2d.view(-1), k=k, largest=True, sorted=True)
                y = (inds // W).to(torch.long)
                x = (inds % W).to(torch.long)
                xy = torch.stack([x, y], dim=1)
                xy = self._clip_round_unique(xy, H, W)
                scores = vals[:xy.shape[0]].to(torch.float32)
                return xy.to(self.device), scores.to(self.device)

            elif method in ['superpoint', 'aliked', 'sift', 'dog_hardnet', 'disk']:
                feat = detector.extract(im_1CHW)
                p = feat['keypoints'][0].cpu().numpy()  # [N,2] (x,y)
                s = feat['keypoint_scores'][0].cpu().numpy()  # [N]
                xy = self._clip_round_unique(p, H, W)
                if xy.shape[0] != p.shape[0]:
                    px = np.clip(np.rint(p[:, 0]), 0, W - 1).astype(np.int64)
                    py = np.clip(np.rint(p[:, 1]), 0, H - 1).astype(np.int64)
                    pix = (py * W + px)
                    first_score = {}
                    for i, fid in enumerate(pix):
                        if fid not in first_score:
                            first_score[fid] = float(s[i])
                    xy_pix = (xy[:, 1] * W + xy[:, 0]).cpu().numpy()
                    scores = torch.tensor([first_score[int(fid)] for fid in xy_pix], dtype=torch.float32)
                else:
                    scores = torch.from_numpy(s).to(torch.float32).to(device)
                return xy.to(self.device), scores.to(self.device)

            elif method == 'xfeat':
                feat = detector.detectAndCompute(im_1CHW, top_k=self.TOPK)[0]
                p = feat['keypoints'].cpu().numpy()
                s = feat['scores'].cpu().numpy()
                xy = self._clip_round_unique(p, H, W)
                if xy.shape[0] != p.shape[0]:
                    px = np.round(np.clip(p[:, 0], 0, W - 1)).astype(np.int64)
                    py = np.round(np.clip(p[:, 1], 0, H - 1)).astype(np.int64)
                    pix = (py * W + px)
                    first_score = {}
                    for i, fid in enumerate(pix):
                        if fid not in first_score:
                            first_score[fid] = float(s[i])
                    xy_pix = (xy[:, 1] * W + xy[:, 0]).cpu().numpy()
                    scores = torch.tensor([first_score[int(fid)] for fid in xy_pix], dtype=torch.float32)
                else:
                    scores = torch.from_numpy(s).to(torch.float32).to(device)
                return xy.to(self.device), scores.to(self.device)

            elif method == 'dad':
                det = detector.detect_from_path(im_1CHW, num_keypoints=self.TOPK)
                feat_xy = detector.to_pixel_coords(det["keypoints"], H, W).squeeze(0).cpu().numpy()
                s = det["keypoint_probs"]
                if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
                s = np.asarray(s)
                if s.ndim == 0:
                    s = s.reshape(1)
                elif s.ndim == 2 and (s.shape[0] == 1 or s.shape[1] == 1):
                    s = s.reshape(-1)
                elif s.ndim > 1:
                    s = s.max(axis=-1)
                s = s.astype(np.float32)

                xy = self._clip_round_unique(feat_xy, H, W)
                if xy.shape[0] != feat_xy.shape[0]:
                    px = np.round(np.clip(feat_xy[:, 0], 0, W - 1)).astype(np.int64)
                    py = np.round(np.clip(feat_xy[:, 1], 0, H - 1)).astype(np.int64)
                    pix = (py * W + px)
                    first_score = {}
                    for i, fid in enumerate(pix):
                        if fid not in first_score: first_score[fid] = float(s[i])
                    xy_pix = (xy[:, 1] * W + xy[:, 0]).cpu().numpy()
                    scores = torch.tensor([first_score.get(int(fid), 0.0) for fid in xy_pix],
                                          dtype=torch.float32, device=device)
                else:
                    scores = torch.from_numpy(s).to(torch.float32).to(self.device)
                return xy.to(self.device), scores.to(self.device)

            else:  # Shi-Tomasi
                x = im_1CHW
                if x.shape[1] == 3:
                    x = torch.mean(x, dim=1, keepdim=True)
                im = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
                p = cv2.goodFeaturesToTrack(im, mask=None, **detector)
                if p is None:
                    return (torch.zeros((0, 2), dtype=torch.long, device=self.device),
                            torch.zeros((0,), dtype=torch.float32, device=self.device))
                p = p.reshape(-1, 2)
                xy = self._clip_round_unique(p, H, W)
                scores = torch.ones((xy.shape[0],), dtype=torch.float32, device=self.device)
                return xy.to(self.device), scores.to(self.device)

    # =================== 融合并集 + 协同得分 ===================
    def _coalesce_union_with_scores(self, xy_list, score_list, weight_list, H, W, device):
        all_pix, all_val = [], []
        for xy, sc, w in zip(xy_list, score_list, weight_list):
            if xy is None or xy.numel() == 0:
                continue
            pix = xy[:, 1] * W + xy[:, 0]
            all_pix.append(pix.to(torch.long))
            all_val.append((sc.to(torch.float32)) * float(w))
        if len(all_pix) == 0:
            return (torch.zeros((0, 2), dtype=torch.long, device=device),
                    torch.zeros((0,), dtype=torch.float32, device=device))

        all_pix = torch.cat(all_pix, dim=0).to(device)
        all_val = torch.cat(all_val, dim=0).to(device)

        uniq, inv = torch.unique(all_pix, return_inverse=True)
        S = torch.zeros_like(uniq, dtype=torch.float32)
        S.scatter_add_(0, inv, all_val)

        x = (uniq % W).to(torch.long)
        y = (uniq // W).to(torch.long)
        xy_union = torch.stack([x, y], dim=1)
        return xy_union, S

    # =================== 多样性选择（逻辑不变，含早停） ===================
    def _select_diverse_topk(self, xy, S, topk, mode='soft', tau=8.0, lam=0.5, sigma=8.0):
        device = xy.device
        M = xy.shape[0]
        if M == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)

        eff = S.clone()
        sel = []
        xyf = xy.to(torch.float32)
        tau2 = tau * tau
        two_sigma2 = 2.0 * (sigma ** 2)

        for _ in range(min(topk, M)):
            idx = torch.argmax(eff)
            best = eff[idx].item()
            if best == float('-inf') or (best <= 0 and len(sel) > 0 and mode == 'soft'):
                break
            sel.append(idx.item())

            d2 = torch.sum((xyf - xyf[idx]) ** 2, dim=1)
            if mode == 'hard':
                eff[d2 <= tau2] = float('-inf')
            else:
                penalty = lam * torch.exp(- d2 / two_sigma2)
                eff.sub_(penalty)
                eff[idx] = float('-inf')

            if torch.isneginf(eff).all():
                break

        return torch.tensor(sel, dtype=torch.long, device=device)

    # =================== 前向（批内/写掩码向量化） ===================
    def forward(self, im_data):
        assert len(im_data.shape) == 4 and im_data.shape[1] in [1, 3]
        device = im_data.device
        B, C, H, W = im_data.shape

        if device != self.device:
            self.device = device
            if getattr(self, 'point_method', None) == 'ensemble':
                for m, det in self.detectors.items():
                    if isinstance(det, nn.Module):
                        self.detectors[m] = det.to(self.device)
            else:
                if isinstance(getattr(self, 'detect', None), nn.Module):
                    self.detect = self.detect.to(self.device)

        # ======== Ensemble 分支 ========
        if getattr(self, 'point_method', None) == 'ensemble':
            with torch.no_grad():
                kpts_list = []
                im_topk = torch.zeros((B, 1, H, W), device=device)
                for b in range(B):
                    im_b = im_data[b:b + 1]
                    xy_each, sc_each, w_each = [], [], []
                    for m in self.ensemble_methods:
                        det = self.detectors[m]
                        xy_m, s_m = self._extract_xy_scores_single(m, det, im_b)
                        xy_each.append(xy_m)
                        sc_each.append(s_m)
                        w_each.append(self.ensemble_weights.get(m, 1.0))

                    xy_union, S = self._coalesce_union_with_scores(xy_each, sc_each, w_each, H, W, device)
                    sel_idx = self._select_diverse_topk(
                        xy_union, S, topk=self.TOPK,
                        mode=self.diversity_mode, tau=self.diversity_tau,
                        lam=self.diversity_lambda, sigma=self.diversity_sigma
                    )
                    xy_sel = xy_union[sel_idx]
                    kpts = self._pack_kpts(b, xy_sel, device)

                    if self.use_ssc and kpts.shape[0] > 0:
                        try:
                            kpts = self.ssc(kpts, num_ret_points=self.TOPK_ssc,
                                            tolerance=self.ssc_tolerance, cols=W, rows=H)
                        except ValueError as e:
                            pass
                            # print(f"SSC警告: {e}, 使用原始关键点")

                    kpts_list.append(kpts)

                    # —— 向量化写掩码 —— #
                    if kpts.numel() > 0:
                        ys = kpts[:, 2]
                        xs = kpts[:, 3]
                        im_topk[b, 0, ys, xs] = im_topk.new_ones(ys.shape)

                return im_topk, kpts_list

        # ======== 单检测器分支 ========
        batch_size = B

        if self.point_method == 'RFdet':
            with torch.no_grad():
                if im_data.shape[1] == 3:
                    im_data = torch.mean(im_data, dim=1, keepdim=True).to(device)

                kpts_list = []
                im_topk = torch.zeros((B, 1, H, W), device=device)

                # 轻量分批（可按显存调整）
                step = 4
                for j in range(0, B, step):
                    clip = im_data[j:j + step]
                    im_rawsc, _, _ = self.detect(clip)
                    im_score = self.detect.process(im_rawsc)[0]  # [b,H,W,1] 或 [b,H,W]
                    if im_score.dim() == 4 and im_score.size(-1) == 1:
                        im_score = im_score.squeeze(-1)

                    for bi in range(im_score.shape[0]):
                        s2d = im_score[bi]
                        k = min(self.TOPK, H * W)
                        vals, inds = torch.topk(s2d.view(-1), k=k, largest=True, sorted=True)
                        y = (inds // W).to(torch.long)
                        x = (inds % W).to(torch.long)
                        xy = torch.stack([x, y], dim=1)
                        xy = self._clip_round_unique(xy, H, W)

                        kpts = self._pack_kpts(j + bi, xy, device)
                        if self.use_ssc and kpts.shape[0] > 0:
                            try:
                                kpts = self.ssc(kpts, num_ret_points=self.TOPK_ssc,
                                                tolerance=self.ssc_tolerance, cols=W, rows=H)
                            except ValueError as e:
                                print(f"SSC警告: {e}, 使用原始关键点")
                        kpts_list.append(kpts)

                        # 向量化写掩码
                        if kpts.numel() > 0:
                            ys = kpts[:, 2]
                            xs = kpts[:, 3]
                            im_topk[j + bi, 0, ys, xs] = im_topk.new_ones(ys.shape)

                return im_topk, kpts_list

        elif self.point_method in ['superpoint', 'aliked', 'sift', 'dog_hardnet', 'disk']:
            with torch.no_grad():
                kpts_list = []
                im_topk = torch.zeros((B, 1, H, W), device=device)
                for b in range(batch_size):
                    feat = self.detect.extract(im_data[b:b + 1])
                    p = feat['keypoints'][0].cpu().numpy()  # [N,2] (x,y)
                    s = feat['keypoint_scores'][0].cpu().numpy()  # [N]
                    xy = self._clip_round_unique(p, H, W)

                    if xy.shape[0] != p.shape[0]:
                        px = np.round(np.clip(p[:, 0], 0, W - 1)).astype(np.int64)
                        py = np.round(np.clip(p[:, 1], 0, H - 1)).astype(np.int64)
                        pix = (py * W + px)
                        first_score = {}
                        for i, fid in enumerate(pix):
                            if fid not in first_score:
                                first_score[fid] = float(s[i])
                        xy_pix = (xy[:, 1] * W + xy[:, 0]).cpu().numpy()
                        scores = torch.tensor([first_score[int(fid)] for fid in xy_pix], dtype=torch.float32, device=device)
                    else:
                        scores = torch.from_numpy(s).to(torch.float32).to(device)

                    xy, scores = self._select_by_score(xy, scores, topk=self.TOPK)
                    kpts = self._pack_kpts(b, xy, device)

                    if self.use_ssc and kpts.shape[0] > 0:
                        try:
                            kpts = self.ssc(kpts, num_ret_points=self.TOPK_ssc,
                                            tolerance=self.ssc_tolerance, cols=W, rows=H)
                        except ValueError as e:
                            print(f"SSC警告: {e}, 使用原始关键点")

                    kpts_list.append(kpts)

                    if kpts.numel() > 0:
                        ys = kpts[:, 2]
                        xs = kpts[:, 3]
                        im_topk[b, 0, ys, xs] = im_topk.new_ones(ys.shape)

                return im_topk, kpts_list

        elif self.point_method == 'xfeat':
            with torch.no_grad():
                kpts_list = []
                im_topk = torch.zeros((B, 1, H, W), device=device)
                for b in range(batch_size):
                    feat = self.detect.detectAndCompute(im_data[b:b + 1], top_k=self.TOPK)[0]
                    p = feat['keypoints'].cpu().numpy()
                    s = feat['scores'].cpu().numpy()
                    xy = self._clip_round_unique(p, H, W)

                    if xy.shape[0] != p.shape[0]:
                        px = np.round(np.clip(p[:, 0], 0, W - 1)).astype(np.int64)
                        py = np.round(np.clip(p[:, 1], 0, H - 1)).astype(np.int64)
                        pix = (py * W + px)
                        first_score = {}
                        for i, fid in enumerate(pix):
                            if fid not in first_score:
                                first_score[fid] = float(s[i])
                        xy_pix = (xy[:, 1] * W + xy[:, 0]).cpu().numpy()
                        scores = torch.tensor([first_score[int(fid)] for fid in xy_pix], dtype=torch.float32, device=device)
                    else:
                        scores = torch.from_numpy(s).to(torch.float32).to(device)

                    xy, scores = self._select_by_score(xy, scores, topk=self.TOPK)
                    kpts = self._pack_kpts(b, xy, device)

                    if self.use_ssc and kpts.shape[0] > 0:
                        try:
                            kpts = self.ssc(kpts, num_ret_points=self.TOPK_ssc,
                                            tolerance=self.ssc_tolerance, cols=W, rows=H)
                        except ValueError as e:
                            print(f"SSC警告: {e}, 使用原始关键点")

                    kpts_list.append(kpts)

                    if kpts.numel() > 0:
                        ys = kpts[:, 2]
                        xs = kpts[:, 3]
                        im_topk[b, 0, ys, xs] = im_topk.new_ones(ys.shape)

                return im_topk, kpts_list

        elif self.point_method == 'dad':
            with torch.no_grad():
                kpts_list = []
                im_topk = torch.zeros((B, 1, H, W), device=device)
                for b in range(batch_size):
                    det = self.detect.detect_from_path(im_data[b:b + 1], num_keypoints=self.TOPK)
                    feat_xy = self.detect.to_pixel_coords(det["keypoints"], H, W).squeeze(0).cpu().numpy()
                    s = det["keypoint_probs"]
                    if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
                    s = np.asarray(s)
                    if s.ndim == 0:
                        s = s.reshape(1)
                    elif s.ndim == 2 and (s.shape[0] == 1 or s.shape[1] == 1):
                        s = s.reshape(-1)
                    elif s.ndim > 1:
                        s = s.max(axis=-1)
                    s = s.astype(np.float32)

                    xy = self._clip_round_unique(feat_xy, H, W)
                    if xy.shape[0] != feat_xy.shape[0]:
                        px = np.round(np.clip(feat_xy[:, 0], 0, W - 1)).astype(np.int64)
                        py = np.round(np.clip(feat_xy[:, 1], 0, H - 1)).astype(np.int64)
                        pix = (py * W + px)
                        first_score = {}
                        for i, fid in enumerate(pix):
                            if fid not in first_score: first_score[fid] = float(s[i])
                        xy_pix = (xy[:, 1] * W + xy[:, 0]).cpu().numpy()
                        scores = torch.tensor([first_score.get(int(fid), 0.0) for fid in xy_pix],
                                              dtype=torch.float32, device=device)
                    else:
                        scores = torch.from_numpy(s).to(torch.float32).to(self.device)

                    xy, scores = self._select_by_score(xy, scores, topk=self.TOPK)
                    kpts = self._pack_kpts(b, xy, device)

                    if self.use_ssc and kpts.shape[0] > 0:
                        try:
                            kpts = self.ssc(kpts, num_ret_points=self.TOPK_ssc,
                                            tolerance=self.ssc_tolerance, cols=W, rows=H)
                        except ValueError as e:
                            print(f"SSC警告: {e}, 使用原始关键点")

                    kpts_list.append(kpts)

                    if kpts.numel() > 0:
                        ys = kpts[:, 2]
                        xs = kpts[:, 3]
                        im_topk[b, 0, ys, xs] = im_topk.new_ones(ys.shape)

                return im_topk, kpts_list

        else:  # Shi-Tomasi
            with torch.no_grad():
                if im_data.shape[1] == 3:
                    im_data = torch.mean(im_data, dim=1, keepdim=True)

                kpts_list = []
                im_topk = torch.zeros((B, 1, H, W), device=device)

                for b in range(batch_size):
                    im = (im_data[b, 0].cpu().numpy() * 255).astype(np.uint8)
                    p = cv2.goodFeaturesToTrack(im, mask=None, **self.detect)
                    if p is None:
                        xy = torch.zeros((0, 2), dtype=torch.long)
                        scores = torch.zeros((0,), dtype=torch.float32)
                    else:
                        p = p.reshape(-1, 2)
                        xy = self._clip_round_unique(p, H, W)
                        scores = torch.ones((xy.shape[0],), dtype=torch.float32)

                    xy, scores = self._select_by_score(xy, scores, topk=self.TOPK)
                    kpts = self._pack_kpts(b, xy, device)

                    if self.use_ssc and kpts.shape[0] > 0:
                        try:
                            kpts = self.ssc(kpts, num_ret_points=self.TOPK_ssc,
                                            tolerance=self.ssc_tolerance, cols=W, rows=H)
                        except ValueError as e:
                            print(f"SSC警告: {e}, 使用原始关键点")

                    kpts_list.append(kpts)

                    if kpts.numel() > 0:
                        ys = kpts[:, 2]
                        xs = kpts[:, 3]
                        im_topk[b, 0, ys, xs] = im_topk.new_ones(ys.shape)

                return im_topk, kpts_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='assets/frame_00000.jpg')
    parser.add_argument('--ssc', type=int, default=True, help='是否开启SSC')
    parser.add_argument('--topk', type=int, default=10000)
    parser.add_argument('--ssc_num', type=int, default=cfg.MODEL.TOPK)
    parser.add_argument('--show', action='store_true', help='显示图像而不保存')
    parser.add_argument('--out', type=str, default='results/vis_kpts.png')
    args = parser.parse_args([])

    # 加载图像
    img = Image.open(args.img).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # [1,3,H,W]

    device = torch.device('cuda')
    img_tensor = img_tensor.to(device, non_blocking=True)
    print('Input shape:', img_tensor.shape, 'Device:', device)

    # 检测器组合
    point_method = ['RFdet', 'superpoint', 'aliked', 'sift', 'disk', 'dog_hardnet', 'xfeat', 'dad']

    # 初始化模型
    det = KeypointDetectionSSC(
        point_method=point_method,
        topK=args.topk,
        ssc_num=args.ssc_num,
        ensemble_weights={'RFdet': 1.0, 'xfeat': 1.0, 'superpoint': 0.8, 'aliked': 0.8,
                          'disk': 0.8, 'dog_hardnet': 0.8, 'sift': 0.7, 'dad': 0.8},
        diversity_mode='soft',
        diversity_lambda=0.6,
        diversity_sigma=7.5,
        diversity_tau=8.0,
        use_ssc=args.ssc,
        ssc_tolerance=0.10
    ).to(device).eval()

    # 前向
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.inference_mode():
        im_topk, kpts = det.forward(img_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.time()

    # 打印信息
    total_k0 = int(kpts[0].shape[0]) if len(kpts) > 0 else 0
    print(f'{point_method} method detecting keypoint shape is {kpts[0].shape if len(kpts)>0 else (0,4)}')
    print(f'Forward time: {(t1 - t0)*1000:.1f} ms, selected points (img0): {total_k0}')

    # 可视化
    pil_img = tensor_to_pil(img_tensor)
    img_with_kpts = draw_keypoints_on_image(pil_img, kpts)
    if args.show:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_kpts)
        plt.axis('off')
        plt.show()
    else:
        img_with_kpts.save(args.out)
        print('Saved to:', args.out)



