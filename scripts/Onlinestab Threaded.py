# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : onlinestab.py
# @Time  : 2025/10/20

import os
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import imageio
import matplotlib
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

from collections import deque

# ========= 你原有的依赖 =========
from model.utils import warpListImage, detect_global_max_crop_tilt
from model.LightKeypointsDetection import KeypointDetectionSSC
from model.LightMotionEsitimation import MotionEstimation
from model.LightMotionPro import EfficientMotionPro as MotionPro
from model.LightOnlineSmoother import Smoother
from model.utils import SingleMotionPropagate, MultiMotionPropagate
from model.utils import generateSmooth_online
from configs.config import cfg

# ========= 多进程相关 =========
import multiprocessing as mp
from multiprocessing import Queue, Event

# =========================
# 模型与组件（保持与你一致）
# =========================
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
        if motion_weight is not None and os.path.isfile(str(motion_weight)):
            if homo=='multi':
                print('正在使用多网格估计网络')
                self.motion_propagate = MotionPro()
            else:
                print('正在使用单网格估计网络')
                self.motion_propagate = MotionPro(globalchoice='single')
            self.motion_propagate.load_state_dict(torch.load(motion_weight, weights_only=False, map_location='cpu'), strict=True)
        else:
            if homo=='multi':
                print('正在使用多网格估计')
                self.motion_propagate = motionPropagate(MultiMotionPropagate)
            else:
                print('正在使用单网格估计')
                self.motion_propagate = motionPropagate(SingleMotionPropagate)

        print('正在载入轨迹平滑模块')
        if smooth_weight is not None and os.path.isfile(str(smooth_weight)):
            print('正在使用深度学习轨迹平滑')
            self.smoother = Smoother()
            self.smoother.load_state_dict(torch.load(smooth_weight, weights_only=False, map_location='cpu'), strict=True)
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
        device = x_RGB.device
        x = x_RGB.to(device, non_blocking=True)   # [1, T, C, H, W]
        x_seq = x.squeeze(0)                      # [T, C, H, W]

        print("detect keypoints ....")
        im_topk, kpts = self.motionModule.forward(x_seq)
        kpts = [kp.to(device, non_blocking=True) for kp in kpts]

        print("estimate motion ....")
        masked_flow = self.flowModule.inference_stab(x, im_topk)  # [T,2,H,W]
        masked_flow = masked_flow.to(device, non_blocking=True)

        print("motion propagation ....")
        origin_list = [
            self.motion_propagate.inference(
                masked_flow[i:i+1, 0:1, :, :],
                masked_flow[i:i+1, 1:2, :, :],
                kpts[i]
            )
            for i in range(len(kpts) - 1)
        ]
        origin_motion = torch.stack(origin_list, dim=2)               # [B,2,T-1,H,W]
        zeros = torch.zeros_like(origin_motion[:, :, 0:1, :, :], device=device)
        origin_motion = torch.cat([zeros, origin_motion], dim=2)      # [B,2,T,H,W]

        origin_motion = origin_motion.cumsum(dim=2)
        minv = origin_motion.amin()
        origin_motion = origin_motion - minv
        maxv = origin_motion.amax() + 1e-5
        origin_motion = origin_motion / maxv

        print("trajectory smoothing ....")
        smoothKernel = self.smoother(origin_motion)
        smooth_list = self.smoother.KernelSmooth(smoothKernel, origin_motion, repeat)
        smoothPath = torch.cat(smooth_list, dim=1)

        origin_motion = origin_motion * maxv + minv
        smoothPath = smoothPath * maxv + minv

        return origin_motion, smoothPath

# =========================
# 多进程：Reader / Preproc / Infer / Writer
# =========================
_SENTINEL = None

def _resize_to_model(img_bgr, inW, inH):
    inter = cv2.INTER_AREA if (img_bgr.shape[1] > inW or img_bgr.shape[0] > inH) else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (inW, inH), interpolation=inter)

def proc_reader(source, q_raw: Queue, stop_ev: Event):
    """读取 BGR 原始帧 -> q_raw"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        q_raw.put(_SENTINEL)
        return
    try:
        while not stop_ev.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            q_raw.put(frame)
    finally:
        cap.release()
        q_raw.put(_SENTINEL)

def proc_preproc(q_raw: Queue, q_proc: Queue, inW: int, inH: int, stop_ev: Event):
    """预处理：BGR -> (CHW float32, 原始BGR) -> q_proc"""
    try:
        while not stop_ev.is_set():
            item = q_raw.get()
            if item is _SENTINEL:
                break
            bgr = item
            small = _resize_to_model(bgr, inW, inH)
            chw = np.transpose(small, (2, 0, 1)).astype(np.float32)  # (C,H,W)
            q_proc.put((chw, bgr))
    finally:
        q_proc.put(_SENTINEL)

def proc_infer(q_proc: Queue,
               q_disp: Queue,
               q_write: Queue,
               inW: int, inH: int,
               fps: float,
               motion_weight,
               smooth_weight,
               point_method,
               flow_method,
               homo,
               use_ssc,
               show_flag: bool,
               stop_ev: Event):
    """
    仅该进程使用 CUDA；维护滑窗=30；输出稳定的中位帧到显示/写盘队列
    """
    # 构建模型（放在子进程内，避免 CUDA 句柄跨进程）
    model = SuperStab(point_method=point_method, flow_method=flow_method,
                      motion_weight=motion_weight, smooth_weight=smooth_weight,
                      homo=homo, use_ssc=use_ssc)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda().eval()
        torch.backends.cudnn.benchmark = True
    else:
        model = model.eval()

    buf_chw = deque(maxlen=30)
    buf_bgr = deque(maxlen=30)
    mid_idx = 14

    last_t = time.time()
    amp = use_cuda
    if use_cuda:
        h2d_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.current_stream()

    # 先填满 30
    try:
        while len(buf_chw) < 30 and not stop_ev.is_set():
            item = q_proc.get()
            if item is _SENTINEL:
                # 源太短
                q_disp.put(_SENTINEL)
                q_write.put(_SENTINEL)
                return
            chw, bgr = item
            buf_chw.append(chw)
            buf_bgr.append(bgr)

        print("[Infer•MP] running...")

        while not stop_ev.is_set():
            # 组装 batch [1,30,C,H,W]
            x_np = np.stack(list(buf_chw), axis=0)
            x_cpu = torch.from_numpy(x_np).unsqueeze(0).float()  # [1,30,C,H,W]
            if use_cuda:
                x_cpu = x_cpu.pin_memory()
                with torch.cuda.stream(h2d_stream):
                    x_dev = x_cpu.to(device='cuda', non_blocking=True)
                compute_stream.wait_stream(h2d_stream)
                x = x_dev
            else:
                x = x_cpu

            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=amp), torch.inference_mode():
                origin_motion, smoothPath = model.inference(x, repeat=50)
            infer_ms = (time.time() - t0) * 1000.0

            # -> [T,H,W,2]
            origin_np = origin_motion[0].permute(2, 3, 1, 0).contiguous().detach().cpu().numpy()
            smooth_np = smoothPath[0].permute(2, 3, 1, 0).contiguous().detach().cpu().numpy()

            dx_all = smooth_np[:, :, :, 0] - origin_np[:, :, :, 0]  # [30,H,W]
            dy_all = smooth_np[:, :, :, 1] - origin_np[:, :, :, 1]  # [30,H,W]

            frames_for_warp = [buf_chw[i][np.newaxis, ...] for i in range(30)]
            out_all = warpListImage(frames_for_warp, dx_all, dy_all)
            if isinstance(out_all, torch.Tensor):
                out_all = out_all.cpu().numpy().astype(np.uint8)
            else:
                out_all = np.asarray(out_all, dtype=np.uint8)

            # 取中位帧并还原到原分辨率
            stab_small = np.transpose(out_all[mid_idx], (1, 2, 0))  # (H,W,C)
            ref_bgr = buf_bgr[mid_idx]
            out_w, out_h = ref_bgr.shape[1], ref_bgr.shape[0]
            inter = cv2.INTER_LANCZOS4 if (stab_small.shape[1] < out_w or stab_small.shape[0] < out_h) else cv2.INTER_AREA
            stab_full = cv2.resize(stab_small, (out_w, out_h), interpolation=inter)

            # HUD
            now = time.time()
            fps_cur = 1.0 / max(1e-6, now - last_t); last_t = now
            cv2.putText(stab_full, f"win=30 mid=14 infer={infer_ms:.1f}ms FPS~{fps_cur:.1f}",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 送显示 & 写盘（显示放主进程，写盘在 Writer 进程）
            if show_flag:
                q_disp.put(stab_full.copy())
            q_write.put(stab_full)

            # 滑窗推进：拉取新帧
            nxt = q_proc.get()
            if nxt is _SENTINEL:
                break
            chw_new, bgr_new = nxt
            buf_chw.popleft(); buf_bgr.popleft()
            buf_chw.append(chw_new); buf_bgr.append(bgr_new)

    except KeyboardInterrupt:
        pass
    finally:
        # 结束信号
        q_disp.put(_SENTINEL)
        q_write.put(_SENTINEL)

def proc_writer(out_path: str, size_wh, fps: float, q_write: Queue, stop_ev: Event):
    """异步落盘，不阻塞推理"""
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, size_wh)
    try:
        while not stop_ev.is_set():
            item = q_write.get()
            if item is _SENTINEL:
                break
            if writer is not None:
                writer.write(item)
    finally:
        if writer is not None:
            writer.release()

# =========================
# 主进程：组织“回字形”流水线 + 显示
# =========================
def run_online_window30_mid14_mp(source, out_path=None, show=True,
                                 point_method=['xfeat'], flow_method='RAFT_small',
                                 motion_weight=None, smooth_weight=None,
                                 homo='multi', use_ssc=True):
    """
    多进程回字形管线入口：Reader -> Preproc -> Infer -> (Disp & Writer)
    """
    # 先探测视频属性
    cap_probe = cv2.VideoCapture(source)
    if not cap_probe.isOpened():
        raise RuntimeError(f"Do not open: {source}")
    fps = cap_probe.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 1e-3 else 30.0
    ok_first, first = cap_probe.read()
    if not ok_first:
        cap_probe.release()
        raise RuntimeError("Failed to read first frame.")
    orig_h, orig_w = first.shape[:2]
    cap_probe.release()

    inW, inH = int(getattr(cfg.MODEL, "WIDTH")), int(getattr(cfg.MODEL, "HEIGHT"))

    # 队列 & 事件
    ctx = mp.get_context("spawn")
    q_raw   = ctx.Queue(maxsize=64)   # Reader -> Preproc
    q_proc  = ctx.Queue(maxsize=64)   # Preproc -> Infer
    q_disp  = ctx.Queue(maxsize=64)   # Infer -> Main(Display)
    q_write = ctx.Queue(maxsize=64)   # Infer -> Writer
    stop_ev = ctx.Event()

    # 子进程
    p_reader = ctx.Process(target=proc_reader, args=(source, q_raw, stop_ev), daemon=True)
    p_pre    = ctx.Process(target=proc_preproc, args=(q_raw, q_proc, inW, inH, stop_ev), daemon=True)
    p_infer  = ctx.Process(target=proc_infer,
                           args=(q_proc, q_disp, q_write, inW, inH, fps,
                                 motion_weight, smooth_weight, point_method, flow_method,
                                 homo, use_ssc, show, stop_ev),
                           daemon=True)
    p_writer = ctx.Process(target=proc_writer, args=(out_path, (orig_w, orig_h), fps, q_write, stop_ev), daemon=True)

    # 启动
    p_reader.start(); p_pre.start(); p_infer.start(); p_writer.start()

    # 主进程负责显示（避免 GUI 在子进程出错）
    try:
        if show:
            print("[Main] Display on. Press 'q' to quit.")
            while True:
                item = q_disp.get()
                if item is _SENTINEL:
                    break
                cv2.imshow("Stabilized(Online-30-mid14 • MP)", item)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_ev.set()
                    break
        else:
            # 不显示则阻塞直到推理完成（监听 infer 的输出结束）
            while True:
                item = q_disp.get()
                if item is _SENTINEL:
                    break
    finally:
        stop_ev.set()
        try:
            q_raw.put(_SENTINEL)
            q_proc.put(_SENTINEL)
            q_write.put(_SENTINEL)
            q_disp.put(_SENTINEL)
        except:
            pass
        for p in [p_reader, p_pre, p_infer, p_writer]:
            if p.is_alive():
                p.join(timeout=2.0)
        cv2.destroyAllWindows()

# ======================
# CLI 入口
# ======================
if __name__ == "__main__":
    import argparse

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="在线稳定(多进程)：窗口=30，取中位(索引14)")
    parser.add_argument("--source", type=str, default="assets/crowd3.mp4",
                        help="视频文件或 RTSP URL（不是摄像头索引）")
    parser.add_argument("--out", type=str, default="results/online_mid14.mp4")
    parser.add_argument("--flow-method", type=str, default="RAFT_small")
    parser.add_argument("--point-method", type=str, nargs="+", default=["xfeat"])
    parser.add_argument("--homo", type=str, default="multi", choices=["multi","single"])
    parser.add_argument("--use-ssc", action="store_true")
    parser.add_argument("--no-ssc", dest="use_ssc", action="store_false")
    parser.set_defaults(use_ssc=True)
    parser.add_argument("--motion-weight", type=str, default="preweights/light_weights/EfficientMotionPro.pth")
    parser.add_argument("--smooth-weight", type=str, default="preweights/LightOnlineSmoother.pth")
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    motion_weight = args.motion_weight if os.path.isfile(args.motion_weight) else None
    smooth_weight = args.smooth_weight if os.path.isfile(args.smooth_weight) else None

    # 直接跑多进程回字形版本
    run_online_window30_mid14_mp(
        source=args.source,
        out_path=args.out,
        show=(not args.no_show),
        point_method=args.point_method,
        flow_method=args.flow_method,
        motion_weight=motion_weight,
        smooth_weight=smooth_weight,
        homo=args.homo,
        use_ssc=args.use_ssc
    )
