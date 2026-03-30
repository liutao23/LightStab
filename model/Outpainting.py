# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : Outpainting.py
# @Time  : 2025/10/17 下午1:56
import os
import argparse
import numpy as np
from typing import List, Tuple
import torch
from tqdm import tqdm
import imageio
import cv2
from PIL import Image

# Project modules (unchanged paths from your repo)
from OffTheShelfModule.outpainting.model.modules.flow_comp_raft import RAFT_bi
from OffTheShelfModule.outpainting.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from OffTheShelfModule.outpainting.model.propainter import InpaintGenerator
from OffTheShelfModule.outpainting.core.utils import to_tensors
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

# -------------------------
# IO helpers
# -------------------------

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def read_frames(input_path: str) -> Tuple[List[Image.Image], float, Tuple[int, int], str]:
    """Read frames from a video file or a directory of frames.
    Priority: OpenCV -> (fallback) torchvision.io.read_video

    Returns:
        frames (list[PIL.Image]), fps (float|None), size (W,H), video_name (str)
    """
    import warnings
    import re

    def _natural_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    input_path = str(input_path)

    # -------- 目录：按常见图片后缀读取 --------
    if os.path.isdir(input_path):
        video_name = os.path.basename(os.path.normpath(input_path))
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        names = sorted(
            [n for n in os.listdir(input_path) if os.path.splitext(n)[1].lower() in exts],
            key=_natural_key
        )
        frames: List[Image.Image] = []
        for n in names:
            fp = os.path.join(input_path, n)
            img_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img_rgb))

        if not frames:
            raise FileNotFoundError(f"No readable frames found in directory: {input_path}")

        fps = None
        size = frames[0].size
        return frames, fps, size, video_name

    # -------- 文件：优先用 OpenCV 解码 --------
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        frames: List[Image.Image] = []
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_val) if fps_val and fps_val > 1e-6 else None
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()

        if frames:
            size = frames[0].size
            return frames, fps, size, video_name
        # 若 OpenCV 打开但读不到帧，则继续走兜底

    # -------- 兜底到 torchvision（局部屏蔽弃用告警）--------
    try:
        from torchvision.io import read_video
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The video decoding and encoding capabilities of torchvision are deprecated"
            )
            vframes, _, info = read_video(filename=input_path, pts_unit='sec')
        frames = [Image.fromarray(f.numpy()) for f in vframes]
        if not frames:
            raise FileNotFoundError(f"Could not decode frames from: {input_path}")
        fps = info.get('video_fps', None)
        size = frames[0].size
        return frames, fps, size, video_name
    except Exception as e:
        raise FileNotFoundError(f"Failed to read video '{input_path}' via OpenCV and torchvision: {e}")


# -------------------------
# Geometry / formatting
# -------------------------

def resize_frames(frames: List[Image.Image], size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
    return frames, process_size, out_size


def extrapolation(video_ori: List[Image.Image], scale: Tuple[float, float]):
    """Pad each frame to a larger canvas according to (scale_h, scale_w)."""
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr -= imgH_extr % 8
    imgW_extr -= imgW_extr % 8

    H_start = (imgH_extr - imgH) // 2
    W_start = (imgW_extr - imgW) // 2

    frames = []
    for v in video_ori:
        canvas = np.zeros((imgH_extr, imgW_extr, 3), dtype=np.uint8)
        np_v = np.asarray(v)
        canvas[H_start:H_start + imgH, W_start:W_start + imgW] = np_v
        frames.append(Image.fromarray(canvas))

    masks_dilated, flow_masks = [], []
    mask = np.ones((imgH_extr, imgW_extr), dtype=np.uint8)

    # Flow mask — a slightly tighter mask to help motion completion
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0

    mask_flow = mask.copy()
    mask_flow[H_start + dilate_h: H_start + imgH - dilate_h,
              W_start + dilate_w: W_start + imgW - dilate_w] = 0
    flow_masks.append(Image.fromarray(mask_flow * 255))

    # Inpaint mask — full missing border
    mask_inpaint = mask.copy()
    mask_inpaint[H_start: H_start + imgH, W_start: W_start + imgW] = 0
    masks_dilated.append(Image.fromarray(mask_inpaint * 255))

    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame

    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


# -------------------------
# Model loading
# -------------------------

def load_outpainting_models(device, raft_ckpt_path: str, flow_complete_ckpt_path: str, propainter_ckpt_path: str):
    if not os.path.isfile(raft_ckpt_path):
        raise FileNotFoundError(f"RAFT checkpoint not found: {raft_ckpt_path}")
    if not os.path.isfile(flow_complete_ckpt_path):
        raise FileNotFoundError(f"FlowComplete checkpoint not found: {flow_complete_ckpt_path}")
    if not os.path.isfile(propainter_ckpt_path):
        raise FileNotFoundError(f"ProPainter checkpoint not found: {propainter_ckpt_path}")

    fix_raft = RAFT_bi(raft_ckpt_path, device)

    fix_flow_complete = RecurrentFlowCompleteNet(flow_complete_ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device).eval()

    model = InpaintGenerator(model_path=propainter_ckpt_path).to(device).eval()

    return fix_raft, fix_flow_complete, model


# -------------------------
# Core pipeline
# -------------------------

def video_outpainting_inference(
    frames: List[Image.Image],
    device: torch.device,
    scale_h: float = 1.2,
    scale_w: float = 1.2,
    fps: int = 24,
    ref_stride: int = 10,
    neighbor_length: int = 10,
    subvideo_length: int = 80,
    raft_iter: int = 20,
    fp16: bool = False,
    output_dir: str = 'results',
    raft_ckpt_path: str = '',
    flow_complete_ckpt_path: str = '',
    propainter_ckpt_path: str = '',
):
    """Run outpainting and save results under output_dir/video_name.

    Returns: list[np.ndarray RGB] outpainted frames (uint8)
    """
    use_half = fp16 and (device.type == 'cuda')

    # Ensure multiples of 8
    h, w = frames[0].size[1], frames[0].size[0]
    frames, size, _ = resize_frames(frames, (w, h))

    # Expand FOV
    frames, flow_masks, masks_dilated, size = extrapolation(frames, (scale_h, scale_w))
    w_ext, h_ext = size

    # Save root
    video_name = 'outpainted'
    save_root = os.path.join(output_dir, video_name)
    os.makedirs(save_root, exist_ok=True)

    # For visualization — masked input overlay
    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]), 2).repeat(3, axis=2) / 255.0
        img = np.array(frames[i])
        green = np.zeros([h_ext, w_ext, 3], dtype=np.float32)
        green[:, :, 1] = 255
        alpha = 0.6
        fuse = (1 - alpha) * img.astype(np.float32) + alpha * green
        viz = mask_ * fuse + (1 - mask_) * img
        masked_frame_for_save.append(viz.astype(np.uint8))

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames_tensor = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks_tensor = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated_tensor = to_tensors()(masks_dilated).unsqueeze(0)

    frames_tensor = frames_tensor.to(device)
    flow_masks_tensor = flow_masks_tensor.to(device)
    masks_dilated_tensor = masks_dilated_tensor.to(device)

    # Load models
    fix_raft, fix_flow_complete, model = load_outpainting_models(
        device,
        raft_ckpt_path=raft_ckpt_path,
        flow_complete_ckpt_path=flow_complete_ckpt_path,
        propainter_ckpt_path=propainter_ckpt_path,
    )

    # Flow computation (fp32 for RAFT)
    video_length = frames_tensor.size(1)
    if frames_tensor.size(-1) <= 640:
        short_clip_len = 12
    elif frames_tensor.size(-1) <= 720:
        short_clip_len = 8
    elif frames_tensor.size(-1) <= 1280:
        short_clip_len = 4
    else:
        short_clip_len = 2

    with torch.no_grad():
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames_tensor[:, f:end_f], iters=raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames_tensor[:, f - 1:end_f], iters=raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames_tensor, iters=raft_iter)
            torch.cuda.empty_cache()

        # Mixed precision for the rest (optional)
        if use_half:
            frames_tensor = frames_tensor.half()
            flow_masks_tensor = flow_masks_tensor.half()
            masks_dilated_tensor = masks_dilated_tensor.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        # Flow completion
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)

                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks_tensor[:, s_f:e_f + 1]
                )
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub, flow_masks_tensor[:, s_f:e_f + 1]
                )

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_tensor)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_tensor)
            torch.cuda.empty_cache()

        # Image propagation (feature propagation + transformer)
        masked_frames = frames_tensor * (1 - masks_dilated_tensor)
        subvideo_length_img_prop = min(100, subvideo_length)

        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated_tensor[:, s_f:e_f].size()
                pred_flows_bi_sub = (
                    pred_flows_bi[0][:, s_f:e_f - 1],
                    pred_flows_bi[1][:, s_f:e_f - 1],
                )

                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f], pred_flows_bi_sub,
                    masks_dilated_tensor[:, s_f:e_f], 'nearest'
                )

                updated_frames_sub = (
                    frames_tensor[:, s_f:e_f] * (1 - masks_dilated_tensor[:, s_f:e_f]) +
                    prop_imgs_sub.view(b, t, 3, h_ext, w_ext) * masks_dilated_tensor[:, s_f:e_f]
                )
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h_ext, w_ext)

                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()

            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated_tensor.size()
            prop_imgs, updated_local_masks = model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated_tensor, 'nearest'
            )
            updated_frames = (
                frames_tensor * (1 - masks_dilated_tensor) +
                prop_imgs.view(b, t, 3, h_ext, w_ext) * masks_dilated_tensor
            )
            updated_masks = updated_local_masks.view(b, t, 1, h_ext, w_ext)
            torch.cuda.empty_cache()

    # Merge with originals to get final composites
    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = neighbor_length // 2
    ref_num = subvideo_length // ref_stride if video_length > subvideo_length else -1

    for f in tqdm(range(0, video_length, neighbor_stride), desc="Outpainting"):
        neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)

        selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
        selected_masks = masks_dilated_tensor[:, neighbor_ids + ref_ids]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1]],
            pred_flows_bi[1][:, neighbor_ids[:-1]],
        )

        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h_ext, w_ext)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated_tensor[0, neighbor_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

            for i, idx in enumerate(neighbor_ids):
                img = pred_img[i].astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (0.5 * comp_frames[idx].astype(np.float32) + 0.5 * img.astype(np.float32)).astype(np.uint8)
        torch.cuda.empty_cache()

    # Save videos
    imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_frame_for_save, fps=fps, quality=7)
    imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)

    print(f'\nOutpainting results are saved in {save_root}')
    torch.cuda.empty_cache()
    return comp_frames


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Standalone video outpainting (ProPainter pipeline).')
    p.add_argument('--input', type=str, default='your_video.mp4',
                   help='Video file or directory of frames')
    p.add_argument('--output-dir', type=str, default='results')
    p.add_argument('--scale-h', type=float, default=1.2)
    p.add_argument('--scale-w', type=float, default=1.2)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--ref-stride', type=int, default=10)
    p.add_argument('--neighbor-length', type=int, default=10)
    p.add_argument('--subvideo-length', type=int, default=80)
    p.add_argument('--raft-iters', type=int, default=20)
    p.add_argument('--fp16', action='store_true')

    p.add_argument('--raft-ckpt', type=str, default='OffTheShelfModule/outpainting/weights/raft-things.pth')
    p.add_argument('--flowcomp-ckpt', type=str, default='OffTheShelfModule/outpainting/weights/recurrent_flow_completion.pth')
    p.add_argument('--propainter-ckpt', type=str, default='OffTheShelfModule/outpainting/weights/ProPainter.pth')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    frames, fps_in, size, video_name = read_frames(args.input)
    fps = args.fps if args.fps is not None else (fps_in or 24)

    _ = video_outpainting_inference(
        frames=frames,
        device=device,
        scale_h=args.scale_h,
        scale_w=args.scale_w,
        fps=fps,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
        raft_iter=args.raft_iters,
        fp16=args.fp16,
        output_dir=args.output_dir,
        raft_ckpt_path=args.raft_ckpt,
        flow_complete_ckpt_path=args.flowcomp_ckpt,
        propainter_ckpt_path=args.propainter_ckpt,
    )


if __name__ == '__main__':
    main()
