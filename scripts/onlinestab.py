# -*- coding: utf-8 -*-
# @Author: anonymous user
# @Unit  : anonymous organization
# @File  : onlinestab.py
# @Time  : 2025/10/18 下午2:47

import time
import imageio
import numpy as np
import matplotlib
import torch
import cv2
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
from model.Outpainting import video_outpainting_inference
from model.utils import warpListImage, detect_global_max_crop_tilt
from model.LightOnlineStab import SuperStab
from tqdm import tqdm
from configs.config import cfg
from OffTheShelfModule.outpainting.model.misc import get_device

def generateStableWithAutoCrop(model, base_path, outPath, outpainting_args=None):
    cap = cv2.VideoCapture(base_path)
    if not cap.isOpened():
        raise RuntimeError(f"Do not open: {base_path}")
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_len = min(n_total if n_total > 0 else 10**9, 200)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read the first frame")

    inW, inH = int(getattr(cfg.MODEL, "WIDTH")), int(getattr(cfg.MODEL, "HEIGHT"))

    rgb_chw_list = []
    def _resize_to_model(img):
        inter = cv2.INTER_AREA if (img.shape[1] > inW or img.shape[0] > inH) else cv2.INTER_LINEAR
        return cv2.resize(img, (inW, inH), interpolation=inter)

    t = 0
    while t < image_len:
        img_resized = _resize_to_model(frame)
        chw = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)
        rgb_chw_list.append(chw)
        t += 1
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()

    if len(rgb_chw_list) == 0:
        raise RuntimeError("No usable frame")

    x_RGB_np = np.stack(rgb_chw_list, axis=0)
    x_RGB = torch.from_numpy(x_RGB_np).unsqueeze(0).float()

    t2 = time.time()
    amp_enable = torch.cuda.is_available()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enable):
        origin_motion, smoothPath = model.inference(x_RGB.cuda(non_blocking=True))
    t3 = time.time()

    image_len = x_RGB.shape[1]
    print(f"Stabilization time per frame: {(t3 - t2) / max(1, image_len):.4f}s")

    origin_motion = origin_motion[0].permute(2, 3, 1, 0).contiguous().cpu().numpy()
    smoothPath   = smoothPath[0].permute(2, 3, 1, 0).contiguous().cpu().numpy()
    x_paths  = origin_motion[..., 0]
    y_paths  = origin_motion[..., 1]
    sx_paths = smoothPath[..., 0]
    sy_paths = smoothPath[..., 1]

    print("Generating stabilized video...")
    new_x_motion_meshes = sx_paths - x_paths
    new_y_motion_meshes = sy_paths - y_paths

    rgbimages_for_warp = [arr[np.newaxis, ...] for arr in rgb_chw_list]
    outImages = warpListImage(rgbimages_for_warp, new_x_motion_meshes, new_y_motion_meshes)
    outImages = outImages.cpu().numpy().astype(np.uint8) if isinstance(outImages, torch.Tensor) \
                else np.asarray(outImages, dtype=np.uint8)
    outImages = [np.transpose(outImages[i], (1, 2, 0)) for i in range(outImages.shape[0])]

    # Auto-crop detection on stabilized frames (returns margins)
    top, bottom, left, right = detect_global_max_crop_tilt(outImages, thr=16, dilate=1)
    print(f"Crop margins: top={top}px, bottom={bottom}px, left={left}px, right={right}px")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_stab = cv2.VideoWriter(outPath, fourcc, fps, (orig_W, orig_H))

    # Crop stabilized frames and restore to original resolution
    cropped_frames = []
    restored_frames = []
    for frame_stab in tqdm(outImages, desc="Stabilizing"):
        hh, ww = frame_stab.shape[:2]
        cropped = frame_stab[top:hh - bottom, left:ww - right]
        cropped_frames.append(cropped)
        inter = cv2.INTER_LANCZOS4 if (cropped.shape[1] < orig_W or cropped.shape[0] < orig_H) else cv2.INTER_AREA
        restored = cv2.resize(cropped, (orig_W, orig_H), interpolation=inter)
        restored_frames.append(restored)
        writer_stab.write(restored)
    writer_stab.release()

    # Optional: outpainting to compensate for cropping
    outpainting_path = None
    outpainted_frames_bgr = None
    if outpainting_args is not None:
        print("Performing video outpainting to compensate for cropping...")
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in cropped_frames]
        scale_h = (orig_H + top + bottom) / max(1, orig_H)
        scale_w = (orig_W + left + right) / max(1, orig_W)
        print(f"Outpainting scale: {scale_h:.2f}x{scale_w:.2f}")

        device = get_device()
        outpainted_frames = video_outpainting_inference(
            pil_frames,
            device=device,
            scale_h=scale_h,
            scale_w=scale_w,
            fps=fps,
            **outpainting_args
        )
        outpainting_path = outPath.replace('.mp4', '_outpainted.mp4')
        for i in range(len(outpainted_frames)):
            outpainted_frames[i] = cv2.resize(outpainted_frames[i], (orig_W, orig_H), interpolation=cv2.INTER_LANCZOS4)
        imageio.mimwrite(outpainting_path, outpainted_frames, fps=fps, quality=7)

        # Convert to BGR for side-by-side stacking
        outpainted_frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in outpainted_frames]

    # Build "crop-visual (black=removed)" column by pasting the cropped frame back on a black canvas
    sx = orig_W / float(inW)
    sy = orig_H / float(inH)
    vis_frames = []
    for cropped in cropped_frames:
        canvas = np.zeros((orig_H, orig_W, 3), dtype=np.uint8)  # full black canvas
        ch, cw = cropped.shape[:2]
        dst_w = max(1, int(round(cw * sx)))
        dst_h = max(1, int(round(ch * sy)))
        resized_crop = cv2.resize(cropped, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        ox = int(round(left * sx))
        oy = int(round(top  * sy))
        x1 = min(orig_W, ox + dst_w)
        y1 = min(orig_H, oy + dst_h)
        canvas[oy:y1, ox:x1] = resized_crop[0:(y1-oy), 0:(x1-ox)]
        vis_frames.append(canvas)

    # Write comparison video: Original | Crop-visual (black = removed) | Outpainted (if available)
    compare_path = outPath.replace('.mp4', '_compare.mp4')
    cmp_cols = 3 if outpainted_frames_bgr is not None else 2
    cmp_size = (orig_W * cmp_cols, orig_H)
    writer_cmp = cv2.VideoWriter(compare_path, fourcc, fps, cmp_size)

    cap_orig = cv2.VideoCapture(base_path)
    if not cap_orig.isOpened():
        print("Warning: Failed to reopen the original video for comparison; only stabilized and crop-visual will be written.")
        cap_orig = None

    font = cv2.FONT_HERSHEY_SIMPLEX
    N = min(len(restored_frames), len(vis_frames))
    if cap_orig is not None and n_total > 0:
        N = min(N, n_total)

    for i in range(N):
        # Original frame
        if cap_orig is not None:
            ok0, raw = cap_orig.read()
        else:
            ok0, raw = False, None
        if not ok0:
            raw = np.zeros((orig_H, orig_W, 3), dtype=np.uint8)
        if raw.shape[:2] != (orig_H, orig_W):
            raw = cv2.resize(raw, (orig_W, orig_H), interpolation=cv2.INTER_AREA)

        col1 = raw
        col2 = vis_frames[i]
        if outpainted_frames_bgr is not None:
            col3 = outpainted_frames_bgr[i] if i < len(outpainted_frames_bgr) else np.zeros_like(col1)
            side = np.hstack([col1, col2, col3])
        else:
            side = np.hstack([col1, col2])

        # Labels
        cv2.putText(side, "Original", (20, 50), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side, "Crop-visual (black = removed)", (orig_W + 20, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        if outpainted_frames_bgr is not None:
            x_off = orig_W * 2
            cv2.putText(side, "Outpainted", (x_off + 20, 50), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        writer_cmp.write(side)

    writer_cmp.release()
    if cap_orig is not None:
        cap_orig.release()

    # Return paths for downstream use
    if outpainting_args is not None:
        return outpainting_path, outPath, compare_path
    else:
        return outPath, compare_path



if __name__ == "__main__":

    MotionProPath = 'preweights/EfficientMotionPro.pth'

    smootherPath = 'preweights/LightOnlineSmoother.pth'

    ensemble_weights = {
        "low_texture_far_lightpoor": {  # 低纹理/远景/光照差
            "RFdet": 1.0,
            "xfeat": 1.0,
            "superpoint": 1.0,
            "aliked": 1.0,
            "disk": 0.9,
            "dog_hardnet": 0.7,
            "sift": 0.7,
            "dad": 1.0,
        },
        "high_texture_daylight": {  # 高纹理/白天/曝光正常
            "RFdet": 1.0,
            "xfeat": 1.0,
            "superpoint": 0.9,
            "aliked": 0.9,
            "disk": 0.8,
            "dog_hardnet": 0.7,
            "sift": 0.5,
            "dad": 0.8,
        },
        "motionblur_lowres": {  # 运动模糊/低清
            "RFdet": 1.0,
            "xfeat": 0.9,
            "superpoint": 1.0,
            "aliked": 1.0,
            "disk": 0.9,
            "dog_hardnet": 0.6,
            "sift": 0.6,
            "dad": 1.0,
        },
    }

    # 低纹理/远景/光照差
    # ensemble_weights_choice = ensemble_weights["low_texture_far_lightpoor"]
    # 高纹理/白天/曝光正常
    ensemble_weights_choice = ensemble_weights["high_texture_daylight"]
    # 运动模糊/低清
    # ensemble_weights_choice = ensemble_weights["motionblur_lowres"]

    # point_method ['RFdet', 'superpoint', 'aliked', 'sift', 'disk', 'dog_hardnet', 'xfeat', 'dad', None]
    # point_method = ['RFdet', 'superpoint', 'aliked', 'sift', 'disk', 'dog_hardnet', 'xfeat', 'dad']
    point_method = ['xfeat']

    # flow_method [None,'PWCNet','liteflownet','RAFT_large','RAFT_small','NeuFlow','flowformer++','Memflow']
    flow_method = 'RAFT_small'

    # motion_weight [None, MotionProPath]
    motion_weight = MotionProPath
    # smooth_weight [None, smootherPath]
    smooth_weight = smootherPath
    # homo ['multi', 'single']
    homo = 'multi'

    if motion_weight is not None:
        motion = 'Motionpro'
    else:
        motion = 'Median'

    if smooth_weight is not None:
        smooth = 'Smoother'
    else:
        smooth = 'JacobiSolver'

    use_ssc = True

    model = SuperStab(point_method=point_method, flow_method=flow_method,
                      ensemble_weights=ensemble_weights_choice,
                      motion_weight=motion_weight, smooth_weight=smooth_weight,
                      homo=homo, use_ssc=use_ssc)
    model.cuda()
    model.eval()

    outpainting_args = {
        'ref_stride': 10,
        'neighbor_length': 10,
        'subvideo_length': 80,
        'raft_iter': 20,
        'fp16': False,
        'output_dir': 'results',
        'raft_ckpt_path': 'OffTheShelfModule/outpainting/weights/raft-things.pth',
        'flow_complete_ckpt_path': 'OffTheShelfModule/outpainting/weights/recurrent_flow_completion.pth',
        'propainter_ckpt_path': 'OffTheShelfModule/outpainting/weights/ProPainter.pth',
    }

    inPath = 'assets/running13.mp4'
    outpath = 'results/running13.mp4'
    generateStableWithAutoCrop(model, inPath, outpath, outpainting_args=outpainting_args)

