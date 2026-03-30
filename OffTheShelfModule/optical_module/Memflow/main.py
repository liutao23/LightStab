from __future__ import print_function, division
import argparse
from loguru import logger as loguru_logger
import random
from core.Networks import build_network
import sys
sys.path.append('core')
from PIL import Image
import os
import numpy as np
import torch
from core.utils import flow_viz
from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate
from inference import inference_core_skflow as inference_core


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def inference(cfg, img1, img2):
    # Initialize the model
    model = build_network(cfg).cuda()
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    # Load checkpoint if provided
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)

    model.eval()

    # Prepare the two images
    print(f"Preparing input images...")
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)

    # Convert grayscale images to RGB
    if len(img1.shape) == 2:
        img1 = np.tile(img1[..., None], (1, 1, 3))
        img2 = np.tile(img2[..., None], (1, 1, 3))
    else:
        img1 = img1[..., :3]
        img2 = img2[..., :3]

    # Convert images to torch tensors
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

    images = torch.stack([img1, img2])

    processor = inference_core.InferenceCore(model, config=cfg)

    images = images.cuda().unsqueeze(0)  # Add batch dimension

    padder = InputPadder(images.shape)
    images = padder.pad(images)

    images = 2 * (images / 255.0) - 1.0  # Normalize to [-1, 1]
    flow_prev = None
    results = []

    print(f"Start inference for two images...")
    flow_low, flow_pre = processor.step(images, end=True, add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
    flow_pre = padder.unpad(flow_pre[0]).cpu()
    results.append(flow_pre)

    # Save the results
    vis_dir = cfg.vis_dir if cfg.vis_dir else 'demo_flow_vis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    print(f"Saving results...")
    flow_img = flow_viz.flow_to_image(results[0].permute(1, 2, 0).numpy())
    image = Image.fromarray(flow_img)
    image.save(f'{vis_dir}/flow_01_to_02.png')

    print(f"Flow visualization saved to {vis_dir}/flow_01_to_02.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet_T', choices=['MemFlowNet', 'MemFlowNet_T'], help="Name your experiment")
    parser.add_argument('--stage', default='kitti', help="Determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', default='/media/B/demo/MemFlow/ckpts/MemFlowNet_T_kitti.pth', help="Restore checkpoint")
    parser.add_argument('--vis_dir', default='demo_flow_vis', help="Directory to save visualized flow")

    # Add arguments for two input images
    parser.add_argument('--img1', default='/home/video/demo/CVPR/LightStab/test_images/0.jpg', help="Path to the first image")
    parser.add_argument('--img2', default='/home/video/demo/CVPR/LightStab/test_images/1.jpg', help="Path to the second image")

    args = parser.parse_args()

    # Load the config based on the chosen network
    if args.name == "MemFlowNet":
        if args.stage == 'things':
            from configs.things_memflownet import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet import get_cfg
        elif args.stage == 'spring_only':
            from configs.spring_memflownet import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet import get_cfg
        else:
            raise NotImplementedError
    elif args.name == "MemFlowNet_T":
        if args.stage == 'things':
            from configs.things_memflownet_t import get_cfg
        elif args.stage == 'things_kitti':
            from configs.things_memflownet_t_kitti import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet_t import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet_t import get_cfg
        else:
            raise NotImplementedError

    cfg = get_cfg()
    cfg.update(vars(args))

    # Initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    # Open the two images
    img1 = Image.open(args.img1)
    img2 = Image.open(args.img2)

    # Perform inference
    inference(cfg, img1, img2)
