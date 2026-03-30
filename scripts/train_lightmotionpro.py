# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : train_lightmotionpro.py
# @Time  : 2025/10/19 下午7:04


# 已删除关键信息, 因为准备会议扩刊，暂时不准备开放，等扩刊以后会上传
import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.data import MotionData
from model.utils import GradualWarmupScheduler
from model.LightMotionPro import EfficientMotionPro

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Control for stabilization model')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--decay', type=int, default=100, help='decay step')
    parser.add_argument('--WeightDecay', type=float, default=0., help='weight decay for L2 Regularization')
    parser.add_argument('--datatype', type=str, choices=['new', 'all'], default='all')
    parser.add_argument('--maxlength', type=int, default=30, help='max number of frames can be dealt with one time')
    parser.add_argument('--totalEpoch', type=int, default=100, help='max number of frames can be dealt with one time')
    parser.add_argument('--Suffix', type=str, default='properMulti_oldloss', help='suffix for saved models')
    parser.add_argument('--reload', type=str, default='', help='path to reload model state dict')
    parser.add_argument('--l1', type=float, default=20.)
    parser.add_argument('--l2', type=float, default=10.)
    parser.add_argument('--l3', type=float, default=10.)
    parser.add_argument('--root', type=str, default="/media/A/Datasets/StabDatasets/US/motionpro")
    parser.add_argument('--output_dir', type=str, default="weights/Motion")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

