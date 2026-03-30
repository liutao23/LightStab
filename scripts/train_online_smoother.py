# -*- coding: utf-8 -*-
# @Author: Tao Liu
# @Unit  : Nanjing University of Science and Technology
# @File  : train_online_smoother.py
# @Time  : 2025/10/18 下午2:35

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.LightOnlineSmoother import Smoother
from model.smoothloss import loss_calculate_Spatial_withKP
from model.utils import GradualWarmupScheduler
from configs.data import SmoothData
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
# 已删除关键信息, 因为准备会议扩刊，暂时不准备开放，等扩刊以后会上传

def parse_args():
    parser = argparse.ArgumentParser(description='Control for stabilization model')
    # dataset paths
    parser.add_argument('--train_motion', type=str, default='TrainData/smoothall/train_mo_smoother.npy',
                        help='Path to train_mo_smoother.npy')
    parser.add_argument('--train_kp',     type=str, default='TrainData/smoothall/train_kp_smoother.npy'
                        , help='Path to train_kp_smoother.npy')
    parser.add_argument('--valid_motion', type=str, default='TrainData/smoothall/valid_mo_smoother.npy',
                        help='Path to valid_mo_smoother.npy')
    parser.add_argument('--valid_kp',     type=str, default='TrainData/smoothall/valid_kp_smoother.npy',
                        help='Path to valid_kp_smoother.npy')

    # output dir (no hardcoding)
    parser.add_argument('--output_dir',   type=str, default='weights/Smoother',
    help='Directory to save weights and plots')

    # grid size
    parser.add_argument('--grid_h', type=int, default=480)
    parser.add_argument('--grid_w', type=int, default=640)

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--decay', type=int, default=100, help='decay step')
    parser.add_argument('--WeightDecay', type=float, default=0., help='weight decay for L2 Regularization')
    parser.add_argument('--datatype', type=str, choices=['new', 'all'], default='all')
    parser.add_argument('--maxlength', type=int, default=100, help='max number of frames per sample')
    parser.add_argument('--repeat', type=int, default=10, help='repeat for KernelSmooth')
    parser.add_argument('--totalEpoch', type=int, default=100)
    parser.add_argument('--Suffix', type=str, default='LossTypeL1', help='suffix for saving weights')
    parser.add_argument('--regu', action='store_true')
    parser.add_argument('--warmup', action='store_false')
    parser.add_argument('--restore', type=str, default='')
    parser.add_argument('--l1', type=float, default=30.)
    parser.add_argument('--l2', type=float, default=10.)
    parser.add_argument('--l3', type=float, default=10.)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)