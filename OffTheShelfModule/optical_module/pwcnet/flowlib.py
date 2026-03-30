#!/usr/bin/python
"""
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import png
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from PIL import Image

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

"""
=============
Flow Section
=============
"""

def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()


def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / (max_flow if max_flow != 0 else 1.0)
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        np.clip(img[:, :, 0:3], 0.0, 1.0, out=img[:, :, 0:3])
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img *= valid[..., None]
        # show
        plt.imshow(img)
        plt.show()

    return None


def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    # 使用 with 确保文件关闭；保持原有打印与返回语义
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        data2d = None

        if magic.size != 1 or magic[0] != 202021.25:
            # 与原实现一致的提示
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            print("Reading %d x %d flo file" % (h[0], w[0]))  # 注意原实现打印(h, w)的顺序
            data2d = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
            # reshape data into 3D array (columns, rows, channels)
            data2d = data2d.reshape((h[0], w[0], 2))
    return data2d


def read_flow_png(flow_file):
    """
    Read optical flow from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    # pypng 转 numpy 向量化读取，减少 Python 循环开销
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    rows_iter = flow_direct[2]
    (w, h) = flow_direct[3]['size']

    # 将行拼成二维数组，再按通道步进切片
    rows = np.vstack([np.asarray(row, dtype=np.uint16)[None, :] for row in rows_iter])  # [h, 3w]
    flow = np.zeros((h, w, 3), dtype=np.float64)

    flow[:, :, 0] = rows[:, 0::3]
    flow[:, :, 1] = rows[:, 1::3]
    flow[:, :, 2] = rows[:, 2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - (1 << 15)) / 64.0
    flow[invalid_idx, 0] = 0.0
    flow[invalid_idx, 1] = 0.0
    return flow


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    h, w = flow.shape[:2]
    with open(filename, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([w], dtype=np.int32).tofile(f)
        np.array([h], dtype=np.int32).tofile(f)
        flow.astype(np.float32, copy=False).tofile(f)


def segment_flow(flow):
    h, w = flow.shape[:2]
    u = flow[:, :, 0].copy()
    v = flow[:, :, 1].copy()

    idx = (np.abs(u) > LARGEFLOW) | (np.abs(v) > LARGEFLOW)
    idx2 = (np.abs(u) == SMALLFLOW)
    class0 = (v == 0) & (u == 0)
    # 避免除零（与原逻辑保持：只在 u==0 处置微小偏移）
    u[idx2] = 1e-5
    tan_value = v / u

    class1 = (tan_value < 1) & (tan_value >= 0) & (u > 0) & (v >= 0)
    class2 = (tan_value >= 1) & (u >= 0) & (v >= 0)
    class3 = (tan_value < -1) & (u <= 0) & (v >= 0)
    class4 = (tan_value < 0) & (tan_value >= -1) & (u < 0) & (v >= 0)
    class8 = (tan_value >= -1) & (tan_value < 0) & (u > 0) & (v <= 0)
    class7 = (tan_value < -1) & (u >= 0) & (v <= 0)
    class6 = (tan_value >= 1) & (u <= 0) & (v <= 0)
    class5 = (tan_value >= 0) & (tan_value < 1) & (u < 0) & (v <= 0)

    seg = np.zeros((h, w), dtype=np.float64)
    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0
    return seg


def flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (np.abs(stu) > UNKNOWN_FLOW_THRESH) | (np.abs(stv) > UNKNOWN_FLOW_THRESH)
    # 局部拷贝避免原地修改传入数组；行为与原逻辑一致（对临时副本赋值）
    stu = stu.copy(); stv = stv.copy(); su = su.copy(); sv = sv.copy()
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = (np.abs(stu) > smallflow) | (np.abs(stv) > smallflow)
    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    mepe = np.mean(epe[ind2]) if np.any(ind2) else 0.0
    return mepe


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0].copy()
    v = flow[:, :, 1].copy()

    # 过滤未知，大幅减小后续运算的异常
    idxUnknow = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # 这些打印与原实现完全一致（包含范围顺序）
    maxu = np.max(u) if u.size else -999.
    minu = np.min(u) if u.size else 999.
    maxv = np.max(v) if v.size else -999.
    minv = np.min(v) if v.size else 999.

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, float(np.max(rad))) if rad.size else -1

    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" %
          (maxrad, minu, maxu, minv, maxv))

    # 归一化（保持原逻辑：避免除零）
    denom = (maxrad + np.finfo(float).eps)
    u = u / denom
    v = v / denom

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, None], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def evaluate_flow_file(gt, pred):
    """
    evaluate the estimated optical flow end point error according to ground truth provided
    :param gt: ground truth file path
    :param pred: estimated flow file path
    :return: end point error, float32
    """
    gt_flow = read_flow(gt)
    eva_flow = read_flow(pred)
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                            eva_flow[:, :, 0], eva_flow[:, :, 1])
    return average_pe


def evaluate_flow(gt_flow, pred_flow):
    """
    gt: ground-truth flow
    pred: estimated flow
    """
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                            pred_flow[:, :, 0], pred_flow[:, :, 1])
    return average_pe


"""
==============
Disparity Section
==============
"""

def read_disp_png(file_name):
    """
    Read optical flow from KITTI .png file
    :param file_name: name of the flow file
    :return: optical flow data in matrix
    """
    image_object = png.Reader(filename=file_name)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]['size']
    # channel 需要整数；保持与原逻辑一致（python3：//）
    channel = len(image_data[0]) // w
    rows = np.vstack([np.asarray(row, dtype=np.uint16)[None, :] for row in image_data])  # [h, channel*w]
    flow = np.zeros((h, w, channel), dtype=np.uint16)
    for j in range(channel):
        flow[:, :, j] = rows[:, j::channel]
    return (flow[:, :, 0] / 256).astype(flow.dtype)


def disp_to_flowfile(disp, filename):
    """
    Read KITTI disparity file in png format
    :param disp: disparity matrix
    :param filename: the flow file name to save
    :return: None
    """
    h, w = disp.shape[:2]
    empty_map = np.zeros((h, w), dtype=np.float32)
    data = np.dstack((disp, empty_map)).astype(np.float32, copy=False)
    with open(filename, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([w], dtype=np.int32).tofile(f)
        np.array([h], dtype=np.int32).tofile(f)
        data.tofile(f)

"""
==============
Image Section
==============
"""

def read_image(filename):
    """
    Read normal image of any format
    :param filename: name of the image file
    :return: image data in matrix uint8 type
    """
    img = Image.open(filename)
    im = np.array(img)
    return im


def warp_image(im, flow):
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height, image_width = im.shape[0], im.shape[1]
    flow_height, flow_width = flow.shape[0], flow.shape[1]
    n = image_height * image_width

    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx = fx + flow[:, :, 0]
    fy = fy + flow[:, :, 1]

    mask = (fx < 0) | (fx > flow_width) | (fy < 0) | (fy > flow_height)
    fx = np.clip(fx, 0, flow_width)
    fy = np.clip(fy, 0, flow_height)

    points = np.concatenate((ix.reshape(n, 1), iy.reshape(n, 1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n, 1)), axis=1)

    warp = np.zeros((image_height, image_width, im.shape[2]), dtype=np.uint8)
    for c in range(im.shape[2]):
        channel = im[:, :, c]
        # 与原实现保持一致：展示 channel
        plt.imshow(channel, cmap='gray')
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, c] = new_channel.astype(np.uint8)

    return warp

"""
==============
Others
==============
"""

def scale_image(image, new_range):
    """
    Linearly scale the image into desired range
    :param image: input image
    :param new_range: the new range to be aligned
    :return: image normalized in new range
    """
    min_val = np.min(image).astype(np.float32)
    max_val = np.max(image).astype(np.float32)
    min_val_new = np.array(min(new_range), dtype=np.float32)
    max_val_new = np.array(max(new_range), dtype=np.float32)
    # 与原行为一致：若 max==min 会产生 inf/NaN，保持不改语义
    scaled_image = (image - min_val) / (max_val - min_val) * (max_val_new - min_val_new) + min_val_new
    return scaled_image.astype(np.uint8)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    h, w = u.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    nanIdx = np.isnan(u) | np.isnan(v)
    if np.any(nanIdx):
        u = u.copy(); v = v.copy()
        u[nanIdx] = 0; v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    # 避免 Python-level 循环中的重复计算：对每个通道仍保持与原实现一致的插值/缩放
    for ch in range(colorwheel.shape[1]):
        tmp = colorwheel[:, ch]
        col0 = tmp[k0 - 1] / 255.0
        col1 = tmp[k1 - 1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col = col.copy()
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] *= 0.75

        # 映射到 0..255，并将 NaN 区域置零（保持原逻辑）
        plane = np.floor(255 * col * (1 - nanIdx)).astype(np.uint8)
        img[:, :, ch] = plane

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.float64)

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel
