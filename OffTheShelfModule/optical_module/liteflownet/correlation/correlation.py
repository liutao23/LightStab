#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import re
import torch
import cupy

# =========================================
# CUDA kernels (NVRTC RawKernel)
# SIZE_i(t) / VALUE_i(t, ...) 在运行时通过 cupy_kernel() 替换为常量
# =========================================

kernel_Correlation_rearrange = r'''
extern "C" __global__ void kernel_Correlation_rearrange(
    const int n,
    const float* __restrict__ input,
    float* __restrict__ output
) {
  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (intIndex >= n) { return; }

  int intSample = blockIdx.z;
  int intChannel = blockIdx.y;

  // input: [N, C, H, W] (NCHW), output: [N, H+6s, W+6s, C]
  float fltValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

  __syncthreads();

  int intPaddedY = (intIndex / SIZE_3(input)) + 3*{{intStride}};
  int intPaddedX = (intIndex % SIZE_3(input)) + 3*{{intStride}};
  int intRearrange = ((SIZE_3(input) + 6*{{intStride}}) * intPaddedY) + intPaddedX;

  // output layout: [N, (H+6s)*(W+6s), C] 展开为 [N, H', W', C] 的线性索引
  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = fltValue;
}
'''

kernel_Correlation_updateOutput = r'''
extern "C" __global__ void kernel_Correlation_updateOutput(
  const int n,
  const float* __restrict__ rbot0,
  const float* __restrict__ rbot1,
  float* __restrict__ top
) {
  extern __shared__ char patch_data_char[];
  float *patch_data = (float *)patch_data_char;

  // 当前 block 对应的中心像素（以 stride 为步长采样）
  int x1 = (blockIdx.x + 3) * {{intStride}};
  int y1 = (blockIdx.y + 3) * {{intStride}};
  int item = blockIdx.z;
  int ch_off = threadIdx.x;  // 0..31 (warp)

  // 将 rbot0 的 1x1xC 小块搬到 shared
  for (int j = 0; j < 1; j++) {
    for (int i = 0; i < 1; i++) {
      int ji_off = (j + i) * SIZE_3(rbot0);
      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) {
        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
        int idxPatchData = ji_off + ch;
        patch_data[idxPatchData] = rbot0[idx1];
      }
    }
  }

  __syncthreads();

  __shared__ float sum[32];

  // top: [N, 49, Hs, Ws] (49=7x7窗口)
  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
    sum[ch_off] = 0.0f;

    int s2o = (top_channel % 7 - 3) * {{intStride}};
    int s2p = (top_channel / 7 - 3) * {{intStride}};

    for (int j = 0; j < 1; j++) {
      for (int i = 0; i < 1; i++) {
        int ji_off = (j + i) * SIZE_3(rbot0);
        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) {
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;

          int idxPatchData = ji_off + ch;
          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;

          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
        }
      }
    }

    __syncthreads();

    if (ch_off == 0) {
      float total_sum = 0.0f;
      for (int idx = 0; idx < 32; idx++) { total_sum += sum[idx]; }

      const int sumelems = SIZE_3(rbot0);
      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
    }

    __syncthreads();
  }
}
'''

kernel_Correlation_updateGradOne = r'''
#define ROUND_OFF 50000

extern "C" __global__ void kernel_Correlation_updateGradOne(
  const int n,
  const int intSample,
  const float* __restrict__ rbot0,
  const float* __restrict__ rbot1,
  const float* __restrict__ gradOutput,
  float* __restrict__ gradOne,
  float* __restrict__ gradTwo
) {
  for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
    int n_ch = intIndex % SIZE_1(gradOne); // channels
    int l    = (intIndex / SIZE_1(gradOne)) % SIZE_3(gradOne) + 3*{{intStride}}; // w-pos
    int m    = (intIndex / SIZE_1(gradOne) / SIZE_3(gradOne)) % SIZE_2(gradOne) + 3*{{intStride}}; // h-pos

    const int round_off = ROUND_OFF;
    const int round_off_s1 = {{intStride}} * round_off;

    int xmin = (l - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil
    int ymin = (m - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil
    int xmax = (l - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor
    int ymax = (m - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor

    float sum = 0.0f;
    if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
      xmin = max(0,xmin);
      xmax = min(SIZE_3(gradOutput)-1,xmax);
      ymin = max(0,ymin);
      ymax = min(SIZE_2(gradOutput)-1,ymax);

      for (int p = -3; p <= 3; p++) {
        for (int o = -3; o <= 3; o++) {
          int s2o = {{intStride}} * o;
          int s2p = {{intStride}} * p;

          int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n_ch;
          float bot1tmp = rbot1[idxbot1];

          int op = (p+3) * 7 + (o+3);
          int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

          for (int y = ymin; y <= ymax; y++) {
            for (int x = xmin; x <= xmax; x++) {
              int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x;
              sum += gradOutput[idxgradOutput] * bot1tmp;
            }
          }
        }
      }
    }
    const int sumelems = SIZE_1(gradOne);
    const int bot0index = ((n_ch * SIZE_2(gradOne)) + (m-3*{{intStride}})) * SIZE_3(gradOne) + (l-3*{{intStride}});
    gradOne[bot0index + intSample*SIZE_1(gradOne)*SIZE_2(gradOne)*SIZE_3(gradOne)] = sum / (float)sumelems;
  }
}
'''

kernel_Correlation_updateGradTwo = r'''
#define ROUND_OFF 50000

extern "C" __global__ void kernel_Correlation_updateGradTwo(
  const int n,
  const int intSample,
  const float* __restrict__ rbot0,
  const float* __restrict__ rbot1,
  const float* __restrict__ gradOutput,
  float* __restrict__ gradOne,
  float* __restrict__ gradTwo
) {
  for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
    int n_ch = intIndex % SIZE_1(gradTwo);
    int l    = (intIndex / SIZE_1(gradTwo)) % SIZE_3(gradTwo) + 3*{{intStride}};
    int m    = (intIndex / SIZE_1(gradTwo) / SIZE_3(gradTwo)) % SIZE_2(gradTwo) + 3*{{intStride}};

    const int round_off = ROUND_OFF;
    const int round_off_s1 = {{intStride}} * round_off;

    float sum = 0.0f;
    for (int p = -3; p <= 3; p++) {
      for (int o = -3; o <= 3; o++) {
        int s2o = {{intStride}} * o;
        int s2p = {{intStride}} * p;

        int xmin = (l - 3*{{intStride}} - s2o + round_off_s1 - 1) / {{intStride}} + 1 - round_off;
        int ymin = (m - 3*{{intStride}} - s2p + round_off_s1 - 1) / {{intStride}} + 1 - round_off;
        int xmax = (l - 3*{{intStride}} - s2o + round_off_s1) / {{intStride}} - round_off;
        int ymax = (m - 3*{{intStride}} - s2p + round_off_s1) / {{intStride}} - round_off;

        if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
          xmin = max(0,xmin);
          xmax = min(SIZE_3(gradOutput)-1,xmax);
          ymin = max(0,ymin);
          ymax = min(SIZE_2(gradOutput)-1,ymax);

          int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n_ch;
          float bot0tmp = rbot0[idxbot0];

          int op = (p+3) * 7 + (o+3);
          int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

          for (int y = ymin; y <= ymax; y++) {
            for (int x = xmin; x <= xmax; x++) {
              int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x;
              sum += gradOutput[idxgradOutput] * bot0tmp;
            }
          }
        }
      }
    }
    const int sumelems = SIZE_1(gradTwo);
    const int bot1index = ((n_ch * SIZE_2(gradTwo)) + (m-3*{{intStride}})) * SIZE_3(gradTwo) + (l-3*{{intStride}});
    gradTwo[bot1index + intSample*SIZE_1(gradTwo)*SIZE_2(gradTwo)*SIZE_3(gradTwo)] = sum / (float)sumelems;
  }
}
'''

# =========================================
# Kernel helpers: 宏替换 / RawKernel 构建 / 与 PyTorch 流对齐
# =========================================

def _torch_cupy_stream():
    """让 CuPy 使用与 PyTorch 相同的 CUDA Stream。"""
    s = torch.cuda.current_stream()
    return cupy.cuda.ExternalStream(s.cuda_stream)

def cupy_kernel(strFunction, objVariables):
    """把 SIZE_i(tensor) / VALUE_i(tensor, ...) 宏替换为常量索引表达式。"""
    strKernel = globals()[strFunction].replace('{{intStride}}', str(objVariables['intStride']))

    # SIZE_i(t) -> 常量尺寸
    while True:
        m = re.search(r'(SIZE_)([0-4])\(([^\)]+)\)', strKernel)
        if m is None:
            break
        dim = int(m.group(2))
        tname = m.group(3)
        sizes = objVariables[tname].size()
        val = sizes[dim] if not torch.is_tensor(sizes[dim]) else sizes[dim].item()
        strKernel = strKernel.replace(m.group(0), str(int(val)))

    # VALUE_i(...) 若有可在更复杂核里展开；本实现不需要，保留框架
    while True:
        m = re.search(r'(VALUE_)([0-4])\(([^\)]+)\)', strKernel)
        if m is None:
            break
        raise NotImplementedError("VALUE_i 宏在本实现中未使用，如需请自行扩展。")

    return strKernel

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    """编译 RawKernel，指定 NVRTC 选项。"""
    return cupy.RawKernel(
        strKernel,
        strFunction,
        options=('-std=c++11',),
        backend='nvrtc'
    )

# =========================================
# Autograd Function
# =========================================

class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, one, two, intStride: int):
        assert one.is_cuda and two.is_cuda, "inputs must be CUDA tensors"
        # 强制 NCHW + float32 + contiguous（禁用 channels_last/AMP）
        one = one.to(dtype=torch.float32)
        two = two.to(dtype=torch.float32)
        if not one.is_contiguous(): one = one.contiguous()
        if not two.is_contiguous(): two = two.contiguous()

        N, C, H, W = one.shape
        # re-arranged buffers: [N, H+6s, W+6s, C]
        rbot0 = one.new_zeros([N, H + 6*intStride, W + 6*intStride, C])
        rbot1 = one.new_zeros([N, H + 6*intStride, W + 6*intStride, C])

        # 输出： [N, 49, ceil(H/s), ceil(W/s)]
        out_H = int(math.ceil(H / intStride))
        out_W = int(math.ceil(W / intStride))
        output = one.new_zeros([N, 49, out_H, out_W])

        # ---- Launch kernels on the SAME stream as PyTorch ----
        with _torch_cupy_stream():
            # rearrange one
            n_pix = H * W
            cupy_launch(
                'kernel_Correlation_rearrange',
                cupy_kernel('kernel_Correlation_rearrange', {
                    'intStride': intStride, 'input': one, 'output': rbot0
                })
            )(
                grid=(int((n_pix + 16 - 1) // 16), C, N),
                block=(16, 1, 1),
                args=[ cupy.int32(n_pix),
                       cupy.uint64(one.data_ptr()),
                       cupy.uint64(rbot0.data_ptr()) ]
            )

            # rearrange two
            n_pix2 = H * W
            cupy_launch(
                'kernel_Correlation_rearrange',
                cupy_kernel('kernel_Correlation_rearrange', {
                    'intStride': intStride, 'input': two, 'output': rbot1
                })
            )(
                grid=(int((n_pix2 + 16 - 1) // 16), C, N),
                block=(16, 1, 1),
                args=[ cupy.int32(n_pix2),
                       cupy.uint64(two.data_ptr()),
                       cupy.uint64(rbot1.data_ptr()) ]
            )

            # correlation updateOutput
            n_top = output.shape[1] * output.shape[2] * output.shape[3]
            cupy_launch(
                'kernel_Correlation_updateOutput',
                cupy_kernel('kernel_Correlation_updateOutput', {
                    'intStride': intStride, 'rbot0': rbot0, 'rbot1': rbot1, 'top': output
                })
            )(
                grid=(output.shape[3], output.shape[2], N),
                block=(32, 1, 1),
                shared_mem=C * 4,  # float32 bytes
                args=[ cupy.int32(n_top),
                       cupy.uint64(rbot0.data_ptr()),
                       cupy.uint64(rbot1.data_ptr()),
                       cupy.uint64(output.data_ptr()) ]
            )

        ctx.intStride = intStride
        ctx.save_for_backward(one, two, rbot0, rbot1)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        one, two, rbot0, rbot1 = ctx.saved_tensors
        intStride = ctx.intStride

        if gradOutput is None:
            return None, None, None
        gradOutput = gradOutput.to(dtype=torch.float32).contiguous()
        assert gradOutput.is_cuda, "gradOutput must be CUDA tensor"

        gradOne = one.new_zeros(one.shape) if ctx.needs_input_grad[0] else None
        gradTwo = one.new_zeros(one.shape) if ctx.needs_input_grad[1] else None

        with _torch_cupy_stream():
            if gradOne is not None:
                for intSample in range(one.shape[0]):
                    n = one.shape[1] * one.shape[2] * one.shape[3]
                    cupy_launch(
                        'kernel_Correlation_updateGradOne',
                        cupy_kernel('kernel_Correlation_updateGradOne', {
                            'intStride': intStride,
                            'rbot0': rbot0, 'rbot1': rbot1,
                            'gradOutput': gradOutput, 'gradOne': gradOne, 'gradTwo': None
                        })
                    )(
                        grid=(int((n + 512 - 1) // 512), 1, 1),
                        block=(512, 1, 1),
                        args=[ cupy.int32(n),
                               cupy.int32(intSample),
                               cupy.uint64(rbot0.data_ptr()),
                               cupy.uint64(rbot1.data_ptr()),
                               cupy.uint64(gradOutput.data_ptr()),
                               cupy.uint64(gradOne.data_ptr()),
                               cupy.uint64(0) ]
                    )

            if gradTwo is not None:
                for intSample in range(one.shape[0]):
                    n = one.shape[1] * one.shape[2] * one.shape[3]
                    cupy_launch(
                        'kernel_Correlation_updateGradTwo',
                        cupy_kernel('kernel_Correlation_updateGradTwo', {
                            'intStride': intStride,
                            'rbot0': rbot0, 'rbot1': rbot1,
                            'gradOutput': gradOutput, 'gradOne': None, 'gradTwo': gradTwo
                        })
                    )(
                        grid=(int((n + 512 - 1) // 512), 1, 1),
                        block=(512, 1, 1),
                        args=[ cupy.int32(n),
                               cupy.int32(intSample),
                               cupy.uint64(rbot0.data_ptr()),
                               cupy.uint64(rbot1.data_ptr()),
                               cupy.uint64(gradOutput.data_ptr()),
                               cupy.uint64(0),
                               cupy.uint64(gradTwo.data_ptr()) ]
                    )

        # 返回与 forward 输入一一对应：gradOne, gradTwo, None(for intStride)
        return gradOne, gradTwo, None

# =========================================
# Public API
# =========================================

def FunctionCorrelation(tenOne, tenTwo, intStride: int):
    """Functional 调用"""
    return _FunctionCorrelation.apply(tenOne, tenTwo, intStride)

class ModuleCorrelation(torch.nn.Module):
    """nn.Module 封装"""
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride
    def forward(self, tenOne, tenTwo):
        return _FunctionCorrelation.apply(tenOne, tenTwo, self.stride)

# =========================================
# Quick self-test
# =========================================
if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA 环境"
    torch.manual_seed(0)

    # 开启梯度！
    x = torch.randn(2, 64, 32, 48, device='cuda', dtype=torch.float32,
                    requires_grad=True).contiguous()
    y = torch.randn(2, 64, 32, 48, device='cuda', dtype=torch.float32,
                    requires_grad=True).contiguous()

    out = FunctionCorrelation(x, y, intStride=1)
    print("out shape:", tuple(out.shape))  # (2, 49, 32, 48)

    out_mean = out.mean()
    out_mean.backward()

    print("x.grad mean abs:", x.grad.abs().mean().item())
    print("y.grad mean abs:", y.grad.abs().mean().item())

