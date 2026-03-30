#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import cupy
import re

# =========================
# CUDA kernels (unchanged math)
# =========================
kernel_Correlation_rearrange = r'''
extern "C" __global__ void kernel_Correlation_rearrange(
    const int n,
    const float* input,
    float* output
) {
  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (intIndex >= n) { return; }

  int intSample = blockIdx.z;
  int intChannel = blockIdx.y;

  float fltValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;

  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = fltValue;
}
''';

kernel_Correlation_updateOutput = r'''
extern "C" __global__ void kernel_Correlation_updateOutput(
  const int n,
  const float* rbot0,
  const float* rbot1,
  float* top
) {
  extern __shared__ char patch_data_char[];
  float *patch_data = (float *)patch_data_char;

  int x1 = blockIdx.x + 4;
  int y1 = blockIdx.y + 4;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;

  // load 1x1xC patch from rbot0 into shared memory
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

  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
    sum[ch_off] = 0.0f;

    int s2o = top_channel % 9 - 4;
    int s2p = top_channel / 9 - 4;

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
  }
}
''';

kernel_Correlation_updateGradFirst = r'''
#define ROUND_OFF 50000
extern "C" __global__ void kernel_Correlation_updateGradFirst(
  const int n,
  const int intSample,
  const float* rbot0,
  const float* rbot1,
  const float* gradOutput,
  float* gradFirst,
  float* gradSecond
) {
  for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
       intIndex < n; intIndex += blockDim.x * gridDim.x) {

    int n = intIndex % SIZE_1(gradFirst);
    int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 4;
    int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 4;

    const int round_off = ROUND_OFF;
    const int round_off_s1 = round_off;

    int xmin = (l - 4 + round_off_s1 - 1) + 1 - round_off;
    int ymin = (m - 4 + round_off_s1 - 1) + 1 - round_off;

    int xmax = (l - 4 + round_off_s1) - round_off;
    int ymax = (m - 4 + round_off_s1) - round_off;

    float sum = 0.0f;
    if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
      xmin = max(0,xmin); xmax = min(SIZE_3(gradOutput)-1,xmax);
      ymin = max(0,ymin); ymax = min(SIZE_2(gradOutput)-1,ymax);

      for (int p = -4; p <= 4; p++) {
        for (int o = -4; o <= 4; o++) {
          int s2o = o, s2p = p;
          int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
          float bot1tmp = rbot1[idxbot1];

          int op = (p+4) * 9 + (o+4);
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
    const int sumelems = SIZE_1(gradFirst);
    const int bot0index = ((n * SIZE_2(gradFirst)) + (m-4)) * SIZE_3(gradFirst) + (l-4);
    gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
  }
}
''';

kernel_Correlation_updateGradSecond = r'''
#define ROUND_OFF 50000
extern "C" __global__ void kernel_Correlation_updateGradSecond(
  const int n,
  const int intSample,
  const float* rbot0,
  const float* rbot1,
  const float* gradOutput,
  float* gradFirst,
  float* gradSecond
) {
  for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
       intIndex < n; intIndex += blockDim.x * gridDim.x) {

    int n = intIndex % SIZE_1(gradSecond);
    int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 4;
    int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 4;

    const int round_off = ROUND_OFF;
    const int round_off_s1 = round_off;

    float sum = 0.0f;
    for (int p = -4; p <= 4; p++) {
      for (int o = -4; o <= 4; o++) {
        int s2o = o, s2p = p;

        int xmin = (l - 4 - s2o + round_off_s1 - 1) + 1 - round_off;
        int ymin = (m - 4 - s2p + round_off_s1 - 1) + 1 - round_off;

        int xmax = (l - 4 - s2o + round_off_s1) - round_off;
        int ymax = (m - 4 - s2p + round_off_s1) - round_off;

        if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
          xmin = max(0,xmin); xmax = min(SIZE_3(gradOutput)-1,xmax);
          ymin = max(0,ymin); ymax = min(SIZE_2(gradOutput)-1,ymax);

          int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
          float bot0tmp = rbot0[idxbot0];

          int op = (p+4) * 9 + (o+4);
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
    const int sumelems = SIZE_1(gradSecond);
    const int bot1index = ((n * SIZE_2(gradSecond)) + (m-4)) * SIZE_3(gradSecond) + (l-4);
    gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
  }
}
''';

# =========================
# Regex pre-compile (minor speed)
# =========================
_re_size = re.compile(r'(SIZE_)([0-4])(\()([^\)]*)(\))')
_re_value = re.compile(r'(VALUE_)([0-4])(\()([^\)]+)(\))')


def cupy_kernel(strFunction, objVariables):
    # inject static sizes/strides
    strKernel = globals()[strFunction]

    while True:
        m = _re_size.search(strKernel)
        if m is None:
            break
        intArg = int(m.group(2))
        strTensor = m.group(4)
        sz = objVariables[strTensor].size()
        strKernel = strKernel.replace(m.group(), str(int(sz[intArg])))
    while True:
        m = _re_value.search(strKernel)
        if m is None:
            break
        intArgs = int(m.group(2))
        strArgs = m.group(4).split(',')
        strTensor = strArgs[0]
        strides = objVariables[strTensor].stride()
        pieces = []
        for k in range(intArgs):
            idx_expr = strArgs[k + 1].replace('{', '(').replace('}', ')').strip()
            pieces.append(f'(({idx_expr})*{int(strides[k])})')
        strKernel = strKernel.replace(m.group(0), f'{strTensor}[' + '+'.join(pieces) + ']')
    return strKernel


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    mod = cupy.RawModule(code=strKernel, backend='nvrtc')
    return mod.get_function(strFunction)


# =========================
# Autograd function
# =========================
class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second):
        # --- input checks (no functional changes) ---
        if not first.is_cuda or not second.is_cuda:
            raise NotImplementedError("This Correlation implementation only supports CUDA tensors.")
        if not first.is_contiguous():
            first = first.contiguous()
        if not second.is_contiguous():
            second = second.contiguous()
        if first.dtype != torch.float32 or second.dtype != torch.float32:
            raise TypeError("first/second must be float32.")

        B, C, H, W = first.shape
        # pre-alloc outputs (as before)
        rbot0 = first.new_zeros([B, H + 8, W + 8, C])
        rbot1 = first.new_zeros([B, H + 8, W + 8, C])
        out = first.new_zeros([B, 81, H, W])

        # save for backward
        ctx.save_for_backward(first, second, rbot0, rbot1)

        # launch on current torch stream
        torch_stream = torch.cuda.current_stream().cuda_stream
        with cupy.cuda.ExternalStream(torch_stream):
            # rearrange first
            n = H * W
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'input': first, 'output': rbot0
            }))(
                grid=( (n + 15) // 16, C, B ),
                block=(16, 1, 1),
                args=( cupy.int32(n),
                       cupy.uint64(first.data_ptr()),
                       cupy.uint64(rbot0.data_ptr()) )
            )

            # rearrange second
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'input': second, 'output': rbot1
            }))(
                grid=( (n + 15) // 16, C, B ),
                block=(16, 1, 1),
                args=( cupy.int32(n),
                       cupy.uint64(second.data_ptr()),
                       cupy.uint64(rbot1.data_ptr()) )
            )

            # correlation
            n_out = out.shape[1] * out.shape[2] * out.shape[3]
            shared_mem_bytes = C * 4  # float32 per channel
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
                'rbot0': rbot0, 'rbot1': rbot1, 'top': out
            }))(
                grid=( W, H, B ),
                block=(32, 1, 1),
                shared_mem=shared_mem_bytes,
                args=( cupy.int32(n_out),
                       cupy.uint64(rbot0.data_ptr()),
                       cupy.uint64(rbot1.data_ptr()),
                       cupy.uint64(out.data_ptr()) )
            )

        return out

    @staticmethod
    def backward(ctx, gradOutput):
        first, second, rbot0, rbot1 = ctx.saved_tensors

        if not gradOutput.is_contiguous():
            gradOutput = gradOutput.contiguous()

        gradFirst = None
        gradSecond = None

        if ctx.needs_input_grad[0]:
            gradFirst = first.new_zeros(first.shape)
        if ctx.needs_input_grad[1]:
            gradSecond = first.new_zeros(first.shape)

        torch_stream = torch.cuda.current_stream().cuda_stream
        with cupy.cuda.ExternalStream(torch_stream):
            if gradFirst is not None:
                n = first.shape[1] * first.shape[2] * first.shape[3]
                grid_x = (n + 511) // 512
                for sample in range(first.shape[0]):
                    cupy_launch('kernel_Correlation_updateGradFirst', cupy_kernel('kernel_Correlation_updateGradFirst', {
                        'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput,
                        'gradFirst': gradFirst, 'gradSecond': None
                    }))(
                        grid=(grid_x, 1, 1),
                        block=(512, 1, 1),
                        args=( cupy.int32(n), cupy.int32(sample),
                               cupy.uint64(rbot0.data_ptr()), cupy.uint64(rbot1.data_ptr()),
                               cupy.uint64(gradOutput.data_ptr()),
                               cupy.uint64(gradFirst.data_ptr()),
                               cupy.uint64(0) )
                    )

            if gradSecond is not None:
                n = first.shape[1] * first.shape[2] * first.shape[3]
                grid_x = (n + 511) // 512
                for sample in range(first.shape[0]):
                    cupy_launch('kernel_Correlation_updateGradSecond', cupy_kernel('kernel_Correlation_updateGradSecond', {
                        'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput,
                        'gradFirst': None, 'gradSecond': gradSecond
                    }))(
                        grid=(grid_x, 1, 1),
                        block=(512, 1, 1),
                        args=( cupy.int32(n), cupy.int32(sample),
                               cupy.uint64(rbot0.data_ptr()), cupy.uint64(rbot1.data_ptr()),
                               cupy.uint64(gradOutput.data_ptr()),
                               cupy.uint64(0),
                               cupy.uint64(gradSecond.data_ptr()) )
                    )

        return gradFirst, gradSecond


# public API (unchanged)
def FunctionCorrelation(tenFirst, tenSecond):
    return _FunctionCorrelation.apply(tenFirst, tenSecond)


class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tenFirst, tenSecond):
        return _FunctionCorrelation.apply(tenFirst, tenSecond)
