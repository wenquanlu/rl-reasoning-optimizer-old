import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
from triton.tools.experimental_descriptor import TmaDescKernelParam
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import pdb

from .kernel import *

# kernel 指南： MXFP4 Tensor 接受的data是x / scale之后的 Float32
# 我需要自己先计算scale存下来

# 先写一个quantization函数，接受一个float32 tensor: x，返回q_x，scale
def fp4_quantization_step1(x : torch.Tensor):
    x = x.clone()
    x_shape = x.shape
    x = x.view(-1,256)
    x_packed_shape = x.shape
    scale = x.abs().max(dim=-1).values.clamp(min = 1e-6)
    q_max = alpha
    q_min = - q_max
    x_clamped = torch.clamp(x, q_min[:, None], q_max[:, None])
    x_bit = x_clamped / scale.unsqueeze(1)
    dummy = MXFP4Tensor(data = x_bit,device = x.device()) #这里data已经转换成uint8形式了
    packed = dummy.to_packed_tensor(dim = 1) # 一个uint8装两个fp4
    return scale, packed, x_shape, x_packed_shape

def fp4_dequantization_step1(scale, packed, x_origin_shape, x_packed_shape):
    dummy = MXFP4Tensor()
    dummy.data = dummy.unpack_packed_tensor(packed_tensor = packed, dim = 1, original_shape = x_packed_shape )
    unpack = dummy.to(torch.float32)
    unpack.view(x_origin_shape)
    return unpack

def fp4_quantization_all(x: torch.Tensor, output_shape = None, transpose = False):
    scale, packed, x_shape, x_packed_shape = fp4_quantization_step1(x)
    unpack = fp4_dequantization_step1(scale, packed, x_shape, x_packed_shape)
    if transpose:
        unpack = unpack.T
    scale_1, packed_1, x_shape_1, x_packed_shape_1 = fp4_quantization_step1(unpack)
    return scale_1, packed_1,x_shape_1, x_packed_shape_1


def fp4_dequantization_all(scale, packed, x_packed_shape):
    dummy = MXFP4Tensor()
    dummy.data = dummy.unpack_packed_tensor(packed_tensor = packed, dim = 1, original_shape = x_packed_shape )
    unpack = dummy.to(torch.float32)
    return unpack

class A8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor):
        # ---- debug
        x_scale, x_packed, x_shape, x_packed_shape = fp4_quantization_all(x)
        w_scale, w_packed, w_shape, w_packed_shape = fp4_quantization_all(w,transpose = True)
        ctx.save_for_backward(x_scale, x_packed, x_shape, x_packed_shape, w_scale, w_packed, w_shape, w_packed_shape,bias)

        x_scale = x_scale.to(torch.float8_e4m3fn)
        w_scale = w_scale.to(torch.float8_e4m3fn)

        x_desc = TmaDescKernelParam(x_packed.data_ptr(), x_shape, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE], 1)
        w_desc = TmaDescKernelParam(w_packed.data_ptr(), w_shape, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE], 1) 
        configs = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 256,
        "num_stages": 4,
        "ELEM_PER_BYTE": 2,
        "VEC_SIZE": 16,
    }
        output = block_scaled_matmul(x_desc, x_scale, w_desc,w_scale, torch.float32,x_shape[0],w_shape[1],x_shape[1],configs)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        quant_x, weight, bias = ctx.saved_tensors

        grad_input =  grad_output @ weight

        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
                                      quant_x.reshape(-1, quant_x.shape[-1])
        
        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias


