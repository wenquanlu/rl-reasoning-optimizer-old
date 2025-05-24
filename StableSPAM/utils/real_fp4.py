import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
# from triton.tools.experimental_descriptor import TmaDescKernelParam
# from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import pdb

from .kernel import *

# kernel 指南： MXFP4 Tensor 接受的data是x / scale之后的 Float32
# 我需要自己先计算scale存下来


def quantize_to_nvfp4(x: torch.Tensor) -> torch.Tensor:
    # 输入float32，输出uint8的格式，并且一个uint8有两个e2m1 fp4
    x_fp32 = x.to(torch.float32)
    x_fp4 = MXFP4Tensor(data=x_fp32, device=x.device)
    
    x_nvfp4 = x_fp4.to_packed_tensor(dim=1)
    return x_nvfp4

def dequantize_from_nvfp4(packed: torch.Tensor, original_shape: tuple, dim: int = 1, output_dtype=torch.float16) -> torch.Tensor:
    #输入 1个uint8中有两个e2m1 fp4，输出float32
    dummy = MXFP4Tensor(size=original_shape, device=packed.device)
    unpacked = dummy.unpack_packed_tensor(packed, dim=dim, original_shape=original_shape)
    dummy.data = unpacked
    result = dummy.to(torch.float32) #暂时不支持bf16的运算
    return result


class triton_fp4_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor, configs):
        M, K = x.shape
        _, N = weight.shape
        x = quantize_to_nvfp4(x)
        weight = quantize_to_nvfp4(weight)
        a_desc = TmaDescKernelParam(x.data_ptr(), x.shape, [configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_K"] // configs["ELEM_PER_BYTE"]], 1)
        b_desc = TmaDescKernelParam(weight.data_ptr(), weight.shape, [configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"] // configs["ELEM_PER_BYTE"]], 1)
        epsilon = 1e-8
        a_scale = torch.rand((M // 128, K // configs["VEC_SIZE"] // 4, 32, 4, 4), device=x.device) + epsilon
        b_scale = torch.rand((N // 128, K // configs["VEC_SIZE"] // 4, 32, 4, 4), device=weight.device) + epsilon
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        output = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, configs)
        ctx.save_for_backward(x, weight, a_scale, b_scale)
        ctx.configs = configs
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, weight, a_scale, b_scale = ctx.saved_tensors
        configs = ctx.configs
        grad_input = block_scaled_matmul(grad_output, weight, a_scale, b_scale, torch.float16, x.shape[0], x.shape[1], weight.shape[0], configs)
        grad_weight = block_scaled_matmul(x.T, grad_output, a_scale, b_scale, torch.float16, weight.shape[0], weight.shape[1], x.shape[0], configs)
        grad_bias = grad_output.sum(dim=0) if grad_output is not None else None
        return grad_input, grad_weight, grad_bias, None


class real_fp4linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        weight_data = None,
        bias_data = None,
        group_size = 256,
        stochastic_round = True,
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.stochastic_round = stochastic_round
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.uint8, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter('bias', None)
        
        if weight_data is not None:
            self.weight.data.copy_(weight_data)
        if bias_data is not None and bias:
            self.bias.data.copy_(bias_data)

        self.configs = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 256,
            "num_stages": 4,
            "ELEM_PER_BYTE": 2,   # nvfp4
            "VEC_SIZE": 16,       # nvfp4
        }
    
    def forward(self, input: Tensor) -> Tensor:
        # 调用自定义的 triton_fp4_linear 进行前向传播
        qweight_fp4 = quantize_to_nvfp4(self.weight)
        qweight = dequantize_from_nvfp4(qweight_fp4)
        qweight = self.weight - self.weight.detach() + qweight.detach()
        return triton_fp4_linear.apply(input, qweight, self.bias, self.configs)
