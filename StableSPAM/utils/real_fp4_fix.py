import argparse
import torch
import triton
import triton.language as tl
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

# 从kernel.py导入必要组件
from .kernel import *

class TritonFP4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, a_scale, b_scale, configs):
        # 将输入转换为MXFP4格式并打包
        x_fp4 = MXFP4Tensor(data=x)
        x_packed = x_fp4.to_packed_tensor(dim=1)
        
        # 创建TMA描述符
        M, K = x_packed.shape
        _, N = weight.shape
        
        a_desc = TmaDescKernelParam(
            x_packed.data_ptr(),
            (M, K),
            [configs['BLOCK_SIZE_M'], configs['BLOCK_SIZE_K'] // configs['ELEM_PER_BYTE']],
            1
        )
        
        b_desc = TmaDescKernelParam(
            weight.data_ptr(),
            (N, K),  # 注意：权重应为转置后的维度
            [configs['BLOCK_SIZE_N'], configs['BLOCK_SIZE_K'] // configs['ELEM_PER_BYTE']],
            1
        )

        # 执行块缩放矩阵乘法
        output = block_scaled_matmul(
            a_desc, a_scale,
            b_desc, b_scale,
            torch.float16, M, N, K, configs
        )
        
        # 保存反向传播所需参数
        ctx.save_for_backward(x_packed, weight, a_scale, b_scale,bias)
        ctx.configs = configs
        
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的参数
        x_packed, weight, a_scale, b_scale,bias = ctx.saved_tensors
        configs = ctx.configs
        M, K = x_packed.shape
        N, _ = weight.shape

        # 计算梯度输入 (dL/dX = dL/dY * W^T)
        # 需要转置权重矩阵
        weight_desc_transposed = TmaDescKernelParam(
            weight.data_ptr(),
            (K, N),  # 转置后的维度
            [configs['BLOCK_SIZE_K'], configs['BLOCK_SIZE_N'] // configs['ELEM_PER_BYTE']],
            1
        )
        
        grad_input_packed = block_scaled_matmul(
            grad_output, None,  # grad_output不需要缩放
            weight_desc_transposed, None,
            torch.float16, M, K, N, configs
        )

        # 解包梯度输入
        grad_input = MXFP4Tensor(data=grad_input_packed.unpack_packed_tensor(
            grad_input_packed, dim=1, original_shape=(M, K*2)  # 假设原始K维度是打包前的两倍
        ).to(torch.float32)

        # 计算梯度权重 (dL/dW = X^T * dL/dY)
        x_desc_transposed = TmaDescKernelParam(
            x_packed.data_ptr(),
            (K, M),  # 转置后的维度
            [configs['BLOCK_SIZE_K'], configs['BLOCK_SIZE_M'] // configs['ELEM_PER_BYTE']],
            1
        )
        
        grad_weight_packed = block_scaled_matmul(
            x_desc_transposed, None,
            grad_output, None,
            torch.float16, K, N, M, configs
        )

        if bias is not None:
            grad_bias = grad_output.sum(dim=0) if grad_output is not None else None
        else:
            grad_bias = None
        return grad_input, grad_weight_packed, grad_bias, None, None, None

class RealFP4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        group_size=256,
        configs=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # 初始化权重为MXFP4格式
        weight_data = torch.empty((out_features, in_features), dtype=torch.float32, device=device)
        weight_fp4 = MXFP4Tensor(data=weight_data)
        self.weight = nn.Parameter(weight_fp4.to_packed_tensor(dim=1))
        
        # 初始化缩放参数
        self.a_scale = nn.Parameter(torch.ones(
            (in_features // 128, group_size // 16 // 4, 32, 4, 4),
            device=device
        ))
        self.b_scale = nn.Parameter(torch.ones(
            (out_features // 128, group_size // 16 // 4, 32, 4, 4),
            device=device
        ))
        
        # 初始化偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter('bias', None)
        
        # 配置参数
        self.configs = configs or {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 256,
            "num_stages": 4,
            "ELEM_PER_BYTE": 2,
            "VEC_SIZE": 16,
        }

        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 权重初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 缩放因子初始化
        nn.init.constant_(self.a_scale, 1.0)
        nn.init.constant_(self.b_scale, 1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 确保输入是MXFP4格式
        if x.dtype != torch.uint8:
            x_fp4 = MXFP4Tensor(data=x)
            x = x_fp4.to_packed_tensor(dim=1)
        
        return TritonFP4LinearFunction.apply(
            x, self.weight, self.bias, 
            self.a_scale, self.b_scale,
            self.configs
        )

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
