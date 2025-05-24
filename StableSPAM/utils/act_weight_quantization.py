import pdb
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .dummy_quant import fake_tensor_quantize, fake_tensor_dequantize

class A8Linear(torch.autograd.Function):
    quant_dim = -1
    quant_group_size = 64
    quant_bit = 4
    quant_topk = 2
    quant_type = 'linear'
    stochastic_round = False
    
    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (BS, N, C_in) in fp16
        # weigth: (C_out, C_in) in fp16??
        # Quantize x with specified parameters
        qx = fake_tensor_quantize(
            x, 
            dim=A8Linear.quant_dim, 
            group_size=A8Linear.quant_group_size, 
            bit=A8Linear.quant_bit, 
            topk=A8Linear.quant_topk, 
            quant_type=A8Linear.quant_type,
            stochastic=A8Linear.stochastic_round
        )
        
        # Save tensors and parameters needed for the backward pass
        ctx.save_for_backward(qx, weight, bias)
        ctx.x_shape = x.shape  # Save original shape separately
        
        # print(f"I am checking x quantization -> {x.shape} -> {qx.shape}")

        # Perform dequantization for forward computation
        def forward_a_float_activation(weight, x_dtype):
            # Dequantize with stored parameters
            float_x = fake_tensor_dequantize(
                qx, 
                dim=A8Linear.quant_dim, 
                shape=ctx.x_shape, 
                group_size=A8Linear.quant_group_size, 
                bit=A8Linear.quant_bit, 
                dtype=x_dtype, 
                topk=A8Linear.quant_topk, 
                quant_type=A8Linear.quant_type
            )
            return float_x @ weight.t() + bias if bias is not None else float_x @ weight.t()

        # Determine x dtype
        if x.dtype == torch.half:
            x_dtype = 'fp16'
        elif x.dtype == torch.bfloat16:
            x_dtype = 'bf16'
        else:
            x_dtype = 'fp32'

        # Apply forward computation with quantized activation
        output = forward_a_float_activation(weight, x_dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with dequantized activations.
        """
        # Retrieve saved tensors
        qx, weight, bias = ctx.saved_tensors
        x_shape = ctx.x_shape  # Retrieve original shape

        # Determine weight dtype for dequantization
        if weight.dtype == torch.half:
            x_dtype = 'fp16'
        elif weight.dtype == torch.bfloat16:
            x_dtype = 'bf16'
        else:
            x_dtype = 'fp32'

        # Dequantize activations for gradient computation
        x = fake_tensor_dequantize(
            qx, 
            dim=A8Linear.quant_dim, 
            shape=x_shape, 
            group_size=A8Linear.quant_group_size, 
            bit=A8Linear.quant_bit, 
            dtype=x_dtype, 
            topk=A8Linear.quant_topk, 
            quant_type=A8Linear.quant_type
        )

        # Compute gradients
        grad_input = grad_output @ weight
        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None
        
        # Compute gradient with respect to weight
        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ x.reshape(-1, x.shape[-1])

        return grad_input, grad_weight, grad_bias

def _quantize_tensor(w, q_group_size=-1, n_bit=4):

    org_w_shape = w.shape
    if q_group_size > 0:
        assert w.nelement() % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape).to(torch.uint8)

    return w, scales, zeros


class QLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                device='cpu', dtype=None, weight_data=None, bias_data=None, num_bits=4, group_size=256, stochastic_round=True,topk=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.weight.__setattr__('stochastic_round', stochastic_round)
        self.weight.__setattr__('group_size', group_size)

        if weight_data is not None:
            self.weight.data.copy_(weight_data)
        if bias_data is not None and bias is not None:
            self.bias.data.copy_(bias_data)

        self.num_bits = num_bits
        self.group_size = group_size

        A8Linear.quant_bit = num_bits
        A8Linear.quant_group_size = group_size
        A8Linear.stochastic_round = stochastic_round
        A8Linear.quant_dim = -1
        A8Linear.quant_topk = 2

    def forward(self, input: Tensor) -> Tensor:
        qweight, scales, zeros = _quantize_tensor(self.weight, q_group_size=self.group_size, n_bit=self.num_bits)
        # dequantize to Bfloat16
        qweight = qweight.to(input.dtype).reshape(-1, self.group_size)
        qweight = (qweight - zeros) * scales
        qweight = qweight.reshape(self.weight.shape)
        # STE backward
        qweight = qweight.detach() + self.weight - self.weight.detach()
        output = A8Linear.apply(input, qweight, self.bias)
        # output = input @ qweight.t()
        # if self.bias is not None:
        #     output += self.bias

        return output


def prepare_model_for_int8_training_simulation_act_weight(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_int8_training_simulation_act_weight(module, args, target_module)

        if isinstance(module, nn.Linear):
            if not name in target_module:
                print('Keep in original linear layer', name, module)
                continue
            
            # NOTE(hanqing): no need to pass those stuffs
            bias_data = module.bias.data if module.bias is not None else None
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            weight_data = module.weight.data
            new_layers = QLinear(in_features, out_features, bias=bias, device='cuda:0', 
                weight_data=weight_data, bias_data=bias_data, 
                num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round)

            model._modules[name] = new_layers

    return model
