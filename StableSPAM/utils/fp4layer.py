import pdb
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def getscale(x: torch.Tensor, Qmin: float, Qmax: float) -> torch.Tensor:
    # make sure Qmin <= Qmax
    if Qmin > Qmax:
        raise ValueError("Qmin should be less than or equal to Qmax.")
    x_clipped = torch.min(torch.max(x,Qmax),Qmin)
    x_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(x))+x_bias)).detach(),1.0)
    
    return x_clipped

def get_x_paramter(x: torch.Tensor)

class A8Linear(torch.autograd.Function):
    
    
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
    def __init__(self,
        infeatures: int,
        outfeatures: int,
        bias: bool = True,
        device = 'cpu',
        dtype = None,
        weight_data = None,
        bias_data = None,
        w_bit = 8,
        a_bit = 8,
        group_size = 256,
        w_exponent_bit = 4,
        a_exponent_bit = 4,
        stochastic_round =True)->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.register_buffer('w_exponent_bit',torch.tensor(w_exponent_bit))
        self.register_buffer('w_mantissa_bit',torch.tensor(w_bit - 1 - w_exponent_bit))
        self.register_buffer('a_exponent_bit',torch.tensor(a_exponent_bit))
        self.register_buffer('a_mantissa_bit',torch.tensor(a_bit - 1 - a_exponent_bit))
        self.register_buffer('w_interval',None)
        self.register_buffer('a_interval',None)
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
    
    def get_maxval_from_bias(self, act_or_weight):
        if act_or_weight == 0 and self.a_interval != None:
            return (2 - 2 ** (-self.a_mantissa_bit)) * 2 ** (
                2**self.a_exponent_bit - 1 - self.a_interval
            )
        elif act_or_weight == 1 and self.w_interval != None:
            return (2 - 2 ** (-self.w_mantissa_bit)) * 2 ** (
                2**self.w_exponent_bit - 1 - self.w_interval
            )
        else:
            raise AssertionError
    
    def get_log_scale(self, x ,act_or_weight):
        
        if act_or_weight == 0:
            a_maxval = self.get_maxval_from_bias(0)
            a_bias = self.a_interval if self.a_interval != None else self.default_bias
            a_bias = a_bias.float()
            a_minval = -a_maxval
            a = torch.min(torch.max(x, a_minval), a_maxval)
            a_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(a)) + a_bias)).detach(), 1.0)
            return a, 2.0 ** (a_log_scales - self.a_mantissa_bit - a_bias)
        
        elif act_or_weight == 1:
            w_maxval = self.get_maxval_from_bias(1)
            w_bias = self.w_interval if self.w_interval != None else self.default_bias
            w_bias = w_bias.float()
            w_minval = -w_maxval
            w = torch.min(torch.max(x, w_minval), w_maxval)
            w_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(w)) + w_bias)).detach(), 1.0)
            return w, 2.0 ** (w_log_scales - self.w_mantissa_bit - w_bias)

    def get_scale(self, input, bits, mantissa_bit, bias):
        
        M = mantissa_bit
        E = bits - 1 - M
        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        minval = -maxval
        input = torch.min(torch.max(input, minval), maxval)
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)).detach(), 1.0)
        return input, 2.0 ** (input_log_scales - M - bias.float())     
    

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
