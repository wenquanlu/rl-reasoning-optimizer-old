import torch
import math
import torch
import math

def fp4_fake_quantize(
    x: torch.Tensor,
    e: int = 2,
    m: int = 2,
    b: int = 1
) -> torch.Tensor:
    alpha = 1.00
    q_max = alpha * (2 - 2**(-m)) * 2**(2**e - b - 1)
    q_min = -q_max
    x_clamped = torch.clamp(x, q_min, q_max)
    alpha_hat = alpha * (2**(-b))
    b_hat = 2**e - math.log2(q_max) + math.log2(2 - 2**(-m)) - 1
    log_v = torch.floor(torch.log2(torch.abs(x_clamped) + 1e-8) + b_hat)
    v = torch.pow(2, torch.clamp(log_v - m, min=1-m))
    x_fp = alpha_hat * v * torch.floor(x_clamped / (alpha_hat * v))
    return x_fp

def fp4_tensor_quantize(x:torch.Tensor):
    y = x.contiguous().view(-1)
    return fp4_fake_quantize(y)

