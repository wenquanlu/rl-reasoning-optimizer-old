import torch

STR_TO_TORCH_DTYPE = {
    "fp16": torch.half,
    "bf16": torch.bfloat16,
    "fp32": torch.float
}

def fake_topk_quantize(x: torch.Tensor, group_size, bit, topk, stochastic=False, output: torch.Tensor = None) -> torch.Tensor:
    x = x.clone() # clone 避免修改原始数据
    x = x.view(-1, group_size) # 重塑成2D, 每行大小group_size
    xabs = x.abs()      # 存储x中元素绝对值
    xabs_topk_index = xabs.topk(topk, dim=-1).indices #找到topk元素的索引
    topk_values = torch.gather(x, 1 , xabs_topk_index)#把topk元素提取出来
    x[torch.arange(0, x.size(0), device=x.device)[:, None].expand(-1, topk), xabs_topk_index] = 0 # 把topk元素置为0
    scale = x.abs().max(dim=-1).values.clamp_min(1e-6) #找到每行最大元素
    if x.dtype in [torch.half, torch.bfloat16]:
        xabs_topk_index = xabs_topk_index.to(dtype=torch.int16)
    else:
        xabs_topk_index = xabs_topk_index.to(dtype=torch.int32)
        
    x *= (2 ** (bit - 1) - 1) / scale[:, None] #进行量化操作
    if stochastic:
        x += torch.rand_like(x) - 0.5
    x = x.round().clamp(-2 ** (bit - 1), 2 ** (bit - 1) - 1).to(dtype=torch.int8) #对x四舍五入然后限制在制定范围 转化为int8
    x = x.view(-1).view(torch.uint8)
    xabs_topk_index = xabs_topk_index.view(-1).view(torch.uint8)
    topk_values = topk_values.view(-1).view(torch.uint8)
    scale = scale.view(-1).view(torch.uint8)
    ret = torch.cat([x, scale, xabs_topk_index, topk_values], dim=0)
    if output is not None:
        output.copy_(ret)
    else:
        output = ret
    return output

def fake_topk_dequantize(x: torch.Tensor, group_size, bit, topk, dtype, output: torch.Tensor = None, reduce_op = None) -> torch.Tensor:
    if reduce_op is None:
        reduce_op = "none"


    if dtype in ["fp16", "bf16"]:
        bytes_per_group = group_size + 2 + 4 * topk #分配字节数
    else:
        bytes_per_group = group_size + 4 + 8 * topk

    num_group = x.numel() // bytes_per_group # 总元素数 / 每个分组占用字节数 = 分组数
    x_ed = num_group * group_size           #去量化操作需要的原始数据长度

    if dtype in ["fp16", "bf16"]:
        scale_ed = x_ed + 2 * num_group
        topk_index_ed = scale_ed + 2 * topk * num_group
        index_torch_dtype = torch.int16
    else:
        scale_ed = x_ed + 4 * num_group
        topk_index_ed = scale_ed + 4 * topk * num_group
        index_torch_dtype = torch.int32

    torch_dtype = STR_TO_TORCH_DTYPE[dtype]
    _x = x[:x_ed].clone().view(dtype=torch.int8).view(-1, group_size) #提取出前x—ed个元素
    _scale = x[x_ed:scale_ed].clone().view(dtype=torch_dtype).view(-1) #提取x-ed到scale-ed的元素
    if topk > 0:
        _topk_index = x[scale_ed:topk_index_ed].clone().view(dtype=index_torch_dtype).to(torch.int64).view(-1, topk)  # 提取出scale-ed到topk-index-ed部分作为topk索引 展开成(num_group,topk)
        _top_values = x[topk_index_ed:].clone().view(dtype=torch_dtype).view(-1, topk)

    _x = _x.to(torch_dtype) * _scale[:, None] / (2 ** (bit - 1) - 1)
    if topk > 0:
        _x[torch.arange(0, _x.size(0), device=_x.device)[:, None].expand(-1, topk), _topk_index] = _top_values
    _x = _x.view(-1)

    if output is not None:
        if reduce_op == "sum":
            output.view(-1).add_(_x)
        elif reduce_op == "min":
            torch.minimum(output.view(-1), _x, out=output.view(-1))
        elif reduce_op == "max":
            torch.maximum(output.view(-1), _x, out=output.view(-1))
        elif reduce_op == "none":
            output.copy_(_x)
        else:
            raise ValueError(f"Unsupport reduce op {reduce_op}")
    
    else:
        output = _x

    return output

def fake_tensor_quantize(x: torch.Tensor, dim: int, group_size: int, bit: int, stochastic=False, topk: int = 0, output_tensor: torch.Tensor = None, quant_type: str = "linear") -> torch.Tensor:
    assert quant_type.lower() == "linear"
    if dim < 0:
        dim += x.dim()
    
    if dim != x.dim() - 1:
        y = x.transpose(-1, dim).contiguous().view(-1)
    else:
        y = x.contiguous().view(-1)

    return fake_topk_quantize(y, group_size, bit, topk, stochastic, output_tensor)

def fake_tensor_dequantize(q: torch.Tensor, dim: int, shape: torch.Size, group_size: int, bit: int, dtype: str, topk: int = 0, output_tensor: torch.Tensor = None, reduce_op = None, quant_type: str = "linear") -> torch.Tensor:
    assert quant_type.lower() == "linear"
    y = fake_topk_dequantize(q, group_size, bit, topk, dtype, output_tensor, reduce_op)

    if output_tensor is None:
        _shape = list(shape)
        if dim < 0:
            dim += len(_shape)

        if dim != len(_shape) - 1:
            _shape[dim], _shape[-1] = _shape[-1], _shape[dim]

        y = y.view(_shape)

        if dim != len(_shape) - 1:
            y = y.transpose(-1, dim).contiguous()

        output_tensor = y

    else:
        output_tensor.copy_(y)

    return output_tensor