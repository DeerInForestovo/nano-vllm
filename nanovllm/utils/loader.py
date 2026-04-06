"""
模型权重加载器 -- 将 HuggingFace safetensors 权重映射到自定义模型结构。

核心挑战：
HuggingFace 模型中 Q/K/V 是三个独立的权重矩阵(q_proj, k_proj, v_proj)，
但在推理优化中，我们将它们合并为一个 qkv_proj（减少 kernel launch）。
同理 gate_proj 和 up_proj 合并为 gate_up_proj。

加载流程：
1. 遍历 safetensors 文件中的每个权重
2. 检查权重名是否匹配 packed_modules_mapping 中的合并规则
3. 如果匹配，调用该参数的 weight_loader 方法，按 shard_id 写入合并后矩阵的对应部分
4. 如果不匹配(如 layernorm, embedding 等)，直接复制
"""
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认加载方式：直接将权重数据拷贝到参数中"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从 safetensors 文件加载权重到模型。

    packed_modules_mapping 定义了 HF 权重名 -> 合并后权重名的映射规则，例如：
    {
        "q_proj": ("qkv_proj", "q"),   # HF 的 q_proj -> 自定义的 qkv_proj, 分片 ID 为 "q"
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),   # gate_proj -> gate_up_proj 的第 0 个分片
        "up_proj": ("gate_up_proj", 1),
    }

    对于需要合并的权重，会调用参数上的 weight_loader 方法（由 QKVParallelLinear 等
    自定义层注册），该方法知道如何将单个 shard 写入合并矩阵的正确位置。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 匹配到合并规则：将 HF 权重名替换为合并后的名称
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # 使用参数自带的 weight_loader，按 shard_id 写入正确位置
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 未匹配合并规则：直接加载（layernorm, embedding 等）
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
