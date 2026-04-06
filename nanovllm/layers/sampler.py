"""
采样器 -- 将 logits 转换为 token ID。

采样策略：Temperature Scaling + Gumbel-Max Trick (等价于 Multinomial Sampling)

流程：
1. logits / temperature（温度越高分布越平坦，随机性越大）
2. softmax 得到概率分布
3. 使用 Gumbel-Max Trick 采样：
   sample = argmax(prob / Exponential(1))
   数学上等价于从 Categorical(prob) 中采样，
   但比 torch.multinomial 更快（可以被 torch.compile 优化）

为什么不用 torch.multinomial？
torch.multinomial 内部有 CPU 同步操作，无法被 torch.compile 融合。
Gumbel-Max Trick 纯用 GPU 张量运算实现，完全在 GPU 上执行。
"""
import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Temperature scaling：logits / temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-Max Trick：
        # 从 Exponential(1) 分布采样，然后 probs / exp_samples，取 argmax
        # 等价于从 Categorical(probs) 中做一次 multinomial 采样
        # clamp_min 防止除零
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
