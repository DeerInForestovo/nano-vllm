"""
SwiGLU 激活函数 -- 现代 LLM 中 MLP 层的标准激活方式。

SwiGLU 的计算：
  输入 x shape = [batch, 2 * intermediate_size]（由 gate_up_proj 输出）
  切分为 gate 和 up 两半：
    gate, up = x.chunk(2)
  输出 = SiLU(gate) * up

其中 SiLU(x) = x * sigmoid(x)，也叫 Swish 激活函数。
gate 分支提供了一个 "门控" 机制，控制 up 分支信息的流通量。

使用 @torch.compile 装饰，PyTorch 会将 chunk + silu + mul
融合为一个 CUDA kernel，避免中间结果的显存分配。
"""
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入按最后一维对半切分为 gate 和 up
        x, y = x.chunk(2, -1)
        # SiLU(gate) * up，即 SwiGLU 激活
        return F.silu(x) * y
