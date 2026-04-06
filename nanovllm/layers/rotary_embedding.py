"""
RoPE (Rotary Position Embedding) -- 旋转位置编码。

核心思想：
通过将 Q/K 向量在复数平面上旋转一个与位置相关的角度，
使得两个 token 的注意力分数自然包含它们之间的相对距离信息。

数学原理：
对于位置 t 的向量 x = [x1, x2, x3, x4, ...]，
将相邻两个维度视为一个复数 (x1 + ix2)，乘以 e^(it*theta)：
  y1 = x1 * cos(t*theta) - x2 * sin(t*theta)
  y2 = x2 * cos(t*theta) + x1 * sin(t*theta)

其中 theta = 1 / (base^(2k/d)) 是与维度 k 相关的频率（低维变化快，高维变化慢）。

实现优化：
- 预计算所有位置的 cos/sin 并缓存到 buffer 中，推理时直接查表
- 使用 @torch.compile 加速（JIT 编译为高效 kernel）
- 通过 @lru_cache 确保全局只创建一个 RoPE 实例（所有层共享）
"""
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对输入向量 x 应用旋转位置编码。

    将 x 沿最后一维对半切分为 (x1, x2)，然后做复数旋转：
      y1 = x1 * cos - x2 * sin
      y2 = x2 * cos + x1 * sin

    使用 float32 计算以保持数值精度，最后转回原始 dtype。
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    预计算式 RoPE 实现。

    初始化时预算 [0, max_position) 所有位置的 cos/sin 值并存为 buffer，
    前向传播时直接用 positions 作为索引查表，避免重复计算。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 频率向量 inv_freq: shape = [rotary_dim / 2]
        # theta_k = 1 / (base^(2k/d))，k=0,1,...,d/2-1
        # base 越大，高维频率越低，能编码的最大距离越远
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 位置序列 t: [0, 1, 2, ..., max_position-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积得到角度矩阵 freqs[i,j] = i * theta_j, shape = [max_pos, rotary_dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 拼接 cos 和 sin 并增加一个维度用于广播
        # cache shape = [max_pos, 1, rotary_dim]（中间维度对应 num_heads，广播到所有 head）
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据 positions 查表获取 cos/sin，然后对 Q 和 K 应用旋转编码。

        Args:
            positions: 每个 token 的绝对位置, shape = [num_tokens]
            query: Q 向量, shape = [num_tokens, num_heads, head_dim]
            key:   K 向量, shape = [num_tokens, num_kv_heads, head_dim]
        """
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: tuple | None = None,
):
    """
    获取全局唯一的 RoPE 实例（通过 lru_cache 缓存）。

    所有 Attention 层共享同一个 RoPE（因为位置编码参数完全相同），
    节省显存且避免重复初始化。
    """
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
    assert rope_scaling is None or rope_scaling.get("rope_type") == "default"
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
