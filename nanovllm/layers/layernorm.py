"""
RMSNorm (Root Mean Square Layer Normalization)。

与标准 LayerNorm 的区别：
- LayerNorm: x_norm = (x - mean) / sqrt(var + eps) * gamma + beta
- RMSNorm:   x_norm = x / sqrt(mean(x^2) + eps) * gamma
RMSNorm 去掉了均值中心化和 beta 偏置，计算更高效，效果相当。

本实现提供两个版本：
1. rms_forward: 标准 RMSNorm（用于第一层的 input_layernorm，此时无前序 residual）
2. add_rms_forward: Fused Residual + RMSNorm（将 residual add 和 norm 合并在一个 kernel 中）

Fused Residual 优化的意义：
标准实现需要两次读写 hidden_states（一次做 residual add，一次做 norm），
Fused 版本只需一次读写，减少了显存带宽消耗。
这对大模型推理的性能影响显著（推理瓶颈往往在显存带宽而非计算）。

两个方法都用 @torch.compile 装饰，PyTorch 会将它们 JIT 编译为
融合的 CUDA kernel（算子融合），进一步减少 kernel launch 和中间结果的显存分配。
"""
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # gamma 缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """标准 RMSNorm（无 residual add）"""
        orig_dtype = x.dtype
        # 转为 float32 计算以保持数值精度
        x = x.float()
        # 计算 RMS = sqrt(mean(x^2) + eps)，使用 rsqrt 直接得到 1/RMS
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原始 dtype 后乘以可学习的缩放参数 gamma
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused Residual Add + RMSNorm。

        计算 residual_new = x + residual，然后对 residual_new 做 RMSNorm。
        返回 (normed_output, residual_new)，其中 residual_new 传递给下一层继续累加。

        这样设计使得每层的 residual 连接形成链式结构：
        Layer 1: residual = embedding
        Layer 2: residual = embedding + attn_1_output
        Layer 3: residual = embedding + attn_1_output + mlp_1_output + attn_2_output
        ...
        """
        orig_dtype = x.dtype
        # Fused: residual add + 类型转换在一步完成
        x = x.float().add_(residual.float())
        # 新的 residual = x + old_residual（保存在原始 dtype 中）
        residual = x.to(orig_dtype)
        # RMSNorm
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """统一入口：有 residual 时走 fused 路径，无 residual 时走标准路径"""
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
