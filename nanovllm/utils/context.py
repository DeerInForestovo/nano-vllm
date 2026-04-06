"""
全局上下文(Context) -- ModelRunner 与 Attention 层之间的通信桥梁。

问题背景：
ModelRunner 在 prepare_prefill/decode 中构造了 FlashAttention 需要的各种张量
(cu_seqlens, slot_mapping, block_tables 等)，但这些张量需要传递到深层的
Attention 模块中使用。如果通过 forward() 参数逐层传递，会污染所有中间层的接口。

解决方案：
使用全局变量 _CONTEXT 存储当前 step 的上下文信息。
ModelRunner.prepare_*() 设置 Context，Attention.forward() 读取 Context。
每个 step 结束后 reset。

这是 vLLM 中类似设计的简化版本(vLLM 使用线程局部存储)。
"""
from dataclasses import dataclass
import torch


@dataclass
class Context:
    # 当前是 prefill 还是 decode（决定 Attention 使用哪个 FlashAttention API）
    is_prefill: bool = False
    # === Prefill 专用字段 ===
    # FlashAttention varlen 接口需要的累积序列长度（前缀和）
    cu_seqlens_q: torch.Tensor | None = None   # Query 侧（去掉 cached tokens 后的长度）
    cu_seqlens_k: torch.Tensor | None = None   # Key 侧（完整上下文长度）
    max_seqlen_q: int = 0                       # batch 中最长的 Q 序列长度
    max_seqlen_k: int = 0                       # batch 中最长的 K 序列长度
    # === Prefill + Decode 共用字段 ===
    # 每个需要写入 KV Cache 的 token 在物理块中的全局槽位索引
    slot_mapping: torch.Tensor | None = None
    # === Decode 专用字段 ===
    # 每个序列的完整上下文长度（Attention 计算范围）
    context_lens: torch.Tensor | None = None
    # 块表：逻辑块 -> 物理块的映射（PagedAttention 的核心数据结构）
    block_tables: torch.Tensor | None = None

# 全局唯一的 Context 实例
_CONTEXT = Context()

def get_context():
    """供 Attention 层读取当前 step 的上下文"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """由 ModelRunner.prepare_prefill/decode 调用，设置当前 step 的上下文"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """每个 step 结束后重置，避免残留状态影响下一步"""
    global _CONTEXT
    _CONTEXT = Context()
