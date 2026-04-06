"""
Attention 层 -- FlashAttention + Triton KV Cache 写入。

本文件是 PagedAttention 在算子层面的实现核心，包含两部分：

1. Triton Kernel (store_kvcache_kernel):
   将当前 step 新计算的 K/V 写入 Paged KV Cache 的对应槽位。
   每个 token 由 slot_mapping 指定写入位置（物理块内的全局偏移）。

2. Attention 模块:
   根据 prefill/decode 阶段选择不同的 FlashAttention API：
   - Prefill: flash_attn_varlen_func (变长序列批处理，支持 prefix cache 的 block_table)
   - Decode:  flash_attn_with_kvcache (从 paged KV Cache 读取历史 KV)
"""
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,           # Key 张量的起始指针
    key_stride,        # Key 张量在 token 维度（dim 0）的 stride
    value_ptr,         # Value 张量的起始指针
    value_stride,      # Value 张量在 token 维度的 stride
    k_cache_ptr,       # KV Cache 中 K 部分的起始指针
    v_cache_ptr,       # KV Cache 中 V 部分的起始指针
    slot_mapping_ptr,  # slot_mapping 数组：每个 token 在 cache 中的全局写入位置
    D: tl.constexpr,   # 每个 token 的 KV 数据总维度 = num_kv_heads * head_dim
):
    """
    Triton kernel：将新计算的 K/V 写入 PagedAttention 的 KV Cache。

    并行策略：每个 Triton program 处理一个 token（program_id = token index）。
    每个 token 的完整 KV 数据（num_kv_heads * head_dim 个元素）在一个 program 内处理。

    slot = slot_mapping[idx] 是该 token 在 KV Cache 中的全局槽位号，
    其计算方式为 block_table[block_idx] * block_size + offset_in_block，
    由 ModelRunner.prepare_prefill/decode 预先算好。
    """
    # 当前处理第 idx 个 token
    idx = tl.program_id(0)
    # 从 slot_mapping 中读取该 token 应写入的缓存位置
    slot = tl.load(slot_mapping_ptr + idx)
    # slot == -1 表示 CUDA Graph padding 的无效 token，跳过
    if slot == -1: return
    # 从 Key/Value 张量中读取当前 token 的完整数据
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 写入 KV Cache 的对应槽位
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    Triton kernel 的 Python wrapper：将 K/V 写入 Paged KV Cache。

    Args:
        key: 当前 step 新计算的 Key, shape = [N, num_kv_heads, head_dim]
        value: 当前 step 新计算的 Value, 同上
        k_cache: 整个 KV Cache 的 K 部分, shape = [num_blocks * block_size, num_kv_heads * head_dim]
                 (经过 view 后是扁平化的)
        v_cache: 整个 KV Cache 的 V 部分, 同上
        slot_mapping: 每个 token 在 cache 中的全局写入位置, shape = [N]
    """
    N, num_heads, head_dim = key.shape
    # D = 每个 token 的 KV 数据总维度，kernel 中一次读写 D 个元素
    D = num_heads * head_dim
    # 确保内存布局连续（Triton kernel 依赖 stride 计算地址偏移）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 N 个 Triton program，每个处理一个 token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    Attention 模块：统一封装 prefill 和 decode 阶段的注意力计算。

    不直接持有 QKV 投影权重（那些在 Qwen3Attention 中），
    只负责：
    1. 将新的 K/V 写入 KV Cache（通过 Triton kernel）
    2. 调用 FlashAttention 计算注意力输出

    k_cache / v_cache 在 ModelRunner.allocate_kv_cache() 中被注入（引用赋值），
    指向全局 KV Cache 大张量中对应层的切片。
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始化为空张量，在 ModelRunner.allocate_kv_cache 中被替换为实际的 cache 引用
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q: Query, shape = [num_tokens, num_heads, head_dim]
            k: Key,   shape = [num_tokens, num_kv_heads, head_dim]
            v: Value, shape = [num_tokens, num_kv_heads, head_dim]
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 第一步：将新的 K/V 写入 KV Cache（warmup 阶段 cache 为空，跳过）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                # 有 prefix cache 命中：Q 是部分 token，但需要 attend 到完整上下文
                # 将 K/V 替换为整个 cache（FlashAttention 通过 block_table 索引物理块）
                k, v = k_cache, v_cache
            # Prefill：使用 varlen API 处理变长输入（多个序列拼接在一起）
            # cu_seqlens_q/k 告诉 FlashAttention 每个序列的边界在哪里
            # causal=True 确保 token 只能 attend 到自身及之前的 token
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # Decode：每个序列只有 1 个新 token（Q），需要 attend 到所有历史 KV
            # q.unsqueeze(1): [bs, head_dim] -> [bs, 1, head_dim]（1 = seqlen_q）
            # FlashAttention 通过 block_table + cache_seqlens 从 paged KV Cache 中读取历史 KV
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
