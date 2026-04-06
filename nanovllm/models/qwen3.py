"""
Qwen3 模型实现 -- 标准 Decoder-Only Transformer 架构。

结构层次：
  Qwen3ForCausalLM          # 最外层：Transformer + LM Head
    -> Qwen3Model            # Transformer 主体：Embedding + N x DecoderLayer + Final Norm
      -> Qwen3DecoderLayer   # 单个 Decoder 层：LayerNorm + Attention + LayerNorm + MLP
        -> Qwen3Attention    # 多头注意力：QKV投影 + RoPE + FlashAttention + Output投影
        -> Qwen3MLP          # 前馈网络：Gate+Up投影 + SiLU激活 + Down投影

与 HuggingFace 实现的关键区别：
1. QKV 权重合并为 qkv_proj（减少 3 次 kernel launch 为 1 次）
2. Gate 和 Up 权重合并为 gate_up_proj（同理）
3. 所有线性层支持张量并行（Column/Row Parallel）
4. 使用 FlashAttention + PagedAttention 代替标准 Attention
5. RMSNorm 实现了 fused residual add（减少一次显存读写）
"""
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 多头注意力模块（支持 GQA / MQA）。

    计算流程：
    hidden_states -> qkv_proj -> [Q, K, V] -> (可选)QK_Norm -> RoPE -> Attention -> o_proj -> output

    张量并行策略：
    - qkv_proj: Column Parallel（每个 rank 持有部分 head 的 QKV）
    - o_proj: Row Parallel（每个 rank 计算部分结果，AllReduce 汇总）
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        # 总 head 数必须能被 TP size 整除
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        # 当前 rank 持有的 Q head 数 = 总 head 数 / TP size
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        # GQA: KV head 数可以少于 Q head 数（多个 Q head 共享一组 KV）
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Attention score 的缩放因子：1/sqrt(head_dim)，防止 softmax 梯度消失
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # QKV 合并投影：一次矩阵乘法同时计算 Q、K、V，减少 kernel launch
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # Output 投影：Row Parallel（每个 rank 计算部分，AllReduce 合并）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # RoPE 旋转位置编码（通过 lru_cache 全局共享同一个实例）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=tuple(rope_scaling.items()) if rope_scaling else None,
        )
        # FlashAttention + PagedAttention 封装
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3 特性：当没有 QKV bias 时，使用 QK Norm（归一化 Q 和 K）
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 一次投影得到 QKV 拼接结果，然后按维度切分
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # reshape 为 [num_tokens, num_heads, head_dim] 供 Attention 使用
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 应用 RoPE 旋转位置编码（让模型感知 token 之间的相对位置）
        q, k = self.rotary_emb(positions, q, k)
        # FlashAttention 计算 + KV Cache 读写
        o = self.attn(q, k, v)
        # Output 投影（包含张量并行的 AllReduce 通信）
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    SwiGLU 前馈网络。

    计算流程：x -> gate_up_proj -> [gate, up] -> SiLU(gate) * up -> down_proj -> output

    gate_up_proj 是 Column Parallel（每个 rank 持有部分 intermediate_size 的列）
    down_proj 是 Row Parallel（每个 rank 持有部分行，结果 AllReduce 合并）
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # Gate 和 Up 合并为一个线性层（减少 kernel launch）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # SiluAndMul: 将 gate_up 输出按一半切分，计算 SiLU(gate) * up
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    单个 Transformer Decoder 层：Pre-Norm 结构。

    计算流程（注意 fused residual 的优化）：
    1. input_layernorm(hidden_states, residual) -> (normed, new_residual)
    2. self_attn(normed) -> hidden_states
    3. post_attention_layernorm(hidden_states, residual) -> (normed, new_residual)
    4. mlp(normed) -> hidden_states

    Fused Residual 的含义：
    标准实现：normed = LayerNorm(x + residual); new_residual = x + residual
    Fused 实现：在一次 kernel 中同时完成 residual add 和 LayerNorm，减少显存读写。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 第一层时 residual 为 None，直接用 hidden_states 作为 residual
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # Fused: LayerNorm(hidden_states + residual)，同时返回 new_residual
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 Transformer 主体：Embedding + N x DecoderLayer + Final LayerNorm。

    输入 token IDs -> Embedding -> 逐层 DecoderLayer -> Final Norm -> hidden_states
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入层（张量并行时按词表维度切分）
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层的 fused residual add + LayerNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    因果语言模型：Transformer + LM Head。

    packed_modules_mapping 定义了 HuggingFace 权重名到合并后权重名的映射，
    供 loader.py 在加载权重时使用。

    forward() 只返回 hidden_states（不含 LM Head），
    compute_logits() 单独计算 logits。
    这样设计是为了 CUDA Graph 只需要捕获 forward()，
    而 compute_logits() 在 Graph 外执行（更灵活，且计算量不大）。
    """
    # HuggingFace 权重名 -> (合并后名称, shard_id) 的映射
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # Q 权重 -> qkv_proj 的 "q" 分片
        "k_proj": ("qkv_proj", "k"),      # K 权重 -> qkv_proj 的 "k" 分片
        "v_proj": ("qkv_proj", "v"),      # V 权重 -> qkv_proj 的 "v" 分片
        "gate_proj": ("gate_up_proj", 0), # Gate 权重 -> gate_up_proj 的第 0 分片
        "up_proj": ("gate_up_proj", 1),   # Up 权重 -> gate_up_proj 的第 1 分片
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # LM Head: 将 hidden_states 映射回词表空间得到 logits
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 权重绑定：如果模型配置要求，LM Head 和 Embedding 共享权重（节省显存）
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：只到 hidden_states，不包含 LM Head（为 CUDA Graph 设计）"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """单独计算 logits = LM_Head(hidden_states)，在 CUDA Graph 外调用"""
        return self.lm_head(hidden_states)
