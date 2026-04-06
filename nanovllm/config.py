"""
全局配置中心。

整合用户传入的参数与 HuggingFace 模型自带的配置（如 hidden_size, num_layers 等），
供引擎层各组件（Scheduler, ModelRunner, BlockManager）共享使用。
"""
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    # 模型权重的本地路径（必须是已下载好的目录）
    model: str
    # 单次调度中允许处理的最大 token 总数（所有序列加在一起）
    # 控制 prefill 阶段的显存峰值和计算量
    max_num_batched_tokens: int = 16384
    # 同时处于 running 状态的最大序列数（即 batch size 上限）
    max_num_seqs: int = 512
    # 单个序列的最大长度（prompt + 生成），超过模型支持的 max_position_embeddings 会被截断
    max_model_len: int = 4096
    # GPU 显存利用率上限（0~1），用于计算可分配的 KV Cache 块数
    # 预留一部分显存给 PyTorch 的临时分配，避免 OOM
    gpu_memory_utilization: float = 0.9
    # 张量并行数（TP），即将模型切分到多少张 GPU 上
    tensor_parallel_size: int = 1
    # 若为 True，禁用 CUDA Graph 优化，所有推理都走 eager 模式（方便调试）
    enforce_eager: bool = False
    # 以下字段在 __post_init__ 或运行时动态填充
    hf_config: AutoConfig | None = None   # HuggingFace 模型配置（含 hidden_size 等）
    eos: int = -1                          # EOS token ID，在 LLMEngine 中从 tokenizer 获取
    kvcache_block_size: int = 256          # KV Cache 的物理块大小（每块存储多少个 token 的 KV）
    num_kvcache_blocks: int = -1           # KV Cache 的物理块总数，在 ModelRunner.allocate_kv_cache 中计算

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # block_size 必须是 256 的倍数，因为 FlashAttention 的 block_table 实现要求对齐
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 从模型目录加载 HuggingFace 配置（config.json），获取模型架构参数
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 实际 max_model_len 不能超过模型支持的最大位置编码长度
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # 单次 batch 的 token 数必须 >= 单个序列的最大长度（否则连一个最长序列都放不下）
        assert self.max_num_batched_tokens >= self.max_model_len
