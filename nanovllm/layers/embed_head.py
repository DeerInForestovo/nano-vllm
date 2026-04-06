"""
并行词嵌入层(Embedding)和语言模型头(LM Head) -- 词表维度的张量并行。

核心挑战：
词表通常很大（如 151936），Embedding 和 LM Head 的权重矩阵 [vocab_size, hidden_size]
占用大量显存。张量并行时需要将词表维度切分到多个 GPU 上。

VocabParallelEmbedding（词嵌入）：
- 每个 rank 只持有 vocab_size / tp_size 个词的 embedding
- 对于不属于当前 rank 词表范围的 token ID，返回零向量
- 通过 AllReduce 将所有 rank 的结果合并（因为每个 rank 只有部分词有非零输出）

ParallelLMHead（语言模型头）：
- 继承 VocabParallelEmbedding 的权重切分方式
- 输出 logits 需要完整词表维度，使用 Gather 收集到 rank 0
- Prefill 优化：只取每个序列最后一个 token 的 hidden_states 计算 logits
  （因为 prefill 阶段只有最后一个 token 需要预测下一个 token）
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行嵌入层：每个 rank 只持有词表的一段。

    词表划分示例（vocab_size=100, tp_size=2）：
    - rank 0: 持有 token ID [0, 49] 的 embedding
    - rank 1: 持有 token ID [50, 99] 的 embedding

    前向计算：
    1. 将不属于当前 rank 范围的 token ID 置零（mask）
    2. 用调整后的局部 ID 查询本地 embedding
    3. 将范围外的 token 的 embedding 置零
    4. AllReduce 合并所有 rank 的结果
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 每个 rank 负责的词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前 rank 负责的词表范围 [start, end)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 只分配本 rank 需要的 embedding 参数
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整词表权重中切出当前 rank 负责的部分"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 生成 mask：标记哪些 token ID 属于当前 rank 的词表范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将 token ID 转换为本地索引（减去起始偏移），范围外的置零
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 将范围外 token 的 embedding 置零，然后 AllReduce 合并
            # 这样每个 token 最终只有一个 rank 贡献了非零 embedding
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型头：将 hidden_states 映射到词表空间得到 logits。

    继承 VocabParallelEmbedding 的词表切分方式和权重加载逻辑，
    但前向计算不同：
    1. Prefill 优化：只取每个序列最后一个 token 的 hidden_states
       （因为只有最后一个位置需要预测下一个 token）
    2. 使用 linear（而非 embedding lookup）计算 logits = hidden_states @ weight^T
    3. 用 Gather（而非 AllReduce）将各 rank 的局部 logits 收集到 rank 0
       （因为只有 rank 0 负责采样，其他 rank 不需要完整 logits）
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # Prefill 优化：从 varlen 拼接的 hidden_states 中
            # 取每个序列的最后一个 token（cu_seqlens_q[1:] - 1 即各序列末尾的索引）
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # logits = hidden_states @ weight^T，每个 rank 只算局部词表的 logits
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 使用 Gather 将各 rank 的局部 logits 收集到 rank 0
            # 比 AllGather 更高效（其他 rank 不需要完整 logits）
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # rank 0 拼接完整的 logits [bs, vocab_size]
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
