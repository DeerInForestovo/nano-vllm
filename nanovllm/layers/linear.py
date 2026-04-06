"""
张量并行线性层 -- Megatron-LM 风格的模型并行实现。

核心概念：
将一个大的线性层 Y = XW + b 切分到多个 GPU 上并行计算，
每个 GPU 只持有 W 的一部分，从而减少单卡显存占用。

两种切分方式：

1. Column Parallel（按列切分输出维度）：
   W = [W1 | W2 | ... | Wn]，每个 rank 持有 Wi
   每个 rank 独立计算 Yi = X @ Wi（无需通信）
   输出是局部结果，后续层可直接使用（或通过 AllGather 拼接完整输出）

2. Row Parallel（按行切分输入维度）：
   W = [W1; W2; ...; Wn]（纵向切分），X = [X1 | X2 | ... | Xn]
   每个 rank 计算局部乘积 Yi = Xi @ Wi
   通过 AllReduce 汇总得到完整输出 Y = sum(Yi)

Transformer 中的搭配策略：
- QKV 投影 / Gate+Up 投影：Column Parallel（每个 rank 持有部分 head/neurons）
- Output 投影 / Down 投影：Row Parallel（结果 AllReduce 汇总）
- 这样 Column -> Row 之间不需要额外的 AllGather/AllReduce，只在 Row 输出时做一次 AllReduce

继承体系：
  LinearBase               # 基类：定义权重形状 + weight_loader 接口
    -> ReplicatedLinear     # 不切分（每个 rank 持有完整副本，用于小权重）
    -> ColumnParallelLinear # 按列切分输出维度
      -> MergedColumnParallelLinear  # 合并多个 Column Parallel（如 gate+up）
      -> QKVParallelLinear           # 合并 QKV 三个投影
    -> RowParallelLinear    # 按行切分输入维度（带 AllReduce）
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除辅助函数，确保可以均匀切分"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    所有并行线性层的基类。

    关键设计：每个 Parameter 上绑定了 weight_loader 方法（作为属性），
    供 loader.py 在加载 HuggingFace 权重时调用。不同的子类有不同的
    weight_loader 实现，负责从完整权重中切出当前 rank 需要的分片。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # tp_dim: 张量并行的切分维度（0=按行/输出维度，1=按列/输入维度）
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 注意：这里的 output_size 和 input_size 已经是切分后的大小
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 将 weight_loader 绑定到 Parameter 上，loader.py 中通过
        # getattr(param, "weight_loader") 获取并调用
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制式线性层：每个 rank 持有完整的权重副本。
    用于不需要切分的小型线性层。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接拷贝完整权重，不做切分"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    按列切分的并行线性层（切分输出维度）。

    完整权重 W shape = [output_size, input_size]
    切分后每个 rank 持有 W_i shape = [output_size / tp_size, input_size]

    前向计算：Y_i = X @ W_i^T（无需通信）
    权重加载：从完整权重的第 tp_dim=0 维切出对应分片
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # tp_dim=0: 沿权重矩阵的第 0 维（output_size）切分
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中按 tp_rank 切出对应的行分片"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并式按列切分线性层（如 gate_proj + up_proj 合并为 gate_up_proj）。

    物理上是一个大矩阵，逻辑上是多个独立的 ColumnParallelLinear 拼接。
    权重布局（以 gate+up 为例，tp_size=2, rank=0）：
      [gate_rank0 | up_rank0]  （取 gate 的前半 + up 的前半）

    weight_loader 需要额外的 loaded_shard_id 参数来确定写入哪个子分片。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 总输出大小 = 所有子分片的大小之和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        将 HuggingFace 的单个子权重（如 gate_proj）写入合并矩阵的对应位置。

        Args:
            loaded_shard_id: 子分片索引（gate=0, up=1）
        """
        param_data = param.data
        # 计算该子分片在合并矩阵中的偏移和大小
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 先定位到合并矩阵中该子分片的区域
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 再从完整权重中切出当前 rank 的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 合并式按列切分线性层。

    将 Q/K/V 三个投影合并为一个矩阵（减少 3 次 kernel launch 为 1 次）。
    支持 GQA（Grouped Query Attention）：Q head 数量可以多于 KV head 数量。

    权重布局（以 tp_size=2, rank=0 为例）：
      [Q_rank0 | K_rank0 | V_rank0]
    其中 Q_rank0 = num_heads/2 * head_dim, K/V_rank0 = num_kv_heads/2 * head_dim

    与 MergedColumnParallelLinear 的区别：
    - QKV 的三个子分片大小可能不同（Q 比 KV 大，因为 GQA）
    - shard_id 使用字符串 "q"/"k"/"v" 而非整数索引
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 当前 rank 持有的 Q/KV head 数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 总输出维度 = Q + K + V = (num_q_heads + 2 * num_kv_heads) * head_dim
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        根据 shard_id ("q"/"k"/"v") 将 HuggingFace 的单个投影权重
        写入合并矩阵中的对应位置。

        合并矩阵布局：[Q 区域 | K 区域 | V 区域]
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        # 定位合并矩阵中的目标区域
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中切出当前 rank 的分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    按行切分的并行线性层（切分输入维度）。

    完整权重 W shape = [output_size, input_size]
    切分后每个 rank 持有 W_i shape = [output_size, input_size / tp_size]

    前向计算：
    1. Y_i = X_i @ W_i^T（X_i 是输入在该 rank 上的局部分片）
    2. Y = AllReduce(Y_i)（将所有 rank 的局部结果求和得到最终输出）

    在 Transformer 中，RowParallelLinear 总是接在 ColumnParallelLinear 之后：
    Column 的输出就是按 head/neuron 维度切分的，恰好是 Row 需要的输入格式。
    因此 Column -> Row 之间不需要额外的 AllGather。

    bias 只在 rank 0 加（否则 AllReduce 后 bias 会被加 tp_size 次）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # tp_dim=1: 沿权重矩阵的第 1 维（input_size）切分
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中按 tp_rank 切出对应的列分片"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias 只在 rank 0 加，避免 AllReduce 后重复累加
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # AllReduce：将所有 rank 的局部结果求和
            dist.all_reduce(y)
        return y
