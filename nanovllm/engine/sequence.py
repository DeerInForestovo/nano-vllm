"""
序列（Sequence）—— 推理引擎中一个请求的核心抽象。

每个用户请求（一条 prompt）对应一个 Sequence 对象，它跟踪：
- token 列表（prompt tokens + 已生成的 completion tokens）
- 生命周期状态（WAITING → RUNNING → FINISHED）
- KV Cache 块表（block_table）—— 记录该序列的 KV 存储在哪些物理块中
- 采样参数（temperature, max_tokens 等）

设计上类似 OS 中的 PCB（进程控制块），是调度器操作的基本单元。
"""
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列生命周期的三个状态（类比进程状态：就绪、运行、终止）"""
    WAITING = auto()    # 在等待队列中，尚未开始或被抢占后重新排队
    RUNNING = auto()    # 正在参与推理计算（prefill 或 decode）
    FINISHED = auto()   # 生成完成（遇到 EOS 或达到 max_tokens）


class Sequence:
    # KV Cache 块大小（每个物理块存多少个 token 的 KV），与 Config.kvcache_block_size 一致
    block_size = 256
    # 全局自增 ID 计数器，确保每个 Sequence 有唯一标识
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # 深拷贝 token 列表，避免外部修改影响内部状态
        self.token_ids = copy(token_ids)
        # 缓存最后一个 token，decode 阶段只需要它作为输入
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        # prompt 长度（固定不变），用于区分 prompt tokens 和 completion tokens
        self.num_prompt_tokens = len(token_ids)
        # 已被 prefix cache 命中的 token 数，命中部分无需重新计算
        self.num_cached_tokens = 0
        # 块表：逻辑块 → 物理块的映射，类似 OS 页表
        # block_table[i] = 物理块 ID，表示该序列第 i 个逻辑块存储在哪个物理块中
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回当前序列总长度（prompt + 已生成 tokens）"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持切片访问 token_ids，如 seq[5:10]"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数量"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """被 prefix cache 命中的完整块数"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """该序列当前所需的总块数（向上取整）"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中实际存储的 token 数（可能不满一整块）"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第 i 个逻辑块对应的 token 列表（用于计算 prefix cache 的 hash）"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """decode 阶段每步生成一个新 token，追加到序列中"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        自定义序列化（pickle），用于张量并行时通过共享内存传输 Sequence。
        优化：decode 阶段不需要传输完整 token_ids（只需 last_token），
        因为 worker 只需要最新的 token 来做下一步推理。
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        反序列化：在非 rank-0 的 worker 上恢复 Sequence。
        prefill 阶段需要完整 token_ids，decode 阶段只需 last_token。
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
