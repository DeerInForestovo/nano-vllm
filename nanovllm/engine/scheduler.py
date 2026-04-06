"""
调度器 —— Continuous Batching 的核心实现。

调度策略：
- 优先调度 waiting 队列中的新请求做 Prefill（首次填充 KV Cache）
- 若没有新请求需要 prefill，则调度 running 队列中的请求做 Decode（逐 token 生成）
- 当显存不足（KV 块不够）时，通过 preempt（抢占）将低优先级序列踢回 waiting

Continuous Batching 的关键优势：
传统 Static Batching 必须等整个 batch 全部生成完成才能接受新请求，
而 Continuous Batching 在每个 step 之间都能动态插入/移除序列，
大幅提高 GPU 利用率。
"""
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 两个队列实现 Continuous Batching：
        # waiting: 等待首次 prefill 的序列（新请求或被抢占的序列）
        # running: 已经完成 prefill、正在逐步 decode 的序列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """两个队列都为空时，所有请求都已处理完毕"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新请求加入 waiting 队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        每个 step 的调度决策，返回 (本次参与计算的序列列表, 是否是 prefill)。

        调度逻辑：
        1. Prefill 优先：如果 waiting 队列有请求，优先为它们做 prefill
           - 受 max_num_seqs（最大并发序列数）和 max_num_batched_tokens（最大 token 数）约束
           - 还需要 BlockManager 有足够的空闲块来存储 KV Cache
        2. Decode：如果没有可 prefill 的请求，调度 running 中的序列做 decode
           - 每个序列只生成 1 个 token
           - 若块不足，则抢占（preempt）最后加入的序列来腾出空间
        """
        # === Prefill 阶段 ===
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查两个约束：token 总数是否超限 + 是否有足够的 KV 块
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 为序列分配 KV Cache 块（同时尝试 prefix cache 匹配）
            self.block_manager.allocate(seq)
            # 实际需要计算的 token 数 = 总长 - 已缓存的 token 数（prefix cache 命中的部分）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True   # True 表示本次是 prefill

        # === Decode 阶段 ===
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查是否有足够块来追加 1 个新 token
            while not self.block_manager.can_append(seq):
                # 块不足，需要抢占其他序列来释放空间
                if self.running:
                    # 抢占最后加入的序列（LIFO 策略，类似栈）
                    self.preempt(self.running.pop())
                else:
                    # 无其他序列可抢占，只能抢占自己（极端情况）
                    self.preempt(seq)
                    break
            else:
                # while...else: 当 can_append 成功时执行
                num_seqs += 1
                # 为 decode 的新 token 管理块（可能需要分配新块或更新 hash）
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 将已调度的序列放回 running 队列头部（保持顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False   # False 表示本次是 decode

    def preempt(self, seq: Sequence):
        """
        抢占：将序列的状态回退到 WAITING，释放其 KV Cache 块，放回 waiting 队列头部。
        被抢占的序列下次被调度时需要重新 prefill（类似 OS 的页面换出）。
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        每个 step 完成后的后处理：
        1. 将新生成的 token 追加到各序列
        2. 检查终止条件（EOS 或达到 max_tokens），标记为 FINISHED 并释放资源
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
