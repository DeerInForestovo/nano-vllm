"""
KV Cache 块管理器 —— PagedAttention 的核心组件。

核心思想（类比操作系统虚拟内存）：
- 物理块（Block）：GPU 显存中固定大小的 KV Cache 存储单元，类比物理页帧
- 逻辑块：序列中按 block_size 分段的 token 组，类比虚拟页
- 块表（block_table）：逻辑块 → 物理块的映射，类比页表
- BlockManager：管理所有物理块的分配/释放，类比操作系统的物理内存管理器

额外实现了 Prefix Caching：
- 对每个满块计算 hash（基于 token 内容 + 前缀 hash 的链式哈希）
- 新请求分配时，若某块的 hash 命中已有块，则直接复用其 KV Cache，跳过计算
"""
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    物理块：GPU 显存中一个固定大小的 KV Cache 存储单元。

    每个 Block 存储 block_size 个 token 的 Key 和 Value 张量。
    实际的 KV 数据存储在 ModelRunner.kv_cache 大张量中，
    Block 对象只保存元数据（引用计数、hash、对应的 token IDs）。
    """

    def __init__(self, block_id):
        self.block_id = block_id
        # 引用计数：有多少个序列正在使用这个物理块
        # 当 ref_count 降为 0 时才能被回收（支持 prefix cache 下的共享）
        self.ref_count = 0
        # 该块内容的 hash 值，用于 prefix caching 匹配（-1 表示未填满/无有效 hash）
        self.hash = -1
        # 该块对应的 token ID 列表，用于 prefix cache 验证（防止 hash 冲突）
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """当块被填满时，记录其 hash 和 token 内容（用于后续 prefix cache 匹配）"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """块被重新分配时，重置元数据（ref_count 设为 1 表示有一个使用者）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    物理块管理器，负责 KV Cache 块的分配、释放和 Prefix Caching。

    维护三个核心数据结构：
    - free_block_ids: 空闲物理块队列
    - used_block_ids: 已使用物理块集合
    - hash_to_block_id: hash → 物理块 ID 的映射（prefix cache 索引）
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 所有物理块对象（下标即 block_id）
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # Prefix Cache 的哈希索引：hash → block_id
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块队列（FIFO），初始时所有块都空闲
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用块集合（用于快速判断某个 block_id 是否在使用中）
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算块的链式哈希值。

        关键设计：每个块的 hash 不仅依赖自身的 token 内容，
        还包含前一个块的 hash 作为前缀。这确保了相同的 token 内容
        在不同上下文（不同前缀）中会产生不同的 hash，避免误匹配。

        类比 Merkle Tree / 区块链中的链式哈希。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前一个块的 hash 作为前缀加入计算
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """从空闲池中取出一个物理块并标记为已使用"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """将物理块归还空闲池"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """判断是否有足够的空闲块来容纳该序列的所有 token（prefill 阶段使用）"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为新序列分配 KV Cache 块（prefill 阶段调用）。

        遍历序列的每个逻辑块，尝试 prefix cache 匹配：
        1. 对满块计算链式 hash
        2. 查找 hash_to_block_id 看是否命中
        3. 命中 → 复用已有物理块（ref_count++），跳过该块的计算
        4. 未命中 → 从空闲池分配新的物理块

        一旦某个块 cache miss，后续所有块都必须重新分配（因为链式 hash 断裂）。
        """
        assert not seq.block_table
        h = -1              # 前一个块的 hash，初始为 -1
        cache_miss = False  # 一旦发生 miss，后续所有块都是 miss
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只对满块（token 数 == block_size）计算 hash；最后一个不满的块 hash 为 -1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # 双重验证：hash 匹配且 token 内容一致（防止 hash 冲突）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # Cache miss：从空闲池分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache hit：复用已有块，该块的 KV 数据不需要重新计算
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 块正被其他序列使用，增加引用计数（共享）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块在空闲池中但 hash 仍有效，重新激活
                    block = self._allocate_block(block_id)
            if h != -1:
                # 记录满块的 hash 和 token 内容，供后续请求匹配
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            # 将物理块 ID 追加到序列的块表中
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的所有 KV Cache 块。
        在序列完成或被抢占时调用。倒序释放以优先回收末尾的不完整块。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 只有引用计数归零才真正回收（可能被其他序列共享）
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        判断 decode 阶段是否有足够块来追加一个新 token。

        只有当新 token 恰好落在新块的第一个位置（len(seq) % block_size == 1）时，
        才需要分配一个新的物理块；否则直接写入当前块的剩余空间。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        decode 阶段为序列追加新 token 时的块管理逻辑。

        三种情况：
        1. 新 token 是新块的第一个 token → 需要分配新的物理块
        2. 新 token 恰好填满当前块 → 计算该块的 hash 并注册到 prefix cache
        3. 其他情况 → 当前块未满，无需额外操作
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 情况 1：当前块已满（上一步刚填满），需要分配新块
            # 上一个块应该已经有 hash（在上一步的情况 2 中设置的）
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 情况 2：刚刚追加的 token 把当前块填满了
            # 计算该块的链式 hash，注册到 prefix cache 索引中
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 情况 3：当前块还有空位，无需操作
            assert last_block.hash == -1
