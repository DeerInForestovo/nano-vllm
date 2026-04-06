"""
模型执行器 —— 整个推理引擎中最核心、最复杂的组件。

职责：
1. 模型初始化：加载权重、预热（warmup）、分配 KV Cache
2. 输入准备：将 Sequence 列表转换为模型需要的张量（input_ids, positions, slot_mapping 等）
3. 模型执行：Prefill 走 eager 模式，Decode 走 CUDA Graph 加速
4. 张量并行通信：rank 0 通过共享内存协调多 GPU 执行
5. 采样：将 logits 转换为下一个 token ID

关键优化技术：
- CUDA Graph：预录制 decode 阶段的 GPU 计算图，replay 时跳过 CPU→GPU 的 kernel launch 开销
- Pin Memory + Non-blocking Transfer：CPU→GPU 数据传输与计算重叠
- 张量并行（Tensor Parallelism）：通过 NCCL 和共享内存实现多 GPU 推理
"""
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型执行器。

        Args:
            config: 全局配置
            rank: 当前进程在张量并行中的 rank（0 为主进程）
            event: 多进程同步事件
                   - rank 0: 传入 list[Event]，每个 Event 对应一个 worker
                   - rank > 0: 传入单个 Event，用于接收 rank 0 的指令
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # ========== 第一步：初始化分布式环境 ==========
        # 使用 NCCL 后端，这是 GPU 间通信的最高效方式
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # ========== 第二步：构建模型并加载权重 ==========
        # 临时将默认 dtype 和 device 设为模型所需的（如 bfloat16 + cuda）
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载 safetensors 权重，自动处理 QKV 合并等映射
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # ========== 第三步：预热 + 分配 KV Cache + 捕获 CUDA Graph ==========
        # 预热的目的是让 PyTorch/CUDA 完成懒初始化（cuDNN 算法选择、内存池分配等），
        # 并记录峰值显存用量，用于后续计算可分配多少 KV Cache 块
        self.warmup_model()
        # 根据 GPU 剩余显存计算并分配 KV Cache
        self.allocate_kv_cache()
        if not self.enforce_eager:
            # 预捕获不同 batch size 的 CUDA Graph（仅 decode 阶段使用）
            self.capture_cudagraph()

        # 恢复默认 dtype 和 device
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ========== 第四步：张量并行同步 ==========
        if self.world_size > 1:
            if rank == 0:
                # rank 0 创建共享内存（1MB），用于向其他 rank 广播调度指令
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等待其他 rank 就绪
            else:
                dist.barrier()  # 等待 rank 0 创建共享内存
                self.shm = SharedMemory(name="nanovllm")
                # 非 rank 0 的 worker 进入事件循环，等待 rank 0 的指令
                self.loop()

    def exit(self):
        """清理资源：关闭共享内存、销毁进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        非 rank 0 的 worker 的主循环。
        不断从共享内存读取 rank 0 发来的方法名和参数，然后执行。
        这样所有 rank 在同一时刻执行相同的方法（如 run），保持 NCCL 通信同步。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        worker 端：从共享内存读取 rank 0 发来的指令。
        协议格式：[4 字节长度 N][N 字节 pickle 数据]
        使用 multiprocessing.Event 做同步等待。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()      # 阻塞直到 rank 0 写入数据并 set event
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()     # 重置 event 为下一次通信准备
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        rank 0 端：将方法名和参数序列化后写入共享内存，并通知所有 worker。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 通知所有 worker（每个 worker 有自己的 Event）
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        统一的方法调用入口。
        rank 0 调用时：先通过共享内存广播给其他 rank，再本地执行。
        其他 rank 调用时：直接本地执行（已在 loop 中从共享内存读取了方法名和参数）。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        模型预热：用最大尺寸的虚拟输入跑一遍前向传播。

        目的：
        1. 触发 PyTorch/CUDA 的懒初始化（JIT 编译 triton kernel、cuDNN 自动调优等）
        2. 记录峰值显存占用，后续用于计算 KV Cache 可用空间
        3. 清空 cache 确保峰值统计准确
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 模拟最大 batch 场景：多个最大长度的序列
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据 GPU 剩余显存计算并分配 KV Cache。

        计算公式：
        可用显存 = 总显存 × gpu_memory_utilization - 已占用 - (峰值 - 当前)
        其中 (峰值 - 当前) 是运行时临时分配的最大开销，需要预留。

        KV Cache 总张量形状：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        其中 2 对应 Key 和 Value 两部分。
        分配后将每层的 KV Cache 引用注入到对应的 Attention 模块中。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 单个物理块的字节数 = 2(K+V) × 层数 × block_size × kv_heads × head_dim × dtype字节
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 可分配的块数 = 可用显存 / 单块字节数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 一次性分配全部 KV Cache（避免碎片化）
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 将每层的 KV Cache 切片注入到对应的 Attention 模块
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        将多个序列的块表（block_table）padding 成统一长度的 2D 张量。
        不足的部分用 -1 填充（FlashAttention 会忽略 -1 对应的块）。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        为 Prefill 阶段准备模型输入。

        Prefill 特点：每个序列可能有不同长度，需要用 flash_attn_varlen_func 处理变长输入。
        核心张量：
        - input_ids: 所有序列的 token 拼接成一维（跳过 prefix cache 命中的部分）
        - positions: 每个 token 的绝对位置（用于 RoPE）
        - cu_seqlens_q/k: FlashAttention 变长接口需要的累积序列长度前缀和
          Q 的长度 = 实际计算的 token 数（去掉 cached 部分）
          K 的长度 = 完整序列长度（包含 cached 部分，因为需要 attend 到完整上下文）
        - slot_mapping: 每个需要写入 KV Cache 的 token 在物理块中的全局槽位索引
          slot = block_table[block_idx] * block_size + offset_in_block
        - block_tables: 仅在有 prefix cache 命中时才需要（Q 长度 < K 长度）
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]    # 累积 Q 序列长度（FlashAttention varlen 接口要求）
        cu_seqlens_k = [0]    # 累积 K 序列长度
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []     # 每个新计算的 token 在 KV Cache 中的写入位置
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # 只取 cache 未命中的 token 作为输入（prefix cache 命中的部分跳过）
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens   # Q 长度：实际需要计算的 token 数
            seqlen_k = seqlen                            # K 长度：完整上下文长度（含 cached）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup 阶段没有 block_table
                continue
            # 构造 slot_mapping：只为非 cached 的块生成槽位映射
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size           # 非最后一块：完整 block_size
                else:
                    end = start + seq.last_block_num_tokens # 最后一块：可能不满
                slot_mapping.extend(list(range(start, end)))
        # 如果 K 总长 > Q 总长，说明有 prefix cache 命中，需要传入 block_tables
        # 让 FlashAttention 从 paged KV Cache 中读取 cached 的 K/V
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        # 将 Python list 转为 CUDA Tensor（pin_memory + non_blocking 实现异步传输）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 将 prefill 上下文信息存入全局 Context，供 Attention 层读取
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        为 Decode 阶段准备模型输入。

        Decode 特点：每个序列只输入最后一个生成的 token（自回归）。
        核心张量：
        - input_ids: 每个序列的 last_token，shape = [batch_size]
        - positions: 每个 token 的位置 = 当前序列长度 - 1
        - slot_mapping: 每个新 token 在 KV Cache 中的写入槽位
        - context_lens: 每个序列的完整上下文长度（用于 FlashAttention 计算 attention range）
        - block_tables: 完整的块表（decode 必须访问所有历史 KV）
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 新 token 写入最后一个块的最后一个槽位
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """收集每个序列的采样温度，转为 CUDA 张量"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向传播。

        路由逻辑：
        1. Prefill 阶段 → eager 模式（因为输入长度可变，无法用固定 shape 的 CUDA Graph）
        2. Decode + enforce_eager → eager 模式（调试用）
        3. Decode + batch_size > 512 → eager 模式（超出预捕获的最大 batch size）
        4. Decode + batch_size ≤ 512 → CUDA Graph replay（高性能路径）

        CUDA Graph 的工作方式：
        - 捕获阶段：录制一次完整的 GPU 操作序列（包括 kernel launch 和内存访问模式）
        - Replay 阶段：直接重放录制的操作，跳过 CPU 侧的 kernel launch 开销
        - 关键约束：replay 时输入张量的地址必须和捕获时一致，因此需要将实际数据
          拷贝到捕获时使用的 graph_vars 缓冲区中
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager 模式：直接跑 Transformer + LM Head
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 选择 >= 当前 batch size 的最小预捕获图（分桶策略）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 将实际数据拷贝到 CUDA Graph 捕获时绑定的缓冲区
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # slot_mapping 和 context_lens 需要先清零再写入（因为 Graph 可能使用了更大的 bs）
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放预录制的 GPU 计算图
            graph.replay()
            # compute_logits 不在 Graph 中（因为它在 Graph 之外更灵活）
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        单次推理步骤的完整流程：准备输入 → 模型前向 → 采样。

        只有 rank 0 执行采样（因为只有 rank 0 需要返回 token IDs）。
        其他 rank 执行相同的前向传播（保持 NCCL AllReduce 同步），但不做采样。
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        预捕获多个 batch size 对应的 CUDA Graph。

        分桶策略：[1, 2, 4, 8, 16, 32, ..., max_bs]
        实际 batch size 会向上取整到最近的桶（如 bs=3 使用 bs=4 的 Graph）。
        从大 bs 到小 bs 逆序捕获，并共享同一个 memory pool 减少显存开销。

        每个 bs 的捕获流程：
        1. 设置 decode context（dummy 数据）
        2. Warmup：先跑一遍确保所有 lazy 初始化完成
        3. Capture：在 torch.cuda.graph 上下文中跑一遍，录制 GPU 操作
        4. 保存 Graph 对象，供后续 replay 使用

        注意：只录制 model() 部分（到 hidden_states），不包含 compute_logits（LM Head），
        因为 LM Head 的计算量不大，且在 Graph 外更灵活。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 预分配固定大小的缓冲区（CUDA Graph 要求地址不变）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 分桶的 batch sizes
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获，确保大 bs 的 Graph 先分配内存，小 bs 复用其子集
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                # 第一个 Graph 创建 memory pool，后续 Graph 共享
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存所有缓冲区的引用，run_model 中通过这些引用写入实际数据
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
