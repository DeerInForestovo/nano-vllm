"""
LLM 引擎 -- 连接用户 API 与底层推理组件的中枢。

职责：
1. 初始化所有组件：ModelRunner(模型执行), Scheduler(调度), Tokenizer(编解码)
2. 管理张量并行的多进程：为每个非 rank-0 的 GPU 创建独立进程
3. 提供 generate() 主循环：反复调用 step() 直到所有请求完成
4. step() 是单步推理：调度 -> 模型执行 -> 后处理

数据流：
  用户 prompts -> add_request() -> Scheduler.waiting 队列
  -> step() 主循环 { schedule() -> ModelRunner.run() -> postprocess() }
  -> 收集所有 FINISHED 的 Sequence -> decode 回文本 -> 返回用户
"""
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 只取 Config 定义的字段，忽略未知参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # ========== 张量并行：为 rank 1~N 创建子进程 ==========
        # 每个子进程直接以 ModelRunner.__init__ 为 target，
        # 在初始化完成后会进入 ModelRunner.loop() 等待 rank 0 的指令
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")   # spawn 模式：子进程不继承父进程的 CUDA 状态
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank 0 的 ModelRunner 在主进程中创建
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # Scheduler 在 ModelRunner 之后创建，因为它需要 config.num_kvcache_blocks
        # (该值在 ModelRunner.allocate_kv_cache 中计算并写入 config)
        self.scheduler = Scheduler(config)
        # 注册退出清理函数，确保多进程资源被正确释放
        atexit.register(self.exit)

    def exit(self):
        """清理：通知所有 worker 退出并等待进程结束"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个推理请求。
        支持传入原始文本(自动 tokenize)或已编码的 token ID 列表。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行单步推理。

        流程：
        1. Scheduler.schedule() -> 决定本步做 prefill 还是 decode，选出参与的序列
        2. ModelRunner.call("run") -> 执行模型前向传播 + 采样
        3. Scheduler.postprocess() -> 将生成的 token 追加到序列，检查终止条件

        返回：
        - outputs: 本步中已完成的序列 [(seq_id, completion_token_ids), ...]
        - num_tokens: 正数=prefill 处理的 token 总数, 负数=decode 的序列数(吞吐量计算用)
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 约定：prefill 返回正的 token 总数，decode 返回负的序列数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """所有请求都已处理完成"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量推理的主入口(Offline Inference)。

        流程：
        1. 将所有 prompts 加入 Scheduler 的 waiting 队列
        2. 循环调用 step()，每步可能完成若干序列
        3. 收集所有完成的序列，按原始顺序排序后返回

        与 vLLM 的 LLM.generate() API 兼容。
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 支持单个 SamplingParams 广播到所有 prompts
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # 计算并显示实时吞吐量
            if use_tqdm:
                if num_tokens > 0:
                    # Prefill 吞吐量(tokens/s)
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # Decode 吞吐量(tokens/s, 即 -num_tokens = batch_size)
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        # 按 seq_id 排序，恢复输入顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 将 token IDs 解码为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
