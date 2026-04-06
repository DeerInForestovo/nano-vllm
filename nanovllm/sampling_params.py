"""
采样参数配置。

控制 LLM 解码阶段的行为：温度（随机性）、最大生成长度、是否忽略 EOS。
这些参数会被存储到每个 Sequence 对象中，在 Sampler 采样时使用。
"""
from dataclasses import dataclass


@dataclass
class SamplingParams:
    # 采样温度：值越高输出越随机，值越低输出越确定
    # 在 Sampler 中用于缩放 logits: logits = logits / temperature
    temperature: float = 1.0
    # 最大生成 token 数量，达到后强制停止（即使未遇到 EOS）
    max_tokens: int = 64
    # 若为 True，则生成时忽略 EOS token，不会因为 EOS 而提前停止
    # 在 benchmark 中常设为 True，以确保生成固定长度的输出来测量吞吐量
    ignore_eos: bool = False

    def __post_init__(self):
        # 禁止 greedy 解码（temperature ≈ 0），因为本项目只实现了 multinomial 采样
        # 若需要 greedy，需在 Sampler 中额外处理 argmax 逻辑
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
