"""
用户 API 的最外层入口。

LLM 类直接继承 LLMEngine，没有添加任何额外逻辑。
这种设计使得用户只需 `from nanovllm import LLM` 即可使用，
同时在未来可以在 LLM 层添加更高级的封装（如异步接口），而不影响 LLMEngine 本身。
"""
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
