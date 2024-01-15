from typing import Callable, Dict, Optional

from proteus.config import LLMsConfig, LLMsName
from proteus.llms.base import BaseLLM


def llm_from_config(config: LLMsConfig, name: Optional[LLMsName] = None) -> BaseLLM:
    def import_openai():
        from proteus.llms.openai import OpenAILLM

        return OpenAILLM

    def import_llama_cpp():
        from proteus.llms.llama_cpp import LlamaCppLLM

        return LlamaCppLLM

    def import_testback():
        from proteus.llms.testback import TestBackLLM

        return TestBackLLM

    def import_gemini():
        from proteus.llms.gemini import GeminiLLM

        return GeminiLLM

    def import_dashscope():
        from proteus.llms.dashscope import DashScopeLLM

        return DashScopeLLM

    def import_mixtral_ins():
        from proteus.llms.replicate_mixtral_ins import ReplicateMixtralInsLLM

        return ReplicateMixtralInsLLM

    def import_qwen14():
        from proteus.llms.replicate_qwen_14b import ReplicateQwen14LLM

        return ReplicateQwen14LLM

    _llm_name_map: Dict[str, Callable[[], BaseLLM]] = {
        "openai": import_openai,
        "llama_cpp": import_llama_cpp,
        "testback": import_testback,
        "gemini": import_gemini,
        "dashscope": import_dashscope,
        "mixtral_ins": import_mixtral_ins,
        "qwen14": import_qwen14,
    }

    if name is not None:
        return _llm_name_map[name]()(getattr(config, name))

    for k, v in _llm_name_map.items():
        if getattr(config, k) is not None:
            return v()(getattr(config, k))
    raise ValueError("No LLMs config provided")


__all__ = ["llm_from_config", "BaseLLM"]
