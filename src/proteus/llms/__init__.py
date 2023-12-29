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

    _llm_name_map: Dict[str, Callable[[], BaseLLM]] = {
        "openai": import_openai,
        "llama_cpp": import_llama_cpp,
        "testback": import_testback,
        "gemini": import_gemini,
    }

    if name is not None:
        return _llm_name_map[name]()(getattr(config, name))

    for k, v in _llm_name_map.items():
        if getattr(config, k) is not None:
            return v()(getattr(config, k))
    raise ValueError("No LLMs config provided")


__all__ = ["llm_from_config", "BaseLLM"]
