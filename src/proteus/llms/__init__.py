from typing import Dict, List, Callable

from proteus.config import BackendsConfig
from proteus.llms.base import BaseLLM


def llm_from_config(name: str, config: BackendsConfig) -> "BaseLLM":
    def import_openai():
        from proteus.llms.openai import OpenAILLM

        return OpenAILLM

    def import_llama_cpp():
        from proteus.llms.llama_cpp import LlamaCppLLM

        return LlamaCppLLM

    def import_testback():
        from proteus.llms.testback import TestBackLLM

        return TestBackLLM

    _llm_name_map: Dict[str, Callable[[], BaseLLM]] = {
        "openai": import_openai,
        "llama_cpp": import_llama_cpp,
        "testback": import_testback,
    }
    return _llm_name_map[name]()(config)


__all__ = ["llm_from_config"]
