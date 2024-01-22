from pathlib import Path
from typing import Any, Dict, Literal, Optional

from msgspec import field
from xdg import XDG_CACHE_HOME

from proteus.spec import ProteusMessagePrompt
from proteus.utils.spec import StructSpec

LLMsName = Literal["openai", "llama_cpp", "testback", "gemini", "dashscope"]


class LLMsConfig(StructSpec, kw_only=True, frozen=True):
    class GeminiConfig(StructSpec, kw_only=True, frozen=True):
        model_name: str = "gemini-pro"
        generation_config: Dict[str, Any] = field(default_factory=dict)

    class TestBackConfig(StructSpec, kw_only=True, frozen=True):
        model: str
        completion_extra: Dict[str, Any] = field(default_factory=dict)

    class OpenAIConfig(StructSpec, kw_only=True, frozen=True):
        model: str = "gpt-3.5-turbo"

    class LlamaCppConfig(StructSpec, kw_only=True, frozen=True):
        model_path: str
        model_extra: Dict[str, Any] = field(default_factory=dict)
        completion_extra: Dict[str, Any] = field(default_factory=dict)

    class DashScopeConfig(StructSpec, kw_only=True, frozen=True):
        model: str = "qwen-turbo"
        parameters: Dict[str, Any] = field(default_factory=dict)

    class ReplicateMixtralInsConfig(StructSpec, kw_only=True, frozen=True):
        max_new_tokens: int = 512
        temperature: float = 0.6
        top_p: float = 0.9
        top_k: int = 50
        presence_penalty: float = 0
        frequency_penalty: float = 0

    class ReplicateQwen14Config(StructSpec, kw_only=True, frozen=True):
        max_tokens: int = 2048
        temperature: float = 0.75
        top_p: float = 0.8

    openai: Optional[OpenAIConfig] = None
    llama_cpp: Optional[LlamaCppConfig] = None
    testback: Optional[TestBackConfig] = None
    gemini: Optional[GeminiConfig] = None
    dashscope: Optional[DashScopeConfig] = None
    mixtral_ins: Optional[ReplicateMixtralInsConfig] = None
    qwen14: Optional[ReplicateQwen14Config] = None

    @classmethod
    def from_path(cls, path: Path) -> "LLMsConfig":
        return cls.from_toml(path.read_bytes())


class PromptsConfig(StructSpec, kw_only=True, frozen=True):
    prompts: Dict[str, ProteusMessagePrompt]

    @classmethod
    def from_path(cls, path: Path) -> "PromptsConfig":
        prompts = {}
        for prompt_path in path.glob("*.json"):
            prompts[prompt_path.stem] = ProteusMessagePrompt.from_json(
                prompt_path.read_bytes()
            )
        return cls(prompts=prompts)


class ManagerConfig(StructSpec, kw_only=True, frozen=True):
    """
    live_history_size: size of live history (context window size)
    cache_folder: cache folder
    cache_history_enabled: whether to cache all history of all talkers
    cache_talkers_enabled: whether to cache states of all talkers, which can be used to resume a manager
    cache_talkers_mem_capacity: capacity of talker cache in memory. -1 means unlimited. If exceeded, the least recently used talker will be saved to cache_folder, or discarded if cache_history_enabled is False.
    """

    live_history_size: int = 0
    cache_folder: str = str(XDG_CACHE_HOME / "proteus")
    cache_history_enabled: bool = False
    cache_talkers_enabled: bool = False
    cache_talkers_mem_capacity: int = -1

    @classmethod
    def from_path(cls, path: Path) -> "ManagerConfig":
        return cls.from_toml(path.read_bytes())
