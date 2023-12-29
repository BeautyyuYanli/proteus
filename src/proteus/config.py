from pathlib import Path
from typing import Any, Dict, Literal, Optional

from msgspec import field
from xdg import XDG_CACHE_HOME

from proteus.spec import ProteusMessagePrompt
from proteus.utils.spec import StructSpec

LLMsName = Literal["openai", "llama_cpp", "testback", "gemini"]


class LLMsConfig(StructSpec, kw_only=True, frozen=True):
    class GeminiConfig(StructSpec, kw_only=True, frozen=True):
        model_name: str = "gemini-pro"

    class TestBackConfig(StructSpec, kw_only=True, frozen=True):
        model: str
        completion_extra: Dict[str, Any] = field(default_factory=dict)

    class OpenAIConfig(StructSpec, kw_only=True, frozen=True):
        model: str = "gpt-3.5-turbo"

    class LlamaCppConfig(StructSpec, kw_only=True, frozen=True):
        model_path: str
        model_extra: Dict[str, Any] = field(default_factory=dict)
        completion_extra: Dict[str, Any] = field(default_factory=dict)

    openai: Optional[OpenAIConfig] = None
    llama_cpp: Optional[LlamaCppConfig] = None
    testback: Optional[TestBackConfig] = None
    gemini: Optional[GeminiConfig] = None

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
    live_history_size: int = 0
    cache_folder: str = str(XDG_CACHE_HOME / "proteus")
    cache_history_enabled: bool = False
    cache_talkers_enabled: bool = False

    @classmethod
    def from_path(cls, path: Path) -> "ManagerConfig":
        return cls.from_toml(path.read_bytes())
