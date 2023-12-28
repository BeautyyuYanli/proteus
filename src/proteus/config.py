from pathlib import Path
from typing import Any, Dict, Optional

from msgspec import field

from proteus.spec import MessagePrompt
from proteus.utils.spec import StructSpec


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


class BackendsConfig(StructSpec, kw_only=True, frozen=True):
    openai: Optional[OpenAIConfig] = None
    llama_cpp: Optional[LlamaCppConfig] = None
    testback: Optional[TestBackConfig] = None
    gemini: Optional[GeminiConfig] = None

    @classmethod
    def from_path(cls, path: Path) -> "BackendsConfig":
        return cls.from_toml(path.read_bytes())


class PromptsConfig(StructSpec, kw_only=True, frozen=True):
    prompts: Dict[str, MessagePrompt]

    @classmethod
    def from_path(cls, path: Path) -> "PromptsConfig":
        prompts = {}
        for prompt_path in path.glob("*.json"):
            prompts[prompt_path.stem] = MessagePrompt.from_json(
                prompt_path.read_bytes()
            )
        return cls(prompts=prompts)


class ManagerConfig(StructSpec, kw_only=True, frozen=True):
    live_history_size: int = 0
