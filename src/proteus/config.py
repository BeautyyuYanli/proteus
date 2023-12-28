from pathlib import Path
from typing import Any, Dict, Literal, Optional

from msgspec import field

from proteus.spec import MessagePrompt
from utils.spec import StructSpec


class TestBackConfig(StructSpec, kw_only=True, frozen=True):
    model: str
    completion_extra: Dict[str, Any] = field(default_factory=dict)


class OpenAIConfig(StructSpec, kw_only=True, frozen=True):
    model: str


class LlamaCppConfig(StructSpec, kw_only=True, frozen=True):
    model_path: str
    model_extra: Dict[str, Any] = field(default_factory=dict)
    completion_extra: Dict[str, Any] = field(default_factory=dict)


class BackendsConfig(StructSpec, kw_only=True, frozen=True):
    openai: Optional[OpenAIConfig]
    llama_cpp: Optional[LlamaCppConfig]
    testback: Optional[TestBackConfig]

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
