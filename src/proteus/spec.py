from typing import List, Literal, Optional

from msgspec import field

from proteus.utils.spec import StructSpec


class ProteusMessage(StructSpec, kw_only=True, frozen=True):
    role: Literal["system", "user", "assistant"]
    content: str


class ProteusMessagePrompt(StructSpec, kw_only=True, frozen=True):
    identity: List[ProteusMessage] = field(default_factory=list)
    examples: List[ProteusMessage] = field(default_factory=list)
    instruct: List[ProteusMessage] = field(default_factory=list)


class ProteusLLMResponse(StructSpec, kw_only=True, frozen=True):
    message: ProteusMessage
    token_cnt: Optional[int] = None
