from typing import List, Literal, Optional

from proteus.utils.spec import StructSpec


class Message(StructSpec, kw_only=True, frozen=True):
    role: Literal["system", "user", "assistant"]
    content: str


class MessagePrompt(StructSpec, kw_only=True, frozen=True):
    identity: List[Message]
    examples: List[Message]
    instruct: List[Message]


class LLMResponse(StructSpec, kw_only=True, frozen=True):
    message: Message
    token_cnt: Optional[int] = None
