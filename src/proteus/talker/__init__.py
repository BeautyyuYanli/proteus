from typing import AsyncGenerator, List, Self
from uuid import uuid4

from proteus.config import PromptsConfig
from proteus.llms import BaseLLM
from proteus.spec import Message, MessagePrompt, StructSpec


class FullHistory:
    _full_history: List[Message]

    def __init__(self, proteus_id: str) -> None:
        self._full_history = []
        self._proteus_id = proteus_id

    def extend(self, msg: List[Message]) -> None:
        self._full_history.extend(msg)


class ProteusTalkerState(StructSpec, kw_only=True, frozen=True):
    id: str
    prompt_name: str
    live_history: List[Message]


class ProteusTalker:
    state: ProteusTalkerState
    config: PromptsConfig
    llm: BaseLLM
    prompt: MessagePrompt
    full_history: FullHistory

    def _finish_init(self) -> None:
        self.prompt = self.config.prompts[self.state.prompt_name]
        self.full_history = FullHistory(self.state.id)

    @classmethod
    def from_new(cls, prompt_name: str, config: PromptsConfig, llm: BaseLLM) -> Self:
        _self = cls()
        # state things that can be serialized
        _self.state = ProteusTalkerState(
            id=uuid4().hex, prompt_name=prompt_name, live_history=[]
        )
        _self.config = config
        _self.llm = llm
        _self._finish_init()

        return _self

    @classmethod
    def from_json(cls, state_json: bytes, config: PromptsConfig, llm: BaseLLM) -> Self:
        state = ProteusTalkerState.from_json(state_json)
        _self = cls()
        _self.state = state
        _self.config = config
        _self.llm = llm
        _self._finish_init()

        return _self

    def to_json(self) -> bytes:
        return self.state.to_json()

    def _construct_prompt_msgs(self, new_inputs: List[Message]) -> List[Message]:
        return (
            self.prompt.identity
            + self.prompt.instruct
            + self.prompt.examples
            + self.state.live_history
            + new_inputs
        )

    async def _extend_history(self, new_turn: List[Message]) -> None:
        self.state.live_history.extend(new_turn)
        self.full_history.extend(new_turn)

    async def say(self, user_input) -> AsyncGenerator[str, None]:
        new_turn = [Message(role="user", content=user_input)]
        msgs = self._construct_prompt_msgs(new_inputs=new_turn)
        resp = await self.llm.request(msgs)
        new_turn.append(resp.message)
        yield resp.message.content

        await self._extend_history(new_turn)


__all__ = ["ProteusTalker"]
