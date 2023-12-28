from threading import Lock
from typing import List, Optional, Self
from uuid import uuid4

from proteus.config import PromptsConfig
from proteus.history_store import BaseHistoryStore, FakeHistoryStore
from proteus.llms.base import BaseLLM
from proteus.spec import Message, MessagePrompt, StructSpec


class ProteusTalkerState(StructSpec, kw_only=True, frozen=True):
    id: str
    prompt_name: str
    live_history: List[Message]
    live_history_size: int


class ProteusTalker:
    state: ProteusTalkerState
    state_lock: Lock
    prompts_config: PromptsConfig
    llm: BaseLLM
    prompt: MessagePrompt
    history_store: BaseHistoryStore

    def _finish_init(self) -> None:
        self.state_lock = Lock()
        self.prompt = self.prompts_config.prompts[self.state.prompt_name]

    @classmethod
    def from_new(
        cls,
        prompt_name: str,
        prompts_config: PromptsConfig,
        llm: BaseLLM,
        history_store: Optional[BaseHistoryStore] = None,
        live_history_size: int = 0,
    ) -> Self:
        _self = cls()
        # state things that can be serialized
        _self.state = ProteusTalkerState(
            id=uuid4().hex,
            prompt_name=prompt_name,
            live_history=[],
            live_history_size=live_history_size,
        )
        _self.prompts_config = prompts_config
        _self.llm = llm
        _self.history_store = history_store or FakeHistoryStore()
        _self._finish_init()

        return _self

    @classmethod
    def from_json(
        cls,
        state_json: bytes,
        prompt_config: PromptsConfig,
        llm: BaseLLM,
        history_store: Optional[BaseHistoryStore] = None,
    ) -> Self:
        _self = cls()
        _self.state = ProteusTalkerState.from_json(state_json)
        _self.prompts_config = prompt_config
        _self.llm = llm
        _self.history_store = history_store or FakeHistoryStore()
        _self._finish_init()

        return _self

    def to_json(self) -> bytes:
        with self.state_lock:
            return self.state.to_json()

    def _construct_prompt_msgs(self, new_inputs: List[Message]) -> List[Message]:
        with self.state_lock:
            return (
                self.prompt.identity
                + self.prompt.instruct
                + self.prompt.examples
                + self.state.live_history
                + new_inputs
            )

    def _extend_history(self, new_turn: List[Message]) -> None:
        with self.state_lock:
            self.state.live_history.extend(new_turn)
            while len(self.state.live_history) > self.state.live_history_size:
                self.state.live_history.pop(0)
            self.history_store.extend(self.state.id, new_turn)

    async def asay(self, user_input) -> str:
        new_turn = [Message(role="user", content=user_input)]
        msgs = self._construct_prompt_msgs(new_inputs=new_turn)
        resp = await self.llm.arequest(msgs)
        new_turn.append(resp.message)
        self._extend_history(new_turn)
        return resp.message.content

    def say(self, user_input) -> str:
        new_turn = [Message(role="user", content=user_input)]
        msgs = self._construct_prompt_msgs(new_inputs=new_turn)
        resp = self.llm.request(msgs)
        new_turn.append(resp.message)
        self._extend_history(new_turn)
        return resp.message.content

    def clear(self) -> None:
        with self.state_lock:
            self.state.live_history = []


__all__ = ["ProteusTalker"]
