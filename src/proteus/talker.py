from threading import Lock
from typing import Callable, List, Self
from uuid import uuid4
from weakref import finalize

from proteus.config import LLMsConfig, PromptsConfig
from proteus.llms import llm_from_config
from proteus.llms.base import BaseLLM
from proteus.spec import ProteusMessage, ProteusMessagePrompt, StructSpec


class ProteusTalkerState(StructSpec, kw_only=True, frozen=True):
    id: str
    prompt_name: str
    live_history: List[ProteusMessage]


class ProteusTalker:
    _state: ProteusTalkerState
    _state_lock: Lock
    _prompts_config: PromptsConfig
    _llm: BaseLLM
    _prompt: ProteusMessagePrompt
    _live_history_size: int
    _save_history: Callable[[str, List[ProteusMessage]], None]
    _persist: Callable[[str, bytes], None]

    @classmethod
    def create(
        cls,
        message_prompt: ProteusMessagePrompt,
        llms_config: LLMsConfig,
        live_history_size: int = 0,
    ):
        """Create an independent ProteusTalker which is not managed by ProteusManager"""
        return cls.from_new(
            prompt_name="default",
            prompts_config=PromptsConfig(prompts={"default": message_prompt}),
            llm=llm_from_config(llms_config),
            live_history_size=live_history_size,
        )

    def _finish_init(self) -> None:
        self._state_lock = Lock()
        self._prompt = self._prompts_config.prompts[self._state.prompt_name]
        finalize(self, self.save)

    @classmethod
    def from_new(
        cls,
        prompt_name: str,
        prompts_config: PromptsConfig,
        llm: BaseLLM,
        live_history_size: int = 0,
        save_history: Callable[[str, List[ProteusMessage]], None] = lambda *args: None,
        persist: Callable[[str, bytes], None] = lambda *args: None,
    ) -> Self:
        _self = cls()
        # state things that can be serialized
        _self._state = ProteusTalkerState(
            id=uuid4().hex,
            prompt_name=prompt_name,
            live_history=[],
        )
        _self._prompts_config = prompts_config
        _self._llm = llm
        _self._live_history_size = live_history_size
        _self._save_history = save_history
        _self._persist = persist
        _self._finish_init()

        return _self

    # @classmethod
    # def from_json(
    #     cls,
    #     state_json: bytes,
    #     prompt_config: PromptsConfig,
    #     llm: BaseLLM,
    #     history_store: Optional[BaseHistoryStore] = None,
    # ) -> Self:
    #     _self = cls()
    #     _self.state = ProteusTalkerState.from_json(state_json)
    #     _self.prompts_config = prompt_config
    #     _self.llm = llm
    #     _self.history_store = history_store or FakeHistoryStore()
    #     _self._finish_init()

    #     return _self

    def _construct_prompt_msgs(
        self, new_inputs: List[ProteusMessage]
    ) -> List[ProteusMessage]:
        with self._state_lock:
            return (
                self._prompt.identity
                + self._prompt.instruct
                + self._prompt.examples
                + self._state.live_history
                + new_inputs
            )

    def _extend_history(self, new_turn: List[ProteusMessage]) -> None:
        with self._state_lock:
            self._state.live_history.extend(new_turn)
            while len(self._state.live_history) > self._live_history_size:
                self._state.live_history.pop(0)
            self._save_history(self._state.id, new_turn)

    async def asay(self, user_input) -> str:
        new_turn = [ProteusMessage(role="user", content=user_input)]
        msgs = self._construct_prompt_msgs(new_inputs=new_turn)
        resp = await self._llm.arequest(msgs)
        new_turn.append(resp.message)
        self._extend_history(new_turn)
        return resp.message.content

    def say(self, user_input) -> str:
        new_turn = [ProteusMessage(role="user", content=user_input)]
        msgs = self._construct_prompt_msgs(new_inputs=new_turn)
        resp = self._llm.request(msgs)
        new_turn.append(resp.message)
        self._extend_history(new_turn)
        return resp.message.content

    def clear(self) -> None:
        with self._state_lock:
            self._state.live_history = []

    def save(self) -> None:
        with self._state_lock:
            self._persist(self._state.id, self._state.to_json())


__all__ = ["ProteusTalker"]
