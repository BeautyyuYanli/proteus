from typing import List

from proteus.llms.base import BaseLLM
from proteus.spec import ProteusMessage, ProteusMessagePrompt
from proteus.storages.history_store import BaseHistoryStore


class ProteusTeller:
    """This is stateless and thread-safe."""

    id: str
    _llm: BaseLLM
    _prompt: ProteusMessagePrompt
    _history: BaseHistoryStore
    _live_history_size: int

    def __init__(
        self,
        id: str,
        llm: BaseLLM,
        prompt: ProteusMessagePrompt,
        history: BaseHistoryStore,
        live_history_size: int = 0,
    ) -> None:
        self.id = id
        self._llm = llm
        self._prompt = prompt
        self._history = history
        self._live_history_size = live_history_size

    def construct_prompt_msgs(
        self, new_inputs: List[ProteusMessage]
    ) -> List[ProteusMessage]:
        """Default implementation. Override this if you want to customize the prompt."""
        return (
            self._prompt.identity
            + self._prompt.instruct
            + self._prompt.examples
            + self._history.get_k(self.id, self._live_history_size)
            + new_inputs
        )

    def say(self, user_input: str) -> str:
        new_turn = [ProteusMessage(role="user", content=user_input)]
        msgs = self.construct_prompt_msgs(new_inputs=new_turn)
        resp = self._llm.request(msgs)
        new_turn.append(resp.message)
        self._history.extend(self.id, new_turn)
        return resp.message.content

    def say_with_template(self, user_input: str, template_name: str) -> str:
        temp_input = self._prompt.templates.get(template_name, "{input}")
        return self.say(temp_input.format(input=user_input))
