from typing import List

from llama_cpp import Llama

from proteus.config import LLMsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import ProteusLLMResponse, ProteusMessage


class LlamaCppLLM(BaseLLM):
    def __init__(
        self,
        config: LLMsConfig.LlamaCppConfig,
    ) -> None:
        self.config = config
        self._llm = Llama(
            self.config.model_path,
            **self.config.model_extra,
        )

    async def arequest(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        return self.request(messages)

    def request(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        completion = self._llm.create_chat_completion(
            [m.to_dict() for m in messages],
            **self.config.completion_extra,
        )
        return ProteusLLMResponse(
            message=ProteusMessage.from_any(completion["choices"][0]["message"]),
            token_cnt=completion["usage"]["completion_tokens"],
        )


__all__ = ["LlamaCppLLM"]
