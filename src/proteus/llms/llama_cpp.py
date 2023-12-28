from typing import List

from llama_cpp import Llama

from proteus.config import BackendsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import LLMResponse, Message


class LlamaCppLLM(BaseLLM):
    def __init__(
        self,
        config: BackendsConfig,
    ) -> None:
        if config.llama_cpp is None:
            raise ValueError("LlamaCpp args not set")
        self.config = config.llama_cpp
        self._llm = Llama(
            self.config.model_path,
            **self.config.model_extra,
        )

    async def arequest(self, messages: List[Message]) -> LLMResponse:
        return self.request(messages)

    def request(self, messages: List[Message]) -> LLMResponse:
        completion = self._llm.create_chat_completion(
            [m.to_dict() for m in messages],
            **self.config.completion_extra,
        )
        return LLMResponse(
            message=Message.from_any(completion["choices"][0]["message"]),
            token_cnt=completion["usage"]["completion_tokens"],
        )


__all__ = ["LlamaCppLLM"]
