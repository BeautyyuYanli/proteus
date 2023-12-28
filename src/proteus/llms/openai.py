from typing import List, cast

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from proteus.config import BackendsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import LLMResponse, Message


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        config: BackendsConfig,
    ) -> None:
        if config.openai is None:
            raise ValueError("OpenAI args not set")
        self.config = config.openai
        self._allm = openai.AsyncOpenAI().chat.completions
        self._llm = openai.OpenAI().chat.completions

    async def arequest(self, messages: List[Message]) -> LLMResponse:
        completion: ChatCompletion = await self._allm.create(
            messages=cast(
                List[ChatCompletionMessageParam], [m.to_dict() for m in messages]
            ),
            **self.config.to_dict(),
        )
        if not isinstance(completion, ChatCompletion):
            raise TypeError("OpenAI returned an unexpected type")
        return LLMResponse(
            message=Message.from_any(completion.choices[0].message),
            token_cnt=None
            if completion.usage is None
            else completion.usage.completion_tokens,
        )

    async def request(self, messages: List[Message]) -> LLMResponse:
        completion: ChatCompletion = self._llm.create(
            messages=cast(
                List[ChatCompletionMessageParam], [m.to_dict() for m in messages]
            ),
            **self.config.to_dict(),
        )
        if not isinstance(completion, ChatCompletion):
            raise TypeError("OpenAI returned an unexpected type")
        return LLMResponse(
            message=Message.from_any(completion.choices[0].message),
            token_cnt=None
            if completion.usage is None
            else completion.usage.completion_tokens,
        )
