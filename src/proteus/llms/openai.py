from typing import List, cast

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from proteus.config import LLMsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import ProteusLLMResponse, ProteusMessage


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        config: LLMsConfig.OpenAIConfig,
    ) -> None:
        self.config = config
        self._allm = openai.AsyncOpenAI().chat.completions
        self._llm = openai.OpenAI().chat.completions

    async def arequest(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        completion: ChatCompletion = await self._allm.create(
            messages=cast(
                List[ChatCompletionMessageParam], [m.to_dict() for m in messages]
            ),
            **self.config.to_dict(),
        )
        if not isinstance(completion, ChatCompletion):
            raise TypeError("OpenAI returned an unexpected type")
        return ProteusLLMResponse(
            message=ProteusMessage.from_any(completion.choices[0].message),
            token_cnt=None
            if completion.usage is None
            else completion.usage.completion_tokens,
        )

    def request(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        completion: ChatCompletion = self._llm.create(
            messages=cast(
                List[ChatCompletionMessageParam], [m.to_dict() for m in messages]
            ),
            **self.config.to_dict(),
        )
        if not isinstance(completion, ChatCompletion):
            raise TypeError("OpenAI returned an unexpected type")
        return ProteusLLMResponse(
            message=ProteusMessage.from_any(completion.choices[0].message),
            token_cnt=None
            if completion.usage is None
            else completion.usage.completion_tokens,
        )
