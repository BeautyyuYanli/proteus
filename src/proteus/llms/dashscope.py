import os
from typing import List

import httpx

from proteus.config import LLMsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import ProteusLLMResponse, ProteusMessage, StructSpec


class DashScopeResponse(StructSpec):
    class Output(StructSpec):
        class Choice(StructSpec):
            message: ProteusMessage

        choices: List[Choice]

    class Usage(StructSpec):
        output_tokens: int
        input_tokens: int

    output: Output
    usage: Usage
    request_id: str


class DashScopeLLM(BaseLLM):
    def __init__(
        self,
        config: LLMsConfig.DashScopeConfig,
    ) -> None:
        self.config = config
        headers = {
            "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}",
        }
        self._aclient = httpx.AsyncClient(headers=headers, timeout=60)
        self._client = httpx.Client(headers=headers, timeout=60)

    async def arequest(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        response = await self._aclient.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            json={
                "model": self.config.model,
                "input": {"messages": [msg.to_json() for msg in messages]},
                "parameters": self.config.parameters,
            },
        )
        try:
            response.raise_for_status()
        except Exception:
            print(await response.aread())
            raise
        completion = DashScopeResponse.from_json(await response.aread())
        return ProteusLLMResponse(
            message=completion.output.choices[0].message,
            token_cnt=completion.usage.input_tokens + completion.usage.output_tokens,
        )

    def request(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        response = self._client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            json={
                "model": self.config.model,
                "input": {"messages": [msg.to_dict() for msg in messages]},
                "parameters": self.config.parameters,
            },
        )
        try:
            response.raise_for_status()
        except Exception:
            print(response.read().decode())
            raise
        completion = DashScopeResponse.from_json(response.read())
        return ProteusLLMResponse(
            message=completion.output.choices[0].message,
            token_cnt=completion.usage.input_tokens + completion.usage.output_tokens,
        )
