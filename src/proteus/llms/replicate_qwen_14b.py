from typing import List

import replicate

from proteus.config import LLMsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import ProteusLLMResponse, ProteusMessage


class ReplicateQwen14LLM(BaseLLM):
    def __init__(
        self,
        config: LLMsConfig.ReplicateMixtralInsConfig,
    ) -> None:
        self.config = config

    async def arequest(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        return self.request(messages)

    def request(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        response = replicate.run(
            "nomagick/qwen-14b-chat:f9e1ed25e2073f72ff9a3f46545d909b1078e674da543e791dec79218072ae70",
            input={
                **self.config.to_dict(),
                "prompt": "\n".join(
                    [
                        f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>"
                        for msg in messages
                    ]
                )
                + "\n<|im_start|>assistant\n",
            },
        )
        return ProteusLLMResponse(
            message=ProteusMessage(
                role="assistant",
                content="".join(response),
            )
        )
