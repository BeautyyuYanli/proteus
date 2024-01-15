from typing import List

import replicate

from proteus.config import LLMsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import ProteusLLMResponse, ProteusMessage


class ReplicateMixtralInsLLM(BaseLLM):
    def __init__(
        self,
        config: LLMsConfig.ReplicateMixtralInsConfig,
    ) -> None:
        self.config = config

    async def arequest(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        return self.request(messages)

    def request(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        prompt = ""
        last_role = "assistant"
        for msg in messages:
            if msg.role != last_role and "assistant" in [msg.role, last_role]:
                if msg.role == "assistant":
                    prompt += "[/INST]\n"
                else:
                    prompt += "[INST]\n"

            prompt += msg.content + "\n"
            last_role = msg.role
        prompt += "[/INST]\n"

        response = replicate.run(
            "mistralai/mixtral-8x7b-instruct-v0.1",
            input={
                **self.config.to_dict(),
                "prompt_template": "<s>{prompt}",
                "prompt": prompt,
            },
        )
        return ProteusLLMResponse(
            message=ProteusMessage(
                role="assistant",
                content="".join(response),
            )
        )
