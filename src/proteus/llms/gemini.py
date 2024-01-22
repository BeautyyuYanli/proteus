from typing import List

import google.generativeai as genai
from google.generativeai.types import ContentDict

from proteus.config import LLMsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import ProteusLLMResponse, ProteusMessage


class GeminiLLM(BaseLLM):
    def __init__(
        self,
        config: LLMsConfig.GeminiConfig,
    ) -> None:
        self.config = config
        self._llm = genai.GenerativeModel(
            model_name=self.config.model_name,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
            generation_config=genai.GenerationConfig(**self.config.generation_config),
        )

    def _adapt_to_gemini(self, messages: List[ProteusMessage]) -> List[ContentDict]:
        contents: List[ContentDict] = [
            {
                "parts": [m.content],
                "role": "model" if m.role == "assistant" else "user",
            }
            for m in messages
        ]
        new_contents = [contents[0]]
        for i in contents[1:]:
            if i["role"] == new_contents[-1]["role"]:
                new_contents[-1]["parts"][0] += "\n\n" + i["parts"][0]
            else:
                new_contents.append(i)
        return new_contents

    async def arequest(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        completion = await self._llm.generate_content_async(
            self._adapt_to_gemini(messages),
        )

        return ProteusLLMResponse(
            message=ProteusMessage(role="assistant", content=completion.text),
            # token_cnt=completion.candidates[0].token_count,
        )

    def request(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        completion = self._llm.generate_content(
            self._adapt_to_gemini(messages),
        )

        return ProteusLLMResponse(
            message=ProteusMessage(role="assistant", content=completion.text),
            # token_cnt=completion.candidates[0].token_count,
        )


__all__ = ["GeminiLLM"]
