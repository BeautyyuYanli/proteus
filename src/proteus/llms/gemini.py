from typing import List

import google.generativeai as genai
from google.generativeai.types import ContentDict

from proteus.config import BackendsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import LLMResponse, Message


class GeminiLLM(BaseLLM):
    def __init__(
        self,
        config: BackendsConfig,
    ) -> None:
        if config.gemini is None:
            raise ValueError("Gemini args not set")
        self.config = config.gemini
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
        )

    def _adapt_to_gemini(self, messages: List[Message]) -> List[ContentDict]:
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

    async def arequest(self, messages: List[Message]) -> LLMResponse:
        completion = await self._llm.generate_content_async(
            self._adapt_to_gemini(messages),
        )

        return LLMResponse(
            message=Message(role="assistant", content=completion.text),
            # token_cnt=completion.candidates[0].token_count,
        )

    def request(self, messages: List[Message]) -> LLMResponse:
        completion = self._llm.generate_content(
            self._adapt_to_gemini(messages),
        )

        return LLMResponse(
            message=Message(role="assistant", content=completion.text),
            # token_cnt=completion.candidates[0].token_count,
        )


__all__ = ["GeminiLLM"]
