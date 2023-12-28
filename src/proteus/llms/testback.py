from typing import List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MistralForCausalLM,
    pipeline,
)

from proteus.config import BackendsConfig
from proteus.llms.base import BaseLLM
from proteus.spec import LLMResponse, Message


class TestBackLLM(BaseLLM):
    def __init__(
        self,
        config: BackendsConfig,
    ) -> None:
        if config.testback is None:
            raise ValueError("TestBack args not set")
        self.config = config.testback

        model: MistralForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            return_dict=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, use_fast=False
        )

        # Define the text generation pipeline
        self.generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )

    async def arequest(self, messages: List[Message]) -> LLMResponse:
        return self.request(messages)

    def request(self, messages: List[Message]) -> LLMResponse:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output = self.generator(
            prompt,
            num_return_sequences=1,
            return_full_text=False,
            **self.config.completion_extra,
        )[0]["generated_text"]
        return LLMResponse(
            message=Message(role="assistant", content=str(output)),
            token_cnt=None,
        )
