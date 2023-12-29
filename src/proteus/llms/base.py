from abc import ABC, abstractmethod
from typing import List

from proteus.config import LLMsConfig
from proteus.spec import ProteusLLMResponse, ProteusMessage


class BaseLLM(ABC):
    @abstractmethod
    def __init__(
        self,
        config: LLMsConfig,
    ) -> None:
        ...

    @abstractmethod
    async def arequest(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        ...

    @abstractmethod
    def request(self, messages: List[ProteusMessage]) -> ProteusLLMResponse:
        ...
