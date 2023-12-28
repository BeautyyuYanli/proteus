from abc import ABC, abstractmethod
from typing import List

from proteus.config import BackendsConfig
from proteus.spec import Message, LLMResponse


class BaseLLM(ABC):
    @abstractmethod
    def __init__(
        self,
        config: BackendsConfig,
    ) -> None:
        ...

    @abstractmethod
    async def request(self, messages: List[Message]) -> LLMResponse:
        ...
