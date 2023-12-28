from abc import ABC, abstractmethod
from typing import List

from proteus.config import BackendsConfig
from proteus.spec import LLMResponse, Message


class BaseLLM(ABC):
    @abstractmethod
    def __init__(
        self,
        config: BackendsConfig,
    ) -> None:
        ...

    @abstractmethod
    async def arequest(self, messages: List[Message]) -> LLMResponse:
        ...

    @abstractmethod
    def request(self, messages: List[Message]) -> LLMResponse:
        ...
