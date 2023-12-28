from abc import abstractmethod
from threading import Lock
from typing import Dict, List, Tuple

from proteus.spec import Message


class BaseHistoryStore:
    @abstractmethod
    def extend(self, proteus_id: str, msg: List[Message]) -> None:
        ...


class FakeHistoryStore(BaseHistoryStore):
    def extend(self, proteus_id: str, msg: List[Message]) -> None:
        pass


class MemoryHistoryStore(BaseHistoryStore):
    _store: Dict[str, Tuple[List[Message], Lock]]

    def __init__(self) -> None:
        self._store = {}

    def extend(self, proteus_id: str, msg: List[Message]) -> None:
        if proteus_id not in self._store:
            self._store[proteus_id] = ([], Lock())
        with self._store[proteus_id][1]:
            self._store[proteus_id][0].extend(msg)
