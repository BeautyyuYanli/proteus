from abc import abstractmethod
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

from proteus.spec import ProteusMessage


class BaseHistoryStore:
    @abstractmethod
    def extend(self, proteus_id: str, msg: List[ProteusMessage]) -> None:
        ...


class FakeHistoryStore(BaseHistoryStore):
    def extend(self, proteus_id: str, msg: List[ProteusMessage]) -> None:
        pass


class MemoryHistoryStore(BaseHistoryStore):
    _store: Dict[str, Tuple[List[ProteusMessage], Lock]]

    def __init__(self) -> None:
        self._store = {}

    def extend(self, proteus_id: str, msg: List[ProteusMessage]) -> None:
        if proteus_id not in self._store:
            self._store[proteus_id] = ([], Lock())
        with self._store[proteus_id][1]:
            self._store[proteus_id][0].extend(msg)


class FileHistoryStore(BaseHistoryStore):
    _cache_folder: Path

    def __init__(self, cache_folder: Path) -> None:
        self._cache_folder = cache_folder
        cache_folder.mkdir(parents=True, exist_ok=True)

    def extend(self, proteus_id: str, msg: List[ProteusMessage]) -> None:
        file_path = self._cache_folder / f"{proteus_id}.jsonl"
        with file_path.open("ab") as file:
            file.write(b"\n".join([m.to_json() for m in msg]) + b"\n")
