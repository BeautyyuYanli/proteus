from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional, Union

from proteus.talker import ProteusTalker


class TalkerStorePersisted:
    _folder_path: Path

    def __init__(self, folder_path: Path) -> None:
        self._folder_path = folder_path

    def upsert(self, talker_id: str, talker_state: bytes) -> None:
        path = self._folder_path / f"{talker_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(talker_state)

    def get(self, talker_id: str) -> bytes:
        path = self._folder_path / f"{talker_id}.json"
        return path.read_bytes()


class TalkerStore:
    _store: OrderedDict[str, ProteusTalker]
    _store_lock: Lock
    _persisted: Optional[TalkerStorePersisted]
    _capacity: int

    def __init__(self, capacity: int, cache_folder: Optional[Path] = None) -> None:
        self._store = OrderedDict()
        self._store_lock = Lock()
        self._persisted = TalkerStorePersisted(cache_folder) if cache_folder else None
        self._capacity = capacity

    def append(self, talker: ProteusTalker) -> None:
        with self._store_lock:
            self._store[talker.state.id] = talker
            self._store.move_to_end(talker.state.id)
            while self._capacity > 0 and len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def get(self, talker_id: str) -> Union[ProteusTalker, bytes]:
        with self._store_lock:
            if talker_id not in self._store:
                if self._persisted:
                    try:
                        talker_state = self._persisted.get(talker_id)
                    except FileNotFoundError as e:
                        raise KeyError(f"Talker {talker_id} not found.") from e
                    return talker_state
                else:
                    raise KeyError(f"Talker {talker_id} not found.")

            self._store.move_to_end(talker_id)
            while self._capacity > 0 and len(self._store) > self._capacity:
                self._store.popitem(last=False)
            return self._store[talker_id]

    def persist(self, talker_id: str, talker_state: bytes) -> None:
        if self._persisted:
            self._persisted.upsert(talker_id, talker_state)
        with self._store_lock:
            if talker_id in self._store:
                self._store[talker_id].state = talker_state

    def persist_all(self) -> None:
        if self._persisted:
            with self._store_lock:
                for talker_id, talker in self._store.items():
                    self._persisted.upsert(talker_id, talker.state.to_json())
