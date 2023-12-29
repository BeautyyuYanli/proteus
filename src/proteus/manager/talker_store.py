from pathlib import Path
from typing import Dict, Optional

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
    _store: Dict[str, ProteusTalker]
    _persisted: Optional[TalkerStorePersisted]

    def __init__(self, cache_folder: Optional[Path] = None) -> None:
        self._store = {}
        self._persisted = TalkerStorePersisted(cache_folder) if cache_folder else None

    def append(self, talker: ProteusTalker) -> None:
        self._store[talker._state.id] = talker

    def get(self, talker_id: str) -> ProteusTalker:
        return self._store[talker_id]

    def persist(self, talker_id: str, talker_state: bytes) -> None:
        if self._persisted:
            self._persisted.upsert(talker_id, talker_state)
