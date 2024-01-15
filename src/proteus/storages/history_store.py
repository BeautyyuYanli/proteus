from abc import abstractmethod
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

from psycopg_pool import ConnectionPool

from proteus.spec import ProteusMessage


class BaseHistoryStore:
    @abstractmethod
    def extend(self, session_id: str, msg: List[ProteusMessage]) -> None:
        ...

    @abstractmethod
    def get_k(self, session_id: str, k: int) -> List[ProteusMessage]:
        ...


class FakeHistoryStore(BaseHistoryStore):
    def extend(self, session_id: str, msg: List[ProteusMessage]) -> None:
        pass

    def get_k(self, session_id: str, k: int) -> List[ProteusMessage]:
        return []


class MemoryHistoryStore(BaseHistoryStore):
    _store: Dict[str, Tuple[List[ProteusMessage], Lock]]
    _size: int

    def __init__(self, size: int = 10) -> None:
        self._store = {}
        self._size = size

    def extend(self, session_id: str, msg: List[ProteusMessage]) -> None:
        if session_id not in self._store:
            self._store[session_id] = ([], Lock())
        with self._store[session_id][1]:
            self._store[session_id][0].extend(msg)
            while len(self._store[session_id][0]) > self._size:
                self._store[session_id][0].pop(0)

    def get_k(self, session_id: str, k: int) -> List[ProteusMessage]:
        if session_id not in self._store:
            return []
        with self._store[session_id][1]:
            return self._store[session_id][0][-k:]


class FileHistoryStore(BaseHistoryStore):
    _cache_folder: Path

    def __init__(self, cache_folder: Path) -> None:
        self._cache_folder = cache_folder
        cache_folder.mkdir(parents=True, exist_ok=True)

    def extend(self, session_id: str, msg: List[ProteusMessage]) -> None:
        file_path = self._cache_folder / f"{session_id}.jsonl"
        with file_path.open("ab") as file:
            file.write(b"\n".join([m.to_json() for m in msg]) + b"\n")

    def get_k(self, session_id: str, k: int) -> List[ProteusMessage]:
        file_path = self._cache_folder / f"{session_id}.jsonl"
        with file_path.open("r") as file:
            lines = file.readlines()
            return [ProteusMessage.from_json(line) for line in lines[-k:]]


class PGHistoryStore(BaseHistoryStore):
    _conn_pool: ConnectionPool

    def __init__(self, conn_pool: ConnectionPool) -> None:
        self._conn_pool = conn_pool
        with self._conn_pool.connection() as conn:
            create_table_query = """
CREATE TABLE IF NOT EXISTS proteus_chat_history (
    MESSAGE_ID      SERIAL  PRIMARY  KEY,
    SESSION_ID      TEXT    NOT NULL,
    ROLE            TEXT    NOT NULL,
    CONTENT         TEXT    NOT NULL
); """
            conn.execute(create_table_query)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS index_session_id ON proteus_chat_history (SESSION_ID);"
            )
            conn.commit()

    def extend(self, session_id: str, msg: List[ProteusMessage]) -> None:
        with self._conn_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO proteus_chat_history (SESSION_ID, ROLE, CONTENT) VALUES (%s, %s, %s)",
                    [(session_id, m.role, m.content) for m in msg],
                )
            conn.commit()

    def get_k(self, session_id: str, k: int) -> List[ProteusMessage]:
        with self._conn_pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT ROLE, CONTENT FROM proteus_chat_history WHERE SESSION_ID = %s ORDER BY MESSAGE_ID DESC LIMIT %s",
                (session_id, k),
            )
            retrieved = [
                ProteusMessage(role=role, content=content)
                for role, content in cur.fetchall()
            ]
            retrieved.reverse()
            return retrieved
