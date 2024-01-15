import shutil
from pathlib import Path
from typing import List

import pytest
from psycopg_pool import ConnectionPool

from proteus.spec import ProteusMessage
from proteus.storages.history_store import (
    FileHistoryStore,
    MemoryHistoryStore,
    PGHistoryStore,
)


@pytest.fixture(scope="module")
def pg_store():
    pool = ConnectionPool(
        "postgresql://postgres:password@localhost:5432/postgres",
        open=True,
    )
    return PGHistoryStore(pool)


@pytest.fixture(scope="module")
def file_store():
    path = Path("tests/file_history_store")
    shutil.rmtree(path, ignore_errors=True)
    return FileHistoryStore(path)


@pytest.fixture(scope="module")
def memory_store():
    return MemoryHistoryStore()


@pytest.fixture(scope="module")
def conversation():
    return [
        ProteusMessage(
            role="user",
            content="Hi",
        ),
        ProteusMessage(
            role="assistant",
            content="Hello",
        ),
        ProteusMessage(
            role="user",
            content="How are you?",
        ),
        ProteusMessage(
            role="assistant",
            content="I'm fine",
        ),
    ]


def test_pg(pg_store: PGHistoryStore, conversation: List[ProteusMessage]):
    pg_store.extend("test", conversation)
    retrieved = pg_store.get_k("test", 2)
    for i, j in zip(conversation[-2:], retrieved, strict=False):
        assert i.to_json() == j.to_json()


def test_file(file_store: FileHistoryStore, conversation: List[ProteusMessage]):
    file_store.extend("test", conversation)
    retrieved = file_store.get_k("test", 2)
    for i, j in zip(conversation[-2:], retrieved, strict=False):
        assert i.to_json() == j.to_json()


def test_memory(memory_store: MemoryHistoryStore, conversation: List[ProteusMessage]):
    memory_store.extend("test", conversation)
    retrieved = memory_store.get_k("test", 2)
    for i, j in zip(conversation[-2:], retrieved, strict=False):
        assert i.to_json() == j.to_json()
