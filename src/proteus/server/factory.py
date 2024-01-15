from typing import Optional

from psycopg_pool import ConnectionPool

from proteus.config import LLMsConfig, PromptsConfig
from proteus.llms import BaseLLM, llm_from_config
from proteus.storages.history_store import (
    PGHistoryStore,
)
from proteus.teller import ProteusTeller


class ProteusFactory:
    _llm: BaseLLM
    _prompts_conf: PromptsConfig
    _pg_conn_pool: ConnectionPool

    def __init__(
        self,
        llm_config: LLMsConfig,
        llm_name: str,
        prompts_config: PromptsConfig,
        pg_conn_pool: ConnectionPool,
    ) -> None:
        self._llm = llm_from_config(llm_config, llm_name)
        self._prompts_conf = prompts_config
        self._pg_conn_pool = pg_conn_pool

    def get_teller(
        self, id: str, prompt_name: str, live_history_size: int = 8
    ) -> Optional[ProteusTeller]:
        ProteusTeller(
            id,
            llm=self._llm,
            prompt=prompt_name,
            history=PGHistoryStore(self._pg_conn_pool),
            live_history_size=live_history_size,
        )
