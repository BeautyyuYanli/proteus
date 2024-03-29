from pathlib import Path
from typing import Optional

from proteus.config import LLMsConfig, LLMsName, ManagerConfig, PromptsConfig
from proteus.llms import BaseLLM, llm_from_config
from proteus.manager.talker_store import TalkerStore
from proteus.storages.history_store import (
    BaseHistoryStore,
    FakeHistoryStore,
    FileHistoryStore,
)
from proteus.talker import ProteusTalker


class ProteusManager:
    llms_conf: LLMsConfig
    prompts_conf: PromptsConfig
    manager_conf: ManagerConfig
    llm: BaseLLM
    talker_store: TalkerStore
    history_store: BaseHistoryStore

    def __init__(
        self,
        llms_conf: LLMsConfig,
        prompts_conf: PromptsConfig,
        manager_conf: ManagerConfig,
        llm_name: Optional[LLMsName] = None,
    ) -> None:
        self.llms_conf = llms_conf
        self.prompts_conf = prompts_conf
        self.manager_conf = manager_conf
        self.llm = llm_from_config(llms_conf, llm_name)
        self.talker_store = TalkerStore(
            manager_conf.cache_talkers_mem_capacity,
            cache_folder=Path(manager_conf.cache_folder) / "talkers"
            if manager_conf.cache_talkers_enabled
            else None,
        )
        self.history_store = (
            FileHistoryStore(Path(manager_conf.cache_folder) / "history")
            if manager_conf.cache_history_enabled
            else FakeHistoryStore()
        )

    def new_talker(self, prompt_name: str) -> str:
        talker = ProteusTalker.from_new(
            prompt_name=prompt_name,
            prompts_config=self.prompts_conf,
            llm=self.llm,
            live_history_size=self.manager_conf.live_history_size,
            save_history=self.history_store.extend,
            persist=self.talker_store.persist,
        )
        self.talker_store.append(talker)
        return talker.state.id

    def get_talker(self, proteus_id: str) -> Optional[ProteusTalker]:
        try:
            talker = self.talker_store.get(proteus_id)
            if isinstance(talker, bytes):
                talker = ProteusTalker.from_json(
                    talker,
                    self.prompts_conf,
                    self.llm,
                    self.manager_conf.live_history_size,
                    self.history_store.extend,
                    self.talker_store.persist,
                )
                self.talker_store.append(talker)
            return talker
        except KeyError:
            return None

    def save_all_talkers(self):
        self.talker_store.persist_all()
