from typing import Dict

from proteus.config import BackendsConfig, ManagerConfig, PromptsConfig
from proteus.history_store import BaseHistoryStore
from proteus.llms import BaseLLM, llm_from_config
from proteus.talker import ProteusTalker


class TalkerStore:
    _store: Dict[str, ProteusTalker]

    def __init__(self) -> None:
        self._store = {}

    def append(self, talker: ProteusTalker) -> None:
        self._store[talker.state.id] = talker

    def get(self, proteus_id: str) -> ProteusTalker:
        return self._store[proteus_id]


class ProteusManager:
    backends_conf: BackendsConfig
    prompts_conf: PromptsConfig
    manager_conf: ManagerConfig
    llm: BaseLLM
    talker_store: TalkerStore
    history_store: BaseHistoryStore

    def __init__(
        self,
        backends_conf: BackendsConfig,
        prompts_conf: PromptsConfig,
        manager_conf: ManagerConfig,
        llm_name: str,
        history_store: BaseHistoryStore,
    ) -> None:
        self.backends_conf = backends_conf
        self.prompts_conf = prompts_conf
        self.manager_conf = manager_conf
        self.llm = llm_from_config(llm_name, backends_conf)
        self.talker_store = TalkerStore()
        self.history_store = history_store

    def new_talker(self, prompt_name: str) -> int:
        talker = ProteusTalker.from_new(
            prompt_name=prompt_name,
            prompts_config=self.prompts_conf,
            llm=self.llm,
            history_store=self.history_store,
            live_history_size=self.manager_conf.live_history_size,
        )
        self.talker_store.append(talker)
        return talker.state.id

    def get_talker(self, proteus_id: str) -> ProteusTalker:
        return self.talker_store.get(proteus_id)


__all__ = ["ProteusManager"]
