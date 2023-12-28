from typing import Dict

from proteus.config import BackendsConfig, PromptsConfig
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
    llm: BaseLLM
    backends_conf: BackendsConfig
    prompts_conf: PromptsConfig
    talker_store: TalkerStore

    def __init__(
        self, llm_name: str, backends_conf: BackendsConfig, prompts_conf: PromptsConfig
    ) -> None:
        self.llm = llm_from_config(llm_name, backends_conf)
        self.backends_conf = backends_conf
        self.prompts_conf = prompts_conf
        self.talker_store = TalkerStore()

    def new_talker(self, prompt_name: str) -> int:
        talker = ProteusTalker.from_new(
            prompt_name,
            self.prompts_conf,
            self.llm,
        )
        self.talker_store.append(talker)
        return talker.state.id

    def get_talker(self, proteus_id: str) -> ProteusTalker:
        return self.talker_store.get(proteus_id)


__all__ = ["ProteusManager"]
