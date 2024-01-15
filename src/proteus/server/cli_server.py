from argparse import ArgumentParser
from pathlib import Path

from proteus.config import LLMsConfig, PromptsConfig
from proteus.llms import llm_from_config
from proteus.storages.history_store import MemoryHistoryStore
from proteus.teller import ProteusTeller
from proteus.utils.logger import logger
from proteus.utils.spec import StructSpec


class Arguments(StructSpec, kw_only=True, frozen=True):
    llm_name: str
    prompt_name: str
    llms_configs: str
    prompts_configs: str
    live_history_size: int


def args_parse() -> Arguments:
    args_parser = ArgumentParser()
    args_parser = ArgumentParser(description="Proteus CLI Playground")
    args_parser.add_argument("--llm_name", type=str, default="openai_llm", nargs="?")
    args_parser.add_argument("--prompt_name", type=str, default="default", nargs="?")
    args_parser.add_argument("--live_history_size", type=int, default=0, nargs="?")
    args_parser.add_argument("--llms_configs", type=str)
    args_parser.add_argument("--prompts_configs", type=str)
    args = Arguments.from_dict(vars(args_parser.parse_args()))
    logger.debug(args)
    return args


def main():
    args = args_parse()
    teller = ProteusTeller(
        id="cli",
        llm=llm_from_config(
            LLMsConfig.from_path(Path(args.llms_configs)), args.llm_name
        ),
        prompt=PromptsConfig.from_path(Path(args.prompts_configs)).prompts[
            args.prompt_name
        ],
        history=MemoryHistoryStore(args.live_history_size),
        live_history_size=args.live_history_size,
    )
    while True:
        user_input = input("User: ")
        print(f"{args.prompt_name}: {teller.say(user_input)}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
