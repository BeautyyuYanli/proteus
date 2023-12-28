import asyncio
from argparse import ArgumentParser
from pathlib import Path

from proteus.config import BackendsConfig, PromptsConfig
from proteus.manager import ProteusManager
from utils.logger import logger
from utils.spec import StructSpec


class Arguments(StructSpec, kw_only=True, frozen=True):
    llm_name: str
    prompt_name: str
    backends_configs: str
    prompts_configs: str


def args_parse() -> Arguments:
    args_parser = ArgumentParser()
    args_parser = ArgumentParser(description="Proteus CLI Playground")
    args_parser.add_argument("--llm_name", type=str, default="openai_llm", nargs="?")
    args_parser.add_argument("--prompt_name", type=str, default="default", nargs="?")
    args_parser.add_argument("--backends_configs", type=str)
    args_parser.add_argument("--prompts_configs", type=str)
    args = Arguments.from_dict(vars(args_parser.parse_args()))
    logger.debug(args)
    return args


async def async_main():
    args = args_parse()
    manager = ProteusManager(
        llm_name=args.llm_name,
        backends_conf=BackendsConfig.from_path(Path(args.backends_configs)),
        prompts_conf=PromptsConfig.from_path(Path(args.prompts_configs)),
    )
    talker_id = manager.new_talker(args.prompt_name)
    talker = manager.get_talker(talker_id)
    while True:
        user_input = input("User: ")
        async for msg in talker.say(user_input):
            print(f"{talker.state.prompt_name}: {msg}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()


__all__ = ["main"]
