import argparse
import sys
from collections.abc import Sequence

SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("setup", "produce a Layout from a Recording"),
    ("analyse", "turn a Recording and a Layout into a list of Events"),
    ("review", "inspect the Events found in a Recording"),
    ("schema", "print the JSON schema of the Analysis output"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="victorious",
        description="Find Matches, Wins and speech in long Fortnite gameplay Recordings.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    for name, help_text in SUBCOMMANDS:
        subparsers.add_parser(name, help=help_text, description=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = build_parser().parse_known_args(argv)
    print(f"victorious {args.command}: not implemented yet", file=sys.stderr)
    return 1
