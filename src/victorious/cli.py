import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from victorious.analysis import analyse
from victorious.layout import LayoutError, load_layout
from victorious.ocr import OcrError
from victorious.schema import json_schema
from victorious.video import RecordingError

SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("setup", "produce a Layout from a Recording"),
    ("analyse", "turn a Recording and a Layout into a list of Events"),
    ("review", "inspect the Events found in a Recording"),
    ("schema", "print the JSON schema of the Analysis output"),
)
STUBS = frozenset({"setup", "review"})


class _CommandParser(argparse.ArgumentParser):
    """Subcommand parser; an unimplemented command tolerates any arguments."""

    def __init__(self, *args: Any, tolerant: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tolerant = tolerant

    def parse_known_args(  # type: ignore[override]
        self, args: Sequence[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> tuple[argparse.Namespace, list[str]]:
        parsed, extras = super().parse_known_args(args, namespace)
        return parsed, [] if self._tolerant else extras


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="victorious",
        description="Find Matches, Wins and speech in long Fortnite gameplay Recordings.",
    )
    subparsers = parser.add_subparsers(
        dest="command", metavar="<command>", required=True, parser_class=_CommandParser
    )
    for name, help_text in SUBCOMMANDS:
        sub = subparsers.add_parser(
            name, help=help_text, description=help_text, tolerant=name in STUBS
        )
        if name == "analyse":
            sub.add_argument("recording", type=Path, help="the Recording to analyse")
            sub.add_argument("--layout", type=Path, required=True, help="the Layout file (YAML)")
        if name in {"analyse", "schema"}:
            sub.add_argument("-o", "--output", type=Path, help="write JSON here, not to stdout")
    return parser


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text)
        return
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        Path(temporary).replace(output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _run_analyse(args: argparse.Namespace) -> str:
    layout = load_layout(args.layout)
    return analyse(args.recording, layout).model_dump_json(indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in STUBS:
        print(f"victorious {args.command}: not implemented yet", file=sys.stderr)
        return 1
    try:
        if args.command == "schema":
            text = json.dumps(json_schema(), indent=2) + "\n"
        else:
            text = _run_analyse(args)
        _emit(text, args.output)
    except (LayoutError, RecordingError, OcrError, ValidationError, OSError) as error:
        print(f"victorious {args.command}: {error}", file=sys.stderr)
        return 1
    return 0
