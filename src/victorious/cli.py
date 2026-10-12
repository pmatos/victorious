import argparse
import json
import os
import secrets
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
from pydantic import ValidationError

from victorious.analysis import analyse_with_observations
from victorious.cache import ObservationCache, default_cache_root
from victorious.categorize import KEY_VARIABLE, jev_from_env
from victorious.errors import SpeechError
from victorious.layout import LayoutError, dump_layout, load_layouts
from victorious.manual import serve_manual
from victorious.observations import records
from victorious.ocr import OcrError
from victorious.preview import render_preview
from victorious.schema import MaskedAudioRecord, json_schema
from victorious.setup import AT_KINDS, SetupResult, TimeError, parse_at, run_setup
from victorious.speakers import DIARIZATION_REPO
from victorious.speech import DEFAULT_MODEL, SpeechOptions, require_ready
from victorious.video import RecordingError

SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("setup", "produce a Layout from a Recording"),
    ("analyse", "turn a Recording and a Layout into a list of Events"),
    ("review", "inspect the Events found in a Recording"),
    ("schema", "print the JSON schema of the Analysis output"),
)
STUBS = frozenset({"review"})


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


def _whole_number(text: str, low: int, high: int | None = None) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if high is None and value < low:
        raise argparse.ArgumentTypeError(f"must be at least {low}, got {value}")
    if high is not None and not low <= value <= high:
        raise argparse.ArgumentTypeError(f"must be between {low} and {high}, got {value}")
    return value


def _positive_int(text: str) -> int:
    return _whole_number(text, 1)


def _non_negative_int(text: str) -> int:
    return _whole_number(text, 0)


def _port(text: str) -> int:
    return _whole_number(text, 0, 65535)


def _at_argument(text: str) -> tuple[str, int]:
    try:
        return parse_at(text)
    except TimeError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


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
            sub.add_argument(
                "--layout",
                type=Path,
                required=True,
                help="a Layout file, or a directory of Layouts (YAML); the best fit is used",
            )
            sub.add_argument(
                "--jobs",
                type=_positive_int,
                default=os.cpu_count() or 1,
                help="decode and read this many chunks at once (default: one per CPU)",
            )
            sub.add_argument(
                "--chunk-seconds", type=_positive_int, default=60, help=argparse.SUPPRESS
            )
            sub.add_argument(
                "--cache-dir",
                type=Path,
                help="keep the cached Observations here (default: $XDG_CACHE_HOME/victorious)",
            )
            sub.add_argument(
                "--no-cache", action="store_true", help="neither read nor write the cache"
            )
            sub.add_argument(
                "--observations",
                type=Path,
                help="write every raw reading here as JSON lines, one per HUD element per second",
            )
            sub.add_argument(
                "--audio-stream",
                type=_non_negative_int,
                default=0,
                metavar="N",
                help="transcribe the Nth audio stream of the Recording, counting from 0 "
                "(default: 0, the first)",
            )
            sub.add_argument(
                "--no-speech",
                action="store_true",
                default=os.environ.get("VICTORIOUS_NO_SPEECH", "").lower() in {"1", "true", "yes"},
                help="skip transcription: no models are loaded and no Hugging Face token is "
                "needed (default when VICTORIOUS_NO_SPEECH is set)",
            )
            sub.add_argument(
                "--voices",
                type=Path,
                metavar="DIR",
                help="enroll known Speakers from short clips in this folder, one per person and "
                "named by file (voices/alice.wav is alice); other voices are unknown_N "
                "(default: ./voices if it exists)",
            )
            sub.add_argument("--whisper-model", default=DEFAULT_MODEL, help=argparse.SUPPRESS)
            sub.add_argument(
                "--diarization-model", default=DIARIZATION_REPO, help=argparse.SUPPRESS
            )
        if name == "setup":
            sub.add_argument("recording", type=Path, help="the Recording to read the HUD from")
            sub.add_argument(
                "--at",
                action="append",
                type=_at_argument,
                default=[],
                metavar="KIND=TIME",
                help="a second showing the in-match HUD (hud=TIME) or the results screen "
                "(results=TIME); both are required, each may repeat. TIME is seconds, "
                "MM:SS or H:MM:SS",
            )
            sub.add_argument(
                "-o", "--output", type=Path, required=True, help="write the Layout (YAML) here"
            )
            sub.add_argument(
                "--preview",
                type=Path,
                help="write the annotated preview PNG here (default: next to the Layout)",
            )
            sub.add_argument(
                "--manual",
                action="store_true",
                help="place the elements by hand on a page served locally, pre-filled with "
                "what auto-detect found",
            )
            sub.add_argument(
                "--no-browser",
                action="store_true",
                help="with --manual, only print the address instead of opening a browser",
            )
            sub.add_argument(
                "--port",
                type=_port,
                default=None,
                help="with --manual, listen on this port of 127.0.0.1 (default: a free one)",
            )
        if name in {"analyse", "schema"}:
            sub.add_argument("-o", "--output", type=Path, help="write JSON here, not to stdout")
    return parser


def _emit(text: str | bytes, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text if isinstance(text, str) else text.decode())
        return
    temporary = output.parent / f".{output.name}.{secrets.token_hex(4)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w" if isinstance(text, str) else "wb") as handle:
            handle.write(text)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_writable_directory(output: Path | None) -> None:
    if output is not None and not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")


def _require_distinct(observations: Path | None, output: Path | None) -> None:
    if (
        observations is not None
        and output is not None
        and observations.resolve() == output.resolve()
    ):
        raise OSError("--observations and --output name the same file")


def _run_analyse(args: argparse.Namespace) -> tuple[str, str | None]:
    speech = None
    categorizer = None
    if not args.no_speech:
        if args.voices is not None and not args.voices.is_dir():
            if args.voices.exists():
                raise SpeechError(
                    f"--voices {args.voices} is not a folder; give the folder of enrollment "
                    "clips or pass --no-speech to skip speech"
                )
            print(
                f"victorious analyse: warning: voices folder {args.voices} does not exist; "
                "every voice will be unknown_N",
                file=sys.stderr,
            )
        speech = SpeechOptions(
            audio_stream=args.audio_stream,
            model=args.whisper_model,
            voices=args.voices or Path("voices"),
            diarization_model=args.diarization_model,
        )
        require_ready(speech)
        categorizer = jev_from_env(os.environ)
    layouts = load_layouts(args.layout)
    cache = None if args.no_cache else ObservationCache(args.cache_dir or default_cache_root())
    started = time.monotonic()
    analysis = analyse_with_observations(
        args.recording,
        layouts,
        jobs=args.jobs,
        chunk_seconds=args.chunk_seconds,
        cache=cache,
        speech=speech,
        categorizer=categorizer,
    )
    state = "off" if cache is None else "hit" if analysis.cached else "miss"
    elapsed = time.monotonic() - started
    chosen = analysis.output.layout
    if chosen is None:
        for warning in analysis.output.warnings:
            print(
                f"victorious analyse: warning: {warning.code}: {warning.message}", file=sys.stderr
            )
    else:
        count = analysis.speech_segments
        spoken = "" if count is None else f", speech {count} segment{'s' if count != 1 else ''}"
        silenced = round(sum(window.end - window.start for window in analysis.masked))
        if silenced:
            spoken += f", music masked {silenced} s"
        if count is not None:
            spoken += f", speakers {analysis.speakers} (enrolled {analysis.enrolled})"
            if categorizer is not None:
                spoken += f", categories {analysis.categorized} of {count}"
        print(
            f"victorious analyse: layout {chosen.path} (score {chosen.score:.2f}, "
            f"{analysis.candidates} candidate{'s' if analysis.candidates != 1 else ''}), "
            f"cache {state}{spoken}, {elapsed:.1f} s",
            file=sys.stderr,
        )
        if count:
            segments = f"{count} segment{'s' if count != 1 else ''}"
            if categorizer is None:
                print(
                    f"victorious analyse: note: no {KEY_VARIABLE} set, so the categories of the "
                    f"{segments} are null",
                    file=sys.stderr,
                )
            elif analysis.categorized < count:
                print(
                    f"victorious analyse: warning: Jev did not categorize "
                    f"{count - analysis.categorized} of {segments}: {analysis.category_failure}",
                    file=sys.stderr,
                )
    text = analysis.output.model_dump_json(indent=2) + "\n"
    if args.observations is None:
        return text, None
    if analysis.layout is None:
        return text, ""
    lines = [
        record.model_dump_json()
        for record in records(analysis.observations, analysis.layout.layout)
    ]
    lines += [
        MaskedAudioRecord(
            element="music_mask",
            start=window.start,
            end=window.end,
            music=window.music,
            speech=window.speech,
        ).model_dump_json()
        for window in analysis.masked
    ]
    return text, "".join(f"{line}\n" for line in lines)


def _setup_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    layout_path: Path = args.output
    preview_path: Path = args.preview or layout_path.with_name(f"{layout_path.stem}.preview.png")
    _require_writable_directory(layout_path)
    _require_writable_directory(preview_path)
    if layout_path.resolve() == preview_path.resolve():
        raise OSError("--preview and --output name the same file")
    if args.recording.resolve() in (layout_path.resolve(), preview_path.resolve()):
        raise OSError("--output and --preview must not name the Recording")
    return layout_path, preview_path


def _write_preview(result: SetupResult, preview_path: Path) -> None:
    if result.found:
        encoded = cv2.imencode(".png", render_preview(result.panels))[1].tobytes()
        _emit(encoded, preview_path)


def _write_layout(result: SetupResult, layout_path: Path, preview_path: Path) -> None:
    """Writes the preview and the Layout of a `result` that has one, then reports on stderr."""
    if result.layout is None:
        raise ValueError("there is no Layout to write")
    _write_preview(result, preview_path)
    _emit(dump_layout(result.layout), layout_path)
    for warning in result.warnings:
        print(f"victorious setup: warning: {warning}", file=sys.stderr)
    print(
        f"victorious setup: wrote {layout_path} ({result.layout.aspect_ratio}, "
        f"{result.found} elements) and {preview_path}",
        file=sys.stderr,
    )


def _run_setup(args: argparse.Namespace) -> int:
    layout_path, preview_path = _setup_paths(args)
    result = run_setup(args.recording, args.at)
    if result.layout is None:
        _write_preview(result, preview_path)
        lines = "\n".join(f"  {name}: {reason}" for name, reason in result.missing.items())
        print(
            f"victorious setup: could not find {len(result.missing)} element"
            f"{'s' if len(result.missing) != 1 else ''}:\n{lines}\n"
            "pass other --at times that show them, or place them by hand with --manual",
            file=sys.stderr,
        )
        return 1
    _write_layout(result, layout_path, preview_path)
    return 0


def _run_manual(args: argparse.Namespace) -> int:
    layout_path, preview_path = _setup_paths(args)
    result = run_setup(args.recording, args.at)
    saved = serve_manual(
        result,
        port=args.port or 0,
        open_browser=not args.no_browser,
        announce=lambda url: print(
            f"victorious setup: serving {url} (Ctrl-C to cancel)", file=sys.stderr, flush=True
        ),
        commit=lambda drawn: _write_layout(drawn, layout_path, preview_path),
    )
    if saved is None:
        print("victorious setup: cancelled, nothing written", file=sys.stderr)
        return 1
    return 0


def _setup_arguments_error(args: argparse.Namespace) -> str | None:
    if not args.manual and (args.no_browser or args.port is not None):
        return "--no-browser and --port only apply with --manual"
    present = {kind for kind, _ in args.at}
    missing = [kind for kind in AT_KINDS if kind not in present]
    if not missing:
        return None
    hints = " and ".join(f"--at {kind}=TIME" for kind in missing)
    return f"setup needs both screen kinds, but {hints} is missing"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in STUBS:
        print(f"victorious {args.command}: not implemented yet", file=sys.stderr)
        return 1
    try:
        if args.command == "setup":
            problem = _setup_arguments_error(args)
            if problem is not None:
                print(f"victorious setup: error: {problem}", file=sys.stderr)
                return 2
            return _run_manual(args) if args.manual else _run_setup(args)
        _require_writable_directory(args.output)
        sidecar: str | None = None
        if args.command == "schema":
            text = json.dumps(json_schema(), indent=2) + "\n"
        else:
            _require_writable_directory(args.observations)
            _require_distinct(args.observations, args.output)
            text, sidecar = _run_analyse(args)
        if sidecar is not None:
            _emit(sidecar, args.observations)
        _emit(text, args.output)
    except (
        LayoutError,
        RecordingError,
        OcrError,
        SpeechError,
        ValidationError,
        OSError,
        cv2.error,
    ) as error:
        print(f"victorious {args.command}: {error}", file=sys.stderr)
        return 1
    return 0
