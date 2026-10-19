"""Cutting the Win Matches of an analysis out of its Recording and joining them into one video."""

import secrets
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from victorious.schema import AnalysisOutput
from victorious.video import tool

DEFAULT_TAIL = 8.0
VIDEO_ARGS = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p")
AUDIO_ARGS = ("-c:a", "aac", "-b:a", "192k")
COPY_ARGS = ("-c", "copy", "-avoid_negative_ts", "make_zero")


class ExtractError(Exception):
    """The Wins cannot be extracted."""


@dataclass(frozen=True)
class Clip:
    """The span of a Recording that holds one Win, from its Match start to past its results."""

    index: int
    start: float
    end: float


def win_clips(output: AnalysisOutput, tail: float, duration: float) -> list[Clip]:
    """One Clip per Win Match, ending `tail` s after the results screen appears.

    A Match ends on its first results second, so without the tail the Victory Royale screen would
    be cut off. The tail never runs into the next Match or past the end of the Recording.
    """
    matches = sorted(output.matches, key=lambda match: match.start)
    clips = []
    for position, match in enumerate(matches):
        if not match.win:
            continue
        limit = duration
        if position + 1 < len(matches):
            limit = min(limit, matches[position + 1].start)
        end = min(match.end + tail, limit)
        clips.append(Clip(match.index, match.start, max(end, match.end)))
    return clips


def _run(command: list[str], what: str) -> None:
    result = subprocess.run(
        command, capture_output=True, encoding="utf-8", errors="replace", check=False
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"ffmpeg exited with status {result.returncode}"
        raise ExtractError(f"cannot {what}: {detail}")


def _listing_line(path: Path) -> str:
    return "file '{}'\n".format(str(path).replace("'", "'\\''"))


def extract_clips(
    recording: Path,
    clips: list[Clip],
    output: Path,
    progress: Callable[[int, Clip], None] = lambda position, clip: None,
    *,
    exact: bool = False,
) -> None:
    """Cuts every Clip of `recording` and joins them into `output`, written whole or not at all.

    The streams are copied, so a cut starts at the keyframe at or before its start second.
    `exact` re-encodes them instead, so each starts on its exact second, at the cost of the time
    to encode the footage.
    """
    if not clips:
        raise ExtractError("there is no Clip to extract")
    if not output.suffix:
        raise ExtractError(f"give the output a file extension such as .mkv: {output}")
    ffmpeg = tool("ffmpeg")
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as scratch:
        cut: list[Path] = []
        for position, clip in enumerate(clips):
            progress(position, clip)
            part = Path(scratch) / f"clip-{position}.mkv"
            _run(
                [
                    ffmpeg, "-v", "error", "-nostdin", "-y",
                    "-ss", f"{clip.start:g}", "-t", f"{clip.end - clip.start:g}",
                    "-i", str(recording), "-map", "0:v:0", "-map", "0:a?",
                    *(VIDEO_ARGS + AUDIO_ARGS if exact else COPY_ARGS), str(part),
                ],
                f"cut Match {clip.index} out of {recording}",
            )  # fmt: skip
            cut.append(part)
        listing = Path(scratch) / "clips.txt"
        listing.write_text("".join(_listing_line(part) for part in cut), encoding="utf-8")
        joined = Path(scratch) / f"{secrets.token_hex(4)}{output.suffix}"
        _run(
            [
                ffmpeg, "-v", "error", "-nostdin", "-y",
                "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined),
            ],
            "join the clips",
        )  # fmt: skip
        joined.replace(output)
