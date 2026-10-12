"""Reviewing the Matches of an analysis: load it, show each Match, save the verdicts."""

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
from pydantic import ValidationError

from victorious.labels import Labels, Verdict, labels_from
from victorious.localpage import Resource, serve_until_saved
from victorious.preview import scaled_frame
from victorious.schema import SCHEMA_VERSION, AnalysisOutput, Match
from victorious.video import RecordingError, VideoInfo, frame_at, probe

RESULTS_SETTLE = 2
JPEG_QUALITY = 85
MAX_WORKERS = 8
VERDICTS: tuple[Verdict, ...] = ("confirmed", "rejected")


class ReviewError(Exception):
    """The analysis cannot be reviewed."""


def load_analysis(path: Path) -> AnalysisOutput:
    try:
        document = json.loads(path.read_text())
    except OSError as error:
        raise ReviewError(f"cannot read analysis {path}: {error.strerror or error}") from error
    except ValueError as error:
        raise ReviewError(f"analysis {path} is not JSON: {error}") from error
    version = document.get("schema_version") if isinstance(document, dict) else None
    if version != SCHEMA_VERSION:
        raise ReviewError(
            f"analysis {path} has schema_version {version!r}, this victorious reads "
            f"{SCHEMA_VERSION!r}; run analyse again"
        )
    try:
        return AnalysisOutput.model_validate(document)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
            for problem in error.errors()
        )
        raise ReviewError(f"analysis {path} is not valid: {problems}") from error


@dataclass(frozen=True)
class MatchFrames:
    """The two frames shown for a Match: how it began and how it ended, with their seconds."""

    start_second: int
    start_jpeg: bytes
    results_second: int
    results_jpeg: bytes


def has_results(match: Match) -> bool:
    """Whether the Match ended on a results screen, not on its last HUD second."""
    return any(event.kind == "match_end" and event.confidence > 0 for event in match.events)


def _jpeg(frame: Any) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", scaled_frame(frame), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise ReviewError("cannot encode a frame as JPEG")
    return bytes(encoded.tobytes())


def _grab(recording: Path, info: VideoInfo, seconds: tuple[int, ...]) -> tuple[int, bytes]:
    """The first of `seconds` the Recording has a frame for."""
    for second in seconds:
        frame = frame_at(recording, info, second)
        if frame is not None:
            return second, _jpeg(frame)
    raise RecordingError(
        f"cannot read Recording {recording}: no frame at {seconds[-1]} s, "
        f"its header claims {info.duration:.0f} s"
    )


def _results_second(match: Match, following: Match | None) -> int:
    end = int(match.end)
    if not has_results(match):
        return end
    settled = end + RESULTS_SETTLE
    if following is not None and settled >= following.start:
        return end
    return settled


def extract_frames(recording: Path, output: AnalysisOutput) -> list[MatchFrames]:
    """Decode the start and results frame of every Match, in parallel."""
    matches = output.matches
    if not matches:
        return []
    info = probe(recording)
    followers = [*matches[1:], None]
    jobs = [
        (
            (int(match.start),),
            (_results_second(match, following), int(match.end)),
        )
        for match, following in zip(matches, followers, strict=True)
    ]
    workers = max(1, min(MAX_WORKERS, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [
            (pool.submit(_grab, recording, info, begun), pool.submit(_grab, recording, info, ended))
            for begun, ended in jobs
        ]
        return [MatchFrames(*first.result(), *second.result()) for first, second in pending]


def _last_value(match: Match, kind: str) -> int | None:
    values = [event.value for event in match.events if event.kind == kind]
    return values[-1] if values else None


def match_cards(output: AnalysisOutput, frames: list[MatchFrames]) -> list[dict[str, Any]]:
    return [
        {
            "index": match.index,
            "start": match.start,
            "end": match.end,
            "win": match.win,
            "placement": match.placement,
            "personal_eliminations": match.personal_eliminations,
            "team_eliminations": _last_value(match, "team_eliminations"),
            "players_left": _last_value(match, "players_left"),
            "unreadable": sum(event.kind == "unreadable" for event in match.events),
            "has_results": has_results(match),
            "start_frame": {
                "second": shown.start_second,
                "image": f"frame/{match.index}-start.jpg",
            },
            "results_frame": {
                "second": shown.results_second,
                "image": f"frame/{match.index}-results.jpg",
            },
        }
        for match, shown in zip(output.matches, frames, strict=True)
    ]


def _verdicts(output: AnalysisOutput, submitted: object) -> dict[int, Verdict] | list[str]:
    chosen = submitted.get("verdicts") if isinstance(submitted, dict) else None
    if not isinstance(chosen, dict):
        return ["verdicts must be an object of Match index to confirmed or rejected"]
    errors: list[str] = []
    known = {str(match.index) for match in output.matches}
    errors.extend(
        f"unknown Match {str(key)[:20]!r} in verdicts" for key in chosen if key not in known
    )
    decided: dict[int, Verdict] = {}
    for match in output.matches:
        value = chosen.get(str(match.index))
        if value is None:
            errors.append(f"no verdict for Match {match.index}")
        elif value not in VERDICTS:
            errors.append(
                f"the verdict of Match {match.index} is {str(value)[:20]!r}, "
                "expected confirmed or rejected"
            )
        else:
            decided[match.index] = value
    return errors or decided


def serve_review(
    output: AnalysisOutput,
    frames: list[MatchFrames],
    *,
    port: int,
    open_browser: bool,
    announce: Callable[[str], None],
    commit: Callable[[Labels], None],
) -> Labels | None:
    """Serve the page until the verdicts are saved, then return the labels; `None` if cancelled."""
    state = {
        "recording": Path(output.recording.path).name,
        "matches": match_cards(output, frames),
    }
    resources: dict[str, Resource] = {
        "": (
            files("victorious").joinpath("data", "review.html").read_bytes(),
            "text/html; charset=utf-8",
        ),
        "state.json": (json.dumps(state).encode(), "application/json"),
    }
    for match, shown in zip(output.matches, frames, strict=True):
        resources[f"frame/{match.index}-start.jpg"] = (shown.start_jpeg, "image/jpeg")
        resources[f"frame/{match.index}-results.jpg"] = (shown.results_jpeg, "image/jpeg")
    saved: list[Labels] = []

    def save(submitted: object) -> tuple[HTTPStatus, dict[str, Any]]:
        decided = _verdicts(output, submitted)
        if isinstance(decided, list):
            return HTTPStatus.BAD_REQUEST, {"errors": decided}
        labels = labels_from(output, decided)
        commit(labels)
        saved.append(labels)
        return HTTPStatus.OK, {"saved": len(labels.matches)}

    done = serve_until_saved(
        resources, "labels", save, port=port, open_browser=open_browser, announce=announce
    )
    return saved[0] if done and saved else None
