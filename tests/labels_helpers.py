"""Hand-built analysis and labels documents for the tests of `review` and `--against`."""

import json
from pathlib import Path
from typing import Any

from victorious.schema import AnalysisOutput, Match, RecordingInfo, frame_event

RECORDING = "/videos/2026-03-01 session.mkv"
DURATION = 3600.0


def match(
    index: int,
    start: float,
    end: float,
    *,
    win: bool = False,
    placement: int | None = 6,
    eliminations: int | None = 0,
    results: bool = True,
) -> Match:
    events = [frame_event("match_start", start, 0.9)]
    if win:
        events.append(frame_event("win", end, 0.95))
    if placement is not None:
        events.append(frame_event("final_placement", end, 0.9, placement))
    events.append(frame_event("match_end", end, 0.9 if results else 0.0))
    return Match(
        index=index,
        start=start,
        end=end,
        win=win,
        placement=placement,
        personal_eliminations=eliminations,
        events=events,
    )


def analysis(*matches: Match, path: str = RECORDING, duration: float = DURATION) -> AnalysisOutput:
    return AnalysisOutput(
        recording=RecordingInfo(path=path, duration=duration, started_at=None),
        layout=None,
        warnings=[],
        matches=list(matches),
    )


def write_analysis(directory: Path, output: AnalysisOutput, name: str = "analysis.json") -> Path:
    path = directory / name
    path.write_text(output.model_dump_json(indent=2) + "\n")
    return path


def label(
    start: float,
    end: float,
    verdict: str = "confirmed",
    *,
    win: bool = False,
    placement: int | None = 6,
    eliminations: int | None = 0,
) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "verdict": verdict,
        "win": win,
        "placement": placement,
        "personal_eliminations": eliminations,
    }


def write_labels(
    directory: Path,
    *labels: dict[str, Any],
    path: str = RECORDING,
    duration: float = DURATION,
    labels_version: str = "1.0",
    name: str = "labels.json",
) -> Path:
    document = {
        "labels_version": labels_version,
        "recording": {"path": path, "duration": duration},
        "analysis": {"schema_version": "1.7", "layout": None},
        "matches": list(labels),
    }
    target = directory / name
    target.write_text(json.dumps(document, indent=2) + "\n")
    return target
