"""Shared helpers for tests that run `victorious analyse` on a synthetic Recording."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.cli_runner import run_cli
from tests.synthetic import RecordingSpec, Shot, build_recording, write_layout

FPS = 5.0
WIDTH = 1920
HEIGHT = 1080
STEADY = ((0.0, 0, 0, 99),)


def small(duration: float, shots: tuple[Shot, ...]) -> RecordingSpec:
    return RecordingSpec(
        duration=duration,
        shots=shots,
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        keyframe_interval=2,
    )


def fast(duration: float, shots: tuple[Shot, ...]) -> RecordingSpec:
    """A 20 fps Recording, fine enough to check times against a 0.25 s tolerance."""
    return RecordingSpec(
        duration=duration,
        shots=shots,
        fps=20.0,
        width=WIDTH,
        height=HEIGHT,
        keyframe_interval=2,
    )


def analyse(
    tmp_path: Path,
    spec: RecordingSpec,
    document: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    recording = build_recording(spec, tmp_path / "recording.mkv")
    layout = write_layout(tmp_path / "layout.yaml", document)
    result = run_cli("analyse", str(recording), "--layout", str(layout), env=env)
    assert result.returncode == 0, result.stderr
    output: dict[str, Any] = json.loads(result.stdout)
    return output
