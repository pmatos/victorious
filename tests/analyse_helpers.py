"""Shared helpers for tests that run `victorious analyse` on a synthetic Recording."""

import json
import os
import shlex
import shutil
import stat
import subprocess
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


def prepare(
    tmp_path: Path, spec: RecordingSpec, document: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    """Build the Recording and write the Layout; returns (recording, layout)."""
    recording = build_recording(spec, tmp_path / "recording.mkv")
    layout = write_layout(tmp_path / "layout.yaml", document)
    return recording, layout


def analyse_text(
    recording: Path, layout: Path, *options: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return run_cli("analyse", str(recording), "--layout", str(layout), *options, env=env)


def is_selection_seek(line: str) -> bool:
    """Layout selection decodes the one frame at a second: a 1 s span at 1 fps."""
    return " -t 1 -i " in line and "-vf fps=1:" in line


class FfmpegShim:
    """An `ffmpeg` first on PATH that logs each invocation, then runs the real one."""

    def __init__(self, directory: Path, fail_when: str | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        real = shutil.which("ffmpeg")
        assert real is not None
        self.log = directory / "invocations.log"
        script = directory / "ffmpeg"
        refusal = ""
        if fail_when is not None:
            words = fail_when.split()
            label = words[words.index("-ss") + 1] if "-ss" in words else words[-1]
            refusal = (
                f'case "$*" in *{shlex.quote(fail_when)}*) '
                f'echo "chunk {label} refused" >&2; exit 1;; esac\n'
            )
        script.write_text(
            f'#!/bin/sh\necho "$*" >> {shlex.quote(str(self.log))}\n{refusal}exec {real} "$@"\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.env = {"PATH": f"{directory}{os.pathsep}{os.environ['PATH']}"}

    def lines(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def decodes(self, matching: str = "", chunks_only: bool = False) -> int:
        """Invocations containing `matching`; `chunks_only` leaves out Layout selection's seeks."""
        return sum(
            matching in line and not (chunks_only and is_selection_seek(line))
            for line in self.lines()
        )

    def chunk_decodes(self) -> int:
        """Decodes of a stretch of the Recording at one frame per second."""
        return self.decodes(matching="-vf fps=1:", chunks_only=True)

    def reset(self) -> None:
        self.log.unlink(missing_ok=True)
