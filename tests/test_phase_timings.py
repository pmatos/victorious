"""The stderr summary says where the wall time went, phase by phase."""

import re
from pathlib import Path

import pytest

from tests.analyse_helpers import analyse_text, fast, prepare
from tests.synthetic import RecordingSpec, match_shots

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

PHASES = re.compile(r"(?P<total>\d+\.\d) s \((?P<phases>[^)]*)\)$")


def with_a_jump() -> RecordingSpec:
    steps = ((0.0, 0, 0, 99), (8.3, 2, 0, 99), (14.0, 2, 1, 98))
    return fast(30, match_shots(2, steps, results_at=22, placement=3))


def summary(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if line.startswith("victorious analyse: layout")]
    assert len(lines) == 1, stderr
    return lines[0]


def phases(line: str) -> dict[str, float]:
    found = PHASES.search(line)
    assert found is not None, line
    return {
        name: float(seconds)
        for name, seconds in re.findall(r"(\w+) (\d+\.\d) s", found.group("phases"))
    }


def test_a_cold_run_times_layout_frames_and_counters(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())

    result = analyse_text(recording, layout, "--no-speech", "--no-cache")

    assert result.returncode == 0, result.stderr
    line = summary(result.stderr)
    assert set(phases(line)) == {"layout", "frames", "counters"}
    total = float(PHASES.search(line).group("total"))  # type: ignore[union-attr]
    assert sum(phases(line).values()) <= total + 0.2


def test_a_cached_run_has_no_frame_phase(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())
    analyse_text(recording, layout, "--no-speech")

    result = analyse_text(recording, layout, "--no-speech")

    assert result.returncode == 0, result.stderr
    assert "cache hit" in summary(result.stderr)
    assert set(phases(summary(result.stderr))) == {"layout"}
