"""Resolving counter jumps of more than one to about 0.25 s, observed through `analyse`."""

import os
import shlex
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from tests.analyse_helpers import analyse, fast
from tests.synthetic import Hud, Results, Shot, match_shots
from victorious.counters import CounterReading, Gap
from victorious.resample import resolve_jump

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

TOLERANCE = 0.25
RESAMPLE_FILTER = "fps=4:"


class FfmpegShim:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.log = directory / "invocations.log"

    def windows(self) -> list[tuple[float, float]]:
        """The (start, duration) of every 4 fps window decode, in invocation order."""
        if not self.log.exists():
            return []
        found: list[tuple[float, float]] = []
        for line in self.log.read_text().splitlines():
            arguments = shlex.split(line)
            if "-vf" not in arguments:
                continue
            if not arguments[arguments.index("-vf") + 1].startswith(RESAMPLE_FILTER):
                continue
            found.append(
                (
                    float(arguments[arguments.index("-ss") + 1]),
                    float(arguments[arguments.index("-t") + 1]),
                )
            )
        return found


@pytest.fixture
def shim(tmp_path: Path) -> FfmpegShim:
    real = shutil.which("ffmpeg")
    assert real is not None
    directory = tmp_path / "shim"
    directory.mkdir()
    shim = FfmpegShim(directory)
    script = directory / "ffmpeg"
    script.write_text(f'#!/bin/sh\necho "$*" >> {shlex.quote(str(shim.log))}\nexec {real} "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return shim


def run(tmp_path: Path, shim: FfmpegShim, duration: float, shots: Any) -> dict[str, Any]:
    env = {"PATH": f"{shim.directory}{os.pathsep}{os.environ['PATH']}"}
    return analyse(tmp_path, fast(duration, shots), env=env)


def counter(match: dict[str, Any], kind: str) -> list[tuple[float, int]]:
    return [(e["time"], e["value"]) for e in match["events"] if e["kind"] == kind]


def test_a_double_elimination_inside_one_second_gives_two_events_within_a_quarter_second(
    tmp_path: Path, shim: FfmpegShim
) -> None:
    steps = ((0.0, 0, 0, 99), (5.3, 1, 0, 99), (5.8, 2, 0, 99))
    shots = match_shots(5, steps, results_at=30, placement=3)

    output = run(tmp_path, shim, 40, shots)

    (match,) = output["matches"]
    ((first_time, first), (second_time, second)) = counter(match, "personal_eliminations")
    assert (first, second) == (1, 2)
    assert 0 <= 10.3 - first_time <= TOLERANCE
    assert 0 <= 10.8 - second_time <= TOLERANCE
    assert match["personal_eliminations"] == 2
    assert shim.windows() == [(10.0, 1.0)]


def steps_of(match: dict[str, Any], kind: str, truth: list[float]) -> list[tuple[int, float]]:
    found = counter(match, kind)
    assert len(found) == len(truth)
    return [(value, true - time) for (time, value), true in zip(found, truth, strict=True)]


@pytest.mark.parametrize(
    ("kind", "steps", "values", "truth"),
    [
        (
            "team_eliminations",
            ((0.0, 0, 0, 99), (5.3, 0, 1, 99), (5.8, 0, 2, 99)),
            [1, 2],
            [10.3, 10.8],
        ),
        (
            "players_left",
            ((0.0, 0, 0, 100), (5.1, 0, 0, 99), (5.4, 0, 0, 98), (5.65, 0, 0, 97)),
            [99, 98, 97],
            [10.1, 10.4, 10.65],
        ),
        (
            "personal_eliminations",
            ((0.0, 0, 0, 99), (5.3, 1, 0, 99), (5.4, 2, 0, 99)),
            [1, 2],
            [10.3, 10.4],
        ),
    ],
    ids=["team", "players-left-three-steps", "both-steps-in-one-cell"],
)
def test_every_step_of_a_jump_is_an_event_within_a_quarter_second(
    tmp_path: Path,
    shim: FfmpegShim,
    kind: str,
    steps: Any,
    values: list[int],
    truth: list[float],
) -> None:
    shots = match_shots(5, steps, results_at=30, placement=3)

    output = run(tmp_path, shim, 40, shots)

    (match,) = output["matches"]
    resolved = steps_of(match, kind, truth)
    assert [value for value, _ in resolved] == values
    assert all(0 <= error <= TOLERANCE for _, error in resolved)


def test_single_step_changes_trigger_no_extra_sampling(tmp_path: Path, shim: FfmpegShim) -> None:
    steps = ((0.0, 0, 0, 100), (10.0, 1, 0, 100), (14.0, 1, 1, 99), (20.0, 2, 1, 99))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = run(tmp_path, shim, 50, shots)

    (match,) = output["matches"]
    assert counter(match, "personal_eliminations") == [(15.0, 1), (25.0, 2)]
    assert counter(match, "players_left") == [(19.0, 99)]
    assert shim.windows() == []


def test_bisect_only_re_samples_the_ambiguous_gaps(tmp_path: Path, shim: FfmpegShim) -> None:
    steps = (
        (0.0, 0, 0, 99),
        (5.3, 1, 0, 99),
        (5.8, 2, 0, 99),
        (15.2, 2, 1, 99),
        (15.7, 2, 2, 99),
        (25.0, 3, 2, 99),
    )
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = run(tmp_path, shim, 50, shots)

    (match,) = output["matches"]
    assert [value for _, value in counter(match, "personal_eliminations")] == [1, 2, 3]
    assert [value for _, value in counter(match, "team_eliminations")] == [1, 2]
    assert sorted(shim.windows()) == [(10.0, 1.0), (20.0, 1.0)]


def test_an_unreadable_second_inside_the_gap_is_not_bisected(
    tmp_path: Path, shim: FfmpegShim
) -> None:
    steps = ((0.0, 0, 0, 99), (5.0, None, 0, 99), (7.0, 2, 0, 99))
    shots = match_shots(5, steps, results_at=30, placement=3)

    output = run(tmp_path, shim, 40, shots)

    (match,) = output["matches"]
    assert counter(match, "personal_eliminations") == [(12.0, 2)]
    assert shim.windows() == []


def test_a_corrupt_second_inside_the_gap_is_not_bisected(tmp_path: Path, shim: FfmpegShim) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 10),
        Shot(Hud(corrupt=True), 10, 12),
        Shot(Hud(2, 0, 99), 12, 30),
        Shot(Results(3), 30, 35),
    )

    output = run(tmp_path, shim, 40, shots)

    (match,) = output["matches"]
    assert counter(match, "personal_eliminations") == [(12.0, 2)]
    assert shim.windows() == []


def test_a_jump_after_a_long_unreadable_span_is_not_bisected(
    tmp_path: Path, shim: FfmpegShim
) -> None:
    steps = ((0.0, 0, 0, 99), (5.0, None, 0, 99), (18.0, 2, 0, 99))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = run(tmp_path, shim, 50, shots)

    (match,) = output["matches"]
    assert counter(match, "personal_eliminations") == [(23.0, 2)]
    unreadable = [e for e in match["events"] if e["kind"] == "unreadable"]
    assert [(e["element"], e["time"], e["end"]) for e in unreadable] == [
        ("personal_eliminations", 10.0, 23.0)
    ]
    assert shim.windows() == []


def test_an_unreadable_second_after_the_new_value_does_not_block_the_bisect(
    tmp_path: Path, shim: FfmpegShim
) -> None:
    steps = ((0.0, 0, 0, 99), (5.3, 1, 0, 99), (5.8, 2, 0, 99), (6.0, None, 0, 99), (7.0, 2, 0, 99))
    shots = match_shots(5, steps, results_at=30, placement=3)

    output = run(tmp_path, shim, 40, shots)

    (match,) = output["matches"]
    resolved = steps_of(match, "personal_eliminations", [10.3, 10.8])
    assert [value for value, _ in resolved] == [1, 2]
    assert all(0 <= error <= TOLERANCE for _, error in resolved)
    assert shim.windows() == [(10.0, 1.0)]


@pytest.mark.parametrize(
    ("steps", "kind", "expected"),
    [
        (((0.0, 0, 0, 80), (5.0, 0, 0, 82)), "players_left", [(10.0, 82)]),
        (((0.0, 0, 0, 99), (5.0, 0, 0, 69)), "players_left", [(10.0, 69)]),
        (((0.0, 0, 0, 99), (5.0, 6, 0, 99)), "personal_eliminations", [(10.0, 6)]),
    ],
    ids=["players-left-rising", "players-left-collapsing", "implausible-eliminations-jump"],
)
def test_implausible_jumps_stay_one_event_without_sampling(
    tmp_path: Path,
    shim: FfmpegShim,
    steps: Any,
    kind: str,
    expected: list[tuple[float, int]],
) -> None:
    shots = match_shots(5, steps, results_at=30, placement=3)

    output = run(tmp_path, shim, 40, shots)

    (match,) = output["matches"]
    assert counter(match, kind) == expected
    assert shim.windows() == []


def test_a_misread_sample_away_from_the_change_does_not_move_a_step() -> None:
    gap = Gap("personal_eliminations", 9.0, 10.0, 0, 2, 0.5)
    samples = [
        (10.0, CounterReading(0, 0.9)),
        (10.25, CounterReading(7, 0.2)),
        (10.5, CounterReading(0, 0.8)),
        (10.75, CounterReading(1, 0.8)),
    ]

    steps = resolve_jump(gap, samples)

    assert [(step.value, step.time) for step in steps] == [(1, 10.75), (2, 10.75)]


def test_an_isolated_misread_between_agreeing_samples_is_ignored() -> None:
    gap = Gap("team_eliminations", 31.0, 34.0, 2, 4, 0.5)
    samples = [
        (32.0, CounterReading(4, 0.9)),
        (32.25, CounterReading(4, 0.9)),
        (32.5, CounterReading(0, 0.2)),
        (32.75, CounterReading(4, 0.9)),
    ]

    steps = resolve_jump(gap, samples)

    assert [(step.value, step.time) for step in steps] == [(3, 32.0), (4, 32.0)]


def test_steps_are_ordered_and_stay_inside_the_window() -> None:
    gap = Gap("players_left", 9.0, 11.0, 90, 87, 0.5)
    samples = [
        (10.0, CounterReading(90, 0.9)),
        (10.25, CounterReading(89, 0.9)),
        (10.5, None),
        (10.75, CounterReading(90, 0.3)),
        (11.0, CounterReading(88, 0.9)),
    ]

    steps = resolve_jump(gap, samples)

    times = [step.time for step in steps]
    assert [step.value for step in steps] == [89, 88, 87]
    assert times == sorted(times)
    assert all(10.0 <= time <= 11.75 for time in times)
