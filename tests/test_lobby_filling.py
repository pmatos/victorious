"""The abandoned-Match rule of the Match loop, without decoding any video."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from tests.synthetic import layout_document, write_layout
from victorious.analysis import _ScheduleRecorder, run_machine
from victorious.counters import CounterReading, lobby_filling
from victorious.layout import load_layout
from victorious.observations import SecondObservation
from victorious.schema import CounterElement
from victorious.screens import ResultsReading

Counters = dict[CounterElement, CounterReading | None]

NO_RESULTS = ResultsReading(banner=0.0, hash=0.0, digits=None)


def second(value: int | None, confidence: float = 0.9) -> Counters:
    return {} if value is None else {"players_left": CounterReading(value, confidence)}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([26, 47, 72, 93, 95, 96], True),
        ([3, 20, 40], True),
        ([80, 79, 79, 78, 78, 77], False),
        ([41, 4, 41, 4, 41], False),
        ([5, 98, 98, 98, 98, 98], False),
        ([], False),
    ],
)
def test_lobby_filling_needs_two_steep_rises_and_a_net_climb(
    values: list[int], expected: bool
) -> None:
    assert lobby_filling([second(value) for value in values]) is expected


def test_lobby_filling_ignores_unconfident_and_missing_readings() -> None:
    seconds = [second(20, 0.2), second(None), second(45, 0.2), second(70), second(95)]

    assert lobby_filling(seconds) is False


class Scripted:
    def __init__(self, readings: dict[int, int]) -> None:
        self._readings = readings
        self.calls: list[int | None] = []

    def reset(self) -> None:
        self.calls.append(None)

    def read(self, second: int) -> Counters:
        self.calls.append(second)
        return {"players_left": CounterReading(self._readings[second], 0.9)}


def observations(hud_seconds: Sequence[int], total: int) -> list[SecondObservation]:
    return [
        SecondObservation(second=s, hud=1.0 if s in hud_seconds else 0.0, results=NO_RESULTS)
        for s in range(total)
    ]


@pytest.mark.parametrize(
    ("after", "matches"),
    [([20, 45, 70, 95, 96, 97, 98, 98], 2), ([16, 16, 16, 15, 15, 15, 15, 15], 1)],
    ids=["lobby_fills", "counters_unchanged"],
)
def test_counter_schedule_does_not_depend_on_the_readings(
    tmp_path: Path, after: list[int], matches: int
) -> None:
    layout = load_layout(write_layout(tmp_path / "layout.yaml", layout_document()))
    seconds = [*range(10), *range(40, 48)]
    readings = dict(zip(seconds, [30] * 10 + after, strict=True))
    frames = observations(seconds, 48)

    recorder = _ScheduleRecorder()
    run_machine(layout, frames, recorder, None)
    scripted = Scripted(readings)
    found = run_machine(layout, frames, scripted, None)

    assert len(found) == matches
    assert scripted.calls == recorder.calls
