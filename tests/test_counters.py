"""Behaviour of the HUD counters in `victorious analyse`, observed through its JSON output."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.analyse_helpers import STEADY, analyse, small
from tests.synthetic import (
    Hud,
    Results,
    Shot,
    build_recording,
    layout_document,
    match_shots,
    write_layout,
)
from victorious import counters
from victorious.analysis import analyse as analyse_recording
from victorious.counters import CounterReader, CounterReading, CounterTracker
from victorious.layout import load_layout, load_layouts
from victorious.ocr import DigitReading, read_digits
from victorious.schema import CounterElement

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

COUNTERS = ("personal_eliminations", "team_eliminations", "players_left")


def counter_events(match: dict[str, Any], kind: str | None = None) -> list[dict[str, Any]]:
    kinds = (kind,) if kind else (*COUNTERS, "unreadable")
    return [event for event in match["events"] if event["kind"] in kinds]


def changes(match: dict[str, Any], kind: str) -> list[tuple[float, int]]:
    return [(event["time"], event["value"]) for event in counter_events(match, kind)]


def test_counter_changes_are_events_at_the_second_they_change(tmp_path: Path) -> None:
    steps = (
        (0.0, 0, 0, 100),
        (10.0, 1, 0, 100),
        (14.0, 1, 1, 99),
        (20.0, 2, 1, 99),
    )
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = analyse(tmp_path, small(50, shots))

    (match,) = output["matches"]
    assert changes(match, "personal_eliminations") == [(15.0, 1), (25.0, 2)]
    assert changes(match, "team_eliminations") == [(19.0, 1)]
    assert changes(match, "players_left") == [(19.0, 99)]
    for event in counter_events(match):
        assert event["source"] == "frame"
        assert 0.0 < event["confidence"] <= 1.0
        assert event["end"] is None
        assert event["element"] is None


def test_unchanged_counters_produce_no_events(tmp_path: Path) -> None:
    shots = match_shots(5, STEADY, results_at=35, placement=3)

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert counter_events(match) == []


def test_a_counter_that_changes_once_is_reported_once(tmp_path: Path) -> None:
    steps = ((0.0, 0, 0, 99), (12.0, 0, 0, 98))
    shots = match_shots(5, steps, results_at=35, placement=3)

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert changes(match, "players_left") == [(17.0, 98)]
    assert counter_events(match, "personal_eliminations") == []
    assert counter_events(match, "team_eliminations") == []


def test_players_left_that_reads_another_value_for_two_seconds_is_not_an_event(
    tmp_path: Path,
) -> None:
    steps = ((0.0, 0, 0, 99), (12.0, 0, 0, 98), (14.0, 0, 0, 99))
    shots = match_shots(5, steps, results_at=35, placement=3)

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert changes(match, "players_left") == []


def test_players_left_multi_step_misread_is_dropped_and_a_later_fall_still_resolved(
    tmp_path: Path,
) -> None:
    steps = ((0.0, 0, 0, 99), (12.0, 0, 0, 96), (14.0, 0, 0, 99), (22.0, 0, 0, 97))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = analyse(tmp_path, small(50, shots))

    (match,) = output["matches"]
    assert [value for _, value in changes(match, "players_left")] == [98, 97]


def test_players_left_misread_after_a_real_fall_is_dropped_and_the_fall_kept(
    tmp_path: Path,
) -> None:
    steps = ((0.0, 0, 0, 99), (8.0, 0, 0, 98), (18.0, 0, 0, 99), (20.0, 0, 0, 98))
    shots = match_shots(5, steps, results_at=35, placement=3)

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert changes(match, "players_left") == [(13.0, 98)]


def test_players_left_that_returns_after_a_long_run_is_kept(tmp_path: Path) -> None:
    steps = ((0.0, 0, 0, 99), (8.0, 0, 0, 98), (20.0, 0, 0, 99))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = analyse(tmp_path, small(50, shots))

    (match,) = output["matches"]
    assert changes(match, "players_left") == [(13.0, 98), (25.0, 99)]


def test_players_left_rising_while_the_lobby_fills_is_kept(tmp_path: Path) -> None:
    steps = ((0.0, 0, 0, 96), (10.0, 0, 0, 97), (20.0, 0, 0, 98))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = analyse(tmp_path, small(50, shots))

    (match,) = output["matches"]
    assert changes(match, "players_left") == [(15.0, 97), (25.0, 98)]


def test_a_personal_count_that_reads_another_value_for_two_seconds_is_not_an_event(
    tmp_path: Path,
) -> None:
    steps = ((0.0, 2, 0, 99), (12.0, 9, 0, 99), (14.0, 2, 0, 99), (22.0, 3, 0, 99))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = analyse(tmp_path, small(50, shots))

    (match,) = output["matches"]
    assert changes(match, "personal_eliminations") == [(27.0, 3)]


def test_a_misread_jump_that_returns_is_dropped_with_the_steps_it_was_resolved_into(
    tmp_path: Path,
) -> None:
    steps = ((0.0, 7, 0, 99), (12.0, 9, 0, 99), (14.0, 7, 0, 99), (22.0, 8, 0, 99))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = analyse(tmp_path, small(50, shots))

    (match,) = output["matches"]
    assert changes(match, "personal_eliminations") == [(27.0, 8)]


def _weaken(monkeypatch: pytest.MonkeyPatch, text: str, confidence: float) -> None:
    real = read_digits

    def read(mask: np.ndarray) -> DigitReading | None:
        reading = real(mask)
        if reading is not None and reading.text == text:
            return DigitReading(text, confidence)
        return reading

    monkeypatch.setattr(counters, "read_digits", read)


def test_a_low_confidence_fall_of_an_eliminations_counter_is_not_an_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _weaken(monkeypatch, "12", 0.23)
    steps = ((0.0, 0, 13, 99), (10.0, 0, 12, 99), (22.0, 0, 13, 99), (30.0, 0, 18, 99))
    shots = match_shots(5, steps, results_at=45, placement=3)
    recording = build_recording(small(55, shots), tmp_path / "recording.mkv")
    layout = load_layouts(write_layout(tmp_path / "layout.yaml"))

    (match,) = analyse_recording(recording, layout).matches

    team = [(e.time, e.value) for e in match.events if e.kind == "team_eliminations"]
    assert team == [(35.0, 18)]


def test_a_confident_fall_of_an_eliminations_counter_is_kept(tmp_path: Path) -> None:
    steps = ((0.0, 8, 0, 99), (12.0, 0, 0, 99))
    shots = match_shots(5, steps, results_at=40, placement=3)

    output = analyse(tmp_path, small(50, shots))

    (match,) = output["matches"]
    assert changes(match, "personal_eliminations") == [(17.0, 0)]


def test_counter_events_sit_between_match_start_and_match_end_in_time_order(
    tmp_path: Path,
) -> None:
    steps = ((0.0, 0, 0, 99), (12.0, 1, 0, 98))
    shots = match_shots(5, steps, results_at=35, placement=3)

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    kinds = [event["kind"] for event in match["events"]]
    assert kinds[0] == "match_start"
    assert kinds[-1] == "match_end"
    times = [event["time"] for event in match["events"]]
    assert times == sorted(times)


def test_one_corrupted_frame_is_skipped_without_an_event(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 15),
        Shot(Hud(corrupt=True), 15, 16),
        Shot(Hud(0, 0, 99), 16, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert counter_events(match) == []


def test_a_change_survives_a_corrupted_frame_on_either_side(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 19),
        Shot(Hud(corrupt=True), 19, 20),
        Shot(Hud(1, 0, 99), 20, 21),
        Shot(Hud(corrupt=True), 21, 22),
        Shot(Hud(1, 0, 99), 22, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert changes(match, "personal_eliminations") == [(20.0, 1)]
    assert counter_events(match, "unreadable") == []


def test_one_wrong_but_plausible_reading_is_not_an_event(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(7, 0, 99), 5, 6),
        Shot(Hud(0, 0, 99), 6, 15),
        Shot(Hud(5, 0, 99), 15, 16),
        Shot(Hud(0, 0, 99), 16, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert counter_events(match) == []
    assert match["personal_eliminations"] == 0


def test_a_counter_missing_for_over_ten_seconds_is_one_unreadable_event_with_its_span(
    tmp_path: Path,
) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 10),
        Shot(Hud(None, 0, 99), 10, 22),
        Shot(Hud(0, 0, 99), 22, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    (event,) = counter_events(match)
    assert event["kind"] == "unreadable"
    assert event["element"] == "personal_eliminations"
    assert (event["time"], event["end"]) == (10.0, 22.0)
    assert event["value"] is None
    assert event["confidence"] == 0.0
    assert event["source"] == "frame"


@pytest.mark.parametrize(("seconds", "expected"), [(8, 0), (10, 0), (11, 1)])
def test_only_spans_longer_than_ten_seconds_are_unreadable(
    tmp_path: Path, seconds: int, expected: int
) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 10),
        Shot(Hud(None, 0, 99), 10, 10 + seconds),
        Shot(Hud(0, 0, 99), 10 + seconds, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert len(counter_events(match, "unreadable")) == expected


def test_all_counters_corrupted_for_over_ten_seconds_gives_one_unreadable_each(
    tmp_path: Path,
) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 10),
        Shot(Hud(corrupt=True), 10, 22),
        Shot(Hud(0, 0, 99), 22, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    unreadable = counter_events(match, "unreadable")
    assert sorted(e["element"] for e in unreadable) == sorted(COUNTERS)
    assert all((e["time"], e["end"]) == (10.0, 22.0) for e in unreadable)
    assert [e["kind"] for e in counter_events(match) if e["kind"] != "unreadable"] == []


def test_a_change_across_an_unreadable_span_is_reported_at_the_first_reading_after_it(
    tmp_path: Path,
) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 10),
        Shot(Hud(None, 0, 99), 10, 22),
        Shot(Hud(2, 0, 99), 22, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    assert changes(match, "personal_eliminations") == [(22.0, 2)]
    assert len(counter_events(match, "unreadable")) == 1


def test_a_span_still_open_when_the_match_ends_is_unreadable_up_to_the_end(
    tmp_path: Path,
) -> None:
    shots = (
        Shot(Hud(0, 0, 99), 5, 10),
        Shot(Hud(None, 0, 99), 10, 35),
        Shot(Results(placement=3), 35, 40),
    )

    output = analyse(tmp_path, small(45, shots))

    (match,) = output["matches"]
    (event,) = counter_events(match)
    assert (event["time"], event["end"]) == (10.0, 35.0)


def test_final_personal_eliminations_are_reported_per_match(tmp_path: Path) -> None:
    first = ((0.0, 0, 0, 99), (8.0, 1, 0, 80), (14.0, 3, 1, 50))
    second = ((0.0, 0, 0, 94), (9.0, 1, 0, 93))
    shots = (
        *match_shots(5, first, results_at=30, placement=2),
        *match_shots(50, second, results_at=75, placement=9),
    )

    output = analyse(tmp_path, small(85, shots))

    first_match, second_match = output["matches"]
    assert first_match["personal_eliminations"] == 3
    assert second_match["personal_eliminations"] == 1
    assert changes(second_match, "personal_eliminations") == [(59.0, 1)]
    assert [e["time"] for e in counter_events(second_match)] == [59.0, 59.0]


def test_match_without_a_personal_eliminations_rect_reports_null(tmp_path: Path) -> None:
    document = layout_document()
    del document["screens"]["hud"]["personal_eliminations"]  # type: ignore[index]
    steps = ((0.0, 0, 0, 99), (8.0, 1, 0, 80))
    shots = match_shots(5, steps, results_at=30, placement=2)

    output = analyse(tmp_path, small(40, shots), document)

    (match,) = output["matches"]
    assert match["personal_eliminations"] is None
    assert counter_events(match, "personal_eliminations") == []
    assert changes(match, "players_left") == [(13.0, 80)]


def test_counters_not_in_the_layout_are_never_unreadable(tmp_path: Path) -> None:
    document = layout_document()
    for name in COUNTERS:
        del document["screens"]["hud"][name]  # type: ignore[index]
    shots = (
        Shot(Hud(0, 0, 99), 5, 10),
        Shot(Hud(corrupt=True), 10, 25),
        Shot(Results(placement=3), 25, 30),
    )

    output = analyse(tmp_path, small(35, shots), document)

    (match,) = output["matches"]
    assert counter_events(match) == []
    assert match["personal_eliminations"] is None


def test_a_reading_from_before_an_unreadable_span_does_not_pair_with_one_after_it() -> None:
    def reading(value: int) -> dict[CounterElement, CounterReading | None]:
        return {"personal_eliminations": CounterReading(value, 0.9)}

    tracker = CounterTracker(("personal_eliminations",))
    for second in range(5):
        tracker.observe(second, reading(0))
    tracker.observe(5, reading(1))
    for second in range(6, 20):
        tracker.observe(second, {"personal_eliminations": None})
    tracker.observe(20, reading(1))

    assert [event.kind for event in tracker.close(30).events] == ["unreadable"]


def _players_left_events(values: list[int]) -> list[tuple[float, int | None]]:
    tracker = CounterTracker(("players_left",))
    for second, value in enumerate(values):
        tracker.observe(float(second), {"players_left": CounterReading(value, 0.9)})
    return [(event.time, event.value) for event in tracker.close(float(len(values))).events]


def test_a_players_left_return_exactly_at_the_window_is_dropped() -> None:
    values = [99] * 5 + [98] * 5 + [99] * 5

    assert _players_left_events(values) == []


def test_a_players_left_return_just_past_the_window_is_kept() -> None:
    values = [99] * 5 + [98] * 6 + [99] * 5

    assert _players_left_events(values) == [(5.0, 98), (11.0, 99)]


def test_a_players_left_misread_through_two_wrong_values_is_dropped_whole() -> None:
    values = [99] * 5 + [98] * 2 + [97] * 2 + [99] * 5

    assert _players_left_events(values) == []


def test_a_real_players_left_fall_before_a_misread_chain_is_kept() -> None:
    values = [100] * 5 + [99] * 10 + [98] * 2 + [97] * 2 + [99] * 5

    assert _players_left_events(values) == [(5.0, 99)]


def test_a_weak_return_to_the_value_just_left_drops_the_misread_it_undoes() -> None:
    tracker = CounterTracker(("personal_eliminations",))
    readings = [(6, 0.9)] * 5 + [(8, 0.3)] * 2 + [(6, 0.3)] * 2 + [(6, 0.9)] * 3
    for second, (value, confidence) in enumerate(readings):
        tracker.observe(float(second), {"personal_eliminations": CounterReading(value, confidence)})

    assert tracker.close(float(len(readings))).events == []


def _counter_events(
    element: CounterElement, readings: list[tuple[int, float]]
) -> list[tuple[float, int | None]]:
    tracker = CounterTracker((element,))
    for second, (value, confidence) in enumerate(readings):
        tracker.observe(float(second), {element: CounterReading(value, confidence)})
    return [(event.time, event.value) for event in tracker.close(float(len(readings))).events]


def test_an_eliminations_return_exactly_at_the_window_is_dropped() -> None:
    readings = [(3, 0.9)] * 5 + [(4, 0.9)] * 5 + [(3, 0.9)] * 5

    assert _counter_events("team_eliminations", readings) == []


def test_an_eliminations_return_just_past_the_window_is_kept() -> None:
    readings = [(3, 0.9)] * 5 + [(4, 0.9)] * 6 + [(3, 0.9)] * 5

    assert _counter_events("team_eliminations", readings) == [(5.0, 4), (11.0, 3)]


def test_a_low_confidence_players_left_fall_is_kept() -> None:
    readings = [(99, 0.9)] * 5 + [(98, 0.3)] * 10

    assert _counter_events("players_left", readings) == [(5.0, 98)]


def test_a_low_confidence_rise_of_an_eliminations_counter_is_kept() -> None:
    readings = [(3, 0.9)] * 5 + [(4, 0.3)] * 10

    assert _counter_events("team_eliminations", readings) == [(5.0, 4)]


def test_a_fall_read_at_low_then_high_confidence_is_timed_at_the_confident_run() -> None:
    readings = [(5, 0.9)] * 5 + [(3, 0.3)] * 2 + [(3, 0.9)] * 4

    assert _counter_events("team_eliminations", readings) == [(7.0, 3)]


def _read_players_left(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    by_mask: dict[int, DigitReading | None],
) -> CounterReading | None:
    masks = [np.full((4, 4), index, np.uint8) for index in range(len(by_mask))]
    monkeypatch.setattr(counters, "counter_masks", lambda frame, rect: masks)
    monkeypatch.setattr(counters, "read_digits", lambda mask: by_mask[int(mask[0, 0])])
    layout = load_layout(write_layout(tmp_path / "layout.yaml"))
    frame = np.zeros((1080, 1920, 3), np.uint8)
    return CounterReader().read(frame, layout)["players_left"]


def test_a_reading_with_a_leading_zero_is_unreadable_not_a_smaller_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reading = _read_players_left(monkeypatch, tmp_path, {0: DigitReading("05", 0.4)})

    assert reading is None


def test_a_leading_zero_reading_does_not_hide_a_valid_reading_from_another_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reading = _read_players_left(
        monkeypatch, tmp_path, {0: DigitReading("04", 0.23), 1: DigitReading("64", 0.19)}
    )

    assert reading == CounterReading(64, 0.19)


def test_a_lone_zero_is_a_valid_reading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reading = _read_players_left(monkeypatch, tmp_path, {0: DigitReading("0", 0.3)})

    assert reading == CounterReading(0, 0.3)
