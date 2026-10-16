"""The second HUD design: the three counters in a horizontal row under the minimap."""

from pathlib import Path
from typing import Any

import pytest

from tests.analyse_helpers import FPS, STEADY, analyse, analyse_facts, setup_at, small
from tests.synthetic import (
    ELEMENT_RECTS,
    Hud,
    RecordingSpec,
    Results,
    Shot,
    build_recording,
    layout_document,
    match_shots,
    write_layout,
)
from tests.synthetic.render import ROW_ELEMENT_RECTS, render_scene
from tests.synthetic.spec import HudDesign
from victorious.layout import load_layout
from victorious.setup import Panel, SetupResult, manual_result

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

HUD_ELEMENTS = ("hud_anchor", "personal_eliminations", "team_eliminations", "players_left")
ROW_STEPS = (
    (0.0, 0, 0, 100),
    (6.0, 1, 0, 97),
    (10.0, 1, 12, 95),
    (14.0, 10, 12, 8),
    (18.0, 10, 25, 6),
)


def counter_changes(match: dict[str, Any], kind: str) -> list[int]:
    return [event["value"] for event in match["events"] if event["kind"] == kind]


def test_counter_row_counters_are_read_while_the_row_moves_with_the_digits(
    tmp_path: Path,
) -> None:
    shots = match_shots(2, ROW_STEPS, results_at=24, placement=1, hud_design="counter_row")

    output = analyse(tmp_path, small(34, shots), layout_document("counter_row"))

    (match,) = output["matches"]
    assert (match["start"], match["end"], match["win"], match["placement"]) == (2.0, 24.0, True, 1)
    assert match["personal_eliminations"] == 10
    assert counter_changes(match, "personal_eliminations") == [1, 10]
    assert counter_changes(match, "team_eliminations") == [12, 25]
    left = counter_changes(match, "players_left")
    assert left == sorted(left, reverse=True)
    assert {97, 95, 8, 6} <= set(left)
    assert left[-1] == 6
    assert output["layout"]["score"] >= 0.7


def test_counter_row_match_ends_on_an_eliminated_results_screen(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(0, 0, 100, design="counter_row"), 2, 10),
        Shot(Hud(2, 1, 40, design="counter_row"), 10, 20),
        Shot(Results(placement=5, style="placed"), 20, 25),
    )

    output = analyse(tmp_path, small(30, shots), layout_document("counter_row"))

    (match,) = output["matches"]
    assert (match["win"], match["placement"], match["personal_eliminations"]) == (False, 5, 2)


def row_scenario() -> RecordingSpec:
    steps = ((0, 0, 0, 99), (6, 1, 0, 95), (10, 1, 1, 90))
    shots = (
        *match_shots(2, steps, 20, placement=1, hud_design="counter_row"),
        *match_shots(
            30, STEADY, results_at=45, placement=6, results_style="placed", hud_design="counter_row"
        ),
    )
    return RecordingSpec(
        duration=55, shots=shots, fps=FPS, width=1920, height=1080, keyframe_interval=2
    )


def test_setup_finds_the_counter_row_and_writes_a_layout_analyse_reads_like_a_hand_written_one(
    tmp_path: Path,
) -> None:
    recording = build_recording(row_scenario(), tmp_path / "recording.mkv")
    layout_path = tmp_path / "layout.yaml"

    result = setup_at(recording, layout_path, "hud=14", "results=21", "results=46")

    assert result.returncode == 0, result.stderr
    layout = load_layout(layout_path)
    assert layout.hud_design == "counter_row"
    hand_written = write_layout(tmp_path / "hand.yaml", layout_document("counter_row"))
    from_setup = analyse_facts(recording, layout_path)
    assert from_setup == analyse_facts(recording, hand_written)
    assert [fact[:5] for fact in from_setup] == [(2.0, 20.0, True, 1, 1), (30.0, 45.0, False, 6, 0)]
    assert any(kind == "players_left" for *_, events in from_setup for kind, *_ in events)


def test_setup_keeps_the_icon_column_for_a_recording_that_has_one(tmp_path: Path) -> None:
    shots = match_shots(2, STEADY, results_at=20, placement=1)
    recording = build_recording(small(30, shots), tmp_path / "recording.mkv")
    layout_path = tmp_path / "layout.yaml"

    result = setup_at(recording, layout_path, "hud=10", "results=21")

    assert result.returncode == 0, result.stderr
    assert load_layout(layout_path).hud_design == "icon_column"


@pytest.mark.parametrize("design", ["icon_column", "counter_row"])
def test_a_layout_drawn_by_hand_takes_the_design_whose_icons_are_under_the_anchor(
    design: HudDesign,
) -> None:
    hud = render_scene(Hud(3, 5, 40, design=design), 1920, 1080)
    results = render_scene(Results(1), 1920, 1080)
    panels = [Panel("hud", 10, (hud,)), Panel("results", 21, (results,))]
    rects = ROW_ELEMENT_RECTS if design == "counter_row" else ELEMENT_RECTS
    elements = {
        name: {"panel": 0 if name in HUD_ELEMENTS else 1, **dict(zip("xywh", rect, strict=True))}
        for name, rect in rects.items()
        if name != "elimination_placement"
    }

    result = manual_result(panels, {"elements": elements})

    assert isinstance(result, SetupResult)
    assert result.layout is not None
    assert result.layout.hud_design == design
    assert result.warnings == []
