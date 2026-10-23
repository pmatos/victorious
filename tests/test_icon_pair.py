"""The third HUD design: the icon column without its team eliminations row."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.analyse_helpers import FPS, STEADY, analyse, analyse_facts, analyse_text, setup_at, small
from tests.synthetic import (
    Hud,
    RecordingSpec,
    Results,
    build_recording,
    layout_document,
    match_shots,
    write_layout,
)
from tests.synthetic.render import PAIR_ELEMENT_RECTS, render_scene
from tests.synthetic.spec import HudDesign
from victorious.cache import layout_hash
from victorious.layout import Layout, dump_layout, load_layout
from victorious.setup import Panel, SetupResult, manual_result

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

PAIR_HUD_ELEMENTS = ("hud_anchor", "personal_eliminations", "players_left")
STEPS = ((0.0, 0, 0, 100), (6.0, 1, 0, 97), (10.0, 4, 0, 60), (14.0, 4, 0, 8))


def counter_changes(match: dict[str, Any], kind: str) -> list[int]:
    return [event["value"] for event in match["events"] if event["kind"] == kind]


def pair_matches() -> RecordingSpec:
    shots = (
        *match_shots(2, STEPS, results_at=24, placement=1, hud_design="icon_pair"),
        *match_shots(
            30, STEADY, results_at=45, placement=6, results_style="placed", hud_design="icon_pair"
        ),
    )
    return small(55, shots)


def column_matches() -> RecordingSpec:
    shots = (
        *match_shots(2, STEPS, results_at=24, placement=1),
        *match_shots(30, STEADY, results_at=45, placement=6, results_style="placed"),
    )
    return small(55, shots)


def test_the_icon_pair_layout_has_no_team_eliminations_rectangle() -> None:
    layout = Layout.model_validate(layout_document("icon_pair"))

    assert layout.hud_design == "icon_pair"
    assert layout.screens.hud.team_eliminations is None
    assert layout.screens.hud.players_left is not None


def test_the_icon_pair_is_part_of_a_layout_identity_and_dumps_back(tmp_path: Path) -> None:
    pair = Layout.model_validate(layout_document("icon_pair"))
    column = Layout.model_validate(layout_document())
    path = tmp_path / "pair.yaml"
    path.write_text(dump_layout(pair))

    assert load_layout(path) == pair
    assert layout_hash(pair) != layout_hash(column)


@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440)])
def test_icon_pair_counters_are_read_without_a_team_eliminations_row(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    shots = match_shots(2, STEPS, results_at=24, placement=1, hud_design="icon_pair")
    spec = RecordingSpec(
        duration=34, shots=shots, fps=FPS, width=size[0], height=size[1], keyframe_interval=2
    )

    output = analyse(tmp_path, spec, layout_document("icon_pair"))

    (match,) = output["matches"]
    assert (match["start"], match["end"], match["win"], match["placement"]) == (2.0, 24.0, True, 1)
    assert match["personal_eliminations"] == 4
    eliminations = counter_changes(match, "personal_eliminations")
    assert eliminations == sorted(eliminations)
    assert (eliminations[0], eliminations[-1]) == (1, 4)
    assert counter_changes(match, "team_eliminations") == []
    left = counter_changes(match, "players_left")
    assert left == sorted(left, reverse=True)
    assert {97, 60, 8} <= set(left)
    assert output["layout"]["score"] >= 0.7


def test_setup_finds_the_icon_pair_and_writes_a_layout_analyse_reads_like_a_hand_written_one(
    tmp_path: Path,
) -> None:
    recording = build_recording(pair_matches(), tmp_path / "recording.mkv")
    layout_path = tmp_path / "layout.yaml"

    result = setup_at(recording, layout_path, "hud=14", "results=25", "results=46")

    assert result.returncode == 0, result.stderr
    layout = load_layout(layout_path)
    assert layout.hud_design == "icon_pair"
    assert layout.screens.hud.team_eliminations is None
    hand_written = write_layout(tmp_path / "hand.yaml", layout_document("icon_pair"))
    from_setup = analyse_facts(recording, layout_path)
    assert from_setup == analyse_facts(recording, hand_written)
    assert [fact[:5] for fact in from_setup] == [(2.0, 24.0, True, 1, 4), (30.0, 45.0, False, 6, 0)]


@pytest.mark.parametrize(
    "order", [("a_column.yaml", "b_pair.yaml"), ("a_pair.yaml", "b_column.yaml")]
)
@pytest.mark.parametrize("hud", ["icon_column", "icon_pair"])
def test_directory_with_both_designs_picks_the_one_the_recording_uses(
    tmp_path: Path, order: tuple[str, str], hud: HudDesign
) -> None:
    recording = build_recording(
        pair_matches() if hud == "icon_pair" else column_matches(), tmp_path / "recording.mkv"
    )
    names = {
        "icon_column": next(name for name in order if "column" in name),
        "icon_pair": next(name for name in order if "pair" in name),
    }
    directory = tmp_path / "layouts"
    directory.mkdir()
    write_layout(directory / names["icon_column"], layout_document())
    write_layout(directory / names["icon_pair"], layout_document("icon_pair"))

    result = analyse_text(recording, directory)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["layout"]["path"] == str(directory / names[hud])
    assert output["layout"]["score"] >= 0.7
    assert [m["win"] for m in output["matches"]] == [True, False]


def test_the_full_icon_column_is_not_taken_for_the_pair(tmp_path: Path) -> None:
    recording = build_recording(column_matches(), tmp_path / "recording.mkv")
    layout = write_layout(tmp_path / "layout.yaml", layout_document("icon_pair"))

    result = analyse_text(recording, layout)

    assert result.returncode == 1
    assert "no Layout matches this Recording's HUD" in result.stderr


def test_the_pair_is_not_read_by_an_icon_column_layout(tmp_path: Path) -> None:
    recording = build_recording(pair_matches(), tmp_path / "recording.mkv")
    layout = write_layout(tmp_path / "layout.yaml", layout_document())

    result = analyse_text(recording, layout)

    assert result.returncode == 1
    assert "no Layout matches this Recording's HUD" in result.stderr


def drawn() -> dict[str, Any]:
    rects = PAIR_ELEMENT_RECTS
    return {
        name: {
            "panel": 0 if name in PAIR_HUD_ELEMENTS else 1,
            **dict(zip("xywh", rect, strict=True)),
        }
        for name, rect in rects.items()
        if name not in ("elimination_placement", "team_eliminations")
    }


def test_a_layout_drawn_by_hand_without_team_eliminations_is_the_icon_pair() -> None:
    hud = render_scene(Hud(3, 0, 40, design="icon_pair"), 1920, 1080)
    results = render_scene(Results(1), 1920, 1080)
    panels = [Panel("hud", 10, (hud,)), Panel("results", 21, (results,))]

    result = manual_result(panels, {"elements": drawn()})

    assert isinstance(result, SetupResult)
    assert result.layout is not None
    assert result.layout.hud_design == "icon_pair"
    assert result.layout.screens.hud.team_eliminations is None
    assert result.warnings == []


def test_a_layout_drawn_by_hand_on_the_icon_column_still_needs_team_eliminations() -> None:
    hud = render_scene(Hud(3, 5, 40), 1920, 1080)
    results = render_scene(Results(1), 1920, 1080)
    panels = [Panel("hud", 10, (hud,)), Panel("results", 21, (results,))]

    result = manual_result(panels, {"elements": drawn()})

    assert result == ["team_eliminations: required, not placed"]
