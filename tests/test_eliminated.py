"""The player's own elimination: an `eliminated` Event, and what it does to the Match."""

from pathlib import Path
from typing import Any

import pytest

from tests.analyse_helpers import STEADY, analyse, small
from tests.synthetic import Hud, Results, Shot, Spectating, match_shots

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")


def eliminations(match: dict[str, Any]) -> list[tuple[float, int | None]]:
    return [(e["time"], e["value"]) for e in match["events"] if e["kind"] == "eliminated"]


def kinds_at(match: dict[str, Any], time: float) -> list[str]:
    return [e["kind"] for e in match["events"] if e["time"] == time]


def test_going_out_while_the_squad_plays_on_is_an_event_and_the_match_ends_at_its_results(
    tmp_path: Path,
) -> None:
    shots = (
        Shot(Hud(), 5, 15),
        Shot(Spectating(), 15, 45),
        Shot(Results(placement=3), 45, 50),
    )

    output = analyse(tmp_path, small(55, shots))

    (match,) = output["matches"]
    assert (match["start"], match["end"], match["win"], match["placement"]) == (5, 45, False, 3)
    assert eliminations(match) == [(15.0, None)]


def test_a_win_after_going_out_keeps_the_elimination(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(), 5, 15),
        Shot(Spectating(), 15, 30),
        Shot(Results(placement=1), 30, 35),
    )

    output = analyse(tmp_path, small(40, shots))

    (match,) = output["matches"]
    assert (match["win"], match["placement"]) == (True, 1)
    assert eliminations(match) == [(15.0, None)]


def test_a_reboot_and_a_second_elimination_are_two_events(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(), 5, 15),
        Shot(Spectating(), 15, 30),
        Shot(Hud(), 30, 45),
        Shot(Spectating(), 45, 55),
        Shot(Results(placement=4), 55, 60),
    )

    output = analyse(tmp_path, small(65, shots))

    (match,) = output["matches"]
    assert eliminations(match) == [(15.0, None), (45.0, None)]
    assert (match["end"], match["placement"]) == (55, 4)


def test_the_whole_squad_out_is_the_elimination_at_the_you_placed_screen(tmp_path: Path) -> None:
    shots = match_shots(5, STEADY, results_at=25, placement=2, results_style="placed")

    output = analyse(tmp_path, small(35, shots))

    (match,) = output["matches"]
    assert (match["end"], match["win"], match["placement"]) == (25, False, 2)
    assert eliminations(match) == [(25.0, 2)]
    assert kinds_at(match, 25.0) == ["eliminated", "final_placement", "match_end"]


def test_a_win_has_no_elimination(tmp_path: Path) -> None:
    shots = match_shots(5, STEADY, results_at=25, placement=1)

    output = analyse(tmp_path, small(35, shots))

    assert eliminations(output["matches"][0]) == []


def test_one_spectating_second_is_not_an_elimination(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(), 5, 15),
        Shot(Spectating(), 15, 16),
        Shot(Hud(), 16, 25),
        Shot(Results(placement=5), 25, 30),
    )

    output = analyse(tmp_path, small(35, shots))

    assert eliminations(output["matches"][0]) == []


def test_a_brief_hud_flicker_while_out_is_the_same_elimination(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(), 5, 15),
        Shot(Spectating(), 15, 30),
        Shot(Hud(), 30, 31),
        Shot(Spectating(), 31, 45),
        Shot(Results(placement=3), 45, 50),
    )

    output = analyse(tmp_path, small(55, shots))

    assert eliminations(output["matches"][0]) == [(15.0, None)]
