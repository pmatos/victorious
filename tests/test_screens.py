"""The results reading on single frames, including the large banner before it settles."""

from pathlib import Path

import pytest

from tests.synthetic import Hud, Lobby, Results, write_layout
from tests.synthetic.render import render_scene
from tests.synthetic.spec import ResultsStyle, Scene
from victorious.layout import Layout, load_layout
from victorious.screens import RESULTS_THRESHOLD, results_reading, scale_range

SIZES = [(1280, 720), (1920, 1080)]


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return load_layout(write_layout(tmp_path / "layout.yaml"))


@pytest.mark.parametrize("size", SIZES, ids=lambda size: f"{size[1]}p")
@pytest.mark.parametrize("style", ["classic", "orange", "orange_large"])
def test_a_win_is_a_results_screen_whatever_the_banner_size(
    layout: Layout, size: tuple[int, int], style: ResultsStyle
) -> None:
    frame = render_scene(Results(1, style), *size)

    assert results_reading(frame, layout, read_placement=False).banner >= RESULTS_THRESHOLD


@pytest.mark.parametrize("size", SIZES, ids=lambda size: f"{size[1]}p")
@pytest.mark.parametrize(
    "scene",
    [Hud(), Lobby(), Results(7, "placed")],
    ids=["hud", "lobby", "eliminated"],
)
def test_the_wide_banner_search_finds_nothing_on_other_screens(
    layout: Layout, size: tuple[int, int], scene: Scene
) -> None:
    frame = render_scene(scene, *size)

    assert results_reading(frame, layout, read_placement=False).banner < RESULTS_THRESHOLD - 0.2


def test_only_the_banner_is_read_at_large_sizes() -> None:
    assert scale_range("results_banner") == (0.9, 1.6)
    assert scale_range("hud_anchor") == (0.9, 1.1)
