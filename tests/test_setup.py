"""`victorious setup` writes a Layout `analyse` accepts from the HUD elements it finds."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

from tests.analyse_helpers import (
    EXPECTED_MATCHES,
    FPS,
    HUD_AT,
    PLACED_RESULTS_AT,
    WIN_RESULTS_AT,
    analyse_facts,
    intersection_over_union,
    scenario_spec,
    setup_at,
)
from tests.cli_runner import run_cli
from tests.synthetic import (
    ELEMENT_RECTS,
    Hud,
    Lobby,
    RecordingSpec,
    Results,
    Shot,
    build_recording,
    decode_frame,
    layout_document,
    write_layout,
)
from victorious.layout import Layout, aspect_ratio_label, load_layout
from victorious.schema import json_schema

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")


@dataclass(frozen=True)
class SetupRun:
    recording: Path
    layout_path: Path
    preview_path: Path
    stderr: str


def run_scenario(directory: Path, width: int = 1920, height: int = 1080) -> SetupRun:
    recording = build_recording(scenario_spec(width, height), directory / "recording.mkv")
    layout_path = directory / "layout.yaml"
    result = setup_at(
        recording, layout_path, f"hud={HUD_AT}", f"results={WIN_RESULTS_AT}",
        f"results={PLACED_RESULTS_AT}",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    return SetupRun(recording, layout_path, layout_path.with_suffix(".preview.png"), result.stderr)


@pytest.fixture(scope="module")
def ran_setup(tmp_path_factory: pytest.TempPathFactory) -> SetupRun:
    return run_scenario(tmp_path_factory.mktemp("setup"))


def test_setup_writes_a_layout_analyse_accepts(ran_setup: SetupRun) -> None:
    layout = load_layout(ran_setup.layout_path)
    assert isinstance(layout, Layout)

    result = run_cli("analyse", str(ran_setup.recording), "--layout", str(ran_setup.layout_path))

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    jsonschema.validate(output, json_schema())
    assert [(m["start"], m["end"], m["win"], m["placement"]) for m in output["matches"]] == [
        (2.0, 20.0, True, 1),
        (30.0, 45.0, False, 6),
        (60.0, 75.0, False, 6),
    ]


def test_setup_layout_gives_the_same_matches_as_a_hand_written_one(
    ran_setup: SetupRun, tmp_path: Path
) -> None:
    hand_written = write_layout(tmp_path / "hand.yaml")

    from_setup = analyse_facts(ran_setup.recording, ran_setup.layout_path)

    assert from_setup == analyse_facts(ran_setup.recording, hand_written)
    assert [fact[:5] for fact in from_setup] == EXPECTED_MATCHES
    assert any(kind == "players_left" for *_, events in from_setup for kind, *_ in events)


def test_setup_layout_gives_the_same_matches_on_a_1440p_recording(tmp_path: Path) -> None:
    run = run_scenario(tmp_path, 2560, 1440)
    hand_written = write_layout(tmp_path / "hand.yaml")

    from_setup = analyse_facts(run.recording, run.layout_path)

    assert from_setup == analyse_facts(run.recording, hand_written)
    assert [fact[:5] for fact in from_setup] == EXPECTED_MATCHES


def test_detected_rectangles_overlap_the_hand_written_ones(ran_setup: SetupRun) -> None:
    document = load_layout(ran_setup.layout_path).model_dump(exclude_none=True)["screens"]
    detected = {name: rect for screen in document.values() for name, rect in screen.items()}

    assert set(detected) == set(ELEMENT_RECTS)
    for name, expected in ELEMENT_RECTS.items():
        assert intersection_over_union(detected[name], expected) >= 0.5, name


def compact(duration: float, shots: tuple[Shot, ...]) -> RecordingSpec:
    """A 720p Recording: fast to search, for the cases that do not need a big frame."""
    return RecordingSpec(duration=duration, shots=shots, fps=FPS, keyframe_interval=2)


@pytest.fixture(scope="module")
def sparse_recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Lobby, then a HUD at 10 s, a Win at 20 s, an elimination at 30 s, a Lobby."""
    shots = (
        Shot(Hud(1, 0, 90), 8, 15),
        Shot(Results(1), 18, 24),
        Shot(Results(9, "placed"), 26, 32),
    )
    directory = tmp_path_factory.mktemp("sparse")
    return build_recording(compact(40, shots), directory / "recording.mkv")


@pytest.mark.parametrize("extra", [(), ("--manual", "--no-browser")])
def test_the_two_screen_kinds_are_required(
    sparse_recording: Path, tmp_path: Path, extra: tuple[str, ...]
) -> None:
    layout = tmp_path / "layout.yaml"

    for given, absent in (((), "hud"), (("hud=10",), "results"), (("results=20",), "hud")):
        result = setup_at(sparse_recording, layout, *given, extra=extra)

        assert result.returncode == 2, result.stderr
        assert f"--at {absent}=" in result.stderr
        assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", ["hud=abc", "bogus=3", "hud", "hud=-1", "results=1:99:99"])
def test_a_malformed_at_is_a_usage_error(sparse_recording: Path, tmp_path: Path, bad: str) -> None:
    result = setup_at(sparse_recording, tmp_path / "layout.yaml", bad, "results=20")

    assert result.returncode == 2, result.stderr
    assert "--at" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("extra", [(), ("--manual", "--no-browser")])
def test_a_time_past_the_end_of_the_recording_names_the_duration(
    sparse_recording: Path, tmp_path: Path, extra: tuple[str, ...]
) -> None:
    result = setup_at(
        sparse_recording, tmp_path / "layout.yaml", "hud=10", "results=900", extra=extra
    )

    assert result.returncode == 1
    assert "results=900" in result.stderr
    assert "40" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_times_may_be_written_as_a_clock(sparse_recording: Path, tmp_path: Path) -> None:
    plain = setup_at(sparse_recording, tmp_path / "plain.yaml", "hud=10", "results=20")
    clock = setup_at(sparse_recording, tmp_path / "clock.yaml", "hud=0:10", "results=0:20.5")

    assert plain.returncode == 0, plain.stderr
    assert clock.returncode == 0, clock.stderr
    assert (tmp_path / "plain.yaml").read_text() == (tmp_path / "clock.yaml").read_text()


def test_elements_that_cannot_be_found_are_named_with_a_pointer_to_manual(
    sparse_recording: Path, tmp_path: Path
) -> None:
    layout = tmp_path / "layout.yaml"

    result = setup_at(sparse_recording, layout, "hud=2", "results=4")

    assert result.returncode == 1
    for name in (
        "hud_anchor",
        "personal_eliminations",
        "team_eliminations",
        "players_left",
        "results_banner",
        "placement",
    ):
        assert name in result.stderr
    assert "elimination_placement" not in result.stderr
    assert "--manual" in result.stderr
    assert not layout.exists()


def test_counters_that_are_not_drawn_are_the_only_ones_named(tmp_path: Path) -> None:
    shots = (Shot(Hud(None, None, None), 2, 10), Shot(Results(1), 12, 18))
    recording = build_recording(compact(20, shots), tmp_path / "recording.mkv")

    result = setup_at(recording, tmp_path / "layout.yaml", "hud=5", "results=14")

    assert result.returncode == 1
    for name in ("personal_eliminations", "team_eliminations", "players_left"):
        assert name in result.stderr
    assert "hud_anchor" not in result.stderr
    assert "--manual" in result.stderr


def test_a_results_screen_of_an_eliminated_player_has_no_banner_or_gold_placement(
    sparse_recording: Path, tmp_path: Path
) -> None:
    result = setup_at(sparse_recording, tmp_path / "layout.yaml", "hud=10", "results=28")

    assert result.returncode == 1
    assert "results_banner" in result.stderr
    assert "placement" in result.stderr
    assert "elimination_placement" not in result.stderr


def test_a_missing_elimination_placement_is_a_warning(
    sparse_recording: Path, tmp_path: Path
) -> None:
    layout = tmp_path / "layout.yaml"

    result = setup_at(sparse_recording, layout, "hud=10", "results=20")

    assert result.returncode == 0, result.stderr
    assert "elimination_placement" in result.stderr
    assert load_layout(layout).screens.results.elimination_placement is None


def test_the_orange_banner_is_found(tmp_path: Path) -> None:
    shots = (Shot(Hud(1, 0, 90), 2, 10), Shot(Results(1, "orange"), 12, 18))
    recording = build_recording(compact(20, shots), tmp_path / "recording.mkv")
    layout = tmp_path / "layout.yaml"

    result = setup_at(recording, layout, "hud=5", "results=14")

    assert result.returncode == 0, result.stderr
    banner = load_layout(layout).screens.results.results_banner
    assert banner.w == pytest.approx(0.1, abs=0.01)


def test_an_ultrawide_recording_gets_the_21_9_label(tmp_path: Path) -> None:
    shots = (Shot(Hud(1, 0, 90), 2, 10), Shot(Results(1), 12, 18))
    spec = RecordingSpec(
        duration=20, shots=shots, fps=FPS, width=2560, height=1080, keyframe_interval=2
    )
    recording = build_recording(spec, tmp_path / "recording.mkv")
    layout = tmp_path / "layout.yaml"

    result = setup_at(recording, layout, "hud=5", "results=14")

    assert result.returncode == 0, result.stderr
    assert load_layout(layout).aspect_ratio == "21:9"
    assert load_layout(layout).fits(2560, 1080)


@pytest.mark.parametrize("extra", [(), ("--manual", "--no-browser")])
def test_an_unreadable_recording_exits_1_and_writes_nothing(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    broken = tmp_path / "recording.mkv"
    broken.write_bytes(b"not a video")

    result = setup_at(broken, tmp_path / "layout.yaml", "hud=1", "results=2", extra=extra)

    assert result.returncode == 1
    assert "recording.mkv" in result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["recording.mkv"]


def test_a_lobby_second_is_not_a_hud(tmp_path: Path) -> None:
    recording = build_recording(compact(10, (Shot(Lobby(), 0, 10),)), tmp_path / "recording.mkv")

    result = setup_at(recording, tmp_path / "layout.yaml", "hud=3", "results=5")

    assert result.returncode == 1
    assert "hud_anchor" in result.stderr


PREVIEW_WIDTH = 1600
SCREEN_OF = {
    "hud_anchor": 0,
    "personal_eliminations": 0,
    "team_eliminations": 0,
    "players_left": 0,
    "results_banner": 1,
    "placement": 1,
    "elimination_placement": 2,
}


def changed(shown: np.ndarray, source: np.ndarray, threshold: int = 30) -> float:
    """The share of pixels that differ visibly between the preview and the frame."""
    difference = np.abs(shown.astype(int) - source.astype(int)).max(axis=2)
    return float(np.mean(difference > threshold))


def columns_changed(shown: np.ndarray, source: np.ndarray) -> float:
    """The share of columns with a visibly changed pixel: a line was drawn across them."""
    difference = np.abs(shown.astype(int) - source.astype(int)).max(axis=2)
    return float(np.mean((difference > 30).any(axis=0)))


def test_the_preview_shows_each_rectangle_with_its_label(ran_setup: SetupRun) -> None:
    preview = cv2.imread(str(ran_setup.preview_path))
    assert preview is not None
    times = [HUD_AT, WIN_RESULTS_AT, PLACED_RESULTS_AT]
    sources = []
    for second in times:
        frame = decode_frame(ran_setup.recording, second)
        scale = PREVIEW_WIDTH / frame.shape[1]
        size = (PREVIEW_WIDTH, round(frame.shape[0] * scale))
        sources.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    panel_height = sources[0].shape[0]
    assert preview.shape == (panel_height * len(times), PREVIEW_WIDTH, 3)
    layout = load_layout(ran_setup.layout_path).model_dump(exclude_none=True)["screens"]
    rects = {name: rect for screen in layout.values() for name, rect in screen.items()}

    for name, rect in rects.items():
        index = SCREEN_OF[name]
        shown = preview[index * panel_height : (index + 1) * panel_height]
        source = sources[index]
        x0, y0 = round(rect["x"] * PREVIEW_WIDTH), round(rect["y"] * panel_height)
        x1 = round((rect["x"] + rect["w"]) * PREVIEW_WIDTH)
        y1 = round((rect["y"] + rect["h"]) * panel_height)

        for edge in (y0, y1):
            band = slice(max(0, edge - 2), edge + 3)
            assert columns_changed(shown[band, x0:x1], source[band, x0:x1]) > 0.9, name
        label_sides = [
            shown[max(0, y0 - 14) : y0 - 2, x0:x1],
            shown[y1 + 2 : y1 + 14, x0:x1],
            shown[y0:y1, max(0, x0 - 40) : x0 - 2],
        ]
        label_sources = [
            source[max(0, y0 - 14) : y0 - 2, x0:x1],
            source[y1 + 2 : y1 + 14, x0:x1],
            source[y0:y1, max(0, x0 - 40) : x0 - 2],
        ]
        assert any(
            side.size and changed(side, original) > 0.3
            for side, original in zip(label_sides, label_sources, strict=True)
        ), name
        if y1 - y0 > 8 and x1 - x0 > 8 and name not in {"placement"}:
            inside = (slice(y0 + 4, y1 - 4), slice(x0 + 4, x1 - 4))
            assert changed(shown[inside], source[inside], 60) < 0.1, name


@pytest.mark.parametrize(
    ("width", "height", "label"),
    [
        (1920, 1080, "16:9"),
        (1366, 768, "16:9"),
        (2560, 1600, "16:10"),
        (2560, 1080, "21:9"),
        (3440, 1440, "43:18"),
        (5120, 1440, "32:9"),
        (1024, 768, "4:3"),
        (1000, 700, "10:7"),
    ],
)
def test_aspect_ratio_labels_are_common_names_or_the_reduced_ratio(
    width: int, height: int, label: str
) -> None:
    assert aspect_ratio_label(width, height) == label
    layout = Layout.model_validate(
        {
            "aspect_ratio": aspect_ratio_label(width, height),
            "screens": layout_document()["screens"],
        }
    )
    assert layout.fits(width, height)
