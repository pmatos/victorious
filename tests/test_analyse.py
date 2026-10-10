"""Behaviour of `victorious analyse`, observed through its JSON output."""

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import pytest

from tests.cli_runner import run_cli
from tests.synthetic import (
    Hud,
    Lobby,
    RecordingSpec,
    Results,
    Shot,
    build_recording,
    layout_document,
    match_shots,
    write_layout,
)
from victorious import analysis as analysis_module
from victorious import cli, screens
from victorious.analysis import analyse as analyse_recording
from victorious.layout import load_layout
from victorious.ocr import OcrError

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

FPS = 5.0
STEADY = ((0.0, 0, 0, 99),)


def small(duration: float, shots: tuple[Shot, ...]) -> RecordingSpec:
    return RecordingSpec(duration=duration, shots=shots, fps=FPS, keyframe_interval=2)


def analyse(tmp_path: Path, spec: RecordingSpec) -> dict[str, Any]:
    recording = build_recording(spec, tmp_path / "recording.mkv")
    layout = write_layout(tmp_path / "layout.yaml")
    result = run_cli("analyse", str(recording), "--layout", str(layout))
    assert result.returncode == 0, result.stderr
    output: dict[str, Any] = json.loads(result.stdout)
    return output


def summary(output: dict[str, Any]) -> list[tuple[float, float, bool, int | None]]:
    return [(m["start"], m["end"], m["win"], m["placement"]) for m in output["matches"]]


@pytest.mark.parametrize("size", [(1280, 720), (2560, 1440)])
def test_two_matches_one_win_one_loss(tmp_path: Path, size: tuple[int, int]) -> None:
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=1),
        *match_shots(45, STEADY, results_at=70, placement=7),
    )
    spec = RecordingSpec(
        duration=80, shots=shots, fps=FPS, width=size[0], height=size[1], keyframe_interval=2
    )

    output = analyse(tmp_path, spec)

    assert summary(output) == [(5.0, 25.0, True, 1), (45.0, 70.0, False, 7)]


def test_eliminated_results_screens_end_each_match_and_give_the_placement(tmp_path: Path) -> None:
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=2, results_style="placed"),
        *match_shots(45, STEADY, results_at=70, placement=13, results_style="placed"),
        *match_shots(90, STEADY, results_at=110, placement=1),
    )

    output = analyse(tmp_path, small(120, shots))

    assert summary(output) == [
        (5.0, 25.0, False, 2),
        (45.0, 70.0, False, 13),
        (90.0, 110.0, True, 1),
    ]


def test_orange_victory_royale_banner_is_a_win(tmp_path: Path) -> None:
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=1, results_style="orange"),
        *match_shots(45, STEADY, results_at=70, placement=5),
    )

    output = analyse(tmp_path, small(80, shots))

    assert summary(output) == [(5.0, 25.0, True, 1), (45.0, 70.0, False, 5)]
    assert "win" in [event["kind"] for event in output["matches"][0]["events"]]


def test_layout_without_an_elimination_placement_still_reads_classic_results(
    tmp_path: Path,
) -> None:
    document = layout_document()
    document["screens"]["results"].pop("elimination_placement")  # type: ignore[index]
    recording = build_recording(
        small(80, (*match_shots(5, STEADY, results_at=25, placement=7),)), tmp_path / "r.mkv"
    )
    layout = write_layout(tmp_path / "layout.yaml", document)

    result = run_cli("analyse", str(recording), "--layout", str(layout))

    assert result.returncode == 0, result.stderr
    assert summary(json.loads(result.stdout)) == [(5.0, 25.0, False, 7)]


def test_match_without_a_hud_for_a_while_is_still_one_match(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(), 5, 15),
        Shot(Lobby(), 15, 23),
        Shot(Hud(), 23, 30),
        Shot(Results(placement=4), 30, 35),
    )

    output = analyse(tmp_path, small(40, shots))

    assert summary(output) == [(5.0, 30.0, False, 4)]


def test_match_with_counters_gone_is_still_one_match(tmp_path: Path) -> None:
    shots = (
        Shot(Hud(), 5, 12),
        Shot(Hud(None, None, None), 12, 25),
        Shot(Results(placement=9), 25, 30),
    )

    output = analyse(tmp_path, small(35, shots))

    assert summary(output) == [(5.0, 25.0, False, 9)]


def test_hud_after_a_win_does_not_open_a_phantom_match(tmp_path: Path) -> None:
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=1),
        Shot(Hud(), 30, 40),
        Shot(Lobby(), 40, 50),
        *match_shots(50, STEADY, results_at=70, placement=7),
    )

    output = analyse(tmp_path, small(80, shots))

    assert summary(output) == [(5.0, 25.0, True, 1), (50.0, 70.0, False, 7)]


def test_output_validates_against_the_exported_schema(tmp_path: Path) -> None:
    spec = small(40, match_shots(5, STEADY, results_at=25, placement=1))
    recording = build_recording(spec, tmp_path / "recording.mkv")
    layout = write_layout(tmp_path / "layout.yaml")
    produced = tmp_path / "out.json"

    analysed = run_cli("analyse", str(recording), "--layout", str(layout), "-o", str(produced))
    schema = json.loads(run_cli("schema").stdout)

    assert analysed.returncode == 0, analysed.stderr
    document = json.loads(produced.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(document, schema)
    assert document["schema_version"] == schema["schema_version"]


def test_events_carry_seconds_source_and_confidence(tmp_path: Path) -> None:
    spec = small(40, match_shots(5, STEADY, results_at=25, placement=1))

    output = analyse(tmp_path, spec)

    (match,) = output["matches"]
    kinds = [event["kind"] for event in match["events"]]
    assert kinds == ["match_start", "win", "final_placement", "match_end"]
    times = {event["kind"]: event["time"] for event in match["events"]}
    assert times == {"match_start": 5.0, "win": 25.0, "final_placement": 25.0, "match_end": 25.0}
    for event in match["events"]:
        assert event["source"] == "frame"
        assert 0.0 <= event["confidence"] <= 1.0
        assert 0.0 <= event["time"] <= output["recording"]["duration"]
    placement = next(e for e in match["events"] if e["kind"] == "final_placement")
    assert placement["value"] == 1


def test_non_win_has_no_win_event_and_reports_placement_value(tmp_path: Path) -> None:
    spec = small(40, match_shots(5, STEADY, results_at=25, placement=7))

    output = analyse(tmp_path, spec)

    (match,) = output["matches"]
    assert [e["kind"] for e in match["events"]] == ["match_start", "final_placement", "match_end"]
    assert match["events"][1]["value"] == 7
    assert match["events"][1]["confidence"] > 0.5


def test_output_option_writes_the_file_and_leaves_stdout_empty(tmp_path: Path) -> None:
    spec = small(40, match_shots(5, STEADY, results_at=25, placement=3))
    recording = build_recording(spec, tmp_path / "recording.mkv")
    layout = write_layout(tmp_path / "layout.yaml")
    produced = tmp_path / "nested.json"

    result = run_cli("analyse", str(recording), "--layout", str(layout), "-o", str(produced))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert json.loads(produced.read_text())["matches"][0]["placement"] == 3


def test_two_recordings_in_one_invocation_is_a_usage_error(tmp_path: Path) -> None:
    spec = small(10, ())
    first = build_recording(spec, tmp_path / "first.mkv")
    second = build_recording(spec, tmp_path / "second.mkv")
    layout = write_layout(tmp_path / "layout.yaml")
    produced = tmp_path / "out.json"

    result = run_cli(
        "analyse", str(first), str(second), "--layout", str(layout), "-o", str(produced)
    )

    assert result.returncode == 2
    assert not produced.exists()


def test_placements_with_repeated_and_similar_digits_are_read(tmp_path: Path) -> None:
    placements = [2, 11, 23, 40, 58, 69, 94, 100, 7]
    shots: list[Shot] = []
    for index, placement in enumerate(placements):
        start = 5 + index * 12
        shots += match_shots(start, STEADY, results_at=start + 4, placement=placement)
    spec = small(5 + len(placements) * 12, tuple(shots))

    output = analyse(tmp_path, spec)

    assert [m["placement"] for m in output["matches"]] == placements
    assert not any(m["win"] for m in output["matches"])


def test_recording_with_no_match_yields_no_matches(tmp_path: Path) -> None:
    output = analyse(tmp_path, small(10, ()))

    assert output["matches"] == []


def test_recording_ending_mid_match_reports_an_unfinished_match(tmp_path: Path) -> None:
    output = analyse(tmp_path, small(30, (Shot(Hud(), 5, 30),)))

    assert summary(output) == [(5.0, 29.0, False, None)]
    (match,) = output["matches"]
    assert [e["kind"] for e in match["events"]] == ["match_start", "match_end"]


def test_missing_recording_fails_cleanly(tmp_path: Path) -> None:
    layout = write_layout(tmp_path / "layout.yaml")
    produced = tmp_path / "out.json"
    missing = tmp_path / "nope.mkv"

    result = run_cli("analyse", str(missing), "--layout", str(layout), "-o", str(produced))

    assert result.returncode == 1
    assert str(missing) in result.stderr
    assert result.stdout == ""
    assert not produced.exists()


def test_file_that_is_not_a_video_fails_cleanly(tmp_path: Path) -> None:
    junk = tmp_path / "notes.mkv"
    junk.write_text("not a video")
    layout = write_layout(tmp_path / "layout.yaml")

    result = run_cli("analyse", str(junk), "--layout", str(layout))

    assert result.returncode == 1
    assert str(junk) in result.stderr
    assert result.stdout == ""


def test_incomplete_layout_names_the_file_and_field(tmp_path: Path) -> None:
    recording = build_recording(small(5, ()), tmp_path / "recording.mkv")
    document = layout_document()
    del document["screens"]["results"]["placement"]  # type: ignore[index]
    layout = write_layout(tmp_path / "layout.yaml", document)

    result = run_cli("analyse", str(recording), "--layout", str(layout))

    assert result.returncode == 1
    assert str(layout) in result.stderr
    assert "placement" in result.stderr
    assert result.stdout == ""


def test_layout_rect_outside_the_frame_is_rejected(tmp_path: Path) -> None:
    recording = build_recording(small(5, ()), tmp_path / "recording.mkv")
    document = layout_document()
    document["screens"]["hud"]["hud_anchor"]["x"] = 0.99  # type: ignore[index]
    layout = write_layout(tmp_path / "layout.yaml", document)

    result = run_cli("analyse", str(recording), "--layout", str(layout))

    assert result.returncode == 1
    assert "hud_anchor" in result.stderr


def test_missing_layout_file_fails_cleanly(tmp_path: Path) -> None:
    recording = build_recording(small(5, ()), tmp_path / "recording.mkv")
    missing = tmp_path / "missing.yaml"

    result = run_cli("analyse", str(recording), "--layout", str(missing))

    assert result.returncode == 1
    assert str(missing) in result.stderr


def test_results_screens_with_unreadable_digits_still_end_each_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screens, "read_digits", lambda mask: None)
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=7),
        *match_shots(45, STEADY, results_at=70, placement=4),
    )
    recording = build_recording(small(80, shots), tmp_path / "recording.mkv")
    layout = load_layout(write_layout(tmp_path / "layout.yaml"))

    output = analyse_recording(recording, layout)

    assert [(m.start, m.end, m.win, m.placement) for m in output.matches] == [
        (5.0, 25.0, False, None),
        (45.0, 70.0, False, None),
    ]


def test_missing_tesseract_fails_before_any_frame_is_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = build_recording(small(10, ()), tmp_path / "recording.mkv")
    layout = load_layout(write_layout(tmp_path / "layout.yaml"))
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "tesseract" else real_which(name)
    )

    def no_decoding(*args: Any) -> None:
        raise AssertionError("frames were decoded")

    monkeypatch.setattr(analysis_module, "sample_frames", no_decoding)

    with pytest.raises(OcrError, match="tesseract"):
        analyse_recording(recording, layout)


def test_layout_for_another_aspect_ratio_is_rejected(tmp_path: Path) -> None:
    recording = build_recording(small(5, ()), tmp_path / "recording.mkv")
    document = layout_document()
    document["aspect_ratio"] = "21:9"
    layout = write_layout(tmp_path / "layout.yaml", document)

    result = run_cli("analyse", str(recording), "--layout", str(layout))

    assert result.returncode == 1
    assert "21:9" in result.stderr
    assert "1280x720" in result.stderr
    assert result.stdout == ""


def test_layout_rect_with_nan_is_rejected(tmp_path: Path) -> None:
    recording = build_recording(small(5, ()), tmp_path / "recording.mkv")
    document = layout_document()
    document["screens"]["hud"]["hud_anchor"]["x"] = float("nan")  # type: ignore[index]
    layout = write_layout(tmp_path / "layout.yaml", document)

    result = run_cli("analyse", str(recording), "--layout", str(layout))

    assert result.returncode == 1
    assert "hud_anchor" in result.stderr
    assert "Traceback" not in result.stderr


def test_opencv_failure_is_reported_as_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = write_layout(tmp_path / "layout.yaml")

    def broken(*args: Any) -> None:
        raise cv2.error("template larger than image")

    monkeypatch.setattr(cli, "analyse", broken)

    status = cli.main(["analyse", str(tmp_path / "r.mkv"), "--layout", str(layout)])

    assert status == 1
    assert "victorious analyse:" in capsys.readouterr().err
