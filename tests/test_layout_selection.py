"""`analyse --layout` takes a Layout or a directory of Layouts and uses the one that fits."""

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from tests.analyse_helpers import FPS, STEADY, FfmpegShim, analyse_text, small
from tests.synthetic import RecordingSpec, build_recording, layout_document, match_shots
from tests.synthetic import write_layout as write_layout_document
from victorious import selection, video
from victorious.cache import ObservationCache
from victorious.cli import main
from victorious.schema import json_schema
from victorious.video import RecordingError

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

DECOY_ANCHOR = {"x": 0.1, "y": 0.5, "w": 0.014, "h": 0.088}


def decoy_document() -> dict[str, Any]:
    document: dict[str, Any] = copy.deepcopy(layout_document())
    document["screens"]["hud"]["hud_anchor"] = dict(DECOY_ANCHOR)
    return document


def two_matches(**sizes: Any) -> RecordingSpec:
    shots = (
        *match_shots(2, STEADY, results_at=20, placement=1),
        *match_shots(30, STEADY, results_at=45, placement=6),
    )
    if sizes:
        return RecordingSpec(duration=55, shots=shots, fps=FPS, keyframe_interval=2, **sizes)
    return small(55, shots)


def summary(output: dict[str, Any]) -> list[tuple[float, float, bool, int | None]]:
    return [(m["start"], m["end"], m["win"], m["placement"]) for m in output["matches"]]


EXPECTED = [(2.0, 20.0, True, 1), (30.0, 45.0, False, 6)]


def layouts_directory(tmp_path: Path, **documents: dict[str, Any]) -> Path:
    directory = tmp_path / "layouts"
    directory.mkdir()
    for name, document in documents.items():
        write_layout_document(directory / name, document)
    return directory


@pytest.mark.parametrize(
    ("real", "decoy"), [("b_real.yaml", "a_decoy.yaml"), ("a_real.yaml", "b_decoy.yaml")]
)
def test_directory_picks_the_layout_matching_the_hud(tmp_path: Path, real: str, decoy: str) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    directory = layouts_directory(tmp_path, **{real: layout_document(), decoy: decoy_document()})

    result = analyse_text(recording, directory)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert summary(output) == EXPECTED
    assert output["layout"]["path"] == str(directory / real)
    assert output["layout"]["score"] >= 0.7
    assert real in result.stderr


def test_directory_ignores_layouts_of_another_aspect_ratio(tmp_path: Path) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    wide = layout_document()
    wide["aspect_ratio"] = "21:9"
    directory = layouts_directory(
        tmp_path, **{"a_ultrawide.yaml": wide, "b.yml": layout_document()}
    )

    result = analyse_text(recording, directory)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["layout"]["path"] == str(directory / "b.yml")
    assert summary(output) == EXPECTED


def test_output_with_the_chosen_layout_validates_against_the_schema(tmp_path: Path) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    layout = write_layout_document(tmp_path / "layout.yaml")

    result = analyse_text(recording, layout)

    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    jsonschema.Draft202012Validator(json_schema()).validate(document)
    assert document["schema_version"] == "1.3"
    assert document["layout"]["path"] == str(layout)
    assert 0.7 <= document["layout"]["score"] <= 1.0


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (2560, 1440)])
def test_a_16x9_layout_works_at_any_16x9_resolution(tmp_path: Path, size: tuple[int, int]) -> None:
    recording = build_recording(
        two_matches(width=size[0], height=size[1]), tmp_path / "recording.mkv"
    )
    layout = write_layout_document(tmp_path / "layout.yaml")

    result = analyse_text(recording, layout)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert summary(output) == EXPECTED
    assert output["layout"]["score"] >= 0.7


def test_32x9_recording_with_a_16x9_layout_errors_naming_setup(tmp_path: Path) -> None:
    spec = two_matches(width=1920, height=540)
    recording = build_recording(spec, tmp_path / "recording.mkv")
    layout = write_layout_document(tmp_path / "layout.yaml")
    produced = tmp_path / "out.json"

    result = analyse_text(recording, layout, "-o", str(produced))

    assert result.returncode == 1
    assert result.stdout == ""
    assert not produced.exists()
    assert "setup" in result.stderr
    assert "16:9" in result.stderr
    assert "1920x540" in result.stderr
    assert "Traceback" not in result.stderr


def test_no_layout_above_the_minimum_errors_without_output(tmp_path: Path) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    layout = write_layout_document(tmp_path / "decoy.yaml", decoy_document())
    produced = tmp_path / "out.json"

    result = analyse_text(recording, layout, "-o", str(produced))

    assert result.returncode == 1
    assert result.stdout == ""
    assert not produced.exists()
    assert "no Layout matches" in result.stderr
    assert "victorious setup" in result.stderr
    assert "Traceback" not in result.stderr


def test_empty_layout_directory_errors(tmp_path: Path) -> None:
    recording = build_recording(small(5, ()), tmp_path / "recording.mkv")
    directory = layouts_directory(tmp_path)

    result = analyse_text(recording, directory)

    assert result.returncode == 1
    assert result.stdout == ""
    assert str(directory) in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_file_in_the_layout_directory_is_named(tmp_path: Path) -> None:
    recording = build_recording(small(5, ()), tmp_path / "recording.mkv")
    directory = layouts_directory(tmp_path, **{"good.yaml": layout_document()})
    broken = directory / "broken.yaml"
    broken.write_text(yaml.safe_dump({"aspect_ratio": "16:9"}))

    result = analyse_text(recording, directory)

    assert result.returncode == 1
    assert str(broken) in result.stderr
    assert result.stdout == ""


def test_layout_directory_rerun_decodes_nothing(tmp_path: Path) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    directory = layouts_directory(
        tmp_path, **{"a_decoy.yaml": decoy_document(), "b_real.yaml": layout_document()}
    )
    shim = FfmpegShim(tmp_path / "shim")

    first = analyse_text(recording, directory, env=shim.env)
    assert shim.decodes() > 0
    shim.reset()
    second = analyse_text(recording, directory, env=shim.env)

    assert second.returncode == 0, second.stderr
    assert shim.decodes() == 0
    assert second.stdout == first.stdout


def test_selection_does_not_depend_on_jobs(tmp_path: Path) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    directory = layouts_directory(
        tmp_path, **{"a_decoy.yaml": decoy_document(), "b_real.yaml": layout_document()}
    )

    one = analyse_text(recording, directory, "--no-cache", "--jobs", "1")
    four = analyse_text(recording, directory, "--no-cache", "--jobs", "4")

    assert one.returncode == four.returncode == 0, one.stderr + four.stderr
    assert one.stdout == four.stdout


def test_identical_layouts_go_to_the_first_file_by_name(tmp_path: Path) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    directory = layouts_directory(
        tmp_path, **{"b.yaml": layout_document(), "a.yaml": layout_document()}
    )

    result = analyse_text(recording, directory)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["layout"]["path"] == str(directory / "a.yaml")


def test_renamed_layout_keeps_the_cached_selection_and_reports_its_new_path(
    tmp_path: Path,
) -> None:
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    directory = layouts_directory(tmp_path, **{"old.yaml": layout_document()})
    shim = FfmpegShim(tmp_path / "shim")
    analyse_text(recording, directory, env=shim.env)
    (directory / "old.yaml").rename(directory / "new.yaml")
    shim.reset()

    result = analyse_text(recording, directory, env=shim.env)

    assert result.returncode == 0, result.stderr
    assert shim.decodes() == 0
    assert json.loads(result.stdout)["layout"]["path"] == str(directory / "new.yaml")


def test_layouts_that_each_fit_half_the_frames_score_below_the_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = {"count": 0}

    def alternating(frame: object, layout: object) -> float:
        index = calls["count"]
        calls["count"] += 1
        frame_number, which = divmod(index, 2)
        return 0.9 if frame_number % 2 == which else 0.1

    monkeypatch.setattr(selection, "hud_score", alternating)
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    directory = layouts_directory(
        tmp_path, **{"a.yaml": layout_document(), "b.yaml": decoy_document()}
    )
    produced = tmp_path / "out.json"

    status = main(
        ["analyse", str(recording), "--layout", str(directory), "--jobs", "1", "-o", str(produced)]
    )

    assert status == 1
    assert "below the minimum" in capsys.readouterr().err
    assert not produced.exists()


def test_a_decode_error_after_the_scored_frames_does_not_fail_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(selection, "SCORED_FRAMES", 3)
    real = video.frame_at

    def truncated(path: Path, info: Any, second: int) -> Any:
        if second >= 8:
            raise RecordingError("truncated frame")
        return real(path, info, second)

    monkeypatch.setattr(selection, "frame_at", truncated)
    recording = build_recording(two_matches(), tmp_path / "recording.mkv")
    layout = write_layout_document(tmp_path / "layout.yaml")
    produced = tmp_path / "out.json"

    status = main(
        [
            *("analyse", str(recording), "--layout", str(layout)),
            *("--jobs", "16", "--no-cache", "-o", str(produced)),
        ]
    )

    assert status == 0
    assert json.loads(produced.read_text())["layout"]["score"] >= 0.7


def test_candidate_frames_never_exceed_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(selection, "MAX_CANDIDATE_FRAMES", 4)
    seconds: list[int] = []
    real = video.frame_at

    def counting(path: Path, info: Any, second: int) -> Any:
        seconds.append(second)
        return real(path, info, second)

    monkeypatch.setattr(selection, "frame_at", counting)
    recording = build_recording(small(10, ()), tmp_path / "recording.mkv")
    layout = write_layout_document(tmp_path / "layout.yaml")

    status = main(["analyse", str(recording), "--layout", str(layout), "--jobs", "1", "--no-cache"])

    assert status == 0
    assert set(seconds) == {0, 3, 6, 9}


@pytest.mark.parametrize("stored", [1.7, -0.1, float("nan"), float("inf"), True, "0.9", None])
def test_a_selection_entry_with_an_impossible_score_is_not_used(
    tmp_path: Path, stored: Any
) -> None:
    recording = tmp_path / "recording.bin"
    recording.write_bytes(b"frames")
    entry = ObservationCache(tmp_path / "cache").selection(recording, ["abc"])
    entry.store([0.9])
    assert entry.load() == [0.9]
    (file,) = (tmp_path / "cache").glob("selection-*.json")
    file.write_text(json.dumps({"abc": stored}))

    assert entry.load() is None
