"""`analyse` copes with Recordings that are not Fortnite, are unreadable, or carry a start time."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from tests.analyse_helpers import STEADY, FfmpegShim, analyse_text, small
from tests.synthetic import Hud, Shot, build_recording, match_shots, write_layout
from victorious import selection
from victorious.cli import main
from victorious.metadata import started_at
from victorious.schema import json_schema

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")


@pytest.fixture(scope="module")
def matched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    shots = (
        *match_shots(2, STEADY, results_at=20, placement=1),
        *match_shots(30, STEADY, results_at=45, placement=6),
    )
    return build_recording(small(55, shots), tmp_path_factory.mktemp("matched") / "matched.mkv")


@pytest.fixture(scope="module")
def short(tmp_path_factory: pytest.TempPathFactory) -> Path:
    shots = match_shots(2, STEADY, results_at=8, placement=1)
    return build_recording(small(14, shots), tmp_path_factory.mktemp("short") / "short.mkv")


@pytest.fixture(scope="module")
def no_hud(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_recording(small(30, ()), tmp_path_factory.mktemp("no-hud") / "no-hud.mkv")


@pytest.fixture
def layout(tmp_path: Path) -> Path:
    return write_layout(tmp_path / "layout.yaml")


def validated(text: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(text)
    jsonschema.Draft202012Validator(json_schema()).validate(document)
    return document


def truncate(source: Path, destination: Path, fraction: float) -> Path:
    data = source.read_bytes()
    destination.write_bytes(data[: int(len(data) * fraction)])
    return destination


def named(source: Path, directory: Path, name: str) -> Path:
    destination = directory / name
    shutil.copy(source, destination)
    return destination


def remux_with_creation_time(source: Path, destination: Path, creation_time: str) -> Path:
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-i", str(source), "-c", "copy"),
            *("-metadata", f"creation_time={creation_time}", str(destination)),
        ],
        check=True,
    )
    return destination


def started(recording: Path, layout: Path) -> Any:
    result = analyse_text(recording, layout)
    assert result.returncode == 0, result.stderr
    document = validated(result.stdout)
    assert "started_at" in document["recording"]
    return document["recording"]["started_at"]


def test_a_recording_without_a_hud_reports_no_matches_and_a_warning(
    no_hud: Path, layout: Path, tmp_path: Path
) -> None:
    produced = tmp_path / "out.json"

    result = analyse_text(no_hud, layout, "-o", str(produced))

    assert result.returncode == 0, result.stderr
    document = validated(produced.read_text())
    assert document["matches"] == []
    assert document["layout"] is None
    assert [warning["code"] for warning in document["warnings"]] == ["not_fortnite"]
    assert document["recording"]["duration"] == pytest.approx(30, abs=0.5)
    assert "not_fortnite" in result.stderr


def test_a_recording_without_a_hud_writes_an_empty_observations_file(
    no_hud: Path, layout: Path, tmp_path: Path
) -> None:
    sidecar = tmp_path / "side.jsonl"

    result = analyse_text(no_hud, layout, "--observations", str(sidecar))

    assert result.returncode == 0, result.stderr
    assert sidecar.read_text() == ""


def test_a_hud_after_the_opening_window_is_not_fortnite(
    tmp_path: Path, layout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(selection, "NOT_FORTNITE_WINDOW", 5)
    late = build_recording(
        small(20, match_shots(8, STEADY, results_at=13, placement=1)), tmp_path / "late.mkv"
    )
    early = build_recording(
        small(20, match_shots(2, STEADY, results_at=8, placement=1)), tmp_path / "early.mkv"
    )
    outputs = {}
    for recording in (late, early):
        produced = tmp_path / f"{recording.stem}.json"
        status = main(
            [*("analyse", str(recording), "--layout", str(layout)), *("--jobs", "1", "--no-cache"),
             *("-o", str(produced))]
        )  # fmt: skip
        assert status == 0
        outputs[recording.stem] = validated(produced.read_text())

    assert outputs["late"]["matches"] == []
    assert outputs["late"]["warnings"][0]["code"] == "not_fortnite"
    assert len(outputs["early"]["matches"]) == 1
    assert outputs["early"]["warnings"] == []


def test_an_ordinary_recording_carries_no_warning(matched: Path, layout: Path) -> None:
    result = analyse_text(matched, layout)

    assert result.returncode == 0, result.stderr
    document = validated(result.stdout)
    assert document["warnings"] == []
    assert document["layout"]["path"] == str(layout)
    assert len(document["matches"]) == 2


def test_the_opening_window_is_sampled_densely_however_long_the_recording(
    tmp_path: Path, layout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(selection, "MAX_CANDIDATE_FRAMES", 2)
    monkeypatch.setattr(selection, "NOT_FORTNITE_WINDOW", 40)
    recording = build_recording(small(40, (Shot(Hud(), 25, 35),)), tmp_path / "brief.mkv")
    produced = tmp_path / "out.json"

    status = main(
        [*("analyse", str(recording), "--layout", str(layout)), *("--jobs", "1", "--no-cache"),
         *("-o", str(produced))]
    )  # fmt: skip

    assert status == 0
    document = validated(produced.read_text())
    assert document["warnings"] == []
    assert len(document["matches"]) == 1


def test_a_truncated_recording_exits_non_zero_without_output(
    matched: Path, layout: Path, tmp_path: Path
) -> None:
    cut = truncate(matched, tmp_path / "cut.mkv", 0.55)
    produced = tmp_path / "out.json"

    result = analyse_text(cut, layout, "-o", str(produced))

    assert result.returncode == 1
    assert result.stdout == ""
    assert not produced.exists()
    assert str(cut) in result.stderr
    assert "truncated" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_truncated_recording_without_a_hud_is_not_reported_as_not_fortnite(
    no_hud: Path, layout: Path, tmp_path: Path
) -> None:
    cut = truncate(no_hud, tmp_path / "cut.mkv", 0.55)
    produced = tmp_path / "out.json"

    result = analyse_text(cut, layout, "-o", str(produced))

    assert result.returncode == 1
    assert not produced.exists()
    assert "truncated" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_failed_run_leaves_no_observations_file(
    matched: Path, layout: Path, tmp_path: Path
) -> None:
    cut = truncate(matched, tmp_path / "cut.mkv", 0.55)
    sidecar = tmp_path / "side.jsonl"

    result = analyse_text(cut, layout, "--observations", str(sidecar))

    assert result.returncode == 1
    assert not sidecar.exists()


@pytest.mark.parametrize(
    ("fraction", "label"),
    [(0.00001, "inside the header"), (0.02, "inside the first frames"), (0.0, "empty")],
)
def test_a_recording_cut_before_its_first_frame_exits_non_zero(
    matched: Path, layout: Path, tmp_path: Path, fraction: float, label: str
) -> None:
    cut = truncate(matched, tmp_path / "cut.mkv", fraction)
    produced = tmp_path / "out.json"

    result = analyse_text(cut, layout, "-o", str(produced))

    assert result.returncode == 1, label
    assert result.stdout == ""
    assert not produced.exists()
    assert "Traceback" not in result.stderr


def test_a_text_file_named_like_a_recording_exits_non_zero(layout: Path, tmp_path: Path) -> None:
    fake = tmp_path / "recording.mkv"
    fake.write_text("this is not a video\n")

    result = analyse_text(fake, layout)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_a_cached_rerun_does_not_check_the_tail_again(
    short: Path, layout: Path, tmp_path: Path
) -> None:
    shim = FfmpegShim(tmp_path / "shim")
    first = analyse_text(short, layout, env=shim.env)
    shim.reset()

    second = analyse_text(short, layout, env=shim.env)

    assert second.returncode == 0, second.stderr
    assert shim.decodes() == 0
    assert second.stdout == first.stdout


def test_the_start_time_is_read_from_the_file_name(
    short: Path, layout: Path, tmp_path: Path
) -> None:
    recording = named(short, tmp_path, "2026-10-09_11-59-53.mkv")

    assert started(recording, layout) == "2026-10-09T11:59:53"


def test_the_start_time_is_read_from_a_file_name_with_a_space(
    short: Path, layout: Path, tmp_path: Path
) -> None:
    recording = named(short, tmp_path, "2026-10-09 11-59-53.mkv")

    assert started(recording, layout) == "2026-10-09T11:59:53"


def test_a_recording_with_no_derivable_time_has_a_null_start(
    short: Path, layout: Path, tmp_path: Path
) -> None:
    recording = named(short, tmp_path, "recording.mkv")

    assert started(recording, layout) is None


def test_a_recording_without_a_hud_still_reports_its_start(
    no_hud: Path, layout: Path, tmp_path: Path
) -> None:
    recording = named(no_hud, tmp_path, "2026-10-09_11-59-53.mkv")

    assert started(recording, layout) == "2026-10-09T11:59:53"


def test_the_start_time_falls_back_to_the_container_metadata(
    short: Path, layout: Path, tmp_path: Path
) -> None:
    recording = remux_with_creation_time(short, tmp_path / "recording.mkv", "2026-10-09T09:59:53Z")

    assert started(recording, layout) == "2026-10-09T09:59:53+00:00"


def test_the_file_name_wins_over_the_container_metadata(
    short: Path, layout: Path, tmp_path: Path
) -> None:
    recording = remux_with_creation_time(
        short, tmp_path / "2026-10-09_11-59-53.mkv", "2026-10-09T09:59:53Z"
    )

    assert started(recording, layout) == "2026-10-09T11:59:53"


def test_an_impossible_date_in_the_file_name_is_ignored(
    short: Path, layout: Path, tmp_path: Path
) -> None:
    recording = named(short, tmp_path, "2026-13-45_99-99-99.mkv")

    assert started(recording, layout) is None


@pytest.mark.parametrize("creation_time", ["not a date", "", "0000-00-00T00:00:00Z"])
def test_an_unparsable_container_time_is_null(creation_time: str) -> None:
    assert started_at(Path("recording.mkv"), creation_time) is None


def test_the_schema_names_the_new_fields() -> None:
    schema = json_schema()

    assert schema["schema_version"] == "1.3"
    assert "started_at" in schema["$defs"]["RecordingInfo"]["properties"]
    assert "warnings" in schema["properties"]
    assert {"type": "null"} in schema["properties"]["layout"]["anyOf"]
