"""Decoding a Recording in parallel chunks gives the same Analysis as one serial decode."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.analyse_helpers import FfmpegShim, analyse_text, prepare, small
from tests.synthetic import Hud, RecordingSpec, Results, Shot, match_shots
from victorious.schema import AnalysisOutput

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

SERIAL = ("--jobs", "1", "--chunk-seconds", "100000")


def two_matches() -> RecordingSpec:
    first = (
        (0.0, 0, 0, 99),
        (8.5, 1, 0, 99),
        (17.2, 3, 0, 97),
        (30.0, 3, 1, 96),
    )
    second = ((0.0, 0, 0, 99), (3.0, 1, 0, 98))
    shots = (
        *match_shots(2, first, results_at=36, placement=1),
        *match_shots(42, second, results_at=47, placement=7),
    )
    return small(55, shots)


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return prepare(tmp_path_factory.mktemp("parallel"), two_matches())


def output_of(recording: tuple[Path, Path], *options: str) -> str:
    result = analyse_text(*recording, "--no-cache", *options)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_chunked_parallel_output_equals_serial_decode(
    recording: tuple[Path, Path], tmp_path: Path
) -> None:
    serial = output_of(recording, *SERIAL)

    shim = FfmpegShim(tmp_path / "shim")
    chunked = analyse_text(
        *recording, "--no-cache", "--jobs", "3", "--chunk-seconds", "10", env=shim.env
    )
    assert chunked.returncode == 0, chunked.stderr
    assert chunked.stdout == serial
    assert shim.chunk_decodes() == 6
    assert output_of(recording, "--jobs", "2", "--chunk-seconds", "7") == serial
    assert output_of(recording, "--jobs", "4") == serial

    document: dict[str, Any] = json.loads(serial)
    AnalysisOutput.model_validate(document)
    assert [(m["start"], m["end"], m["win"], m["placement"]) for m in document["matches"]] == [
        (2.0, 36.0, True, 1),
        (42.0, 47.0, False, 7),
    ]
    kinds = [event["kind"] for event in document["matches"][0]["events"]]
    assert kinds.count("personal_eliminations") == 3
    assert document["matches"][0]["personal_eliminations"] == 3


@pytest.mark.parametrize("jobs", ["1", "2", "8"])
def test_jobs_has_no_effect_on_output(recording: tuple[Path, Path], jobs: str) -> None:
    assert output_of(recording, "--jobs", jobs, "--chunk-seconds", "13") == output_of(
        recording, *SERIAL
    )


def test_a_failing_chunk_fails_the_run_without_output(tmp_path: Path) -> None:
    shots = (Shot(Hud(), 2, 30), Shot(Results(placement=3), 30, 35))
    recording, layout = prepare(tmp_path, small(40, shots))
    shim = FfmpegShim(tmp_path / "shim", fail_when="-ss 20 -t 5")
    output = tmp_path / "out.json"

    result = analyse_text(
        recording, layout, "--jobs", "4", "--chunk-seconds", "5", "-o", str(output), env=shim.env
    )

    assert result.returncode == 1
    assert result.stderr.startswith("victorious analyse: cannot decode Recording")
    assert "chunk 20 refused" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "options", [("--jobs", "0"), ("--jobs", "-2"), ("--chunk-seconds", "0"), ("--jobs", "many")]
)
def test_invalid_jobs_or_chunk_seconds_is_a_usage_error(
    tmp_path: Path, options: tuple[str, ...]
) -> None:
    recording, layout = prepare(tmp_path, small(5, ()))

    result = analyse_text(recording, layout, *options)

    assert result.returncode == 2
    assert result.stdout == ""
