"""Behaviour of `victorious extract --wins`."""

from pathlib import Path

import pytest

from tests.cli_runner import run_cli
from tests.synthetic import RecordingSpec, build_recording, match_shots
from victorious.extract import win_clips
from victorious.schema import AnalysisOutput, Match, RecordingInfo
from victorious.video import probe

DURATION = 85.0
KEYFRAME_INTERVAL = 2
SPANS = [(2.0, 20.0, True), (30.0, 45.0, False), (60.0, 75.0, True)]


def _match(index: int, start: float, end: float, win: bool) -> Match:
    return Match(
        index=index,
        start=start,
        end=end,
        win=win,
        placement=1 if win else 6,
        personal_eliminations=0,
        events=[],
    )


def _analysis(
    recording: Path, spans: list[tuple[float, float, bool]] = SPANS, duration: float = DURATION
) -> AnalysisOutput:
    return AnalysisOutput(
        recording=RecordingInfo(path=str(recording), duration=duration, started_at=None),
        layout=None,
        warnings=[],
        matches=[_match(index, *span) for index, span in enumerate(spans)],
    )


def _write(path: Path, output: AnalysisOutput) -> Path:
    path.write_text(output.model_dump_json(indent=2))
    return path


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory, require_ffmpeg: None) -> Path:
    shots = (
        *match_shots(2, ((0, 0, 0, 99),), 20, placement=1),
        *match_shots(30, ((0, 0, 0, 99),), 45, placement=6, results_style="placed"),
        *match_shots(60, ((0, 0, 0, 99),), 75, placement=1),
    )
    spec = RecordingSpec(
        duration=DURATION,
        shots=shots,
        fps=5.0,
        width=640,
        height=360,
        keyframe_interval=KEYFRAME_INTERVAL,
    )
    return build_recording(spec, tmp_path_factory.mktemp("recording") / "recording.mkv")


def test_the_wins_are_cut_and_joined_with_a_tail_after_each(
    recording: Path, tmp_path: Path
) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(recording))
    output = tmp_path / "wins.mkv"

    result = run_cli("extract", str(analysis), "--wins", "-o", str(output), "--tail", "3")

    assert result.returncode == 0, result.stderr
    assert "cutting Match 0" in result.stderr
    assert "cutting Match 2" in result.stderr
    assert "wrote" in result.stderr
    wanted = (23 - 2) + (78 - 60)
    assert wanted - 1.0 <= probe(output).duration <= wanted + 2 * KEYFRAME_INTERVAL + 1.0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "analysis.json",
        "wins.mkv",
    ]


def test_exact_cuts_are_re_encoded_to_start_on_their_second(
    recording: Path, tmp_path: Path
) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(recording, [(3.0, 20.0, True)]))
    output = tmp_path / "wins.mkv"

    result = run_cli(
        "extract", str(analysis), "--wins", "--exact", "-o", str(output), "--tail", "3"
    )

    assert result.returncode == 0, result.stderr
    assert probe(output).duration == pytest.approx(23 - 3, abs=1.0)
    assert probe(output).codec == "h264"


def test_copied_cuts_keep_the_codec_of_the_recording(recording: Path, tmp_path: Path) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(recording, [(3.0, 20.0, True)]))
    output = tmp_path / "wins.mkv"

    result = run_cli("extract", str(analysis), "--wins", "-o", str(output), "--tail", "3")

    assert result.returncode == 0, result.stderr
    early = probe(output).duration - (23 - 3)
    assert -1.0 <= early <= KEYFRAME_INTERVAL + 1.0
    assert probe(output).codec == probe(recording).codec


def test_the_recording_can_be_given_when_the_analysis_path_is_stale(
    recording: Path, tmp_path: Path
) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(tmp_path / "moved.mkv"))
    output = tmp_path / "wins.mkv"

    missing = run_cli("extract", str(analysis), "--wins", "-o", str(output))
    found = run_cli(
        "extract", str(analysis), "--wins", "-o", str(output), "--recording", str(recording)
    )

    assert missing.returncode == 1
    assert "--recording" in missing.stderr
    assert found.returncode == 0, found.stderr
    assert output.is_file()


def test_a_tail_never_runs_into_the_next_match_or_past_the_recording() -> None:
    output = _analysis(
        Path("recording.mkv"),
        [(0.0, 10.0, True), (12.0, 20.0, False), (30.0, 40.0, True)],
        duration=43.0,
    )

    clips = win_clips(output, tail=8.0, duration=43.0)

    assert [(clip.index, clip.start, clip.end) for clip in clips] == [
        (0, 0.0, 12.0),
        (2, 30.0, 43.0),
    ]


def test_a_recording_with_no_win_writes_nothing(recording: Path, tmp_path: Path) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(recording, [(30.0, 45.0, False)]))
    output = tmp_path / "wins.mkv"

    result = run_cli("extract", str(analysis), "--wins", "-o", str(output))

    assert result.returncode == 1
    assert "no Win" in result.stderr
    assert not output.exists()


def test_a_failed_cut_leaves_no_output_and_no_scratch_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mkv"
    broken.write_bytes(b"not a video")
    analysis = _write(tmp_path / "analysis.json", _analysis(broken))
    output = tmp_path / "wins.mkv"

    result = run_cli("extract", str(analysis), "--wins", "-o", str(output))

    assert result.returncode == 1
    assert result.stderr.startswith("victorious extract:")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["analysis.json", "broken.mkv"]


def test_the_output_may_not_overwrite_the_recording(recording: Path, tmp_path: Path) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(recording))
    before = recording.read_bytes()

    result = run_cli("extract", str(analysis), "--wins", "-o", str(recording))

    assert result.returncode == 1
    assert "must not name" in result.stderr
    assert recording.read_bytes() == before


def test_the_output_needs_an_extension(recording: Path, tmp_path: Path) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(recording))

    result = run_cli("extract", str(analysis), "--wins", "-o", str(tmp_path / "wins"))

    assert result.returncode == 1
    assert "extension" in result.stderr
    assert not (tmp_path / "wins").exists()


def test_the_output_folder_must_exist(recording: Path, tmp_path: Path) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis(recording))

    result = run_cli("extract", str(analysis), "--wins", "-o", str(tmp_path / "nope" / "w.mkv"))

    assert result.returncode == 1
    assert "output directory does not exist" in result.stderr


def test_wins_must_be_asked_for(tmp_path: Path) -> None:
    result = run_cli("extract", "analysis.json", "-o", str(tmp_path / "wins.mkv"))

    assert result.returncode == 2
    assert "--wins" in result.stderr
