"""`analyse` transcribes the speech of each Match, in English or Portuguese."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from tests.analyse_helpers import STEADY, FfmpegShim, analyse_text, small
from tests.synthetic import AudioClip, build_recording, match_shots, write_layout
from victorious.schema import json_schema
from victorious.speech import is_speech

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

SPEECH = Path(__file__).parent / "fixtures" / "speech"
ENGLISH = SPEECH / "en.opus"
PORTUGUESE = SPEECH / "pt.opus"
CLIP_SECONDS = 4.0


@pytest.fixture(autouse=True)
def speech_on(speech_is_off_and_models_are_kept: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTORIOUS_NO_SPEECH")
    monkeypatch.setenv("HF_TOKEN", "test-token")


def ffmpeg(*arguments: str | Path) -> None:
    tool = shutil.which("ffmpeg")
    assert tool is not None
    subprocess.run([tool, "-v", "error", "-y", *map(str, arguments)], check=True)


def recording(tmp_path: Path, *clips: AudioClip, name: str = "recording.mkv") -> Path:
    """Two Matches, 5-25 s and 65-85 s, in a 100 s Recording with `clips` mixed in."""
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=1),
        *match_shots(65, STEADY, results_at=85, placement=6),
    )
    return build_recording(replace(small(100, shots), audio=clips), tmp_path / name)


def run(recording: Path, tmp_path: Path, *options: str) -> dict[str, Any]:
    layout = write_layout(tmp_path / "layout.yaml")
    result = analyse_text(recording, layout, "--no-cache", *options)
    assert result.returncode == 0, result.stderr
    document: dict[str, Any] = json.loads(result.stdout)
    jsonschema.Draft202012Validator(json_schema()).validate(document)
    return document


def transcripts(match: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in match["events"] if event["kind"] == "transcript"]


def test_english_and_portuguese_are_tagged(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8), AudioClip(PORTUGUESE, at=70))

    first, second = run(source, tmp_path)["matches"]

    [english] = transcripts(first)
    [portuguese] = transcripts(second)
    assert english["language"] == "en"
    assert "players" in english["text"].lower()
    assert portuguese["language"] == "pt"
    assert "jogadores" in portuguese["text"].lower()
    for event, started in ((english, 8), (portuguese, 70)):
        assert event["source"] == "speech"
        assert abs(event["time"] - started) < 1.5
        assert event["time"] < event["end"] <= started + CLIP_SECONDS + 2
        assert 0 < event["confidence"] <= 1


def test_audio_outside_matches_is_not_transcribed(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=40))

    document = run(source, tmp_path)

    assert [transcripts(match) for match in document["matches"]] == [[], []]


def test_speech_just_after_the_results_screen_is_kept(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=29))

    first, second = run(source, tmp_path)["matches"]

    [event] = transcripts(first)
    assert event["time"] > first["end"]
    assert event["language"] == "en"
    assert transcripts(second) == []


def test_speech_between_two_matches_belongs_to_the_nearer_one(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=60), AudioClip(ENGLISH, at=29))

    first, second = run(source, tmp_path)["matches"]

    assert [event["time"] for event in transcripts(first)] == pytest.approx([29], abs=1.5)
    assert [event["time"] for event in transcripts(second)] == pytest.approx([60], abs=1.5)


def test_no_speech_skips_transcription_and_models(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8))
    layout = write_layout(tmp_path / "layout.yaml")
    output = tmp_path / "out.json"
    program = (
        "import sys; from victorious.cli import main; "
        "code = main(sys.argv[1:]); "
        "print(code, 'faster_whisper' in sys.modules)"
    )

    result = subprocess.run(
        [
            sys.executable, "-c", program, "analyse", str(source), "--layout", str(layout),
            "--no-cache", "--no-speech", "-o", str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_TOKEN": "", "HF_HOME": str(tmp_path / "no-models")},
        check=False,
    )  # fmt: skip

    assert result.stdout.split() == ["0", "False"], result.stderr
    document = json.loads(output.read_text())
    assert [transcripts(match) for match in document["matches"]] == [[], []]
    assert not (tmp_path / "no-models").exists()


def test_missing_token_fails_before_any_decode(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8))
    layout = write_layout(tmp_path / "layout.yaml")
    output = tmp_path / "out.json"
    shim = FfmpegShim(tmp_path / "shim")
    env = {
        **shim.env,
        "HF_TOKEN": "",
        "HUGGING_FACE_HUB_TOKEN": "",
        "HF_HOME": str(tmp_path / "no-token"),
    }

    result = analyse_text(source, layout, "-o", str(output), env=env)

    assert result.returncode == 1
    assert "Hugging Face" in result.stderr
    assert "HF_TOKEN" in result.stderr
    assert "--no-speech" in result.stderr
    assert "Traceback" not in result.stderr
    assert shim.lines() == []
    assert not output.exists()


def with_second_audio_stream(first: Path, second: Path, destination: Path) -> Path:
    ffmpeg("-i", first, "-i", second, "-map", "0", "-map", "1:a", "-c", "copy", destination)
    return destination


def test_audio_stream_selects_the_stream(tmp_path: Path) -> None:
    portuguese = recording(tmp_path, AudioClip(PORTUGUESE, at=8), name="pt.mkv")
    english = recording(tmp_path, AudioClip(ENGLISH, at=8), name="en.mkv")
    both = with_second_audio_stream(portuguese, english, tmp_path / "both.mkv")

    default = run(both, tmp_path)["matches"][0]
    second = run(both, tmp_path, "--audio-stream", "1")["matches"][0]

    assert [event["language"] for event in transcripts(default)] == ["pt"]
    assert [event["language"] for event in transcripts(second)] == ["en"]


def test_an_audio_stream_the_recording_lacks_fails_before_any_decode(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8))
    layout = write_layout(tmp_path / "layout.yaml")
    output = tmp_path / "out.json"
    shim = FfmpegShim(tmp_path / "shim")

    result = analyse_text(source, layout, "--audio-stream", "2", "-o", str(output), env=shim.env)

    assert result.returncode == 1
    assert "1 audio stream" in result.stderr
    assert "--no-speech" in result.stderr
    assert shim.lines() == []
    assert not output.exists()


def test_a_recording_without_audio_names_no_speech(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8))
    silent = tmp_path / "silent.mkv"
    ffmpeg("-i", source, "-c:v", "copy", "-an", silent)
    layout = write_layout(tmp_path / "layout.yaml")
    shim = FfmpegShim(tmp_path / "shim")

    result = analyse_text(silent, layout, env=shim.env)

    assert result.returncode == 1
    assert "no audio stream" in result.stderr
    assert "--no-speech" in result.stderr
    assert shim.lines() == []


def test_silence_yields_no_transcript(tmp_path: Path) -> None:
    document = run(recording(tmp_path), tmp_path)

    assert [transcripts(match) for match in document["matches"]] == [[], []]


def test_speech_does_not_change_the_frame_events(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8), AudioClip(PORTUGUESE, at=70))

    spoken = run(source, tmp_path)
    silent = run(source, tmp_path, "--no-speech")

    def frames(document: dict[str, Any]) -> list[Any]:
        return [
            [event for event in match["events"] if event["kind"] != "transcript"]
            for match in document["matches"]
        ]

    assert frames(spoken) == frames(silent)
    assert any(transcripts(match) for match in spoken["matches"])


@pytest.mark.parametrize(
    ("no_speech_prob", "avg_logprob", "text", "kept"),
    [
        (0.85, -0.5, " Nice shot", True),
        (0.5, -0.3, " Nice shot", True),
        (0.5, -1.9, " Thanks for watching", False),
        (0.97, -0.4, " Nice shot", False),
        (0.2, -0.2, "   ", False),
    ],
)
def test_low_confidence_segments_are_dropped(
    no_speech_prob: float, avg_logprob: float, text: str, kept: bool
) -> None:
    assert is_speech(no_speech_prob, avg_logprob, text) is kept
