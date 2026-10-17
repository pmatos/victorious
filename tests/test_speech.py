"""`analyse` transcribes the speech of each Match, in English or Portuguese."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema
import numpy as np
import pytest

from tests.analyse_helpers import STEADY, FfmpegShim, analyse_text, small
from tests.synthetic import AudioClip, build_recording, match_shots, write_layout
from victorious.schema import SpeechLanguage, json_schema
from victorious.speech import (
    Segment,
    _transcribe_window,
    candidate_languages,
    drop_repeated,
    group_speech,
    is_speech,
    outside_spans,
    pick_reading,
)
from victorious.video import AUDIO_RATE, sample_audio

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

SPEECH = Path(__file__).parent / "fixtures" / "speech"
ENGLISH = SPEECH / "en.opus"
PORTUGUESE = SPEECH / "pt.opus"
CLIP_SECONDS = 4.0


@pytest.fixture(autouse=True)
def speech_on(speech_is_off_and_models_are_kept: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTORIOUS_NO_SPEECH")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("VICTORIOUS_NO_CATEGORIES", "1")


def ffmpeg(*arguments: str | Path) -> None:
    tool = shutil.which("ffmpeg")
    assert tool is not None
    subprocess.run([tool, "-v", "error", "-y", *map(str, arguments)], check=True)


def recording(
    tmp_path: Path, *clips: AudioClip, name: str = "recording.mkv", second: int = 65
) -> Path:
    """Two Matches, 5-25 s and `second` + 20 s, in a 100 s Recording with `clips` mixed in."""
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=1),
        *match_shots(second, STEADY, results_at=second + 20, placement=6),
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


@pytest.mark.parametrize(("at", "owner"), [(27, 0), (31, 1)])
def test_speech_in_two_overlapping_margins_belongs_to_the_nearer_match(
    tmp_path: Path, at: int, owner: int
) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=at), second=40)

    matches = run(source, tmp_path)["matches"]

    assert [len(transcripts(match)) for match in matches] == [1 - owner, owner]
    [event] = transcripts(matches[owner])
    assert abs(event["time"] - at) < 1.5


def test_no_speech_skips_transcription_and_models(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8))
    layout = write_layout(tmp_path / "layout.yaml")
    output = tmp_path / "out.json"
    program = (
        "import sys; from victorious.cli import main; "
        "code = main(sys.argv[1:]); "
        "print(code, *(name in sys.modules "
        "for name in ('faster_whisper', 'onnxruntime', 'pyannote', 'torch')))"
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

    assert result.stdout.split() == ["0", "False", "False", "False", "False"], result.stderr
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


def test_an_unknown_whisper_model_fails_cleanly(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8))
    layout = write_layout(tmp_path / "layout.yaml")

    result = analyse_text(source, layout, "--no-cache", "--whisper-model", "no-such-model")

    assert result.returncode == 1
    assert "no-such-model" in result.stderr
    assert "--no-speech" in result.stderr
    assert "Traceback" not in result.stderr


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
    ("no_speech_prob", "avg_logprob", "compression_ratio", "text", "kept"),
    [
        (0.85, -0.5, 1.0, " Nice shot", True),
        (0.5, -0.3, 1.0, " Nice shot", True),
        (0.4, -1.3, 1.0, " Nice shot", True),
        (0.3, -1.95, 1.0, " the", False),
        (0.97, -0.4, 1.0, " Nice shot", False),
        (0.2, -0.2, 1.0, "   ", False),
        (0.4, -0.45, 3.42, " Someone killed me, ayyyyyyyyyyyyyyyyyyyyyyyyyyyyyy", False),
        (0.4, -0.45, 2.3, " Nice shot", True),
    ],
)
def test_low_confidence_segments_are_dropped(
    no_speech_prob: float, avg_logprob: float, compression_ratio: float, text: str, kept: bool
) -> None:
    assert is_speech(no_speech_prob, avg_logprob, compression_ratio, text) is kept


def _said(*texts: str, language: SpeechLanguage = "en", confidence: float = 0.4) -> list[Segment]:
    return [
        Segment(
            text=text,
            language=language,
            start=2.0 * i,
            end=2.0 * i + 2.0,
            confidence=confidence,
        )
        for i, text in enumerate(texts)
    ]


def test_a_phrase_repeated_back_to_back_is_a_hallucinated_loop() -> None:
    loop = _said("Nice shot", "the", "the", "The.", "the", "Thank you.", "Thank you.")

    assert [s.text for s in drop_repeated(loop)] == ["Nice shot", "Thank you.", "Thank you."]


def test_a_repeat_that_is_not_back_to_back_is_kept() -> None:
    said = _said("No", "No", "Go left", "No", "No", "Nice shot", "No")

    assert drop_repeated(said) == said


def test_nearby_speech_is_grouped_up_to_one_window() -> None:
    spans = [(0, 100), (150, 300), (320, 900), (1000, 1100), (5000, 5100)]

    assert group_speech(spans, max_length=1000, max_gap=200) == [
        (0, 900),
        (1000, 1100),
        (5000, 5100),
    ]
    assert group_speech(spans, max_length=500, max_gap=200) == [
        (0, 300),
        (320, 900),
        (1000, 1100),
        (5000, 5100),
    ]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"en": 0.9, "pt": 0.01, "es": 0.02}, ("en",)),
        ({"en": 0.28, "pt": 0.57}, ("pt", "en")),
        ({"en": 0.33, "pt": 0.31}, ("en", "pt")),
        ({"en": 0.4, "es": 0.4}, ("en", "pt")),
        ({"en": 0.5, "es": 0.4}, ("en",)),
        ({"es": 0.9}, ("en", "pt")),
    ],
)
def test_both_languages_are_tried_when_detection_is_unsure(
    scores: dict[str, float], expected: tuple[str, ...]
) -> None:
    assert candidate_languages(scores) == expected


def test_the_language_that_reads_the_speech_best_wins() -> None:
    english = _said("Oh, they're healing", "Okay, right", language="en", confidence=0.57)
    portuguese = _said("Ah, ele esta a fome", "Ok, certo", language="pt", confidence=0.36)

    assert pick_reading({"pt": portuguese, "en": english}) == english
    assert pick_reading({"pt": english, "en": portuguese}) == english


def test_a_language_that_only_loops_does_not_win() -> None:
    english = _said("Oh, they're healing", "Okay, right", language="en", confidence=0.5)
    loop = _said("oh", "Oh.", "oh", language="pt", confidence=0.9)

    assert pick_reading({"pt": loop, "en": english}) == english
    assert pick_reading({"pt": loop}) == []


def test_a_language_that_only_repeats_one_word_loses_to_one_that_does_not() -> None:
    english = _said("Oh, they're healing", "Okay, right", language="en", confidence=0.4)
    chant = _said("oi oi oi oi oi oi", "oi oi oi oi", language="pt", confidence=0.9)

    assert pick_reading({"pt": chant, "en": english}) == english
    assert pick_reading({"pt": chant}) == chant


def test_a_word_repeated_inside_a_segment_is_kept_beside_real_speech() -> None:
    said = _said("No, no, no, no, no.", "Go left", language="en", confidence=0.5)

    assert pick_reading({"en": said}) == said


def test_the_padding_beyond_the_match_windows_is_cut_from_the_diarization_audio() -> None:
    windows = [(10.0, 20.0), (30.0, 40.0)]

    cuts = outside_spans(windows, 9.0, 31.0)

    assert [(cut.start, cut.end) for cut in cuts] == [(9.0, 10.0), (20.0, 30.0)]
    assert outside_spans(windows, 12.0, 18.0) == []
    assert [(c.start, c.end) for c in outside_spans(windows, 38.0, 41.0)] == [(40.0, 41.0)]


def _part(
    start: float,
    end: float,
    text: str,
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=1.1,
    )


class ScriptedModel:
    """Stands in for `WhisperModel`: a fixed language guess and one script per language."""

    def __init__(self, scores: dict[str, float], scripts: dict[str, list[SimpleNamespace]]) -> None:
        self.scores = scores
        self.scripts = scripts
        self.calls: list[dict[str, Any]] = []

    def detect_language(self, audio: Any) -> tuple[str, float, list[tuple[str, float]]]:
        return "en", 0.0, list(self.scores.items())

    def transcribe(
        self,
        audio: Any,
        *,
        language: str,
        clip_timestamps: list[float],
        condition_on_previous_text: bool,
    ) -> tuple[Any, None]:
        self.calls.append(
            {
                "language": language,
                "clip_timestamps": clip_timestamps,
                "condition_on_previous_text": condition_on_previous_text,
            }
        )
        return iter(self.scripts[language]), None


def test_speech_is_decoded_in_order_with_the_text_before_it_as_context() -> None:
    clip = sample_audio(ENGLISH, 0.0, CLIP_SECONDS, 0)
    model = ScriptedModel({"en": 0.95, "pt": 0.01}, {"en": [_part(0.2, 3.0, " Hello there.")]})

    segments = _transcribe_window(model, clip, 100.0)

    [call] = model.calls
    assert call["condition_on_previous_text"] is True
    start, end = call["clip_timestamps"]
    assert 0.0 <= start < end <= CLIP_SECONDS
    assert [(s.text, s.language, s.start, s.end) for s in segments] == [
        ("Hello there.", "en", 100.2, 103.0)
    ]


def test_a_segment_that_is_not_speech_is_not_kept_after_sequential_decoding() -> None:
    clip = sample_audio(ENGLISH, 0.0, CLIP_SECONDS, 0)
    script = [_part(0.2, 1.0, " Hello there."), _part(1.0, 2.0, " the", avg_logprob=-1.9)]
    model = ScriptedModel({"en": 0.95, "pt": 0.01}, {"en": script})

    segments = _transcribe_window(model, clip, 0.0)

    assert [s.text for s in segments] == ["Hello there."]


def test_an_unsure_language_is_decoded_twice_and_the_better_reading_is_kept() -> None:
    clip = sample_audio(ENGLISH, 0.0, CLIP_SECONDS, 0)
    scripts = {
        "en": [_part(0.2, 3.0, " Hello there.", avg_logprob=-0.2)],
        "pt": [_part(0.2, 3.0, " Ola ali.", avg_logprob=-1.2)],
    }
    model = ScriptedModel({"en": 0.4, "pt": 0.38}, scripts)

    segments = _transcribe_window(model, clip, 0.0)

    assert [call["language"] for call in model.calls] == ["en", "pt"]
    assert [(s.text, s.language) for s in segments] == [("Hello there.", "en")]


def test_stretches_of_speech_apart_are_decoded_in_one_ordered_call() -> None:
    clip = sample_audio(ENGLISH, 0.0, CLIP_SECONDS, 0)
    gap = np.zeros(int(10 * AUDIO_RATE), dtype=clip.dtype)
    audio = np.concatenate([clip, gap, clip])
    script = [_part(0.2, 3.0, " First."), _part(14.2, 17.0, " Second.")]
    model = ScriptedModel({"en": 0.95, "pt": 0.01}, {"en": script})

    segments = _transcribe_window(model, audio, 50.0)

    [call] = model.calls
    first, second = call["clip_timestamps"][:2], call["clip_timestamps"][2:]
    assert len(first) == len(second) == 2
    assert first[0] < first[1] < second[0] < second[1]
    assert [(s.text, s.start) for s in segments] == [("First.", 50.2), ("Second.", 64.2)]
