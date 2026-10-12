"""`analyse` withholds music-dominant audio from transcription and reports what it masked."""

import os
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from pydantic import Field, TypeAdapter

from tests.analyse_helpers import analyse_text
from tests.synthetic import AudioClip, music_bed, music_with_speech, write_layout, write_wav
from tests.test_speech import ENGLISH, PORTUGUESE, recording, run, transcripts
from victorious.music import (
    MODEL_FILE,
    MaskedWindow,
    MusicTagger,
    SecondScore,
    masked_windows,
    merge_windows,
    silence,
    unmasked,
)
from victorious.schema import MaskedAudioRecord, ObservationRecord
from victorious.speech import SpeechError

CLIP_SECONDS = 4.0
SIDECAR_RECORD: TypeAdapter[ObservationRecord | MaskedAudioRecord] = TypeAdapter(
    Annotated[ObservationRecord | MaskedAudioRecord, Field(discriminator="element")]
)

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")


@pytest.fixture(autouse=True)
def speech_on(speech_is_off_and_models_are_kept: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTORIOUS_NO_SPEECH")
    monkeypatch.setenv("HF_TOKEN", "test-token")


def bed(tmp_path: Path, at: float, seconds: float, name: str = "bed.wav") -> AudioClip:
    return AudioClip(write_wav(tmp_path / name, music_bed(seconds)), at=at)


def sidecar(
    source: Path, tmp_path: Path, *options: str
) -> tuple[list[ObservationRecord], list[MaskedAudioRecord]]:
    """The sidecar's HUD records and its `music_mask` records, checking they come last."""
    layout = write_layout(tmp_path / "layout.yaml")
    observations = tmp_path / "observations.jsonl"
    result = analyse_text(
        source,
        layout,
        "--no-cache",
        *options,
        "--observations",
        str(observations),
        "-o",
        str(tmp_path / "out.json"),
    )
    assert result.returncode == 0, result.stderr
    records = [SIDECAR_RECORD.validate_json(line) for line in observations.read_text().splitlines()]
    hud = [record for record in records if isinstance(record, ObservationRecord)]
    masks = [record for record in records if isinstance(record, MaskedAudioRecord)]
    assert records == [*hud, *masks]
    return hud, masks


def test_music_only_produces_no_transcript(tmp_path: Path) -> None:
    source = recording(tmp_path, bed(tmp_path, at=8, seconds=14))

    document = run(source, tmp_path)

    assert [transcripts(match) for match in document["matches"]] == [[], []]


def test_masked_windows_are_reported_in_the_observations_output(tmp_path: Path) -> None:
    source = recording(tmp_path, bed(tmp_path, at=10, seconds=14))

    hud, masks = sidecar(source, tmp_path)
    silent_hud, silent_masks = sidecar(source, tmp_path, "--no-speech")

    [mask] = masks
    assert abs(mask.start - 10) < 1.5
    assert abs(mask.end - 24) < 1.5
    assert mask.music >= 0.5
    assert 0 <= mask.speech < 0.25
    assert hud == silent_hud
    assert silent_masks == []


def speech_over_music(
    tmp_path: Path, speech: Path, *, start: float, speech_at: float, name: str, seconds: float = 18
) -> AudioClip:
    """`speech` at `speech_at` s into `seconds` of music at 0.3 gain, which starts at `start`."""
    mixed = music_with_speech(
        tmp_path / f"{name}.wav",
        speech,
        at=speech_at,
        seconds=seconds,
        speech_gain=1.0,
        music_gain=0.3,
    )
    return AudioClip(mixed, at=start)


def test_speech_over_music_is_kept(tmp_path: Path) -> None:
    source = recording(
        tmp_path,
        speech_over_music(tmp_path, ENGLISH, start=8, speech_at=2, name="en"),
        speech_over_music(tmp_path, PORTUGUESE, start=68, speech_at=2, name="pt"),
    )

    first, second = run(source, tmp_path)["matches"]

    [english] = transcripts(first)
    [portuguese] = transcripts(second)
    assert english["language"] == "en"
    assert "players" in english["text"].lower()
    assert abs(english["time"] - 10) < 1.5
    assert portuguese["language"] == "pt"
    assert "jogadores" in portuguese["text"].lower()
    assert abs(portuguese["time"] - 70) < 1.5


def test_speech_over_music_is_not_reported_as_masked(tmp_path: Path) -> None:
    source = recording(
        tmp_path, speech_over_music(tmp_path, ENGLISH, start=8, speech_at=2, name="en")
    )

    _, masks = sidecar(source, tmp_path)

    spoken = (10.0, 10.0 + CLIP_SECONDS)
    assert all(mask.end <= spoken[0] or mask.start >= spoken[1] for mask in masks)
    assert masks, "the music-only seconds around the speech are masked"


def test_speech_only_recording_is_unaffected(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8), AudioClip(PORTUGUESE, at=70))

    first, second = run(source, tmp_path)["matches"]
    _, masks = sidecar(source, tmp_path)

    [english] = transcripts(first)
    [portuguese] = transcripts(second)
    assert "players" in english["text"].lower()
    assert "jogadores" in portuguese["text"].lower()
    assert masks == []


def test_music_outside_matches_is_neither_tagged_nor_reported(tmp_path: Path) -> None:
    source = recording(tmp_path, bed(tmp_path, at=40, seconds=14))

    document = run(source, tmp_path)
    _, masks = sidecar(source, tmp_path)

    assert [transcripts(match) for match in document["matches"]] == [[], []]
    assert masks == []


def test_overlapping_margins_make_one_mask_decision(tmp_path: Path) -> None:
    across = speech_over_music(tmp_path, ENGLISH, start=26, speech_at=7, name="across", seconds=16)
    source = recording(tmp_path, across, second=40)

    matches = run(source, tmp_path)["matches"]
    _, masks = sidecar(source, tmp_path)

    assert masks
    assert sum(len(transcripts(match)) for match in matches) == 1
    for first, second in pairwise(masks):
        assert first.end < second.start
    for match in matches:
        for event in transcripts(match):
            middle = (event["time"] + event["end"]) / 2
            assert all(not mask.start < middle < mask.end for mask in masks)


def test_a_missing_audio_tagger_fails_cleanly(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=8))
    layout = write_layout(tmp_path / "layout.yaml")
    output = tmp_path / "out.json"
    env = {**os.environ, "HF_HOME": str(tmp_path / "no-models"), "HF_HUB_OFFLINE": "1"}

    result = analyse_text(source, layout, "--no-cache", "-o", str(output), env=env)

    assert result.returncode == 1
    assert "audio tagger" in result.stderr
    assert "--no-speech" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_a_corrupt_audio_tagger_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / MODEL_FILE
    model.write_bytes(b"this is not an onnx model")
    names = tmp_path / "names.csv"
    names.write_text("display_name\nSpeech\nMusic\n")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, name, **options: str(model if name == MODEL_FILE else names),
    )

    with pytest.raises(SpeechError, match="audio tagger"):
        MusicTagger.load()


def scores(*pairs: tuple[float, float]) -> list[SecondScore]:
    return [SecondScore(second, music, speech) for second, (music, speech) in enumerate(pairs)]


MUSIC = (0.9, 0.0)
CALL = (0.4, 0.8)
DIP = (0.3, 0.0)
SILENT = (0.0, 0.0)


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        pytest.param([MUSIC] * 5, [(0, 5)], id="a run"),
        pytest.param([MUSIC] * 3, [(0, 3)], id="three seconds are enough"),
        pytest.param([MUSIC] * 2, [], id="a stinger is not masked"),
        pytest.param([SILENT, *[MUSIC] * 4, SILENT], [(1, 5)], id="silence is not taken in"),
        pytest.param([MUSIC] * 3 + [DIP] + [MUSIC] * 3, [(0, 7)], id="a dip is bridged"),
        pytest.param([MUSIC] * 4 + [CALL] + [MUSIC] * 4, [(0, 3), (6, 9)], id="a call is kept"),
        pytest.param([CALL, *[MUSIC] * 4, CALL], [], id="speech on both sides of a short run"),
        pytest.param([CALL, *[MUSIC] * 5, CALL], [(2, 5)], id="speech on both sides is spared"),
        pytest.param([MUSIC] * 3 + [SILENT] * 2 + [MUSIC] * 3, [(0, 3), (5, 8)], id="long gap"),
        pytest.param([(0.5, 0.24)] * 3, [(0, 3)], id="just music"),
        pytest.param([(0.5, 0.25)] * 3, [], id="just speech"),
        pytest.param([(0.49, 0.0)] * 3, [], id="just not music"),
        pytest.param([(0.3, 0.0), *[MUSIC] * 3, (0.3, 0.0)], [(0, 5)], id="music at the edges"),
        pytest.param([(0.3, 0.3), *[MUSIC] * 4, (0.1, 0.0)], [(2, 5)], id="no edge near speech"),
    ],
)
def test_masked_windows_follow_the_music(
    pairs: list[tuple[float, float]], expected: list[tuple[int, int]]
) -> None:
    found = masked_windows(scores(*pairs))

    assert [(w.start, w.end) for w in found] == [(float(a), float(b)) for a, b in expected]


def test_masked_windows_are_in_recording_seconds_and_end_with_the_audio() -> None:
    [found] = masked_windows(scores(*[MUSIC] * 5), offset=100.5, duration=4.25)

    assert (found.start, found.end) == (100.5, 104.75)
    assert found.music == pytest.approx(0.9)
    assert found.speech == 0.0


def test_overlapping_and_touching_windows_merge_into_one() -> None:
    first = MaskedWindow(10.0, 20.0, 0.8, 0.0)
    second = MaskedWindow(15.0, 25.0, 0.6, 0.2)
    third = MaskedWindow(25.0, 30.0, 0.9, 0.0)
    apart = MaskedWindow(40.0, 45.0, 0.7, 0.1)

    merged = merge_windows([apart, third, first, second])

    assert [(w.start, w.end) for w in merged] == [(10.0, 30.0), (40.0, 45.0)]
    assert merged[1] == apart


def test_silence_zeroes_exactly_the_masked_samples() -> None:
    rate = 16000
    audio = np.ones(10 * rate, dtype="float32")
    audio.setflags(write=False)

    quiet = silence(audio, [MaskedWindow(102.0, 105.0, 0.9, 0.0)], offset=100.0)

    assert len(quiet) == len(audio)
    assert not quiet[2 * rate : 5 * rate].any()
    assert quiet[: 2 * rate].all()
    assert quiet[5 * rate :].all()
    assert audio.all()


def test_spans_are_cut_at_the_masked_seconds() -> None:
    rate = 16000
    masked = [MaskedWindow(102.0, 104.0, 0.9, 0.0), MaskedWindow(110.0, 111.0, 0.9, 0.0)]
    spans = [(0, rate), (rate, 5 * rate), (3 * rate, 9 * rate), (10 * rate, 10 * rate + 1000)]

    kept = unmasked(spans, masked, offset=100.0)

    assert kept == [(0, rate), (rate, 2 * rate), (4 * rate, 5 * rate), (4 * rate, 9 * rate)]
