"""`analyse` tells Speakers apart, names enrolled voices and numbers the others per Recording."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from tests.analyse_helpers import FfmpegShim
from tests.cli_runner import run_cli
from tests.synthetic import AudioClip, write_layout
from tests.test_speech import ENGLISH, PORTUGUESE, SPEECH, recording, transcripts
from victorious.errors import SpeechError
from victorious.schema import SCHEMA_VERSION, Event, json_schema
from victorious.speakers import (
    DIARIZATION_REPO,
    MATCH_DISTANCE,
    Diarizer,
    EnrolledVoice,
    EnrollmentClip,
    ParallelEmbedding,
    Piece,
    Turn,
    clamped,
    fetch_diarizer,
    name_speakers,
    padded_spans,
    pieces_for,
    speaker_for_segment,
)

ENROLLMENT = SPEECH / "en_enroll.opus"
UNKNOWN = re.compile(r"^unknown_\d+$")

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")


@pytest.fixture(autouse=True)
def speech_on(speech_is_off_and_models_are_kept: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTORIOUS_NO_SPEECH")
    monkeypatch.setenv("HF_TOKEN", "test-token")


def enrol(tmp_path: Path, **clips: Path) -> Path:
    directory = tmp_path / "voices"
    directory.mkdir()
    for name, clip in clips.items():
        shutil.copy(clip, directory / f"{name}{clip.suffix}")
    return directory


def analyse(
    source: Path, tmp_path: Path, *options: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    layout = write_layout(tmp_path / "layout.yaml")
    return run_cli(
        "analyse", str(source), "--layout", str(layout), "--no-cache", *options, cwd=cwd or tmp_path
    )


def speakers_of(
    source: Path, tmp_path: Path, *options: str, cwd: Path | None = None
) -> list[list[str | None]]:
    result = analyse(source, tmp_path, *options, cwd=cwd)
    assert result.returncode == 0, result.stderr
    document: dict[str, Any] = json.loads(result.stdout)
    jsonschema.Draft202012Validator(json_schema()).validate(document)
    return [[event["speaker"] for event in transcripts(m)] for m in document["matches"]]


def test_two_voices_yield_two_distinct_speakers(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6), AudioClip(PORTUGUESE, at=17))
    nobody = tmp_path / "nobody"

    first, second = speakers_of(source, tmp_path, "--voices", str(nobody))

    assert second == []
    assert [bool(label and UNKNOWN.match(label)) for label in first] == [True, True]
    assert first[0] != first[1]


def test_an_enrolled_voice_is_labelled_by_name(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6), AudioClip(PORTUGUESE, at=17))
    voices = enrol(tmp_path, alice=ENROLLMENT)

    first, _ = speakers_of(source, tmp_path, "--voices", str(voices))

    assert first == ["alice", "unknown_1"]


def test_unknown_labels_are_stable_within_a_recording(tmp_path: Path) -> None:
    source = recording(
        tmp_path,
        AudioClip(ENGLISH, at=6),
        AudioClip(PORTUGUESE, at=17),
        AudioClip(ENGLISH, at=70),
    )

    first, second = speakers_of(source, tmp_path, "--voices", str(tmp_path / "nobody"))

    assert first == ["unknown_1", "unknown_2"]
    assert second == ["unknown_1"]


def test_a_missing_voices_folder_is_not_an_error(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))
    quiet = tmp_path / "quiet"
    quiet.mkdir()

    default = analyse(source, tmp_path)
    empty = analyse(source, tmp_path, "--voices", str(quiet))
    missing = analyse(source, tmp_path, "--voices", str(tmp_path / "nowhere"))

    for result in (default, empty, missing):
        assert result.returncode == 0, result.stderr
        [match, _] = json.loads(result.stdout)["matches"]
        [event] = transcripts(match)
        assert event["speaker"] == "unknown_1"
        assert "Traceback" not in result.stderr
    assert "warning" not in default.stderr
    assert "warning" not in empty.stderr
    assert "warning" in missing.stderr
    assert "nowhere" in missing.stderr


def test_a_voices_path_that_is_a_file_is_refused_as_not_a_folder(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))
    clip = tmp_path / "alice.opus"
    shutil.copy(ENROLLMENT, clip)

    result = analyse(source, tmp_path, "--voices", str(clip))

    assert result.returncode == 1
    assert "is not a folder" in result.stderr
    assert "does not exist" not in result.stderr
    assert "Traceback" not in result.stderr


def test_a_voices_folder_in_the_working_directory_is_used(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))
    enrol(tmp_path, alice=ENROLLMENT)

    [first, _] = speakers_of(source, tmp_path)

    assert first == ["alice"]


def test_an_undecodable_enrollment_clip_fails_before_any_decode(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))
    voices = enrol(tmp_path, alice=ENROLLMENT)
    (voices / "bad.wav").write_bytes(b"not audio")
    shim = FfmpegShim(tmp_path / "shim")
    output = tmp_path / "out.json"

    layout = write_layout(tmp_path / "layout.yaml")

    result = run_cli(
        "analyse", str(source), "--layout", str(layout), "--no-cache",
        "--voices", str(voices), "-o", str(output), env=shim.env,
    )  # fmt: skip

    assert result.returncode == 1
    assert "bad.wav" in result.stderr
    assert "--no-speech" in result.stderr
    assert "Traceback" not in result.stderr
    assert shim.decodes(str(source)) == 0
    assert not output.exists()


@pytest.mark.parametrize("name", ["unknown_3", "unknown_x"])
def test_an_enrollment_named_like_an_unknown_voice_is_refused(tmp_path: Path, name: str) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))
    voices = enrol(tmp_path, **{name: ENROLLMENT})

    result = analyse(source, tmp_path, "--voices", str(voices))

    assert result.returncode == 1
    assert name in result.stderr
    assert "Traceback" not in result.stderr


def test_two_clips_for_one_person_are_refused(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))
    voices = enrol(tmp_path, alice=ENROLLMENT)
    shutil.copy(ENGLISH, voices / "alice.ogg")

    result = analyse(source, tmp_path, "--voices", str(voices))

    assert result.returncode == 1
    assert "alice" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_diarization_model_fails_cleanly(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))
    output = tmp_path / "out.json"

    result = analyse(source, tmp_path, "--diarization-model", "/nonexistent", "-o", str(output))

    assert result.returncode == 1
    assert "diarization" in result.stderr
    assert "--no-speech" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_the_speaker_is_part_of_the_schema() -> None:
    assert SCHEMA_VERSION == "1.6"
    assert "speaker" in Event.model_fields
    assert "speaker" in json_schema()["$defs"]["Event"]["properties"]


def test_frame_events_have_no_speaker(tmp_path: Path) -> None:
    source = recording(tmp_path, AudioClip(ENGLISH, at=6))

    result = analyse(source, tmp_path, "--no-speech")

    document = json.loads(result.stdout)
    events = [event for match in document["matches"] for event in match["events"]]
    assert events
    assert all(event["speaker"] is None for event in events)


def turn(start: float, end: float, speaker: str) -> Turn:
    return Turn(start, end, speaker)


def test_a_segment_takes_the_speaker_with_the_most_overlap() -> None:
    turns = [turn(0, 4, "a"), turn(4, 5.5, "b"), turn(5.5, 9, "a")]

    assert speaker_for_segment(3.0, 5.0, turns) == "a"
    assert speaker_for_segment(3.9, 6.0, turns) == "b"
    assert speaker_for_segment(7.0, 8.0, turns) == "a"


def test_a_segment_overlapping_no_turn_takes_the_nearest_one() -> None:
    turns = [turn(0, 2, "a"), turn(10, 12, "b")]

    assert speaker_for_segment(3.0, 4.0, turns) == "a"
    assert speaker_for_segment(8.5, 9.5, turns) == "b"


def test_a_segment_has_no_speaker_only_when_there_are_no_turns() -> None:
    assert speaker_for_segment(1.0, 2.0, []) is None


def voice(name: str, *vector: float) -> EnrolledVoice:
    return EnrolledVoice(name, np.array(vector, dtype="float32"))


def test_the_nearest_enrolled_voice_names_a_speaker_within_the_match_distance() -> None:
    enrolled = [voice("alice", 1.0, 0.0), voice("bob", 0.0, 1.0)]
    close = np.array([0.1, 1.0], dtype="float32")

    names = name_speakers({"s0": close}, enrolled, ["s0"])

    assert names == {"s0": "bob"}


def test_a_speaker_too_far_from_every_enrolled_voice_is_unknown() -> None:
    enrolled = [voice("alice", 1.0, 0.0)]
    far = np.array([-1.0, 0.2], dtype="float32")
    assert MATCH_DISTANCE < 1.0

    assert name_speakers({"s0": far}, enrolled, ["s0"]) == {"s0": "unknown_1"}


def test_unknown_speakers_are_numbered_by_first_appearance() -> None:
    embeddings = {
        "late": np.array([0.0, 1.0], dtype="float32"),
        "early": np.array([1.0, 0.0], dtype="float32"),
        "silent": np.array([-1.0, 0.0], dtype="float32"),
    }

    names = name_speakers(embeddings, [], ["early", "late", "early"])

    assert names == {"early": "unknown_1", "late": "unknown_2"}


def test_a_speaker_without_an_embedding_is_unknown() -> None:
    names = name_speakers({}, [voice("alice", 1.0, 0.0)], ["s0"])

    assert names == {"s0": "unknown_1"}


def test_several_speakers_may_be_one_enrolled_voice() -> None:
    enrolled = [voice("alice", 1.0, 0.0)]
    embeddings = {
        "s0": np.array([1.0, 0.05], dtype="float32"),
        "s1": np.array([1.0, -0.05], dtype="float32"),
    }

    assert name_speakers(embeddings, enrolled, ["s0", "s1"]) == {"s0": "alice", "s1": "alice"}


def test_padded_spans_merge_neighbours_and_stay_inside_the_recording() -> None:
    spans = padded_spans([(0.5, 2.0), (3.0, 4.0), (20.0, 21.0), (99.0, 99.8)], 1.0, 100.0)

    assert spans == [(0.0, 5.0), (19.0, 22.0), (98.0, 100.0)]


def test_a_source_time_maps_into_the_joined_audio_or_its_nearest_edge() -> None:
    pieces = pieces_for([(10.0, 14.0), (30.0, 33.0)], gap=0.5)

    assert pieces == [Piece(10.0, 0.0, 4.0), Piece(30.0, 4.5, 3.0)]
    assert clamped(pieces, 11.0) == 1.0
    assert clamped(pieces, 31.0) == 5.5
    assert clamped(pieces, 20.0) == 4.0
    assert clamped(pieces, 28.0) == 4.5
    assert clamped(pieces, 5.0) == 0.0
    assert clamped(pieces, 99.0) == 7.5


class Torch:
    @staticmethod
    def from_numpy(array: np.ndarray) -> np.ndarray:
        return array


def failing(_: object) -> object:
    raise RuntimeError("boom")


def test_the_pinned_mirror_is_fetched_without_a_token_and_other_models_with_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def snapshot_download(model: str, **options: Any) -> str:
        calls.append({"model": model, **options})
        return "/cache/model"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    fetch_diarizer(DIARIZATION_REPO)
    fetch_diarizer("pyannote/speaker-diarization-community-1")

    [mirror, original] = calls
    assert mirror["token"] is False
    assert mirror["revision"] is not None
    assert original["token"] is None
    assert original["revision"] is None


def test_an_enrollment_clip_that_cannot_be_embedded_is_an_error_naming_it() -> None:
    diarizer = Diarizer(object(), failing, Torch)
    clip = EnrollmentClip("alice", np.zeros(16000, dtype="float32"))

    with pytest.raises(SpeechError, match=r"alice.*--no-speech"):
        diarizer.enroll([clip])


def test_a_pipeline_without_the_expected_result_is_a_speech_error() -> None:
    diarizer = Diarizer(lambda _: object(), failing, Torch)

    with pytest.raises(SpeechError, match="diarization model failed"):
        diarizer.diarize(np.zeros(16000, dtype="float32"))


def test_a_diarization_folder_without_an_embedder_is_a_speech_error(tmp_path: Path) -> None:
    with pytest.raises(SpeechError, match="embedding/"):
        Diarizer.load(tmp_path)


class _Embedder:
    """A stand-in for a pyannote embedding model: one row per sample in the batch."""

    dimension = 3
    sample_rate = 16000

    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, waveforms: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
        self.batches.append(len(waveforms))
        weights = np.ones(len(waveforms)) if masks is None else masks.sum(axis=1)
        return np.stack([waveforms.mean(axis=(1, 2)), weights, np.arange(len(waveforms))], axis=1)


def test_embedding_a_batch_in_parts_gives_the_rows_of_the_whole_batch_in_order() -> None:
    waveforms = np.random.default_rng(0).normal(size=(7, 1, 40)).astype("float32")
    masks = np.random.default_rng(1).uniform(size=(7, 5)).astype("float32")
    whole = _Embedder()(waveforms, masks)
    inner = _Embedder()

    parts = ParallelEmbedding(inner, workers=3)(waveforms, masks=masks)

    assert max(inner.batches) < 7
    assert sum(inner.batches) == 7
    np.testing.assert_allclose(parts[:, :2], whole[:, :2])


def test_a_parallel_embedder_still_answers_for_the_model_it_wraps() -> None:
    embedder = ParallelEmbedding(_Embedder(), workers=2)

    assert embedder.dimension == 3
    assert embedder.sample_rate == 16000
    assert embedder(np.zeros((1, 1, 8), dtype="float32")).shape == (1, 3)


def test_an_empty_batch_is_embedded_as_an_empty_batch() -> None:
    inner = _Embedder()

    rows = ParallelEmbedding(inner, workers=3)(np.zeros((0, 1, 8), dtype="float32"))

    assert rows.shape == (0, 3)
