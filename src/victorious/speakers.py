"""Telling Speakers apart: diarization, and naming voices enrolled from a `voices/` folder."""

import functools
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from victorious.errors import SpeechError
from victorious.video import AUDIO_RATE, RecordingError, sample_audio

DIARIZATION_REPO = "pyannote-community/speaker-diarization-community-1"
DIARIZATION_REVISION = "8a527374977391da736e0daaef26855d949d9685"
MATCH_DISTANCE = 0.6
SEGMENT_PAD = 1.0
JOIN_GAP = 0.5
MIN_TURN = 0.5
MAX_TURNS = 10
MAX_ENROLLMENT = 30.0
UNKNOWN_PREFIX = "unknown_"


@dataclass(frozen=True)
class EnrollmentClip:
    name: str
    audio: np.ndarray


@dataclass(frozen=True)
class EnrolledVoice:
    name: str
    embedding: np.ndarray


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Piece:
    """A span of the Recording (`source`, seconds) as it sits in the joined audio (`joined`)."""

    source: float
    joined: float
    length: float


def speaker_for_segment(start: float, end: float, turns: Sequence[Turn]) -> str | None:
    """The speaker whose turns overlap [start, end) most, else the nearest turn in time."""
    if not turns:
        return None
    overlap: dict[str, float] = {}
    for turn in turns:
        shared = min(end, turn.end) - max(start, turn.start)
        if shared > 0:
            overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + shared
    if overlap:
        return max(overlap, key=lambda speaker: overlap[speaker])
    nearest = min(turns, key=lambda turn: max(turn.start - end, start - turn.end))
    return nearest.speaker


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - float(a @ b) / norm if norm else 1.0


def name_speakers(
    embeddings: dict[str, np.ndarray], enrolled: Sequence[EnrolledVoice], owners: Sequence[str]
) -> dict[str, str]:
    """A name for each diarized speaker in `owners` (one per segment, in time order).

    A speaker takes the nearest enrolled voice when it is within `MATCH_DISTANCE`; the others
    are `unknown_N`, numbered by the first segment they own.
    """
    names: dict[str, str] = {}
    unknown = 0
    for speaker in owners:
        if speaker in names:
            continue
        embedding = embeddings.get(speaker)
        nearest = (
            min(
                ((_distance(embedding, voice.embedding), voice.name) for voice in enrolled),
                key=lambda scored: scored[0],
                default=None,
            )
            if embedding is not None
            else None
        )
        if nearest is not None and nearest[0] <= MATCH_DISTANCE:
            names[speaker] = nearest[1]
        else:
            unknown += 1
            names[speaker] = f"{UNKNOWN_PREFIX}{unknown}"
    return names


def padded_spans(
    spans: Sequence[tuple[float, float]], pad: float, duration: float
) -> list[tuple[float, float]]:
    """Each span widened by `pad` seconds within [0, duration], overlapping ones merged."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        begin, stop = max(0.0, start - pad), min(duration, end + pad)
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((begin, stop))
    return merged


def pieces_for(spans: Sequence[tuple[float, float]], gap: float) -> list[Piece]:
    pieces: list[Piece] = []
    joined = 0.0
    for start, end in spans:
        pieces.append(Piece(start, joined, end - start))
        joined += end - start + gap
    return pieces


def clamped(pieces: Sequence[Piece], source: float) -> float:
    """Where `source` sits in the joined audio, or its nearest edge when it falls in a gap."""
    piece = min(pieces, key=lambda p: max(p.source - source, source - p.source - p.length, 0.0))
    return piece.joined + min(max(source - piece.source, 0.0), piece.length)


@functools.cache
def load_clips(directory: Path | None) -> tuple[EnrollmentClip, ...]:
    """The clips of an enrollment folder, named by file stem; none when it is absent."""
    if directory is None or not directory.is_dir():
        return ()
    clips: list[EnrollmentClip] = []
    taken: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        name = path.stem
        if name.startswith(UNKNOWN_PREFIX):
            raise SpeechError(
                f"voice clip {path}: the name {name!r} is reserved for unenrolled voices; "
                "rename the file or pass --no-speech to skip speech"
            )
        if name in taken:
            raise SpeechError(
                f"voice clips {taken[name].name} and {path.name} both enroll {name!r}: "
                "keep one clip per person or pass --no-speech to skip speech"
            )
        taken[name] = path
        try:
            audio = sample_audio(path, 0.0, MAX_ENROLLMENT, 0)
        except RecordingError as error:
            raise SpeechError(
                f"cannot decode voice clip {path}: {error}; "
                "remove it or pass --no-speech to skip speech"
            ) from error
        if len(audio) == 0:
            raise SpeechError(
                f"voice clip {path} has no audio; remove it or pass --no-speech to skip speech"
            )
        clips.append(EnrollmentClip(name, audio))
    return tuple(clips)


def fetch_diarizer(model: str) -> Path:
    """The local folder of the diarization pipeline: a folder given, else a cached or fetched
    snapshot of the pinned repository."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    if Path(model).is_dir():
        return Path(model)

    def fetch(local_only: bool) -> Path:
        return Path(
            snapshot_download(
                model,
                revision=DIARIZATION_REVISION if model == DIARIZATION_REPO else None,
                token=False if model == DIARIZATION_REPO else None,
                local_files_only=local_only,
                ignore_patterns=["*.gif", "*.md"],
            )
        )

    try:
        try:
            return fetch(True)
        except LocalEntryNotFoundError:
            return fetch(False)
    except Exception as error:
        raise SpeechError(
            f"cannot load the diarization model {model!r}: {error}; pass --no-speech to skip speech"
        ) from error


class ParallelEmbedding:
    """An embedding model that runs the parts of each batch on threads, in order.

    Torch scales poorly inside one large batch on a few threads, but its operators release the
    GIL, so independent parts of the batch run side by side; the rows are those of the whole.
    """

    def __init__(self, model: Any, workers: int) -> None:
        self._model = model
        self._workers = max(1, workers)
        self._pool = ThreadPoolExecutor(max_workers=self._workers)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, waveforms: Any, masks: Any = None) -> np.ndarray:
        size = max(1, -(-len(waveforms) // self._workers))
        parts = [
            (
                waveforms[start : start + size],
                None if masks is None else masks[start : start + size],
            )
            for start in range(0, len(waveforms), size)
        ]
        rows = self._pool.map(lambda part: self._model(part[0], masks=part[1]), parts)
        return np.vstack(list(rows))


class Diarizer:
    """The pyannote community-1 pipeline and its embedder, run on CPU."""

    def __init__(self, pipeline: Any, embedder: Any, torch: Any) -> None:
        self._pipeline = pipeline
        self._embedder = embedder
        self._torch = torch

    @classmethod
    def load(cls, folder: Path) -> "Diarizer":
        if not (folder / "embedding").is_dir():
            raise SpeechError(
                f"the diarization model {str(folder)!r} has no embedding/ folder, "
                "which a community-1 pipeline ships; pass --no-speech to skip speech"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import torch
                from pyannote.audio import Inference, Model, Pipeline

                pipeline = Pipeline.from_pretrained(str(folder))
                model = Model.from_pretrained(str(folder / "embedding"))
                embedder = Inference(model, window="whole") if model is not None else None
        except Exception as error:
            raise SpeechError(
                f"cannot load the diarization model {str(folder)!r}: {error}; "
                "pass --no-speech to skip speech"
            ) from error
        if pipeline is None or embedder is None:
            raise SpeechError(
                f"cannot load the diarization model {str(folder)!r}; "
                "pass --no-speech to skip speech"
            )
        workers = max(1, (os.cpu_count() or 2) // 2)
        torch.set_num_threads(1)
        pipeline._embedding = ParallelEmbedding(pipeline._embedding, workers)
        return cls(pipeline, embedder, torch)

    def _input(self, audio: np.ndarray) -> dict[str, Any]:
        waveform = self._torch.from_numpy(np.array(audio, dtype="float32"))[None]
        return {"waveform": waveform, "sample_rate": AUDIO_RATE}

    def diarize(self, audio: np.ndarray) -> list[Turn]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = self._pipeline(self._input(audio))
            tracks = result.exclusive_speaker_diarization.itertracks(yield_label=True)
            return [
                Turn(float(span.start), float(span.end), str(label)) for span, _, label in tracks
            ]
        except Exception as error:
            raise SpeechError(
                f"the diarization model failed: {error}; pass --no-speech to skip speech"
            ) from error

    def embed(self, audio: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(self._embedder(self._input(audio)), dtype="float32").ravel()
        except Exception as error:
            raise SpeechError(
                f"the voice embedder failed: {error}; pass --no-speech to skip speech"
            ) from error

    def enroll(self, clips: Sequence[EnrollmentClip]) -> list[EnrolledVoice]:
        voices: list[EnrolledVoice] = []
        for clip in clips:
            try:
                voices.append(EnrolledVoice(clip.name, self.embed(clip.audio)))
            except SpeechError as error:
                raise SpeechError(
                    f"cannot embed the voice clip for {clip.name!r}: {error}"
                ) from error
        return voices


@functools.cache
def load_diarizer(model: str) -> Diarizer:
    return Diarizer.load(fetch_diarizer(model))


def _speaker_embeddings(
    diarizer: Diarizer, audio: np.ndarray, turns: Sequence[Turn], speakers: set[str]
) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for speaker in speakers:
        longest = sorted(
            (t for t in turns if t.speaker == speaker and t.end - t.start >= MIN_TURN),
            key=lambda t: t.end - t.start,
            reverse=True,
        )[:MAX_TURNS]
        vectors: list[np.ndarray] = []
        for t in longest:
            try:
                vectors.append(
                    diarizer.embed(audio[int(t.start * AUDIO_RATE) : int(t.end * AUDIO_RATE)])
                )
            except SpeechError:
                continue
        if vectors:
            found[speaker] = np.mean(vectors, axis=0)
    return found


def label_segments(
    segments: Sequence[tuple[float, float]],
    duration: float,
    read_audio: Callable[[float, float], np.ndarray],
    diarizer: Callable[[], Diarizer],
    clips: Sequence[EnrollmentClip],
) -> list[str | None]:
    """A Speaker for each segment (start, end in Recording seconds).

    Only the audio around the segments is diarized, once, so one voice keeps one identity
    across the Recording; `read_audio(start, end)` gives that audio as it was transcribed.
    """
    if not segments:
        return []
    pieces_audio: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []
    for begin, end in padded_spans(segments, SEGMENT_PAD, duration):
        audio = read_audio(begin, end)
        if len(audio):
            pieces_audio.append(audio)
            spans.append((begin, begin + len(audio) / AUDIO_RATE))
    if not spans:
        return [None] * len(segments)
    pieces = pieces_for(spans, JOIN_GAP)
    gap = np.zeros(round(JOIN_GAP * AUDIO_RATE), dtype="float32")
    joined = np.concatenate([part for audio in pieces_audio for part in (audio, gap)])
    model = diarizer()
    turns = model.diarize(joined)
    labels = [
        speaker_for_segment(clamped(pieces, start), clamped(pieces, end), turns)
        for start, end in segments
    ]
    order = sorted(range(len(segments)), key=lambda i: segments[i])
    owners = [label for i in order if (label := labels[i]) is not None]
    embeddings: dict[str, np.ndarray] = {}
    enrolled: list[EnrolledVoice] = []
    if clips and owners:
        enrolled = model.enroll(clips)
        embeddings = _speaker_embeddings(model, joined, turns, set(owners))
    names = name_speakers(embeddings, enrolled, owners)
    return [None if label is None else names[label] for label in labels]
