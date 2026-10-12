"""Transcribing what is said within each Match, in English or Portuguese."""

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from victorious.schema import Event, Match, SpeechLanguage, transcript_event
from victorious.video import AUDIO_RATE, VideoInfo, sample_audio

MARGIN = 10.0
LANGUAGES: tuple[SpeechLanguage, ...] = ("en", "pt")
MAX_CHUNK = 30.0
MAX_GAP = 5.0
BATCH_SIZE = 16
MAX_NO_SPEECH_PROB = 0.9
MIN_AVG_LOGPROB = -1.5
DEFAULT_MODEL = "small"


class SpeechError(Exception):
    """Speech cannot be transcribed as asked."""


@dataclass(frozen=True)
class SpeechOptions:
    audio_stream: int = 0
    model: str = DEFAULT_MODEL


@dataclass(frozen=True)
class Segment:
    text: str
    language: SpeechLanguage
    start: float
    end: float
    confidence: float


def require_ready() -> None:
    """Fails when no Hugging Face token is set, before anything is read or decoded."""
    from huggingface_hub import get_token

    if not get_token():
        raise SpeechError(
            "no Hugging Face token found: set HF_TOKEN (or run `hf auth login`), "
            "or pass --no-speech to skip speech"
        )


def require_audio_stream(info: VideoInfo, options: SpeechOptions) -> None:
    if info.audio_streams == 0:
        raise SpeechError("the Recording has no audio stream; pass --no-speech to skip speech")
    if options.audio_stream >= info.audio_streams:
        count = info.audio_streams
        raise SpeechError(
            f"--audio-stream {options.audio_stream}: the Recording has {count} audio "
            f"stream{'s' if count != 1 else ''} (0 to {count - 1}); "
            "pass --no-speech to skip speech"
        )


def is_speech(no_speech_prob: float, avg_logprob: float, text: str) -> bool:
    return (
        bool(text.strip())
        and no_speech_prob <= MAX_NO_SPEECH_PROB
        and avg_logprob >= MIN_AVG_LOGPROB
    )


def _load_model(name: str) -> Any:
    from faster_whisper import WhisperModel
    from huggingface_hub.errors import LocalEntryNotFoundError

    def load(local_only: bool) -> Any:
        return WhisperModel(
            name,
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 1,
            use_auth_token=False,
            local_files_only=local_only,
        )

    try:
        return load(True)
    except LocalEntryNotFoundError:
        return load(False)


def _language(model: Any, audio: np.ndarray) -> SpeechLanguage:
    _, _, probabilities = model.detect_language(audio)
    scores = dict(probabilities)
    return max(LANGUAGES, key=lambda code: scores.get(code, 0.0))


def group_speech(
    spans: Sequence[tuple[int, int]], max_length: int, max_gap: int
) -> list[tuple[int, int]]:
    """Joins neighbouring spans of speech (in samples) into stretches of at most `max_length`."""
    groups: list[tuple[int, int]] = []
    for start, end in spans:
        if groups:
            first, last = groups[-1]
            if end - first <= max_length and start - last <= max_gap:
                groups[-1] = (first, end)
                continue
        groups.append((start, end))
    return groups


def _transcribe_window(model: Any, audio: np.ndarray, offset: float) -> list[Segment]:
    from faster_whisper import BatchedInferencePipeline
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    vad = VadOptions(max_speech_duration_s=MAX_CHUNK)
    spans = [(t["start"], t["end"]) for t in get_speech_timestamps(audio, vad, AUDIO_RATE)]
    groups = group_speech(spans, int(MAX_CHUNK * AUDIO_RATE), int(MAX_GAP * AUDIO_RATE))
    by_language: dict[SpeechLanguage, list[dict[str, float]]] = {}
    for start, end in groups:
        language = _language(model, audio[start:end])
        by_language.setdefault(language, []).append(
            {"start": start / AUDIO_RATE, "end": end / AUDIO_RATE}
        )
    pipeline = BatchedInferencePipeline(model)
    segments: list[Segment] = []
    for language, clips in by_language.items():
        spoken, _ = pipeline.transcribe(
            audio,
            language=language,
            clip_timestamps=clips,
            condition_on_previous_text=False,
            without_timestamps=False,
            batch_size=BATCH_SIZE,
        )
        for part in spoken:
            if is_speech(part.no_speech_prob, part.avg_logprob, part.text):
                segments.append(
                    Segment(
                        text=part.text.strip(),
                        language=language,
                        start=offset + part.start,
                        end=offset + part.end,
                        confidence=math.exp(part.avg_logprob),
                    )
                )
    return sorted(segments, key=lambda segment: segment.start)


def _window(match: Match, duration: float) -> tuple[float, float]:
    return max(0.0, match.start - MARGIN), min(duration, match.end + MARGIN)


def _distance(second: float, match: Match) -> float:
    return max(match.start - second, second - match.end, 0.0)


def _owner(
    segment: Segment, windows: Sequence[tuple[float, float]], matches: Sequence[Match]
) -> int:
    """The Match a segment belongs to: of those whose window it touches, the nearest to it."""
    midpoint = (segment.start + segment.end) / 2
    touching = [
        index
        for index, (begin, end) in enumerate(windows)
        if segment.start < end and segment.end > begin
    ]
    return min(touching, key=lambda index: (_distance(midpoint, matches[index]), index))


def _merged(events: list[Event], transcripts: list[Event]) -> list[Event]:
    """The Events in time order, never moving an Event that was already there."""
    merged: list[Event] = []
    pending = iter(sorted(transcripts, key=lambda event: event.time))
    upcoming = next(pending, None)
    for event in events:
        while upcoming is not None and upcoming.time < event.time:
            merged.append(upcoming)
            upcoming = next(pending, None)
        merged.append(event)
    while upcoming is not None:
        merged.append(upcoming)
        upcoming = next(pending, None)
    return merged


def transcribe_matches(
    path: Path, info: VideoInfo, matches: list[Match], options: SpeechOptions
) -> tuple[list[Match], int]:
    """The Matches with a `transcript` Event per segment spoken within [start - 10, end + 10].

    Also returns how many segments were kept. The model is loaded on the first window only.
    """
    windows = [_window(match, info.duration) for match in matches]
    segments: list[list[Segment]] = [[] for _ in matches]
    model: Any = None
    for index, (begin, end) in enumerate(windows):
        if end <= begin:
            continue
        audio = sample_audio(path, begin, end, options.audio_stream)
        if len(audio) == 0:
            continue
        if model is None:
            model = _load_model(options.model)
        for segment in _transcribe_window(model, audio, begin):
            if _owner(segment, windows, matches) == index:
                segments[index].append(segment)
    result = [
        match.model_copy(
            update={
                "events": _merged(
                    match.events,
                    [
                        transcript_event(s.text, s.language, s.start, s.end, s.confidence)
                        for s in found
                    ],
                )
            }
        )
        for match, found in zip(matches, segments, strict=True)
    ]
    return result, sum(len(found) for found in segments)
