"""Transcribing what is said within each Match, in English or Portuguese."""

import bisect
import heapq
import math
import os
import re
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
MAX_COMPRESSION_RATIO = 2.4
LOOP_LENGTH = 3
AMBIGUITY = 0.5
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


def is_speech(
    no_speech_prob: float, avg_logprob: float, compression_ratio: float, text: str
) -> bool:
    return (
        bool(text.strip())
        and no_speech_prob <= MAX_NO_SPEECH_PROB
        and avg_logprob >= MIN_AVG_LOGPROB
        and compression_ratio <= MAX_COMPRESSION_RATIO
    )


def drop_repeated(segments: list[Segment]) -> list[Segment]:
    """Drops runs of `LOOP_LENGTH` or more back-to-back segments with the same words.

    Whisper loops on music and noise ("the", "Thank you.", ...) one short segment at a time,
    which the per-segment compression ratio cannot see.
    """
    keys = [re.sub(r"\W+", " ", segment.text.lower()).strip() for segment in segments]
    kept: list[Segment] = []
    start = 0
    while start < len(segments):
        stop = start + 1
        while stop < len(segments) and keys[stop] == keys[start]:
            stop += 1
        if stop - start < LOOP_LENGTH:
            kept.extend(segments[start:stop])
        start = stop
    return kept


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
        try:
            return load(True)
        except LocalEntryNotFoundError:
            return load(False)
    except (ValueError, RuntimeError) as error:
        raise SpeechError(
            f"cannot load the Whisper model {name!r}: {error}; pass --no-speech to skip speech"
        ) from error


def candidate_languages(scores: dict[str, float]) -> tuple[SpeechLanguage, ...]:
    """The languages worth decoding: the likeliest, and any within `AMBIGUITY` of it."""
    ranked = sorted(LANGUAGES, key=lambda code: -scores.get(code, 0.0))
    best = scores.get(ranked[0], 0.0)
    return tuple(code for code in ranked if best - scores.get(code, 0.0) < AMBIGUITY)


def _chant(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return len(words) >= LOOP_LENGTH and len(set(words)) == 1


def pick_reading(readings: dict[SpeechLanguage, list[Segment]]) -> list[Segment]:
    """The reading of one stretch of speech whose segments are most confident on average.

    Loops are dropped first, and a segment that only repeats one word counts as no confidence
    at all, so a language that chants loses to one that does not. Ties go to the first
    language, which is the one detection found likeliest.
    """

    def score(segments: list[Segment]) -> float:
        total = sum(0.0 if _chant(s.text) else s.confidence for s in segments)
        return total / len(segments)

    kept = [drop_repeated(segments) for segments in readings.values()]
    spoken = [segments for segments in kept if segments]
    return max(spoken, key=score) if spoken else []


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
    candidates = [
        candidate_languages(dict(model.detect_language(audio[start:end])[2]))
        for start, end in groups
    ]
    pipeline = BatchedInferencePipeline(model)
    readings: list[dict[SpeechLanguage, list[Segment]]] = [
        {language: [] for language in languages} for languages in candidates
    ]
    for language in LANGUAGES:
        indices = [i for i, languages in enumerate(candidates) if language in languages]
        if not indices:
            continue
        starts = [groups[i][0] / AUDIO_RATE for i in indices]
        spoken, _ = pipeline.transcribe(
            audio,
            language=language,
            clip_timestamps=[
                {"start": groups[i][0] / AUDIO_RATE, "end": groups[i][1] / AUDIO_RATE}
                for i in indices
            ],
            condition_on_previous_text=False,
            without_timestamps=False,
            batch_size=BATCH_SIZE,
        )
        for part in spoken:
            if not is_speech(
                part.no_speech_prob, part.avg_logprob, part.compression_ratio, part.text
            ):
                continue
            middle = (part.start + part.end) / 2
            owner = indices[max(bisect.bisect_right(starts, middle) - 1, 0)]
            readings[owner][language].append(
                Segment(
                    text=part.text.strip(),
                    language=language,
                    start=offset + part.start,
                    end=offset + part.end,
                    confidence=math.exp(part.avg_logprob),
                )
            )
    chosen = [segment for reading in readings for segment in pick_reading(reading)]
    return sorted(chosen, key=lambda segment: segment.start)


def _window(match: Match, duration: float) -> tuple[float, float]:
    return max(0.0, match.start - MARGIN), min(duration, match.end + MARGIN)


def _distance(second: float, match: Match) -> float:
    return max(match.start - second, second - match.end, 0.0)


def _owner(
    segment: Segment, source: int, windows: Sequence[tuple[float, float]], matches: Sequence[Match]
) -> int:
    """The Match a segment belongs to: of those whose window it touches, the nearest to it."""
    midpoint = (segment.start + segment.end) / 2
    touching = {
        index
        for index, (begin, end) in enumerate(windows)
        if segment.start < end and segment.end > begin
    } | {source}
    return min(touching, key=lambda index: (_distance(midpoint, matches[index]), index))


def _merged(events: list[Event], transcripts: list[Event]) -> list[Event]:
    """The Events in time order, never moving an Event that was already there."""
    ordered = sorted(transcripts, key=lambda event: event.time)
    return list(heapq.merge(events, ordered, key=lambda event: event.time))


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
            if _owner(segment, index, windows, matches) == index:
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
