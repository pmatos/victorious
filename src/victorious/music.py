"""Finding music-dominant audio, so that emote and lobby music is not read as lyrics."""

import csv
import functools
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from victorious.errors import SpeechError
from victorious.video import AUDIO_RATE

MODEL_REPO = "audiomagic/yamnet-onnx"
MODEL_REVISION = "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd"
MODEL_FILE = "yamnet.onnx"
CLASS_MAP_FILE = "yamnet_class_map.csv"
MUSIC_MIN = 0.5
SPEECH_KEEP = 0.25
MUSIC_EDGE = 0.25
MIN_MASK = 3
MIN_SPEECH = 0.25
BRIDGE = 1
FRAME_HOP = 0.48
FRAME_LENGTH = 0.96


@dataclass(frozen=True)
class SecondScore:
    """How music-like and how speech-like one second of audio is, each in [0, 1]."""

    second: int
    music: float
    speech: float

    @property
    def dominated_by_music(self) -> bool:
        return self.music >= MUSIC_MIN and self.speech < SPEECH_KEEP


@dataclass(frozen=True)
class MaskedWindow:
    """A span of a Recording, in seconds from its start, withheld from transcription."""

    start: float
    end: float
    music: float
    speech: float


def masked_windows(
    scores: Sequence[SecondScore], offset: float = 0.0, duration: float | None = None
) -> list[MaskedWindow]:
    """The runs of music-dominant seconds, in Recording seconds (`offset` + the second).

    A run keeps going over one second that is not music-dominant when it has no speech, since
    the music merely dipped, but never over a second with speech. Runs shorter than `MIN_MASK`
    seconds are stingers and single beats, which must not blank speech. A run gives up
    its first or last second when speech is heard in the next one, because speech starts and
    ends inside the second the tagger still calls music. A kept run takes in the second before
    and after it when music is still audible there and nobody speaks, because the music starts
    and stops inside that second.
    """
    ordered = sorted(scores, key=lambda score: score.second)
    runs: list[list[SecondScore]] = []
    gap: list[SecondScore] = []
    for score in ordered:
        if score.dominated_by_music:
            contiguous = runs and runs[-1][-1].second + 1 + len(gap) == score.second
            clear = all(skipped.speech < SPEECH_KEEP for skipped in gap)
            if contiguous and len(gap) <= BRIDGE and clear:
                runs[-1].extend([*gap, score])
            else:
                runs.append([score])
            gap = []
        elif runs and runs[-1][-1].second + 1 + len(gap) == score.second:
            gap.append(score)
    by_second = {score.second: score for score in ordered}

    def edge(second: int) -> int:
        score = by_second.get(second)
        return 1 if score and score.music >= MUSIC_EDGE and score.speech < SPEECH_KEEP else 0

    def spoken(second: int) -> bool:
        score = by_second.get(second)
        return score is not None and score.speech >= SPEECH_KEEP

    found = []
    for whole in runs:
        run = whole[1:] if spoken(whole[0].second - 1) else whole
        run = run[:-1] if run and spoken(run[-1].second + 1) else run
        if len(run) < MIN_MASK:
            continue
        start = offset + run[0].second - edge(whole[0].second - 1)
        end = offset + run[-1].second + 1 + edge(whole[-1].second + 1)
        if duration is not None:
            end = min(end, offset + duration)
        found.append(
            MaskedWindow(
                start,
                end,
                sum(score.music for score in run) / len(run),
                sum(score.speech for score in run) / len(run),
            )
        )
    return found


def merge_windows(windows: Iterable[MaskedWindow]) -> list[MaskedWindow]:
    """The windows with those that overlap or touch joined, scores weighted by length."""
    groups: list[list[MaskedWindow]] = []
    reach = -math.inf
    for window in sorted(windows, key=lambda w: (w.start, w.end)):
        if groups and window.start <= reach:
            groups[-1].append(window)
            reach = max(reach, window.end)
        else:
            groups.append([window])
            reach = window.end
    merged = []
    for group in groups:
        total = sum(w.end - w.start for w in group)
        merged.append(
            MaskedWindow(
                group[0].start,
                max(w.end for w in group),
                sum((w.end - w.start) * w.music for w in group) / total,
                sum((w.end - w.start) * w.speech for w in group) / total,
            )
        )
    return merged


def silence(audio: np.ndarray, windows: Iterable[MaskedWindow], offset: float) -> np.ndarray:
    """A copy of `audio` (which starts `offset` seconds in) with the `windows` zeroed."""
    result = np.array(audio, copy=True)
    for window in windows:
        first = max(0, round((window.start - offset) * AUDIO_RATE))
        last = min(len(result), round((window.end - offset) * AUDIO_RATE))
        if last > first:
            result[first:last] = 0.0
    return result


def unmasked(
    spans: Iterable[tuple[int, int]], windows: Iterable[MaskedWindow], offset: float
) -> list[tuple[int, int]]:
    """Speech `spans` (in samples of audio that starts `offset` seconds in) outside `windows`.

    A span that is cut keeps its parts outside the windows, unless they are too short to be
    speech.
    """
    cuts = sorted(
        (round((w.start - offset) * AUDIO_RATE), round((w.end - offset) * AUDIO_RATE))
        for w in windows
    )
    kept: list[tuple[int, int]] = []
    for start, end in spans:
        position = start
        for cut_start, cut_end in cuts:
            if cut_end <= position or cut_start >= end:
                continue
            if cut_start > position:
                kept.append((position, min(cut_start, end)))
            position = max(position, cut_end)
        if position < end:
            kept.append((position, end))
    return [(a, b) for a, b in kept if b - a >= MIN_SPEECH * AUDIO_RATE]


def _frame_range(second: int, frames: int) -> range:
    first = max(0, (second * 100 - round(FRAME_LENGTH * 100)) // round(FRAME_HOP * 100) + 1)
    last = min(frames, -(-(second + 1) * 100 // round(FRAME_HOP * 100)))
    return range(first, last)


class MusicTagger:
    """YAMNet (AudioSet) run on CPU with onnxruntime: per-second music and speech scores."""

    def __init__(self, session: Any, music: int, speech: int) -> None:
        self._session = session
        self._music = music
        self._speech = speech

    @classmethod
    def load(cls) -> "MusicTagger":
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        def fetch(name: str, local_only: bool) -> Path:
            return Path(
                hf_hub_download(
                    MODEL_REPO,
                    name,
                    revision=MODEL_REVISION,
                    token=False,
                    local_files_only=local_only,
                )
            )

        def download(name: str) -> Path:
            try:
                return fetch(name, True)
            except LocalEntryNotFoundError:
                return fetch(name, False)

        try:
            model = download(MODEL_FILE)
            with download(CLASS_MAP_FILE).open(newline="", encoding="utf-8") as handle:
                names = [row["display_name"] for row in csv.DictReader(handle)]
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = os.cpu_count() or 1
            session = onnxruntime.InferenceSession(
                str(model), options, providers=["CPUExecutionProvider"]
            )
            return cls(session, names.index("Music"), names.index("Speech"))
        except Exception as error:
            raise SpeechError(
                f"cannot load the audio tagger {MODEL_REPO!r}: {error}; "
                "pass --no-speech to skip speech"
            ) from error

    def tag(self, audio: np.ndarray, first_second: int = 0) -> list[SecondScore]:
        """One score per started second of `audio` (16 kHz mono float32), numbered from
        `first_second`."""
        if len(audio) == 0:
            return []
        try:
            frames = self._session.run(None, {"waveform": np.ascontiguousarray(audio, "float32")})[
                0
            ]
        except Exception as error:
            raise SpeechError(
                f"the audio tagger {MODEL_REPO!r} failed: {error}; pass --no-speech to skip speech"
            ) from error
        seconds = math.ceil(len(audio) / AUDIO_RATE)
        scores = []
        for second in range(seconds):
            span = _frame_range(second, len(frames))
            if not span:
                span = range(len(frames) - 1, len(frames))
            part = frames[span.start : span.stop]
            scores.append(
                SecondScore(
                    first_second + second,
                    float(part[:, self._music].mean()),
                    float(part[:, self._speech].max()),
                )
            )
        return scores


@functools.cache
def load_tagger() -> MusicTagger:
    return MusicTagger.load()
