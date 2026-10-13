"""Turn a Recording and a Layout into Matches with Events."""

import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from victorious.cache import ObservationCache
from victorious.categorize import JevClient, categorize_matches
from victorious.counters import (
    CounterTracker,
    Resolver,
    ReuseTracker,
    lobby_filling,
    tracked_elements,
)
from victorious.layout import Layout, LoadedLayout
from victorious.metadata import started_at
from victorious.music import MaskedWindow
from victorious.observations import Counters, SecondObservation
from victorious.ocr import require_ocr
from victorious.parallel import extract_seconds, read_counters
from victorious.resample import JumpResolver
from victorious.schema import (
    AnalysisOutput,
    AnalysisWarning,
    ChosenLayout,
    CounterElement,
    Event,
    Match,
    RecordingInfo,
    frame_event,
)
from victorious.screens import HUD_THRESHOLD, RESULTS_THRESHOLD, ResultsReading
from victorious.selection import NOT_FORTNITE_WINDOW, NoHud, select_layout
from victorious.speech import SpeechOptions, require_audio_stream, transcribe_matches
from victorious.video import VideoInfo, probe, require_complete

PLACEMENT_READINGS_PER_RUN = 5
MAX_PLACEMENT = 100
MAX_PLACEMENT_ATTEMPTS = 15
NEW_MATCH_ABSENCE_SECONDS = 20
LOBBY_PEEK_SECONDS = 6
ABANDONED_LOBBY_SECONDS = 30


@dataclass
class _OpenMatch:
    start: int
    start_confidence: float
    last_hud: int
    tracker: CounterTracker
    results_start: int | None = None
    readings: list[ResultsReading] = field(default_factory=list)
    opening: list[Counters] = field(default_factory=list)

    def abandoned_lobby(self) -> bool:
        """A short HUD run that only showed a lobby filling: the player left before the game."""
        return self.last_hud - self.start < ABANDONED_LOBBY_SECONDS and lobby_filling(self.opening)


def _vote(readings: list[ResultsReading]) -> tuple[int, float] | None:
    weights: dict[int, list[float]] = defaultdict(list)
    for reading in readings:
        if reading.digits is None or not reading.digits.text.isdigit():
            continue
        placement = int(reading.digits.text)
        if 1 <= placement <= MAX_PLACEMENT:
            weights[placement].append(reading.digits.confidence)
    if not weights:
        return None
    placement, confidences = max(weights.items(), key=lambda item: (sum(item[1]), -item[0]))
    return placement, sum(confidences) / len(confidences)


def _finish(index: int, match: _OpenMatch, resolver: Resolver | None) -> Match:
    outcome: list[Event] = []
    if match.results_start is None:
        end = float(match.last_hud)
        win, placement = False, None
        closing = frame_event("match_end", end, 0.0)
    else:
        end = float(match.results_start)
        banner = max(reading.banner for reading in match.readings)
        voted = _vote(match.readings)
        if banner >= RESULTS_THRESHOLD:
            win, placement, placement_confidence = True, 1, banner
        elif voted is not None:
            placement, placement_confidence = voted
            win = placement == 1
        else:
            win, placement, placement_confidence = False, None, 0.0

        if win:
            outcome.append(frame_event("win", end, placement_confidence))
        if placement is not None:
            outcome.append(frame_event("final_placement", end, placement_confidence, placement))
        results_confidence = max(max(r.banner, r.hash) for r in match.readings)
        closing = frame_event("match_end", end, results_confidence)

    counted = match.tracker.close(end, resolver)
    middle = sorted([*counted.events, *outcome], key=lambda event: event.time)
    start = float(match.start)
    opening = frame_event("match_start", start, match.start_confidence)
    return Match(
        index=index,
        start=start,
        end=end,
        win=win,
        placement=placement,
        personal_eliminations=counted.personal_eliminations,
        events=[opening, *middle, closing],
    )


def _wants_digits(current: _OpenMatch | None) -> bool:
    if current is None:
        return False
    read = sum(r.digits is not None for r in current.readings)
    return read < PLACEMENT_READINGS_PER_RUN and len(current.readings) < MAX_PLACEMENT_ATTEMPTS


class CounterSource(Protocol):
    def reset(self) -> None: ...

    def read(self, second: int) -> Counters: ...


def _peek_lobby(
    observations: Sequence[SecondObservation], second: int, counters: CounterSource
) -> dict[int, Counters]:
    """Reads the counters of the in-match seconds right after the HUD returns, in time order."""
    ahead: dict[int, Counters] = {}
    for upcoming in observations[second : second + LOBBY_PEEK_SECONDS]:
        if upcoming.hud >= HUD_THRESHOLD and not upcoming.results.is_results:
            ahead[upcoming.second] = counters.read(upcoming.second)
    return ahead


def run_machine(
    layout: Layout,
    observations: Sequence[SecondObservation],
    counters: CounterSource,
    resolver: Resolver | None,
) -> list[Match]:
    matches: list[Match] = []
    current: _OpenMatch | None = None
    in_results = False
    quiet_seen = True
    elements = tracked_elements(layout)

    peeked: dict[int, Counters] = {}

    def read(second: int) -> Counters:
        return peeked.pop(second) if second in peeked else counters.read(second)

    for observation in observations:
        second = observation.second
        reading = observation.results
        if reading.is_results:
            if current is not None:
                if current.results_start is None:
                    current.results_start = second
                if not _wants_digits(current):
                    reading = replace(reading, digits=None)
                current.readings.append(reading)
            in_results = True
            continue

        if in_results:
            in_results = False
            if current is not None:
                matches.append(_finish(len(matches), current, resolver))
                current = None
            quiet_seen = False
        hud = observation.hud
        present = hud >= HUD_THRESHOLD
        if present:
            returned = False
            if current is not None and second - current.last_hud > NEW_MATCH_ABSENCE_SECONDS:
                returned = True
                counters.reset()
                ahead = _peek_lobby(observations, second, counters)
                peeked.update(ahead)
                if lobby_filling(list(ahead.values())):
                    if not current.abandoned_lobby():
                        matches.append(_finish(len(matches), current, resolver))
                    current = None
            if current is not None:
                current.last_hud = second
            elif quiet_seen:
                if not returned:
                    counters.reset()
                current = _OpenMatch(
                    start=second,
                    start_confidence=hud,
                    last_hud=second,
                    tracker=CounterTracker(elements),
                )
        else:
            quiet_seen = True
        if current is not None:
            taken = read(second) if present else {}
            if second - current.start < ABANDONED_LOBBY_SECONDS:
                current.opening.append(taken)
            current.tracker.observe(second, taken)

    if current is not None:
        matches.append(_finish(len(matches), current, resolver))
    return matches


class _ScheduleRecorder:
    """Stands in for the counter source to learn when the Match loop resets and reads."""

    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def reset(self) -> None:
        self.calls.append(None)

    def read(self, second: int) -> Counters:
        self.calls.append(second)
        return {}


class _Readings:
    def __init__(self, observations: Sequence[SecondObservation]) -> None:
        self._observations = observations

    def reset(self) -> None:
        pass

    def read(self, second: int) -> Counters:
        return self._observations[second].counters


class Phases:
    """Wall seconds spent in each named phase of an analysis, in the order they ran."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.monotonic() - started


def _with_counters(
    path: Path, info: VideoInfo, layout: Layout, jobs: int, chunk_seconds: int, phases: Phases
) -> list[SecondObservation]:
    """The Observations of every second, reading counters exactly where a serial pass would.

    The Match loop is run once with a recorder to learn which seconds it reads counters at, the
    skip-OCR rule is replayed over that schedule, and only the pixels that rule keeps are OCR'd.
    """
    with phases.timed("frames"):
        seconds = extract_seconds(path, info, layout, jobs, chunk_seconds)
    observations = [second.observation for second in seconds]
    recorder = _ScheduleRecorder()
    run_machine(layout, observations, recorder, None)

    elements = tracked_elements(layout)
    reuse = ReuseTracker()
    fresh: list[list[np.ndarray]] = []
    source: dict[tuple[int, CounterElement], int] = {}
    latest: dict[CounterElement, int] = {}
    for call in recorder.calls:
        if call is None:
            reuse.reset()
            continue
        masks = seconds[call].masks
        for element in elements:
            tight, loose = masks[element]
            band = tight.unpack()
            if reuse.needs_ocr(element, band):
                fresh.append([band, loose.unpack()])
                latest[element] = len(fresh) - 1
            source[call, element] = latest[element]

    with phases.timed("counters"):
        readings = read_counters(fresh, jobs)
    counters: dict[int, Counters] = {}
    for (second, element), index in source.items():
        counters.setdefault(second, {})[element] = readings[index]
    return [replace(o, counters=counters.get(o.second, {})) for o in observations]


@dataclass(frozen=True)
class Analysis:
    output: AnalysisOutput
    observations: list[SecondObservation]
    cached: bool
    layout: LoadedLayout | None
    candidates: int
    speech_segments: int | None = None
    masked: list[MaskedWindow] = field(default_factory=list)
    speakers: int = 0
    enrolled: int = 0
    categorized: int = 0
    category_failure: str | None = None
    phases: dict[str, float] = field(default_factory=dict)


def analyse_with_observations(
    path: Path,
    layouts: Sequence[LoadedLayout],
    *,
    jobs: int = 1,
    chunk_seconds: int = 60,
    cache: ObservationCache | None = None,
    speech: SpeechOptions | None = None,
    categorizer: JevClient | None = None,
) -> Analysis:
    info = probe(path)
    if speech is not None:
        require_audio_stream(info, speech)
    require_ocr()
    recording = RecordingInfo(
        path=str(path), duration=info.duration, started_at=started_at(path, info.creation_time)
    )
    phases = Phases()
    with phases.timed("layout"):
        selection = select_layout(path, info, layouts, jobs=jobs, cache=cache)
    if isinstance(selection, NoHud):
        require_complete(path, info)
        shown = ", ".join(str(loaded.path) for loaded in selection.tried)
        tried = (
            f"Layouts tried: {shown}"
            if selection.aspect_fits
            else f"no Layout fits this {info.width}x{info.height} Recording: {shown}"
        )
        warning = AnalysisWarning(
            code="not_fortnite",
            message=(
                f"no in-match HUD in the first {NOT_FORTNITE_WINDOW // 60} minutes "
                f"({selection.frames} frames sampled, {tried}); "
                "this does not look like a Fortnite Recording, or its HUD needs a new Layout "
                f"from `victorious setup {path}`"
            ),
        )
        output = AnalysisOutput(recording=recording, layout=None, warnings=[warning], matches=[])
        return Analysis(output, [], False, None, len(selection.tried))
    layout = selection.chosen.layout
    entry = cache.entry(path, layout) if cache is not None else None
    observations = entry.load() if entry is not None else None
    cached = observations is not None
    if observations is None:
        require_complete(path, info)
        observations = _with_counters(path, info, layout, jobs, chunk_seconds, phases)
    resolver = JumpResolver(path, info, layout, entry)
    matches = run_machine(layout, observations, _Readings(observations), resolver)
    if entry is not None and not cached:
        entry.store(observations)
    segments = None
    masked: list[MaskedWindow] = []
    speakers = enrolled = categorized = 0
    category_failure: str | None = None
    if speech is not None:
        with phases.timed("speech"):
            spoken = transcribe_matches(path, info, matches, speech)
        matches, segments, masked = spoken.matches, spoken.segments, spoken.masked
        speakers, enrolled = spoken.speakers, spoken.enrolled
        judged = categorize_matches(matches, categorizer)
        matches = judged.matches
        categorized, category_failure = judged.categorized, judged.first_failure
    output = AnalysisOutput(
        recording=recording,
        layout=ChosenLayout(path=str(selection.chosen.path), score=selection.score),
        warnings=[],
        matches=matches,
    )
    return Analysis(
        output,
        observations,
        cached,
        selection.chosen,
        selection.fitting,
        segments,
        masked,
        speakers,
        enrolled,
        categorized,
        category_failure,
        phases.seconds,
    )


def analyse(
    path: Path,
    layouts: Sequence[LoadedLayout],
    *,
    jobs: int = 1,
    chunk_seconds: int = 60,
    cache: ObservationCache | None = None,
    speech: SpeechOptions | None = None,
    categorizer: JevClient | None = None,
) -> AnalysisOutput:
    return analyse_with_observations(
        path,
        layouts,
        jobs=jobs,
        chunk_seconds=chunk_seconds,
        cache=cache,
        speech=speech,
        categorizer=categorizer,
    ).output
