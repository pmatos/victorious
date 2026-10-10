"""Turn a Recording and a Layout into Matches with Events."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from victorious.counters import CounterReader, CounterTracker, tracked_elements
from victorious.layout import Layout
from victorious.ocr import find_tesseract
from victorious.schema import AnalysisOutput, Event, EventKind, Match, RecordingInfo
from victorious.screens import (
    HUD_THRESHOLD,
    RESULTS_THRESHOLD,
    ResultsReading,
    hud_score,
    results_reading,
)
from victorious.video import probe, sample_frames

PLACEMENT_READINGS_PER_RUN = 5
MAX_PLACEMENT = 100
MAX_PLACEMENT_ATTEMPTS = 15


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _is_results(reading: ResultsReading) -> bool:
    return reading.banner >= RESULTS_THRESHOLD or reading.hash >= RESULTS_THRESHOLD


@dataclass
class _OpenMatch:
    start: int
    start_confidence: float
    last_hud: int
    tracker: CounterTracker
    results_start: int | None = None
    readings: list[ResultsReading] = field(default_factory=list)


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


def _event(kind: EventKind, time: float, confidence: float, value: int | None = None) -> Event:
    return Event(
        kind=kind,
        time=time,
        source="frame",
        confidence=_clamp(confidence),
        value=value,
        end=None,
        element=None,
    )


def _assemble(
    start: float, match: _OpenMatch, end: float, outcome: list[Event], closing: Event
) -> tuple[list[Event], int | None]:
    counted = match.tracker.close(end)
    middle = sorted([*counted.events, *outcome], key=lambda event: event.time)
    opening = _event("match_start", start, match.start_confidence)
    return [opening, *middle, closing], counted.personal_eliminations


def _finish(index: int, match: _OpenMatch) -> Match:
    start = float(match.start)
    if match.results_start is None:
        end = float(match.last_hud)
        events, personal = _assemble(start, match, end, [], _event("match_end", end, 0.0))
        return Match(
            index=index,
            start=start,
            end=end,
            win=False,
            placement=None,
            personal_eliminations=personal,
            events=events,
        )

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

    outcome: list[Event] = []
    if win:
        outcome.append(_event("win", end, placement_confidence))
    if placement is not None:
        outcome.append(_event("final_placement", end, placement_confidence, placement))
    results_confidence = max(max(r.banner, r.hash) for r in match.readings)
    closing = _event("match_end", end, results_confidence)
    events, personal = _assemble(start, match, end, outcome, closing)
    return Match(
        index=index,
        start=start,
        end=end,
        win=win,
        placement=placement,
        personal_eliminations=personal,
        events=events,
    )


def _wants_digits(current: _OpenMatch | None) -> bool:
    if current is None:
        return False
    read = sum(r.digits is not None for r in current.readings)
    return read < PLACEMENT_READINGS_PER_RUN and len(current.readings) < MAX_PLACEMENT_ATTEMPTS


def analyse(path: Path, layout: Layout) -> AnalysisOutput:
    info = probe(path)
    layout.require_aspect_ratio(info.width, info.height)
    find_tesseract()
    matches: list[Match] = []
    current: _OpenMatch | None = None
    in_results = False
    quiet_seen = True
    reader = CounterReader()
    elements = tracked_elements(layout)

    for second, frame in sample_frames(path, info):
        reading = results_reading(frame, layout, read_placement=_wants_digits(current))
        if _is_results(reading):
            if current is not None:
                if current.results_start is None:
                    current.results_start = second
                current.readings.append(reading)
            in_results = True
            continue

        if in_results:
            in_results = False
            if current is not None:
                matches.append(_finish(len(matches), current))
                current = None
            quiet_seen = False
        hud = hud_score(frame, layout)
        present = hud >= HUD_THRESHOLD
        if present:
            if current is not None:
                current.last_hud = second
            elif quiet_seen:
                reader.reset()
                current = _OpenMatch(
                    start=second,
                    start_confidence=hud,
                    last_hud=second,
                    tracker=CounterTracker(elements),
                )
        else:
            quiet_seen = True
        if current is not None:
            current.tracker.observe(second, reader.read(frame, layout) if present else {})

    if current is not None:
        matches.append(_finish(len(matches), current))
    return AnalysisOutput(
        recording=RecordingInfo(path=str(path), duration=info.duration), matches=matches
    )
