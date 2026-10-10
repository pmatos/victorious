"""Turn a Recording and a Layout into Matches with Events."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from victorious.layout import Layout
from victorious.ocr import find_tesseract
from victorious.schema import AnalysisOutput, Event, Match, RecordingInfo
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


def _finish(index: int, match: _OpenMatch) -> Match:
    if match.results_start is None:
        end = float(match.last_hud)
        events = [
            Event(
                kind="match_start",
                time=float(match.start),
                source="frame",
                confidence=_clamp(match.start_confidence),
                value=None,
            ),
            Event(kind="match_end", time=end, source="frame", confidence=0.0, value=None),
        ]
        return Match(
            index=index,
            start=float(match.start),
            end=end,
            win=False,
            placement=None,
            events=events,
        )

    end = float(match.results_start)
    banner = max(reading.banner for reading in match.readings)
    voted = _vote(match.readings)
    if banner >= RESULTS_THRESHOLD:
        win, placement, placement_confidence = True, 1, _clamp(banner)
    elif voted is not None:
        placement, placement_confidence = voted
        win = placement == 1
    else:
        win, placement, placement_confidence = False, None, 0.0

    results_confidence = _clamp(max(max(r.banner, r.hash) for r in match.readings))
    events = [
        Event(
            kind="match_start",
            time=float(match.start),
            source="frame",
            confidence=_clamp(match.start_confidence),
            value=None,
        )
    ]
    if win:
        events.append(
            Event(
                kind="win",
                time=end,
                source="frame",
                confidence=_clamp(banner if banner >= RESULTS_THRESHOLD else placement_confidence),
                value=None,
            )
        )
    if placement is not None:
        events.append(
            Event(
                kind="final_placement",
                time=end,
                source="frame",
                confidence=_clamp(placement_confidence),
                value=placement,
            )
        )
    events.append(
        Event(kind="match_end", time=end, source="frame", confidence=results_confidence, value=None)
    )
    return Match(
        index=index, start=float(match.start), end=end, win=win, placement=placement, events=events
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
        if hud >= HUD_THRESHOLD:
            if current is not None:
                current.last_hud = second
            elif quiet_seen:
                current = _OpenMatch(start=second, start_confidence=hud, last_hud=second)
        else:
            quiet_seen = True

    if current is not None:
        matches.append(_finish(len(matches), current))
    return AnalysisOutput(
        recording=RecordingInfo(path=str(path), duration=info.duration), matches=matches
    )
