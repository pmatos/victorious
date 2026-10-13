"""Reading the HUD counters and turning their readings into Events."""

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import get_args

import numpy as np

from victorious.layout import Layout, Rect
from victorious.ocr import DigitReading, read_digits
from victorious.schema import CounterElement, Event, frame_event
from victorious.screens import counter_masks

COUNTER_ELEMENTS: tuple[CounterElement, ...] = get_args(CounterElement)
MIN_COUNTER_CONFIDENCE = 0.05
RETRY_CONFIDENCE = 0.5
MAX_COUNTER_DIGITS = 3
UNREADABLE_SECONDS = 10.0
REUSE_MISMATCH = 0.04
MAX_RESOLVED_JUMP: dict[CounterElement, int] = {
    "personal_eliminations": 3,
    "team_eliminations": 3,
    "players_left": 10,
}
MAX_GAP_SECONDS = 3.0
TRANSIENT_SECONDS = 5.0
RISING: dict[CounterElement, int] = {
    "personal_eliminations": 1,
    "team_eliminations": 1,
    "players_left": -1,
}
_POOL = ThreadPoolExecutor(max_workers=len(COUNTER_ELEMENTS))


@dataclass(frozen=True)
class CounterReading:
    value: int
    confidence: float


@dataclass(frozen=True)
class Step:
    value: int
    time: float
    confidence: float


@dataclass(frozen=True)
class Gap:
    """A jump of more than one: `old` was last read at `start`, `new` first read at `end`."""

    element: CounterElement
    start: float
    end: float
    old: int
    new: int
    confidence: float


Resolver = Callable[[Sequence[Gap]], list[list[Step]]]


def _further(element: CounterElement, current: int | None, value: int) -> int:
    """The value that is further along the direction the counter moves in.

    Only the eliminations counters never move back within a Match; the players-left counter
    rises while the lobby fills, so it has no such memory.
    """
    if element == "players_left":
        return value
    if current is None or (value - current) * RISING[element] > 0:
        return value
    return current


def is_ambiguous(element: CounterElement, old: int, new: int) -> bool:
    return 1 < (new - old) * RISING[element] <= MAX_RESOLVED_JUMP[element]


@dataclass(frozen=True)
class TrackerResult:
    events: list[Event]
    personal_eliminations: int | None


def counter_rect(layout: Layout, element: CounterElement) -> Rect | None:
    rect: Rect | None = getattr(layout.screens.hud, element)
    return rect


def tracked_rects(layout: Layout) -> list[tuple[CounterElement, Rect]]:
    found = ((element, counter_rect(layout, element)) for element in COUNTER_ELEMENTS)
    return [(element, rect) for element, rect in found if rect is not None]


def tracked_elements(layout: Layout) -> tuple[CounterElement, ...]:
    return tuple(element for element, _ in tracked_rects(layout))


def _same_pixels(previous: np.ndarray, current: np.ndarray) -> bool:
    if previous.shape != current.shape:
        return False
    mismatch = int(np.count_nonzero(previous != current))
    union = int(np.count_nonzero(previous | current))
    return mismatch <= REUSE_MISMATCH * max(1, union)


def _to_reading(reading: DigitReading) -> CounterReading | None:
    text = reading.text
    plausible = (
        text.isdigit()
        and len(text) <= MAX_COUNTER_DIGITS
        and (len(text) == 1 or text[0] != "0")
        and reading.confidence >= MIN_COUNTER_CONFIDENCE
    )
    return CounterReading(int(text), reading.confidence) if plausible else None


def best_reading(masks: list[np.ndarray]) -> CounterReading | None:
    best: CounterReading | None = None
    for mask in masks:
        digits = read_digits(mask)
        reading = _to_reading(digits) if digits is not None else None
        if reading is not None and (best is None or reading.confidence > best.confidence):
            best = reading
        if best is not None and best.confidence >= RETRY_CONFIDENCE:
            break
    return best


def read_counter_series(
    frames: Iterable[tuple[float, np.ndarray]], layout: Layout, element: CounterElement
) -> list[tuple[float, CounterReading | None]]:
    """One counter read in each frame, skipping Tesseract while its pixels did not change."""
    rect = counter_rect(layout, element)
    if rect is None:
        return [(time, None) for time, _ in frames]
    series: list[tuple[float, CounterReading | None]] = []
    previous: np.ndarray | None = None
    reading: CounterReading | None = None
    for time, frame in frames:
        masks = counter_masks(frame, rect)
        if previous is None or not _same_pixels(previous, masks[0]):
            previous, reading = masks[0], best_reading(masks)
        series.append((time, reading))
    return series


class ReuseTracker:
    """Decides when a counter's pixels differ enough from the last OCR'd ones to read again."""

    def __init__(self) -> None:
        self._seen: dict[CounterElement, np.ndarray] = {}

    def reset(self) -> None:
        self._seen.clear()

    def needs_ocr(self, element: CounterElement, mask: np.ndarray) -> bool:
        seen = self._seen.get(element)
        if seen is not None and _same_pixels(seen, mask):
            return False
        self._seen[element] = mask
        return True


class CounterReader:
    """Reads each counter once per frame, skipping Tesseract when its pixels did not change."""

    def __init__(self) -> None:
        self._reuse = ReuseTracker()
        self._readings: dict[CounterElement, CounterReading | None] = {}

    def reset(self) -> None:
        self._reuse.reset()

    def read(
        self, frame: np.ndarray, layout: Layout
    ) -> dict[CounterElement, CounterReading | None]:
        readings: dict[CounterElement, CounterReading | None] = {}
        fresh: dict[CounterElement, list[np.ndarray]] = {}
        for element in COUNTER_ELEMENTS:
            rect = counter_rect(layout, element)
            if rect is None:
                continue
            masks = counter_masks(frame, rect)
            if self._reuse.needs_ocr(element, masks[0]):
                fresh[element] = masks
            else:
                readings[element] = self._readings[element]
        for element, reading in zip(fresh, _POOL.map(best_reading, fresh.values()), strict=True):
            self._readings[element] = reading
            readings[element] = reading
        return readings


@dataclass
class _Pending:
    value: int
    time: float
    start: float | None
    confidences: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _Change:
    """A committed change of a counter: its Event and the value it left."""

    index: int
    previous: int
    time: float


@dataclass
class _State:
    committed: int | None = None
    recent_changes: list[_Change] = field(default_factory=list)
    pending: _Pending | None = None
    gap_start: float | None = None
    last_seen: float | None = None
    furthest: int | None = None


class CounterTracker:
    """Emits an Event when a counter's value changes, once the new value is read twice in a row.

    A jump of more than one between two readable seconds is recorded as a Gap; `close` hands the
    Gaps to a resolver and replaces each jump's single Event by one Event per step. A
    `players_left` change that goes back to the value it left within `TRANSIENT_SECONDS` is a
    misread, not a change: neither it, nor any change made since, nor the return becomes an Event.
    """

    def __init__(self, elements: tuple[CounterElement, ...]) -> None:
        self._states: dict[CounterElement, _State] = {element: _State() for element in elements}
        self._events: list[Event] = []
        self._gaps: dict[int, Gap] = {}
        self._dropped: set[int] = set()

    def observe(self, second: float, readings: dict[CounterElement, CounterReading | None]) -> None:
        for element, state in self._states.items():
            reading = readings.get(element)
            if reading is None:
                state.last_seen = None
                if state.gap_start is None:
                    state.gap_start = second
                continue
            self._close_gap(element, state, second)
            self._accept(element, state, second, reading)

    def _close_gap(self, element: CounterElement, state: _State, end: float) -> None:
        start, state.gap_start = state.gap_start, None
        if start is None or end - start <= UNREADABLE_SECONDS:
            return
        state.pending = None
        state.recent_changes.clear()
        self._events.append(frame_event("unreadable", start, 0.0, end=end, element=element))

    def _accept(
        self, element: CounterElement, state: _State, second: float, reading: CounterReading
    ) -> None:
        if state.committed is not None and reading.value == state.committed:
            state.pending = None
            state.last_seen = second
            return
        pending = state.pending
        if pending is None or pending.value != reading.value:
            state.pending = _Pending(reading.value, second, state.last_seen, [reading.confidence])
            return
        pending.confidences.append(reading.confidence)
        previous = state.committed
        if previous is not None:
            returned_from = self._return_start(element, state, pending)
            if returned_from is not None:
                self._drop_transient(state, returned_from)
            else:
                self._commit_change(element, state, pending, previous)
        state.committed = pending.value
        state.furthest = _further(element, state.furthest, pending.value)
        state.pending = None
        state.last_seen = second

    def _commit_change(
        self, element: CounterElement, state: _State, pending: _Pending, previous: int
    ) -> None:
        confidence = sum(pending.confidences) / len(pending.confidences)
        self._events.append(frame_event(element, pending.time, confidence, pending.value))
        state.recent_changes = [
            change
            for change in state.recent_changes
            if pending.time - change.time <= TRANSIENT_SECONDS
        ]
        state.recent_changes.append(_Change(len(self._events) - 1, previous, pending.time))
        self._record_gap(element, state, pending, confidence)

    @staticmethod
    def _return_start(element: CounterElement, state: _State, pending: _Pending) -> int | None:
        if element != "players_left":
            return None
        for position, change in enumerate(state.recent_changes):
            if change.previous == pending.value:
                within = pending.time - change.time <= TRANSIENT_SECONDS
                return position if within else None
        return None

    def _drop_transient(self, state: _State, start: int) -> None:
        for change in state.recent_changes[start:]:
            self._dropped.add(change.index)
            self._gaps.pop(change.index, None)
        del state.recent_changes[start:]

    def _record_gap(
        self, element: CounterElement, state: _State, pending: _Pending, confidence: float
    ) -> None:
        start, committed = pending.start, state.committed
        if start is None or committed is None or pending.time - start > MAX_GAP_SECONDS:
            return
        old = _further(element, state.furthest, committed)
        if is_ambiguous(element, old, pending.value):
            gap = Gap(element, start, pending.time, old, pending.value, confidence)
            self._gaps[len(self._events) - 1] = gap

    def close(self, end: float, resolver: Resolver | None = None) -> TrackerResult:
        for element, state in self._states.items():
            self._close_gap(element, state, end)
        resolved: dict[int, list[Step]] = {}
        if self._gaps and resolver is not None:
            resolved = dict(zip(self._gaps, resolver(list(self._gaps.values())), strict=True))
        events: list[Event] = []
        for index, event in enumerate(self._events):
            if index in self._dropped:
                continue
            steps = resolved.get(index)
            if steps:
                events.extend(
                    frame_event(event.kind, step.time, step.confidence, step.value)
                    for step in steps
                )
            else:
                events.append(event)
        personal = self._states.get("personal_eliminations")
        return TrackerResult(
            events=events,
            personal_eliminations=personal.committed if personal else None,
        )
