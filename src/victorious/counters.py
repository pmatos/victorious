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


def is_ambiguous(element: CounterElement, old: int, new: int) -> bool:
    jump = new - old
    return 1 < abs(jump) <= MAX_RESOLVED_JUMP[element] and (jump > 0) == (RISING[element] > 0)


@dataclass(frozen=True)
class TrackerResult:
    events: list[Event]
    personal_eliminations: int | None


def _rect(layout: Layout, element: CounterElement) -> Rect | None:
    rect: Rect | None = getattr(layout.screens.hud, element)
    return rect


def tracked_elements(layout: Layout) -> tuple[CounterElement, ...]:
    return tuple(element for element in COUNTER_ELEMENTS if _rect(layout, element) is not None)


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


def _best_reading(masks: list[np.ndarray]) -> CounterReading | None:
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
    rect = _rect(layout, element)
    series: list[tuple[float, CounterReading | None]] = []
    seen: tuple[np.ndarray, CounterReading | None] | None = None
    for time, frame in frames:
        if rect is not None:
            masks = counter_masks(frame, rect)
            if seen is None or not _same_pixels(seen[0], masks[0]):
                seen = (masks[0], _best_reading(masks))
        series.append((time, seen[1] if seen is not None else None))
    return series


class CounterReader:
    """Reads each counter once per frame, skipping Tesseract when its pixels did not change."""

    def __init__(self) -> None:
        self._seen: dict[CounterElement, tuple[np.ndarray, CounterReading | None]] = {}

    def reset(self) -> None:
        self._seen.clear()

    def read(
        self, frame: np.ndarray, layout: Layout
    ) -> dict[CounterElement, CounterReading | None]:
        readings: dict[CounterElement, CounterReading | None] = {}
        fresh: dict[CounterElement, list[np.ndarray]] = {}
        for element in COUNTER_ELEMENTS:
            rect = _rect(layout, element)
            if rect is None:
                continue
            masks = counter_masks(frame, rect)
            seen = self._seen.get(element)
            if seen is not None and _same_pixels(seen[0], masks[0]):
                readings[element] = seen[1]
            else:
                fresh[element] = masks
        for (element, masks), reading in zip(
            fresh.items(), _POOL.map(_best_reading, fresh.values()), strict=True
        ):
            self._seen[element] = (masks[0], reading)
            readings[element] = reading
        return readings


@dataclass
class _Pending:
    value: int
    time: float
    blind: bool
    confidences: list[float] = field(default_factory=list)


@dataclass
class _State:
    committed: int | None = None
    pending: _Pending | None = None
    gap_start: float | None = None
    last_seen: float | None = None
    none_since_seen: bool = False


class CounterTracker:
    """Emits an Event when a counter's value changes, once the new value is read twice in a row.

    A jump of more than one between two readable seconds is recorded as a Gap; `close` hands the
    Gaps to a resolver and replaces each jump's single Event by one Event per step.
    """

    def __init__(self, elements: tuple[CounterElement, ...]) -> None:
        self._states: dict[CounterElement, _State] = {element: _State() for element in elements}
        self._events: list[Event] = []
        self._gaps: dict[int, Gap] = {}

    def observe(self, second: float, readings: dict[CounterElement, CounterReading | None]) -> None:
        for element, state in self._states.items():
            reading = readings.get(element)
            if reading is None:
                state.none_since_seen = True
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
        state.last_seen = None
        self._events.append(frame_event("unreadable", start, 0.0, end=end, element=element))

    def _accept(
        self, element: CounterElement, state: _State, second: float, reading: CounterReading
    ) -> None:
        if state.committed is not None and reading.value == state.committed:
            state.pending = None
            state.last_seen = second
            state.none_since_seen = False
            return
        pending = state.pending
        if pending is None or pending.value != reading.value:
            state.pending = _Pending(
                reading.value, second, state.none_since_seen, [reading.confidence]
            )
            return
        pending.confidences.append(reading.confidence)
        if state.committed is not None:
            confidence = sum(pending.confidences) / len(pending.confidences)
            self._events.append(frame_event(element, pending.time, confidence, pending.value))
            self._record_gap(element, state, pending, confidence)
        state.committed = pending.value
        state.pending = None
        state.last_seen = second
        state.none_since_seen = False

    def _record_gap(
        self, element: CounterElement, state: _State, pending: _Pending, confidence: float
    ) -> None:
        start, old = state.last_seen, state.committed
        if start is None or old is None or pending.blind:
            return
        if not 0 < pending.time - start <= MAX_GAP_SECONDS:
            return
        if is_ambiguous(element, old, pending.value):
            gap = Gap(element, start, pending.time, old, pending.value, confidence)
            self._gaps[len(self._events) - 1] = gap

    def close(self, end: float, resolver: Resolver | None = None) -> TrackerResult:
        for element, state in self._states.items():
            self._close_gap(element, state, end)
        events = list(self._events)
        if self._gaps and resolver is not None:
            indices = list(self._gaps)
            resolved = dict(zip(indices, resolver([self._gaps[i] for i in indices]), strict=True))
            events = []
            for index, event in enumerate(self._events):
                steps = resolved.get(index)
                if not steps:
                    events.append(event)
                    continue
                gap = self._gaps[index]
                events.extend(
                    frame_event(gap.element, step.time, step.confidence, step.value)
                    for step in steps
                )
        personal = self._states.get("personal_eliminations")
        return TrackerResult(
            events=events,
            personal_eliminations=personal.committed if personal else None,
        )
