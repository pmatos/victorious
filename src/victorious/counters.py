"""Reading the HUD counters and turning their readings into Events."""

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
_POOL = ThreadPoolExecutor(max_workers=len(COUNTER_ELEMENTS))


@dataclass(frozen=True)
class CounterReading:
    value: int
    confidence: float


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
    confidences: list[float] = field(default_factory=list)


@dataclass
class _State:
    committed: int | None = None
    pending: _Pending | None = None
    gap_start: float | None = None


class CounterTracker:
    """Emits an Event when a counter's value changes, once the new value is read twice in a row."""

    def __init__(self, elements: tuple[CounterElement, ...]) -> None:
        self._states: dict[CounterElement, _State] = {element: _State() for element in elements}
        self._events: list[Event] = []

    def observe(self, second: float, readings: dict[CounterElement, CounterReading | None]) -> None:
        for element, state in self._states.items():
            reading = readings.get(element)
            if reading is None:
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
        self._events.append(frame_event("unreadable", start, 0.0, end=end, element=element))

    def _accept(
        self, element: CounterElement, state: _State, second: float, reading: CounterReading
    ) -> None:
        if state.committed is not None and reading.value == state.committed:
            state.pending = None
            return
        pending = state.pending
        if pending is None or pending.value != reading.value:
            state.pending = _Pending(reading.value, second, [reading.confidence])
            return
        pending.confidences.append(reading.confidence)
        if state.committed is not None:
            confidence = sum(pending.confidences) / len(pending.confidences)
            self._events.append(frame_event(element, pending.time, confidence, pending.value))
        state.committed = pending.value
        state.pending = None

    def close(self, end: float) -> TrackerResult:
        for element, state in self._states.items():
            self._close_gap(element, state, end)
        personal = self._states.get("personal_eliminations")
        return TrackerResult(
            events=list(self._events),
            personal_eliminations=personal.committed if personal else None,
        )
