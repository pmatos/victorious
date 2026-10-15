"""What is read from each sampled second of a Recording, before it becomes Events."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from victorious.counters import (
    COUNTER_ELEMENTS,
    CounterReading,
    tracked_rects,
)
from victorious.layout import Layout, Rect
from victorious.ocr import DigitReading
from victorious.schema import CounterElement, ObservationRecord, ObservedElement
from victorious.screens import (
    HUD_THRESHOLD,
    SPECTATING_BAND,
    ResultsReading,
    counter_masks,
    hud_score,
    results_reading,
    spectating_score,
)

Counters = dict[CounterElement, CounterReading | None]


@dataclass(frozen=True)
class SecondObservation:
    """Everything read at one sampled second.

    `counters` holds the readings the Match loop used, so it is empty at a second where no Match
    was open or the HUD was absent. `spectating` is the "SPECTATING" label's score, read only at
    seconds that show neither the HUD nor a results screen.
    """

    second: int
    hud: float
    results: ResultsReading
    counters: Counters = field(default_factory=dict)
    spectating: float = 0.0


@dataclass(frozen=True)
class PackedMask:
    bits: bytes
    shape: tuple[int, int]

    @staticmethod
    def pack(mask: np.ndarray) -> "PackedMask":
        height, width = mask.shape
        return PackedMask(np.packbits(mask > 0).tobytes(), (height, width))

    def unpack(self) -> np.ndarray:
        height, width = self.shape
        bits = np.unpackbits(np.frombuffer(self.bits, np.uint8), count=height * width)
        return np.asarray(bits.reshape(height, width) * np.uint8(255), dtype=np.uint8)


@dataclass(frozen=True)
class ChunkSecond:
    """One second as read in parallel: everything but the counter readings.

    `masks` holds each tracked counter's two bands when the HUD was on screen and a Match
    could read counters from it, else nothing.
    """

    observation: SecondObservation
    masks: dict[CounterElement, tuple[PackedMask, PackedMask]]


class Aborted(Exception):
    """Another chunk failed, so this one stopped."""


def extract_chunk(
    frames: Iterable[tuple[int, np.ndarray]],
    layout: Layout,
    should_stop: Callable[[], bool] = lambda: False,
) -> list[ChunkSecond]:
    rects = tracked_rects(layout)
    seconds: list[ChunkSecond] = []
    for second, frame in frames:
        if should_stop():
            raise Aborted
        results = results_reading(frame, layout, read_placement=True)
        hud = hud_score(frame, layout)
        spectating = 0.0 if hud >= HUD_THRESHOLD or results.is_results else spectating_score(frame)
        masks: dict[CounterElement, tuple[PackedMask, PackedMask]] = {}
        if not results.is_results and hud >= HUD_THRESHOLD:
            for element, rect in rects:
                tight, loose = counter_masks(frame, rect)
                masks[element] = (PackedMask.pack(tight), PackedMask.pack(loose))
        observation = SecondObservation(second, hud, results, spectating=spectating)
        seconds.append(ChunkSecond(observation, masks))
    return seconds


def reading_to_json(reading: CounterReading | None) -> list[float] | None:
    return None if reading is None else [reading.value, reading.confidence]


def reading_from_json(raw: list[float] | None) -> CounterReading | None:
    return None if raw is None else CounterReading(int(raw[0]), float(raw[1]))


def to_json(observation: SecondObservation) -> dict[str, Any]:
    results = observation.results
    digits = results.digits
    return {
        "second": observation.second,
        "hud": observation.hud,
        "banner": results.banner,
        "hash": results.hash,
        "placement_score": results.placement_score,
        "elimination_placement_score": results.elimination_placement_score,
        "digits": None if digits is None else [digits.text, digits.confidence],
        "digits_element": results.digits_element,
        "counters": {k: reading_to_json(v) for k, v in observation.counters.items()},
        "spectating": observation.spectating,
    }


def from_json(raw: dict[str, Any]) -> SecondObservation:
    digits = raw["digits"]
    results = ResultsReading(
        banner=float(raw["banner"]),
        hash=float(raw["hash"]),
        digits=None if digits is None else DigitReading(str(digits[0]), float(digits[1])),
        placement_score=float(raw["placement_score"]),
        elimination_placement_score=float(raw["elimination_placement_score"]),
        digits_element=raw["digits_element"],
    )
    counters: Counters = {}
    for element in COUNTER_ELEMENTS:
        if element in raw["counters"]:
            counters[element] = reading_from_json(raw["counters"][element])
    return SecondObservation(
        int(raw["second"]), float(raw["hud"]), results, counters, float(raw["spectating"])
    )


def _placement_digits(
    results: ResultsReading, element: ObservedElement
) -> tuple[int | None, float]:
    digits = results.digits
    if digits is None or results.digits_element != element or not digits.text.isdigit():
        return None, 0.0
    return int(digits.text), digits.confidence


def records(
    observations: Iterable[SecondObservation], layout: Layout
) -> Iterator[ObservationRecord]:
    """One record per second per HUD element the Layout describes."""
    regions: dict[str, Rect | None] = {
        **dict(layout.screens.hud),
        **dict(layout.screens.results),
        "spectating": SPECTATING_BAND,
    }
    for observation in observations:
        results = observation.results
        rows: list[tuple[ObservedElement, float | None, int | None, float]] = [
            ("hud_anchor", observation.hud, None, 0.0)
        ]
        for counter in COUNTER_ELEMENTS:
            reading = observation.counters.get(counter)
            if reading is None:
                rows.append((counter, None, None, 0.0))
            else:
                rows.append((counter, None, reading.value, reading.confidence))
        rows.append(("results_banner", results.banner, None, 0.0))
        rows.append(
            ("placement", results.placement_score, *_placement_digits(results, "placement"))
        )
        rows.append(
            (
                "elimination_placement",
                results.elimination_placement_score,
                *_placement_digits(results, "elimination_placement"),
            )
        )
        rows.append(("spectating", observation.spectating, None, 0.0))
        for element, score, value, confidence in rows:
            region = regions[element]
            if region is not None:
                yield ObservationRecord(
                    second=observation.second,
                    element=element,
                    value=value,
                    confidence=min(1.0, max(0.0, confidence)),
                    score=score,
                    region=region,
                )
