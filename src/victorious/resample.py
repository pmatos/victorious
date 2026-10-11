"""Narrowing the time of each step of a counter jump by re-sampling only that gap."""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from victorious.counters import CounterReading, Gap, Step, read_counter_series
from victorious.layout import Layout
from victorious.video import VideoInfo, sample_window

RESAMPLE_RATE = 4
MAX_WORKERS = 8
CORES_PER_DECODE = 3


def _without_spikes(
    gap: Gap, samples: Sequence[tuple[float, CounterReading]]
) -> list[tuple[float, CounterReading]]:
    """Drop a sample that differs from both neighbours when the neighbours agree."""
    values = [gap.old, *(reading.value for _, reading in samples), gap.new]
    return [
        sample
        for index, sample in enumerate(samples, start=1)
        if values[index - 1] != values[index + 1] or values[index] == values[index - 1]
    ]


def resolve_jump(gap: Gap, samples: Sequence[tuple[float, CounterReading | None]]) -> list[Step]:
    """One Step per value from just after `gap.old` through `gap.new`.

    A step is timed at the first sample from which every later readable sample is at or beyond
    its value, so one misread sample cannot move it. The last slot of the window is the second
    that confirmed `gap.new`, so every step has a time.
    """
    confirmed = (gap.end + 1 - 1 / RESAMPLE_RATE, CounterReading(gap.new, gap.confidence))
    low, high = sorted((gap.old, gap.new))
    readable = [
        (time, reading)
        for time, reading in samples
        if reading is not None and low <= reading.value <= high
    ]
    readable = [*_without_spikes(gap, readable), confirmed]
    direction = 1 if gap.new > gap.old else -1
    steps: list[Step] = []
    for value in range(gap.old + direction, gap.new + direction, direction):
        first = len(readable) - 1
        while first > 0 and (readable[first - 1][1].value - value) * direction >= 0:
            first -= 1
        time, reading = readable[first]
        steps.append(Step(value, time, reading.confidence))
    return steps


class JumpResolver:
    """Decodes a short window around each Gap and times its steps; Gaps run in parallel."""

    def __init__(self, path: Path, info: VideoInfo, layout: Layout) -> None:
        self._path = path
        self._info = info
        self._layout = layout
        self._workers = max(1, min(MAX_WORKERS, (os.cpu_count() or 1) // CORES_PER_DECODE))

    def __call__(self, gaps: Sequence[Gap]) -> list[list[Step]]:
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(self._resolve, gaps))

    def _resolve(self, gap: Gap) -> list[Step]:
        frames = sample_window(self._path, self._info, gap.start + 1, gap.end + 1, RESAMPLE_RATE)
        return resolve_jump(gap, read_counter_series(frames, self._layout, gap.element))
