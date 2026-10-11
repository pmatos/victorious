"""Reading the Recording's seconds in parallel chunks, and OCR of counters in parallel."""

import math
import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import (
    FIRST_EXCEPTION,
    BrokenExecutor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from pathlib import Path

import cv2
import numpy as np

from victorious.counters import CounterReading, best_reading
from victorious.layout import Layout
from victorious.observations import Aborted, ChunkSecond, extract_chunk
from victorious.ocr import PASSES
from victorious.video import RecordingError, VideoInfo, sample_chunk

_ABORT: Event | None = None
_WORKER_DIED = "a decoding worker died unexpectedly"


@dataclass(frozen=True)
class _Chunk:
    path: Path
    info: VideoInfo
    layout: Layout
    start: int
    length: int | None
    threads: int


def _initialise_worker(abort: Event) -> None:
    global _ABORT
    _ABORT = abort
    cv2.setNumThreads(1)


def _stopped() -> bool:
    return _ABORT is not None and _ABORT.is_set()


def _extract(chunk: _Chunk) -> list[ChunkSecond]:
    frames = sample_chunk(chunk.path, chunk.info, chunk.start, chunk.length, chunk.threads)
    return extract_chunk(frames, chunk.layout, _stopped)


def plan_chunks(duration: float, chunk_seconds: int) -> list[tuple[int, int | None]]:
    """Whole-second (start, length) pairs; the last one runs to the end of the Recording."""
    starts = list(range(0, math.floor(duration) + 1, chunk_seconds))
    return [(start, chunk_seconds) for start in starts[:-1]] + [(starts[-1], None)]


def _run_in_pool(chunks: list[_Chunk], workers: int) -> list[list[ChunkSecond]]:
    context = multiprocessing.get_context("spawn")
    abort = context.Event()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialise_worker,
        initargs=(abort,),
    ) as pool:
        futures: list[Future[list[ChunkSecond]]] = []
        try:
            futures.extend(pool.submit(_extract, c) for c in chunks)
            wait(futures, return_when=FIRST_EXCEPTION)
        except BrokenExecutor as broken:
            raise RecordingError(_WORKER_DIED) from broken
        finally:
            abort.set()
            for future in futures:
                future.cancel()
        wait(futures)
        failures = [
            error
            for future in futures
            if not future.cancelled() and (error := future.exception()) is not None
        ]
        real = [error for error in failures if not isinstance(error, Aborted)]
        if failures:
            error = (real or failures)[0]
            if isinstance(error, BrokenExecutor):
                raise RecordingError(_WORKER_DIED) from error
            raise error
        return [future.result() for future in futures]


def extract_seconds(
    path: Path, info: VideoInfo, layout: Layout, jobs: int, chunk_seconds: int
) -> list[ChunkSecond]:
    """Every sampled second of the Recording, read in chunks on up to `jobs` processes."""
    plan = plan_chunks(info.duration, chunk_seconds)
    workers = max(1, min(jobs, len(plan)))
    threads = max(1, (os.cpu_count() or 1) // workers)
    chunks = [_Chunk(path, info, layout, start, length, threads) for start, length in plan]
    if workers == 1:
        results = [_extract(chunk) for chunk in chunks]
    else:
        results = _run_in_pool(chunks, workers)
    seconds = [second for chunk in results for second in chunk]
    if [second.observation.second for second in seconds] != list(range(len(seconds))):
        raise RecordingError(f"cannot decode Recording {path}: frames are missing")
    return seconds


def read_counters(masks: Sequence[list[np.ndarray]], jobs: int) -> list[CounterReading | None]:
    """Tesseract on each list of masks, `jobs` at a time."""
    readers = max(1, jobs // len(PASSES))
    if readers == 1 or len(masks) < 2:
        return [best_reading(candidates) for candidates in masks]
    with ThreadPoolExecutor(max_workers=readers) as pool:
        return list(pool.map(best_reading, masks))
