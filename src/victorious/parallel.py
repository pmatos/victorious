"""Reading the Recording's seconds in parallel chunks, and OCR of counters in parallel."""

import math
import multiprocessing
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    FIRST_EXCEPTION,
    BrokenExecutor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from multiprocessing.synchronize import BoundedSemaphore, Event
from pathlib import Path

import cv2
import numpy as np

from victorious import hwaccel
from victorious.counters import CounterReading, best_reading
from victorious.hwaccel import HwDecoder
from victorious.layout import Layout
from victorious.observations import Aborted, ChunkSecond, extract_chunk
from victorious.ocr import concurrent_readers
from victorious.video import (
    TAIL_TOLERANCE,
    RecordingError,
    Route,
    RouteKind,
    VideoInfo,
    resolve_hardware,
    sample_chunk,
)

HW_SLOTS = 2
# NVDEC on an RTX 4090 kept gaining up to 8 (frames of a 3 h Recording: 462 s at 4, 434 s at 8,
# 893 s with every chunk on the GPU), so an NVIDIA GPU gets more slots than VAAPI.
CUDA_SLOTS = 8
SLOTS_VARIABLE = "VICTORIOUS_HW_SLOTS"

_ABORT: Event | None = None
_GPU: BoundedSemaphore | None = None
_WORKER_DIED = "a decoding worker died unexpectedly"


@dataclass(frozen=True)
class _Chunk:
    path: Path
    info: VideoInfo
    layout: Layout
    start: int
    length: int | None
    threads: int
    hardware: bool = False
    gpu_only: bool = False


def _initialise_worker(abort: Event, shared: hwaccel.Shared, gpu: BoundedSemaphore) -> None:
    global _ABORT, _GPU
    _ABORT = abort
    _GPU = gpu
    hwaccel.adopt(shared)
    cv2.setNumThreads(1)


def _stopped() -> bool:
    return _ABORT is not None and _ABORT.is_set()


def _extract(chunk: _Chunk) -> tuple[list[ChunkSecond], RouteKind]:
    """Decodes on the GPU when a slot is free and on the CPU otherwise.

    The last chunks wait for a slot instead: a CPU decode started late would outlast the GPU's
    queue and leave the other workers idle.
    """
    gate = _GPU if chunk.hardware and not hwaccel.is_disabled() else None
    on_gpu = gate is not None and gate.acquire(block=chunk.gpu_only)
    route = Route()
    try:
        frames = sample_chunk(
            chunk.path,
            chunk.info,
            chunk.start,
            chunk.length,
            chunk.threads,
            hardware=chunk.hardware and (gate is None or on_gpu),
            route=route,
        )
        return extract_chunk(frames, chunk.layout, _stopped), route.kind
    finally:
        if on_gpu and gate is not None:
            gate.release()


def gpu_slots(env: Mapping[str, str], decoder: HwDecoder | None = None) -> int:
    """How many chunks decode on the GPU at once: a VAAPI device saturates at about two, NVDEC
    at about eight; the CPU decodes the chunks that find no free slot."""
    default = CUDA_SLOTS if decoder is not None and decoder.api == "cuda" else HW_SLOTS
    try:
        return max(0, int(env.get(SLOTS_VARIABLE, default)))
    except ValueError:
        return default


def plan_chunks(duration: float, chunk_seconds: int) -> list[tuple[int, int | None]]:
    """Whole-second (start, length) pairs; the last one runs to the end of the Recording."""
    starts = list(range(0, math.floor(duration) + 1, chunk_seconds))
    return [(start, chunk_seconds) for start in starts[:-1]] + [(starts[-1], None)]


def _run_in_pool(
    chunks: list[_Chunk], workers: int, slots: int
) -> list[tuple[list[ChunkSecond], RouteKind]]:
    context = multiprocessing.get_context("spawn")
    abort = context.Event()
    shared = hwaccel.share(context.Event())
    gpu = context.BoundedSemaphore(max(1, slots))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialise_worker,
        initargs=(abort, shared, gpu),
    ) as pool:
        futures: list[Future[tuple[list[ChunkSecond], RouteKind]]] = []
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
) -> tuple[list[ChunkSecond], dict[RouteKind, int]]:
    """Every sampled second of the Recording, read in chunks on up to `jobs` processes.

    Also how many chunks were decoded on the `gpu`, on the `cpu`, and `resumed` on the CPU after
    the GPU failed part-way.
    """
    decoder = resolve_hardware(path, info) if gpu_slots(os.environ) > 0 else None
    slots = gpu_slots(os.environ, decoder)
    hardware = decoder is not None
    plan = plan_chunks(info.duration, chunk_seconds)
    workers = max(1, min(jobs, len(plan)))
    threads = max(1, (os.cpu_count() or 1) // workers)
    chunks = [
        _Chunk(path, info, layout, start, length, threads, hardware, k >= len(plan) - workers)
        for k, (start, length) in enumerate(plan)
    ]
    if workers == 1:
        results = [_extract(chunk) for chunk in chunks]
    else:
        results = _run_in_pool(chunks, workers, slots)
    seconds = [second for chunk, _ in results for second in chunk]
    routes = Counter(route for _, route in results)
    if [second.observation.second for second in seconds] != list(range(len(seconds))):
        raise RecordingError(f"cannot decode Recording {path}: frames are missing")
    if len(seconds) < math.floor(info.duration) - TAIL_TOLERANCE:
        raise RecordingError(
            f"cannot read Recording {path}: it is truncated (decoded {len(seconds)} s, "
            f"its header claims {info.duration:.0f} s)"
        )
    return seconds, dict(routes)


def read_counters(masks: Sequence[list[np.ndarray]], jobs: int) -> list[CounterReading | None]:
    """Tesseract on each list of masks, `jobs` at a time."""
    readers = concurrent_readers(jobs)
    if readers == 1 or len(masks) < 2:
        return [best_reading(candidates) for candidates in masks]
    with ThreadPoolExecutor(max_workers=readers) as pool:
        return list(pool.map(best_reading, masks))
