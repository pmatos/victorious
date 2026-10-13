"""Reading digits from a binary mask with Tesseract, one glyph at a time.

Line mode confuses adjacent identical digits (11 reads as 1), so each connected
component is recognised on its own. Tesseract's fast models drop some glyphs
(a 0 under a digit whitelist, a 4 at one size), so every glyph is read with a few
independent settings and the digits vote by confidence. The glyph is smoothed
after scaling because the fast models misread the hard edges of a binary mask
(a 6 whose tip is missing reads as 0, a 4 as 1).
"""

import atexit
import contextlib
import ctypes
import ctypes.util
import functools
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

GLYPH_PADDING = 16
MIN_GLYPH_AREA = 12
MIN_GLYPH_HEIGHT_RATIO = 0.5
SMOOTHING_SIGMA = 0.05  # of the scaled glyph height
LOOKALIKES = {"O": "0", "o": "0"}


@dataclass(frozen=True)
class _Pass:
    page_segmentation_mode: int
    glyph_height: int
    whitelist: bool


PASSES = (_Pass(8, 96, True), _Pass(8, 80, True), _Pass(10, 64, False))
# One slot per pass for each concurrent reader: parallel counter OCR shares this pool.
_PASS_POOL = ThreadPoolExecutor(max_workers=len(PASSES) * (os.cpu_count() or 1))
_TESSERACT_ENV = {**os.environ, "OMP_THREAD_LIMIT": "1"}


class OcrError(Exception):
    """Tesseract is missing or failed."""


class _Engine:
    """One Tesseract instance configured as a pass, used from a single thread."""

    def __init__(self, library: ctypes.CDLL, settings: _Pass) -> None:
        self._library = library
        self._handle = library.TessBaseAPICreate()
        if library.TessBaseAPIInit3(self._handle, None, b"eng") != 0:
            library.TessBaseAPIDelete(self._handle)
            raise OcrError("tesseract failed to load its English model")
        library.TessBaseAPISetPageSegMode(self._handle, settings.page_segmentation_mode)
        if settings.whitelist:
            library.TessBaseAPISetVariable(self._handle, b"tessedit_char_whitelist", b"0123456789")

    def tsv(self, image: np.ndarray) -> str:
        _single_threaded()
        pixels = np.ascontiguousarray(image)
        height, width = pixels.shape
        library = self._library
        library.TessBaseAPISetImage(self._handle, pixels.ctypes.data, width, height, 1, width)
        text = library.TessBaseAPIGetTsvText(self._handle, 0)
        if not text:
            raise OcrError("tesseract produced no output")
        try:
            return ctypes.string_at(text).decode(errors="replace")
        finally:
            library.TessDeleteText(text)
            library.TessBaseAPIClear(self._handle)

    def close(self) -> None:
        self._library.TessBaseAPIEnd(self._handle)
        self._library.TessBaseAPIDelete(self._handle)


_ENGINES_LOCK = threading.Lock()
_UNUSABLE = threading.Event()
_THREAD = threading.local()
_IDLE: dict[tuple[int, bool], list[_Engine]] = {}
_ALL: list[_Engine] = []


_HANDLE = ctypes.c_void_p
_SIGNATURES: dict[str, tuple[type | None, list[type]]] = {
    "TessBaseAPICreate": (_HANDLE, []),
    "TessBaseAPIInit3": (ctypes.c_int, [_HANDLE, ctypes.c_char_p, ctypes.c_char_p]),
    "TessBaseAPISetVariable": (ctypes.c_bool, [_HANDLE, ctypes.c_char_p, ctypes.c_char_p]),
    "TessBaseAPISetPageSegMode": (None, [_HANDLE, ctypes.c_int]),
    "TessBaseAPISetImage": (
        None,
        [_HANDLE, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ),
    "TessBaseAPIGetTsvText": (ctypes.c_void_p, [_HANDLE, ctypes.c_int]),
    "TessBaseAPIClear": (None, [_HANDLE]),
    "TessBaseAPIEnd": (None, [_HANDLE]),
    "TessBaseAPIDelete": (None, [_HANDLE]),
    "TessDeleteText": (None, [ctypes.c_void_p]),
}


def _load_library(name: str) -> ctypes.CDLL | None:
    try:
        library = ctypes.CDLL(name)
        for function, (result, arguments) in _SIGNATURES.items():
            member = getattr(library, function)
            member.restype = result
            member.argtypes = arguments
    except (OSError, AttributeError):
        return None
    return library


@functools.cache
def _load_tesseract() -> ctypes.CDLL | None:
    for name in ("libtesseract.so.5", ctypes.util.find_library("tesseract")):
        if name is not None and (library := _load_library(name)) is not None:
            return library
    return None


def _library() -> ctypes.CDLL | None:
    """libtesseract, loaded once, or None when only the command line is installed or the
    library could not load its model."""
    return None if _UNUSABLE.is_set() else _load_tesseract()


@functools.cache
def _openmp() -> ctypes.CDLL | None:
    try:
        runtime = ctypes.CDLL("libgomp.so.1")
        runtime.omp_set_num_threads.argtypes = [ctypes.c_int]
        runtime.omp_set_num_threads.restype = None
    except (OSError, AttributeError):
        return None
    return runtime


def _single_threaded() -> None:
    """Keeps Tesseract's OpenMP team to the calling thread: the readers are already parallel.

    The setting belongs to the calling thread; a thread that later runs a library sharing
    this runtime inherits it.
    """
    if getattr(_THREAD, "limited", False):
        return
    runtime = _openmp()
    if runtime is not None:
        runtime.omp_set_num_threads(1)
    _THREAD.limited = True


@contextlib.contextmanager
def _engine(settings: _Pass) -> Iterator[_Engine | None]:
    """An idle instance configured as `settings` for the caller's use alone, or None when
    libtesseract is unavailable. Instances are created on demand, one per concurrent reader."""
    library = _library()
    if library is None:
        yield None
        return
    key = (settings.page_segmentation_mode, settings.whitelist)
    with _ENGINES_LOCK:
        idle = _IDLE.setdefault(key, [])
        engine = idle.pop() if idle else None
    if engine is None:
        try:
            engine = _Engine(library, settings)
        except OcrError:
            _UNUSABLE.set()
            yield None
            return
        with _ENGINES_LOCK:
            _ALL.append(engine)
    try:
        yield engine
    finally:
        with _ENGINES_LOCK:
            _IDLE.setdefault(key, []).append(engine)


@atexit.register
def _close_engines() -> None:
    with _ENGINES_LOCK:
        for engine in _ALL:
            engine.close()
        _ALL.clear()
        _IDLE.clear()


def concurrent_readers(jobs: int) -> int:
    """How many masks to read at once: an in-process pass is one thread, a spawned one is three."""
    return max(1, jobs if _library() is not None else jobs // len(PASSES))


@dataclass(frozen=True)
class DigitReading:
    text: str
    confidence: float  # 0..1, the weakest glyph


def find_tesseract() -> str:
    path = shutil.which("tesseract")
    if path is None:
        raise OcrError("tesseract not found on PATH; install tesseract-ocr")
    return path


def require_ocr() -> None:
    """Raises OcrError unless digits can be read, in-process or through the command line."""
    if _library() is None:
        find_tesseract()


def glyph_boxes(mask: np.ndarray, max_height: int | None = None) -> list[tuple[int, int, int, int]]:
    """The glyph-sized components of `mask`, left to right; none taller than `max_height`.

    The height cap applies before the relative size filter, so a tall bright blob that is
    no glyph cannot make the real digits next to it look small.
    """
    if mask.size == 0:
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, area in (stats[i] for i in range(1, count))
        if area >= MIN_GLYPH_AREA and (max_height is None or h <= max_height)
    ]
    if not boxes:
        return []
    tallest = max(h for _, _, _, h in boxes)
    return sorted(box for box in boxes if box[3] >= MIN_GLYPH_HEIGHT_RATIO * tallest)


def _glyph_image(glyph: np.ndarray, height: int) -> np.ndarray:
    scale = height / glyph.shape[0]
    resized = cv2.resize(
        glyph, (max(1, round(glyph.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC
    )
    resized = cv2.GaussianBlur(resized, (0, 0), SMOOTHING_SIGMA * height)
    return cv2.copyMakeBorder(255 - resized, *(GLYPH_PADDING,) * 4, cv2.BORDER_CONSTANT, value=255)


def _tsv_from_command_line(image: np.ndarray, tesseract: str, settings: _Pass) -> str:
    encoded = cv2.imencode(".png", image)[1].tobytes()
    command = [tesseract, "stdin", "stdout", "--psm", str(settings.page_segmentation_mode)]
    if settings.whitelist:
        command += ["-c", "tessedit_char_whitelist=0123456789"]
    result = subprocess.run(
        [*command, "tsv"], input=encoded, capture_output=True, check=False, env=_TESSERACT_ENV
    )
    if result.returncode != 0:
        raise OcrError(f"tesseract failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode(errors="replace")


def _recognise(image: np.ndarray, settings: _Pass) -> tuple[str, float] | None:
    with _engine(settings) as engine:
        if engine is not None:
            text = engine.tsv(image)
        else:
            text = _tsv_from_command_line(image, find_tesseract(), settings)
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 12:
            continue
        glyph = fields[11].strip()
        if len(glyph) != 1:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        digit = LOOKALIKES.get(glyph, glyph)
        if confidence >= 0 and digit.isdigit():
            return digit, confidence / 100
    return None


def _read_glyph(glyph: np.ndarray) -> tuple[str, float] | None:
    def run(settings: _Pass) -> tuple[str, float] | None:
        return _recognise(_glyph_image(glyph, settings.glyph_height), settings)

    votes: dict[str, float] = {}
    readings = map(run, PASSES) if _library() is not None else _PASS_POOL.map(run, PASSES)
    for reading in readings:
        if reading is not None:
            votes[reading[0]] = votes.get(reading[0], 0.0) + reading[1]
    if not votes:
        return None
    digit, total = max(votes.items(), key=lambda item: item[1])
    return digit, total / len(PASSES)


def read_digits(mask: np.ndarray) -> DigitReading | None:
    """The digits in `mask` (white glyphs on black), or None when nothing readable is there."""
    if mask.size == 0:
        return None
    boxes = glyph_boxes(mask)
    if not boxes:
        return None
    readings = [_read_glyph(mask[y : y + h, x : x + w]) for x, y, w, h in boxes]
    found = [reading for reading in readings if reading is not None]
    if len(found) != len(readings):
        return None
    return DigitReading(
        text="".join(text for text, _ in found),
        confidence=min(confidence for _, confidence in found),
    )
