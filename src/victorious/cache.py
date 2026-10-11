"""Keeping the Observations of a Recording so a rerun with the same Layout skips decoding."""

import hashlib
import json
import os
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from victorious.counters import CounterReading
from victorious.layout import Layout
from victorious.observations import (
    SecondObservation,
    from_json,
    reading_from_json,
    reading_to_json,
    to_json,
)
from victorious.schema import CounterElement

OBSERVATION_VERSION = 1
SELECTION_VERSION = 1
FINGERPRINT_SAMPLE = 1 << 20

Series = list[tuple[float, CounterReading | None]]


def default_cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "victorious"


def fingerprint(path: Path) -> str:
    """The size and the first, middle and last MiB of the file: Recordings are too big to hash."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        for offset in (
            0,
            max(0, size // 2 - FINGERPRINT_SAMPLE // 2),
            max(0, size - FINGERPRINT_SAMPLE),
        ):
            handle.seek(offset)
            digest.update(handle.read(FINGERPRINT_SAMPLE))
    return digest.hexdigest()


def layout_hash(layout: Layout) -> str:
    document = json.dumps(layout.model_dump(), sort_keys=True)
    return hashlib.sha256(document.encode()).hexdigest()


def _write_atomically(path: Path, text: str) -> None:
    """Best effort: a cache that cannot be written must not fail the analysis."""
    temporary = path.parent / f".{path.name}.{secrets.token_hex(4)}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(text)
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class CacheEntry:
    """What is kept for one Recording read through one Layout."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(self) -> list[SecondObservation] | None:
        """The cached Observations of every second, or None when missing or unreadable."""
        document = _read_json(self._directory / "observations.json")
        if not isinstance(document, list):
            return None
        try:
            observations = [from_json(raw) for raw in document]
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if [o.second for o in observations] != list(range(len(observations))):
            return None
        return observations

    def store(self, observations: list[SecondObservation]) -> None:
        text = json.dumps([to_json(observation) for observation in observations])
        _write_atomically(self._directory / "observations.json", text)

    def _series_path(self, element: CounterElement, start: float, end: float) -> Path:
        return self._directory / f"window-{element}-{start:g}-{end:g}.json"

    def load_series(self, element: CounterElement, start: float, end: float) -> Series | None:
        document = _read_json(self._series_path(element, start, end))
        if not isinstance(document, list):
            return None
        try:
            return [(float(time), reading_from_json(raw)) for time, raw in document]
        except (TypeError, ValueError, IndexError):
            return None

    def store_series(
        self, element: CounterElement, start: float, end: float, series: Series
    ) -> None:
        document = [[time, reading_to_json(reading)] for time, reading in series]
        _write_atomically(self._series_path(element, start, end), json.dumps(document))


class ObservationCache:
    def __init__(self, root: Path) -> None:
        self._root = root

    def entry(self, recording: Path, layout: Layout) -> CacheEntry:
        name = f"{fingerprint(recording)[:24]}-{layout_hash(layout)[:24]}-v{OBSERVATION_VERSION}"
        return CacheEntry(self._root / name)

    def selection(self, recording: Path, layout_hashes: Sequence[str]) -> "SelectionEntry":
        digest = hashlib.sha256("\n".join(sorted(set(layout_hashes))).encode()).hexdigest()
        name = f"selection-{fingerprint(recording)[:24]}-{digest[:24]}-v{SELECTION_VERSION}.json"
        return SelectionEntry(self._root / name, layout_hashes)


class SelectionEntry:
    """The score of each candidate Layout of one Recording, in the order of their hashes."""

    def __init__(self, file: Path, layout_hashes: Sequence[str]) -> None:
        self._file = file
        self._hashes = list(layout_hashes)

    def load(self) -> list[float] | None:
        document = _read_json(self._file)
        if not isinstance(document, dict):
            return None
        scores = [document.get(digest) for digest in self._hashes]
        if not all(isinstance(s, int | float) and not isinstance(s, bool) for s in scores):
            return None
        return [float(s) for s in scores if s is not None]

    def store(self, scores: Sequence[float]) -> None:
        document = dict(zip(self._hashes, scores, strict=True))
        _write_atomically(self._file, json.dumps(document, sort_keys=True))
