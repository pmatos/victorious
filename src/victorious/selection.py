"""Choosing which Layout fits a Recording by scoring each HUD anchor on in-match frames."""

import math
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from victorious.cache import ObservationCache, layout_hash
from victorious.detect import shows_hud
from victorious.layout import LayoutError, LoadedLayout
from victorious.screens import HUD_THRESHOLD, hud_score
from victorious.video import VideoInfo, frame_at

SCORED_FRAMES = 20
MAX_CANDIDATE_FRAMES = 240
MIN_LAYOUT_SCORE = HUD_THRESHOLD
NOT_FORTNITE_WINDOW = 600
OPENING_STRIDE = 10
ICON_SCAN_FRAMES = 12


class LayoutSelectionError(LayoutError):
    """No Layout can be used for the Recording."""


@dataclass(frozen=True)
class Selection:
    chosen: LoadedLayout
    score: float
    fitting: int


@dataclass(frozen=True)
class NoHud:
    """No in-match HUD of any kind in the opening window: the Recording is not Fortnite."""

    tried: list[LoadedLayout]
    frames: int
    aspect_fits: bool = True


def _scores_at(
    path: Path, info: VideoInfo, second: int, layouts: Sequence[LoadedLayout]
) -> list[float] | None:
    frame = frame_at(path, info, second)
    if frame is None:
        return None
    return [hud_score(frame, loaded.layout) for loaded in layouts]


def _candidate_seconds(info: VideoInfo) -> tuple[list[int], list[int]]:
    """The seconds to sample in the opening window and in the rest of the Recording.

    The opening window is sampled at least every OPENING_STRIDE seconds however long the
    Recording is, the rest at the stride that keeps the total near MAX_CANDIDATE_FRAMES.
    """
    whole = math.floor(info.duration)
    stride = max(1, math.ceil(whole / MAX_CANDIDATE_FRAMES))
    window = min(max(whole, 1), NOT_FORTNITE_WINDOW)
    opening = list(range(0, window, min(stride, OPENING_STRIDE)))
    rest = list(range(NOT_FORTNITE_WINDOW, whole, stride))
    return opening, rest


def _groups(seconds: Sequence[int], jobs: int) -> Iterator[Sequence[int]]:
    size = max(1, jobs)
    for offset in range(0, len(seconds), size):
        yield seconds[offset : offset + size]


def _in_match_scores(
    path: Path,
    info: VideoInfo,
    layouts: Sequence[LoadedLayout],
    opening: Sequence[int],
    rest: Sequence[int],
    jobs: int,
) -> tuple[list[list[float]], int]:
    """The scores of the first in-match frames in time order, and how many frames were tried.

    The rest of the Recording is only scanned when a frame in the opening window showed a HUD.
    """
    found: list[list[float]] = []
    tried = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for segment in (opening, rest):
            for group in _groups(segment, jobs):
                for scores in pool.map(lambda s: _scores_at(path, info, s, layouts), group):
                    tried += 1
                    if scores is not None and max(scores) >= HUD_THRESHOLD:
                        found.append(scores)
                        if len(found) >= SCORED_FRAMES:
                            return found, tried
            if not found:
                break
    return found, tried


def _shows_hud(path: Path, info: VideoInfo, second: int) -> bool:
    frame = frame_at(path, info, second)
    return frame is not None and shows_hud(frame)


def _icon_scan_seconds(seconds: Sequence[int]) -> Sequence[int]:
    return seconds[:: max(1, math.ceil(len(seconds) / ICON_SCAN_FRAMES))]


def _any_hud(path: Path, info: VideoInfo, spread: Sequence[int], jobs: int) -> bool:
    """Whether the reference HUD icons appear in any of the spread frames, whatever the Layout.

    Locating the icons costs seconds a frame, and a HUD that is up at all is up for minutes,
    so a sparse look is enough.
    """
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return any(
            any(pool.map(lambda s: _shows_hud(path, info, s), group))
            for group in _groups(spread, jobs)
        )


def _best(layouts: Sequence[LoadedLayout], scores: Sequence[float]) -> tuple[LoadedLayout, float]:
    return max(zip(layouts, scores, strict=True), key=lambda pair: pair[1])


def _no_fit(path: Path, info: VideoInfo, layouts: Sequence[LoadedLayout]) -> LayoutSelectionError:
    ratios = ", ".join(f"{loaded.path} is {loaded.layout.aspect_ratio}" for loaded in layouts)
    return LayoutSelectionError(
        f"no Layout fits this {info.width}x{info.height} Recording ({ratios}); "
        f"run `victorious setup {path}` to produce a Layout for its aspect ratio"
    )


def select_layout(
    path: Path,
    info: VideoInfo,
    layouts: Sequence[LoadedLayout],
    *,
    jobs: int,
    cache: ObservationCache | None,
) -> Selection | NoHud:
    fitting = sorted(
        (loaded for loaded in layouts if loaded.layout.fits(info.width, info.height)),
        key=lambda loaded: str(loaded.path),
    )
    opening, rest = _candidate_seconds(info)
    if not fitting:
        spread = _icon_scan_seconds(opening)
        if not _any_hud(path, info, spread, jobs):
            return NoHud(list(layouts), len(spread), aspect_fits=False)
        raise _no_fit(path, info, layouts)

    hashes = [layout_hash(loaded.layout) for loaded in fitting]
    entry = cache.selection(path, hashes) if cache is not None else None
    stored = entry.load() if entry is not None else None
    if stored is not None:
        best, score = _best(fitting, stored)
        return Selection(best, score, len(fitting))

    frames, tried = _in_match_scores(path, info, fitting, opening, rest, jobs)
    if not frames:
        if not _any_hud(path, info, _icon_scan_seconds(opening), jobs):
            return NoHud(fitting, tried)
        raise LayoutSelectionError(
            f"no Layout matches this Recording's HUD: none of {tried} sampled frames shows "
            f"a HUD at the anchor of {', '.join(str(loaded.path) for loaded in fitting)}; "
            f"run `victorious setup {path}` to produce a Layout for this HUD"
        )
    means = [min(1.0, max(0.0, float(np.mean(column)))) for column in zip(*frames, strict=True)]
    best, score = _best(fitting, means)
    if score < MIN_LAYOUT_SCORE:
        raise LayoutSelectionError(
            f"no Layout matches this Recording's HUD: the best score is {score:.2f} "
            f"for {best.path}, below the minimum {MIN_LAYOUT_SCORE:.2f} "
            f"({len(frames)} in-match frames scored)"
        )
    if entry is not None:
        entry.store(means)
    return Selection(best, score, len(fitting))
