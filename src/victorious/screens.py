"""Pure per-frame detection of the in-match HUD and the results screen."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

import cv2
import numpy as np

from victorious.layout import HudDesign, Layout, Rect
from victorious.matcher import (
    TemplateHit,
    locate_template_multiscale,
    match_template_in_frame_multiscale,
)
from victorious.ocr import DigitReading, glyph_boxes, read_digits
from victorious.references import load_reference, reference_scale

HUD_THRESHOLD = 0.7
RESULTS_THRESHOLD = 0.7
SPECTATING_THRESHOLD = 0.7
SPECTATING_BAND = Rect(x=0.4, y=0.0, w=0.2, h=0.09)
ICON_NAMES = ("icon_skull", "icon_squad", "icon_person")
ROW_ICON_NAMES = ("row_person", "row_crosshair", "row_squad")
BANNER_NAMES = ("victory_royale", "victory_royale_orange")
SCALE_SWEEP = (0.9, 1.0, 1.1)
BANNER_WIDE_SWEEP = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
BANNER_WIDE_MARGIN = 1.0
BANNER_WIDE_SHRINK = 0.5
ANCHOR_MARGIN = 0.1
RESULTS_MARGIN = 0.2
GOLD_LOW = (15, 90, 200)
GOLD_HIGH = (35, 255, 255)
WHITE_LOW = (0, 0, 230)
WHITE_HIGH = (180, 40, 255)
DIGIT_BAND = 0.25
COUNTER_MARGIN = 0.1
COUNTER_BANDS = (
    ((0, 0, 235), (180, 12, 255)),
    ((0, 0, 225), (180, 16, 255)),
)
ROW_COUNTER_BANDS = (
    ((0, 0, 225), (180, 16, 255)),
    ((0, 0, 205), (180, 30, 255)),
)
ROW_ANCHOR_MARGIN = (0.5, 0.25)
COUNTER_GLYPH_HEIGHT = (0.3, 0.9)
COUNTER_GAP = 0.6


PlacementElement = Literal["placement", "elimination_placement"]
Band = tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class DesignSpec:
    """What tells one HUD design from another: its icons, from first to last on screen, which
    counter sits beside each, and the white bands that keep its digits."""

    icons: tuple[str, ...]
    counter_icons: Mapping[str, str]
    bands: tuple[Band, ...]


DESIGNS: dict[HudDesign, DesignSpec] = {
    "icon_column": DesignSpec(
        ICON_NAMES,
        {
            "icon_skull": "personal_eliminations",
            "icon_squad": "team_eliminations",
            "icon_person": "players_left",
        },
        COUNTER_BANDS,
    ),
    "counter_row": DesignSpec(
        ROW_ICON_NAMES,
        {
            "row_crosshair": "personal_eliminations",
            "row_squad": "team_eliminations",
            "row_person": "players_left",
        },
        ROW_COUNTER_BANDS,
    ),
}


@dataclass(frozen=True)
class ResultsReading:
    """What the results screen shows: `hash` is the best "#" score of the two styles."""

    banner: float
    hash: float
    digits: DigitReading | None
    placement_score: float = 0.0
    elimination_placement_score: float = 0.0
    digits_element: PlacementElement | None = None

    @property
    def is_results(self) -> bool:
        return self.banner >= RESULTS_THRESHOLD or self.hash >= RESULTS_THRESHOLD


def _box(
    frame: np.ndarray, rect: Rect, margin: float, margin_y: float | None = None
) -> tuple[int, int, int, int]:
    """`rect` grown by a share of its size on each side, as pixels (x0, y0, x1, y1)."""
    height, width = frame.shape[:2]
    margin_y = margin if margin_y is None else margin_y
    x0 = max(0, int((rect.x - margin * rect.w) * width))
    y0 = max(0, int((rect.y - margin_y * rect.h) * height))
    x1 = min(width, int((rect.x + rect.w * (1 + margin)) * width) + 1)
    y1 = min(height, int((rect.y + rect.h * (1 + margin_y)) * height) + 1)
    return x0, y0, x1, y1


def _crop(frame: np.ndarray, rect: Rect, margin: float) -> np.ndarray:
    x0, y0, x1, y1 = _box(frame, rect, margin)
    return np.asarray(frame[y0:y1, x0:x1])


def _score(
    region: np.ndarray,
    name: str,
    color: bool,
    frame_height: int,
    sweep: Sequence[float] = SCALE_SWEEP,
    shrink: float = 1.0,
) -> tuple[float, float]:
    base = reference_scale(name, frame_height) * shrink
    template = load_reference(name, color)
    _, confidence, scale = match_template_in_frame_multiscale(
        region, template, 0.0, [base * factor for factor in sweep]
    )
    return float(np.nan_to_num(confidence)), scale


def scale_range(element: str) -> tuple[float, float]:
    """The smallest and largest size, as a factor of the reference, `analyse` reads."""
    factors = (*SCALE_SWEEP, *BANNER_WIDE_SWEEP) if element == "results_banner" else SCALE_SWEEP
    return min(factors), max(factors)


def _best_banner(
    region: np.ndarray,
    height: int,
    sweep: Sequence[float] = SCALE_SWEEP,
    shrink: float = 1.0,
) -> float:
    return max(_score(region, name, True, height, sweep, shrink)[0] for name in BANNER_NAMES)


def _banner_score(frame: np.ndarray, rect: Rect) -> float:
    """The banner at its settled size next to the Layout rect, else large and low on the frame.

    The wide search only runs when the narrow one misses, and on a shrunken copy, since most
    seconds show no banner at all.
    """
    height = frame.shape[0]
    settled = _best_banner(_crop(frame, rect, RESULTS_MARGIN), height)
    if settled >= RESULTS_THRESHOLD:
        return settled
    small = cv2.resize(
        _crop(frame, rect, BANNER_WIDE_MARGIN),
        None,
        fx=BANNER_WIDE_SHRINK,
        fy=BANNER_WIDE_SHRINK,
        interpolation=cv2.INTER_AREA,
    )
    return max(settled, _best_banner(small, height, BANNER_WIDE_SWEEP, BANNER_WIDE_SHRINK))


def _row_hits(frame: np.ndarray, layout: Layout) -> dict[str, TemplateHit]:
    """Where each icon of the counter row sits, in frame pixels, searched around the anchor.

    The icons move sideways with the width of the numbers, so the search is wide.
    """
    height = frame.shape[0]
    x0, y0, x1, y1 = _box(frame, layout.screens.hud.hud_anchor, *ROW_ANCHOR_MARGIN)
    region = np.asarray(frame[y0:y1, x0:x1])
    hits: dict[str, TemplateHit] = {}
    for name in ROW_ICON_NAMES:
        base = reference_scale(name, height)
        hit = locate_template_multiscale(
            region, load_reference(name, True), [base * factor for factor in SCALE_SWEEP]
        )
        if hit is not None:
            score = float(np.nan_to_num(hit.score))
            hits[name] = TemplateHit(hit.x + x0, hit.y + y0, hit.w, hit.h, score, hit.scale)
    return hits


def in_row_order(hits: Sequence[TemplateHit]) -> bool:
    """Whether `hits`, in the order the row draws its icons, are three separate icons."""
    return all(left.x + left.w <= right.x for left, right in pairwise(hits))


def leading_glyph_run(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """The first run of glyph boxes from the left, as (left, top, right, bottom).

    A gap wider than `COUNTER_GAP` of the tallest glyph ends the run: what follows is the next
    icon or scenery, not part of the counter.
    """
    ordered = sorted(boxes)
    tallest = max(h for _, _, _, h in ordered)
    left, top = ordered[0][0], ordered[0][1]
    right, bottom = left + ordered[0][2], top + ordered[0][3]
    for x, y, w, h in ordered[1:]:
        if x - right > COUNTER_GAP * tallest:
            break
        top, bottom, right = min(top, y), max(bottom, y + h), max(right, x + w)
    return left, top, right, bottom


def _leading_run(mask: np.ndarray, icon_height: int) -> np.ndarray:
    """`mask` cut to the digits next to its left edge, so the cut is the same wherever the row
    sits and nothing after the digits is read."""
    boxes = glyph_boxes(mask, max_height=int(COUNTER_GLYPH_HEIGHT[1] * icon_height))
    if not boxes:
        return mask
    left, _, right, _ = leading_glyph_run(boxes)
    return np.asarray(mask[:, left:right])


def counter_icons(frame: np.ndarray, layout: Layout) -> Mapping[str, TemplateHit]:
    """The icons the counters of the Layout's design are read beside, found in this frame.

    The icon column has none to find: its counters sit at the Layout's rectangles.
    """
    return _row_hits(frame, layout) if layout.hud_design == "counter_row" else {}


def counter_masks(
    frame: np.ndarray, layout: Layout, element: str, icons: Mapping[str, TemplateHit]
) -> list[np.ndarray]:
    """One counter's white digits as binary masks, tightest white band first.

    In the counter row the counter is read between its icon and the next as found in this frame
    (`icons`, from `counter_icons`), and over the Layout rectangle's width when it is the last.
    """
    rect: Rect = getattr(layout.screens.hud, element)
    if layout.hud_design == "icon_column":
        return counter_band_masks(_crop(frame, rect, COUNTER_MARGIN))
    design = DESIGNS[layout.hud_design]
    height, width = frame.shape[:2]
    icon = next((name for name, counter in design.counter_icons.items() if counter == element), "")
    hit = icons.get(icon)
    region = _crop(frame, rect, COUNTER_MARGIN)
    icon_height = max(1, round(rect.h * height))
    if hit is not None:
        left = hit.x + hit.w
        following = [other.x for other in icons.values() if other.x > hit.x]
        right = min(following) if following else left + round(rect.w * width)
        top, bottom = max(0, int(rect.y * height)), min(height, int((rect.y + rect.h) * height) + 1)
        beside = np.asarray(frame[top:bottom, left:right])
        if beside.size:
            region, icon_height = beside, hit.h
    return [_leading_run(mask, icon_height) for mask in band_masks(region, design.bands)]


@dataclass(frozen=True)
class HudReading:
    """The HUD at one frame: how surely it is on screen, and the icons found beside the counters."""

    score: float
    icons: Mapping[str, TemplateHit] = field(default_factory=dict)


def _column_score(frame: np.ndarray, layout: Layout) -> float:
    region = _crop(frame, layout.screens.hud.hud_anchor, ANCHOR_MARGIN)
    height = frame.shape[0]
    return min(_score(region, name, True, height)[0] for name in ICON_NAMES)


def hud_reading(frame: np.ndarray, layout: Layout) -> HudReading:
    """How surely the Layout's HUD is on screen: its weakest icon's match, 0..1."""
    if layout.hud_design == "icon_column":
        return HudReading(_column_score(frame, layout))
    hits = _row_hits(frame, layout)
    ordered = [hits[name] for name in ROW_ICON_NAMES if name in hits]
    shown = len(ordered) == len(ROW_ICON_NAMES) and in_row_order(ordered)
    return HudReading(min(hit.score for hit in ordered) if shown else 0.0, hits)


def hud_score(frame: np.ndarray, layout: Layout) -> float:
    """How surely the in-match HUD is on screen: its weakest icon's match, 0..1."""
    return hud_reading(frame, layout).score


def spectating_score(frame: np.ndarray) -> float:
    """How surely the white "SPECTATING" label of an eliminated player is on screen, 0..1.

    The label sits at the top centre whatever the HUD, so it is searched in a fixed band.
    """
    region = white_mask(_crop(frame, SPECTATING_BAND, 0.0))
    return _score(region, "spectating", False, frame.shape[0])[0]


def band_masks(region: np.ndarray, bands: Sequence[Band]) -> list[np.ndarray]:
    """`region`'s white pixels in each band, as binary masks."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return [np.asarray(cv2.inRange(hsv, low, high)) for low, high in bands]


def gold_mask(region: np.ndarray) -> np.ndarray:
    return band_masks(region, [(GOLD_LOW, GOLD_HIGH)])[0]


def white_mask(region: np.ndarray) -> np.ndarray:
    return band_masks(region, [(WHITE_LOW, WHITE_HIGH)])[0]


def counter_band_masks(region: np.ndarray) -> list[np.ndarray]:
    """White digits in `region` as binary masks, tightest white band first.

    The tight band keeps digits apart from pale sky and cloud; the loose one keeps thin strokes.
    """
    return band_masks(region, COUNTER_BANDS)


def _digits_right_of_hash(
    mask: np.ndarray, template: np.ndarray, scale: float
) -> DigitReading | None:
    size = (max(1, int(template.shape[1] * scale)), max(1, int(template.shape[0] * scale)))
    if size[0] > mask.shape[1] or size[1] > mask.shape[0]:
        return None
    resized = cv2.resize(template, size, interpolation=cv2.INTER_AREA)
    scores = cv2.matchTemplate(mask, resized, cv2.TM_CCOEFF_NORMED)
    _, _, _, (x, y) = cv2.minMaxLoc(scores)
    band = round(DIGIT_BAND * size[1])
    return read_digits(mask[max(0, y - band) : y + size[1] + band, x + size[0] :])


@dataclass(frozen=True)
class _HashStyle:
    element: PlacementElement
    template: str
    rect: Callable[[Layout], Rect | None]
    mask: Callable[[np.ndarray], np.ndarray]


HASH_STYLES = (
    _HashStyle("placement", "hash", lambda layout: layout.screens.results.placement, gold_mask),
    _HashStyle(
        "elimination_placement",
        "hash_white",
        lambda layout: layout.screens.results.elimination_placement,
        white_mask,
    ),
)


def results_reading(frame: np.ndarray, layout: Layout, read_placement: bool) -> ResultsReading:
    """The Victory Royale banner and "#" scores, and the placement digits when asked for."""
    height = frame.shape[0]
    banner = _banner_score(frame, layout.screens.results.results_banner)
    best: tuple[float, np.ndarray, float, _HashStyle] | None = None
    scores: dict[str, float] = {}
    for style in HASH_STYLES:
        rect = style.rect(layout)
        if rect is None:
            continue
        mask = style.mask(_crop(frame, rect, RESULTS_MARGIN))
        score, scale = _score(mask, style.template, False, height)
        scores[style.element] = score
        if best is None or score > best[0]:
            best = (score, mask, scale, style)
    placement_score = scores.get("placement", 0.0)
    elimination_score = scores.get("elimination_placement", 0.0)
    if best is None:
        return ResultsReading(banner=banner, hash=0.0, digits=None)
    hash_score, mask, scale, style = best
    wanted = read_placement and banner < RESULTS_THRESHOLD and hash_score >= RESULTS_THRESHOLD
    digits = (
        _digits_right_of_hash(mask, load_reference(style.template, False), scale)
        if wanted
        else None
    )
    return ResultsReading(
        banner=banner,
        hash=hash_score,
        digits=digits,
        placement_score=placement_score,
        elimination_placement_score=elimination_score,
        digits_element=style.element if digits is not None else None,
    )
