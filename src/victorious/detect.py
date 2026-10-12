"""Locating each HUD element in one frame, from reference icons and OCR anchors."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from victorious.counters import best_reading
from victorious.layout import EPSILON, Rect
from victorious.matcher import TemplateHit, locate_template_multiscale
from victorious.ocr import glyph_boxes, read_digits
from victorious.references import load_reference, reference_scale
from victorious.screens import (
    BANNER_NAMES,
    HUD_THRESHOLD,
    ICON_NAMES,
    RESULTS_THRESHOLD,
    counter_band_masks,
    gold_mask,
    white_mask,
)

SCALE_FACTORS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
COUNTER_FOR_ICON = {
    "icon_skull": "personal_eliminations",
    "icon_squad": "team_eliminations",
    "icon_person": "players_left",
}
RESULTS_REGION = (0.15, 0.0, 0.85, 0.45)
COLUMN_SLACK = 0.5
COUNTER_WINDOW_ICONS = 3
COUNTER_ROW_MARGIN = 0.2
COUNTER_GLYPH_HEIGHT = (0.3, 0.9)
COUNTER_GAP = 0.6
COUNTER_WIDTH_PER_HEIGHT = 4
HASH_ROW_MARGIN = 0.75
PLACEMENT_DIGIT_HEIGHT = (0.6, 2.0)
HASH_GAP = 0.15
DIGIT_ADVANCE = 1.0
MAX_PLACEMENT_DIGITS = 3
PLACEMENT_HEIGHT_PER_HASH = 2.25
PLACEMENT_RANGE = (1, 100)


@dataclass(frozen=True)
class Found:
    """An element's rectangle, how well it matched, and the matched size over the reference."""

    rect: Rect
    score: float
    scale_factor: float


@dataclass(frozen=True)
class Detection:
    found: dict[str, Found] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)


def _normalised(x0: float, y0: float, x1: float, y1: float, frame: np.ndarray) -> Rect | None:
    height, width = frame.shape[:2]
    left, top = max(0.0, x0) / width, max(0.0, y0) / height
    right, bottom = min(float(width), x1) / width, min(float(height), y1) / height
    x, y = round(left, 4), round(top, 4)
    w, h = round(right - x, 4), round(bottom - y, 4)
    if w <= 0 or h <= 0:
        return None
    w, h = min(w, 1 - x), min(h, 1 - y)
    if x + w > 1 + EPSILON or y + h > 1 + EPSILON or w <= 0 or h <= 0:
        return None
    return Rect(x=x, y=y, w=w, h=h)


def _scales(name: str, frame_height: int) -> list[float]:
    base = reference_scale(name, frame_height)
    return [base * factor for factor in SCALE_FACTORS]


def _locate(
    region: np.ndarray, name: str, color: bool, frame_height: int
) -> tuple[TemplateHit | None, float]:
    hit = locate_template_multiscale(
        region, load_reference(name, color), _scales(name, frame_height)
    )
    if hit is None:
        return None, 0.0
    return hit, float(np.nan_to_num(hit.score))


def _factor(hit: TemplateHit, name: str, frame_height: int) -> float:
    return hit.scale / reference_scale(name, frame_height)


@dataclass(frozen=True)
class _Icon:
    name: str
    hit: TemplateHit
    score: float


def _locate_icons(frame: np.ndarray) -> tuple[dict[str, _Icon], dict[str, str]]:
    height, width = frame.shape[:2]
    first: dict[str, tuple[TemplateHit | None, float]] = {
        name: _locate(frame, name, True, height) for name in ICON_NAMES
    }
    anchor = max(first, key=lambda name: first[name][1])
    anchor_hit = first[anchor][0]
    icons: dict[str, _Icon] = {}
    problems: dict[str, str] = {}
    for name in ICON_NAMES:
        hit, score = first[name]
        if name != anchor and anchor_hit is not None and first[anchor][1] >= HUD_THRESHOLD:
            left = max(0, int(anchor_hit.x - COLUMN_SLACK * anchor_hit.w))
            right = min(width, int(anchor_hit.x + (1 + COLUMN_SLACK) * anchor_hit.w) + 1)
            hit, score = _locate(frame[:, left:right], name, True, height)
            if hit is not None:
                hit = TemplateHit(hit.x + left, hit.y, hit.w, hit.h, hit.score, hit.scale)
        if hit is not None and score >= HUD_THRESHOLD:
            icons[name] = _Icon(name, hit, score)
        else:
            problems[name] = f"reference icon {name} not found (best match {score:.2f})"
    return icons, problems


def _rightmost_cluster(
    boxes: Sequence[tuple[int, int, int, int]], icon_height: int
) -> tuple[int, int, int, int] | None:
    """The right-most run of digit-sized glyph pieces, as a box; a cracked glyph stays whole.

    `boxes` come from `glyph_boxes` already capped at the tallest digit.
    """
    low = COUNTER_GLYPH_HEIGHT[0]
    tallest = max((box[3] for box in boxes), default=0)
    if tallest < low * icon_height:
        return None
    pieces = sorted(boxes, key=lambda box: box[0] + box[2], reverse=True)
    runs: list[tuple[int, int, int, int]] = []
    for x, y, w, h in pieces:
        if runs and runs[-1][0] - (x + w) <= COUNTER_GAP * tallest:
            left, top, right, bottom = runs[-1]
            runs[-1] = (min(left, x), min(top, y), max(right, x + w), max(bottom, y + h))
        else:
            runs.append((x, y, x + w, y + h))
    for left, top, right, bottom in runs:
        if bottom - top >= low * icon_height and bottom - top >= 0.6 * tallest:
            return left, top, right - left, bottom - top
    return None


def _counter(frame: np.ndarray, icon: _Icon) -> tuple[Rect | None, str]:
    hit = icon.hit
    height = frame.shape[0]
    right = max(1, hit.x - max(1, round(0.05 * hit.w)))
    left = max(0, right - COUNTER_WINDOW_ICONS * hit.w)
    margin = round(COUNTER_ROW_MARGIN * hit.h)
    top, bottom = max(0, hit.y - margin), min(height, hit.y + hit.h + margin)
    for mask in counter_band_masks(frame[top:bottom, left:right]):
        cluster = _rightmost_cluster(
            glyph_boxes(mask, max_height=int(COUNTER_GLYPH_HEIGHT[1] * hit.h)), hit.h
        )
        if cluster is None:
            continue
        x, y, w, h = cluster
        glyphs = mask[y : y + h, x : x + w]
        if best_reading([glyphs]) is None:
            continue
        x_right = left + x + w
        centre = top + y + h / 2
        rect = _normalised(
            x_right - COUNTER_WIDTH_PER_HEIGHT * h, centre - h / 2, x_right, centre + h / 2, frame
        )
        if rect is not None:
            return rect, ""
    return None, f"no readable digits left of {icon.name}"


def shows_hud(frame: np.ndarray) -> bool:
    """Whether all the reference HUD icons are on screen, without reading the counters."""
    return not _locate_icons(frame)[1]


def find_hud(frame: np.ndarray) -> Detection:
    """The HUD anchor and the three counters in one in-match frame."""
    height = frame.shape[0]
    icons, problems = _locate_icons(frame)
    found: dict[str, Found] = {}
    missing: dict[str, str] = {}
    if problems:
        missing["hud_anchor"] = "; ".join(problems.values())
    else:
        hits = [icon.hit for icon in icons.values()]
        rect = _normalised(
            min(h.x for h in hits),
            min(h.y for h in hits),
            max(h.x + h.w for h in hits),
            max(h.y + h.h for h in hits),
            frame,
        )
        score = min(icon.score for icon in icons.values())
        factor = float(np.mean([_factor(i.hit, i.name, height) for i in icons.values()]))
        if rect is None:
            missing["hud_anchor"] = "the reference icons sit outside the frame"
        else:
            found["hud_anchor"] = Found(rect, score, factor)
    for name in ICON_NAMES:
        element = COUNTER_FOR_ICON[name]
        icon = icons.get(name)
        if icon is None:
            missing[element] = problems[name]
            continue
        rect, detail = _counter(frame, icon)
        if rect is None:
            missing[element] = detail
        else:
            found[element] = Found(rect, icon.score, _factor(icon.hit, name, height))
    return Detection(found, missing)


def _placement_digits(
    mask: np.ndarray, hit: TemplateHit, banner_found: bool
) -> tuple[int, int, int] | None:
    """Digit count and vertical span of the placement right of the "#" at `hit`, if it is one.

    On a Win the banner already says the placement is 1, so a lone digit-sized glyph is enough
    where OCR cannot read the big gold "1".
    """
    advance = DIGIT_ADVANCE * hit.h
    left = hit.x + hit.w
    margin = round(HASH_ROW_MARGIN * hit.h)
    top, bottom = max(0, hit.y - margin), min(mask.shape[0], hit.y + hit.h + margin)
    window = mask[
        top:bottom, left : left + int(HASH_GAP * hit.h + (MAX_PLACEMENT_DIGITS + 1) * advance)
    ]
    low, high = PLACEMENT_DIGIT_HEIGHT
    boxes = [b for b in glyph_boxes(window, max_height=int(high * hit.h)) if low * hit.h <= b[3]]
    if not boxes:
        return None
    boxes.sort()
    x0, y0, x1, y1 = boxes[0][0], boxes[0][1], boxes[0][0] + boxes[0][2], boxes[0][1] + boxes[0][3]
    count = 1
    for x, y, w, h in boxes[1:]:
        if x - x1 > HASH_GAP * hit.h + 0.5 * advance:
            break
        x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h)
        count += 1
    reading = read_digits(window[y0:y1, x0:x1])
    if reading is None:
        return (1, top + y0, top + y1) if banner_found and count == 1 else None
    if not reading.text.isdigit():
        return None
    value = int(reading.text)
    if not PLACEMENT_RANGE[0] <= value <= PLACEMENT_RANGE[1] or len(reading.text) != count:
        return None
    return count, top + y0, top + y1


def _placement_rect(
    hit: TemplateHit,
    digits: tuple[int, int, int],
    region_origin: tuple[int, int],
    frame: np.ndarray,
) -> Rect | None:
    count, digit_top, digit_bottom = digits
    advance = DIGIT_ADVANCE * hit.h
    gap = HASH_GAP * hit.h
    ox, oy = region_origin
    left = ox + hit.x - (MAX_PLACEMENT_DIGITS - count) / 2 * advance
    right = ox + hit.x + hit.w + gap + MAX_PLACEMENT_DIGITS * advance
    top, bottom = min(hit.y, digit_top), max(hit.y + hit.h, digit_bottom)
    half = max(PLACEMENT_HEIGHT_PER_HASH * hit.h, 1.1 * (bottom - top)) / 2
    centre = oy + (top + bottom) / 2
    return _normalised(left, centre - half, right, centre + half, frame)


def find_results(frame: np.ndarray) -> Detection:
    """The results banner and the two placement styles in one results frame."""
    height, width = frame.shape[:2]
    x0, y0 = int(RESULTS_REGION[0] * width), int(RESULTS_REGION[1] * height)
    x1, y1 = int(RESULTS_REGION[2] * width), int(RESULTS_REGION[3] * height)
    region = np.asarray(frame[y0:y1, x0:x1])
    found: dict[str, Found] = {}
    missing: dict[str, str] = {}

    best: tuple[str, TemplateHit, float] | None = None
    for name in BANNER_NAMES:
        hit, score = _locate(region, name, True, height)
        if hit is not None and (best is None or score > best[2]):
            best = (name, hit, score)
    if best is not None and best[2] >= RESULTS_THRESHOLD:
        name, hit, score = best
        rect = _normalised(x0 + hit.x, y0 + hit.y, x0 + hit.x + hit.w, y0 + hit.y + hit.h, frame)
        if rect is not None:
            found["results_banner"] = Found(rect, score, _factor(hit, name, height))
    if "results_banner" not in found:
        score = best[2] if best is not None else 0.0
        missing["results_banner"] = f"no Victory Royale banner (best match {score:.2f})"

    styles = (
        ("placement", "hash", gold_mask, "gold #N"),
        ("elimination_placement", "hash_white", white_mask, "white YOU PLACED #N"),
    )
    for element, template, mask_of, label in styles:
        mask = mask_of(region)
        hit, score = _locate(mask, template, False, height)
        rect = None
        if hit is not None and score >= RESULTS_THRESHOLD:
            digits = _placement_digits(mask, hit, "results_banner" in found and template == "hash")
            if digits is not None:
                rect = _placement_rect(hit, digits, (x0, y0), frame)
        if hit is not None and rect is not None:
            found[element] = Found(rect, score, _factor(hit, template, height))
        else:
            missing[element] = f"no {label} (best match {score:.2f})"
    return Detection(found, missing)
