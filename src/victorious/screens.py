"""Pure per-frame detection of the in-match HUD and the results screen."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from victorious.layout import Layout, Rect
from victorious.matcher import match_template_in_frame_multiscale
from victorious.ocr import DigitReading, read_digits
from victorious.references import load_reference, reference_scale

HUD_THRESHOLD = 0.7
RESULTS_THRESHOLD = 0.7
ICON_NAMES = ("icon_skull", "icon_squad", "icon_person")
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


PlacementElement = Literal["placement", "elimination_placement"]


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


def _crop(frame: np.ndarray, rect: Rect, margin: float) -> np.ndarray:
    height, width = frame.shape[:2]
    x0 = max(0, int((rect.x - margin * rect.w) * width))
    y0 = max(0, int((rect.y - margin * rect.h) * height))
    x1 = min(width, int((rect.x + rect.w * (1 + margin)) * width) + 1)
    y1 = min(height, int((rect.y + rect.h * (1 + margin)) * height) + 1)
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


def _banner_score(frame: np.ndarray, rect: Rect) -> float:
    """The banner at its settled size next to the Layout rect, else large and low on the frame.

    The wide search only runs when the narrow one misses, and on a shrunken copy, since most
    seconds show no banner at all.
    """
    height = frame.shape[0]
    region = _crop(frame, rect, RESULTS_MARGIN)
    settled = max(_score(region, name, True, height)[0] for name in BANNER_NAMES)
    if settled >= RESULTS_THRESHOLD:
        return settled
    wide = _crop(frame, rect, BANNER_WIDE_MARGIN)
    small = cv2.resize(
        wide, None, fx=BANNER_WIDE_SHRINK, fy=BANNER_WIDE_SHRINK, interpolation=cv2.INTER_AREA
    )
    large = max(
        _score(small, name, True, height, BANNER_WIDE_SWEEP, BANNER_WIDE_SHRINK)[0]
        for name in BANNER_NAMES
    )
    return max(settled, large)


def hud_score(frame: np.ndarray, layout: Layout) -> float:
    """How surely the in-match HUD is on screen: its weakest icon's match, 0..1."""
    region = _crop(frame, layout.screens.hud.hud_anchor, ANCHOR_MARGIN)
    height = frame.shape[0]
    return min(_score(region, name, True, height)[0] for name in ICON_NAMES)


def _band_masks(
    region: np.ndarray, bands: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]]
) -> list[np.ndarray]:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return [np.asarray(cv2.inRange(hsv, low, high)) for low, high in bands]


def gold_mask(region: np.ndarray) -> np.ndarray:
    return _band_masks(region, [(GOLD_LOW, GOLD_HIGH)])[0]


def white_mask(region: np.ndarray) -> np.ndarray:
    return _band_masks(region, [(WHITE_LOW, WHITE_HIGH)])[0]


def counter_band_masks(region: np.ndarray) -> list[np.ndarray]:
    """White digits in `region` as binary masks, tightest white band first.

    The tight band keeps digits apart from pale sky and cloud; the loose one keeps thin strokes.
    """
    return _band_masks(region, COUNTER_BANDS)


def counter_masks(frame: np.ndarray, rect: Rect) -> list[np.ndarray]:
    """One HUD counter's white digits as binary masks, tightest white band first."""
    return counter_band_masks(_crop(frame, rect, COUNTER_MARGIN))


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
