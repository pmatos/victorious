"""Draw one scene as a BGR frame, composited from committed crops of the real HUD."""

from functools import cache
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic.spec import Hud, Lobby, Results, Scene
from victorious.references import load_reference

CROPS = Path(__file__).parent.parent / "fixtures" / "crops"

NATIVE_WIDTH = 2560
NATIVE_HEIGHT = 1440
BANNER_CROP_SCALE = 4.0
LARGE_BANNER_ZOOM = 1.45
LARGE_BANNER_OFFSET = (-2, 82)
DIGIT_GAP_NATIVE = 3.0
DIGIT_HEIGHT_NATIVE = 17
ICON_NATIVE_SIZE = 36
ICON_CENTRE_X = 2199
GOLD = (60, 190, 250)
WHITE = (255, 255, 255)

# Pixel rects (x, y, w, h) measured on a native 2560x1440 Recording.
NATIVE_RECTS: dict[str, tuple[int, int, int, int]] = {
    "hud_anchor": (2181, 43, 36, 126),
    "personal_eliminations": (2104, 53, 68, DIGIT_HEIGHT_NATIVE),
    "team_eliminations": (2104, 94, 68, DIGIT_HEIGHT_NATIVE),
    "players_left": (2104, 143, 68, DIGIT_HEIGHT_NATIVE),
    "results_banner": (1105, 110, 380, 190),
    "placement": (918, 150, 280, 140),
    "elimination_placement": (1290, 150, 320, 160),
}
ELEMENT_RECTS: dict[str, tuple[float, float, float, float]] = {
    name: (x / NATIVE_WIDTH, y / NATIVE_HEIGHT, w / NATIVE_WIDTH, h / NATIVE_HEIGHT)
    for name, (x, y, w, h) in NATIVE_RECTS.items()
}

ICONS = {
    "icon_skull": "personal_eliminations",
    "icon_squad": "team_eliminations",
    "icon_person": "players_left",
}


@cache
def load_crop(name: str, color: bool = False) -> np.ndarray:
    if not name.startswith("digit_"):
        return load_reference(name, color)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(CROPS / f"{name}.png"), flag)
    if image is None:
        raise FileNotFoundError(f"missing digit crop {name}.png in {CROPS}")
    return image


@cache
def _background(
    width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int], seed: int
) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    gradient = np.array(top, np.float32) * (1 - ramp) + np.array(bottom, np.float32) * ramp
    noise = np.random.default_rng(seed).normal(0.0, 2.5, (height, width, 1)).astype(np.float32)
    return np.clip(gradient + noise, 0, 255).astype(np.uint8)


def _overlap(
    frame: np.ndarray, tile: np.ndarray, x: int, y: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    height, width = frame.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + tile.shape[1], width), min(y + tile.shape[0], height)
    if x0 >= x1 or y0 >= y1:
        return None
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))


def _blend(
    frame: np.ndarray, alpha: np.ndarray, x: int, y: int, color: tuple[int, int, int]
) -> None:
    overlap = _overlap(frame, alpha, x, y)
    if overlap is None:
        return
    dst, src = overlap
    a = alpha[src][..., None].astype(np.float32) / 255.0
    target = frame[dst].astype(np.float32)
    frame[dst] = (target * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def _paste(frame: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    overlap = _overlap(frame, tile, x, y)
    if overlap is not None:
        dst, src = overlap
        frame[dst] = tile[src]


def _pixel_rect(name: str, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = ELEMENT_RECTS[name]
    return round(x * width), round(y * height), round(w * width), round(h * height)


def _scaled(glyph: np.ndarray, height_px: int) -> np.ndarray:
    ratio = height_px / glyph.shape[0]
    width_px = max(1, round(glyph.shape[1] * ratio))
    return cv2.resize(glyph, (width_px, height_px), interpolation=cv2.INTER_AREA)


def _draw_text(
    frame: np.ndarray,
    glyphs: list[np.ndarray],
    rect: tuple[int, int, int, int],
    *,
    height: int,
    align: str,
    color: tuple[int, int, int],
) -> None:
    x, y, w, h = rect
    while True:
        height = max(1, height)
        gap = max(1, round(DIGIT_GAP_NATIVE * height / DIGIT_HEIGHT_NATIVE))
        scaled = [_scaled(g, height) for g in glyphs]
        total = sum(g.shape[1] for g in scaled) + gap * (len(scaled) - 1)
        if total <= w or height == 1:
            break
        height = min(height - 1, int(height * w / total))
    cursor = x + w - total if align == "right" else x
    top = y + (h - height) // 2
    for glyph in scaled:
        _blend(frame, glyph, cursor, top, color)
        cursor += glyph.shape[1] + gap


def _digits(value: int) -> list[np.ndarray]:
    return [load_crop(f"digit_{c}") for c in str(value)]


def render_scene(scene: Scene, width: int, height: int) -> np.ndarray:
    in_match = isinstance(scene, Hud | Results)
    frame = (
        _background(width, height, (70, 30, 5), (110, 55, 15), seed=2)
        if in_match
        else _background(width, height, (180, 140, 60), (120, 60, 120), seed=1)
    ).copy()
    if isinstance(scene, Hud):
        _draw_hud(frame, scene)
    elif isinstance(scene, Results):
        _draw_results(frame, scene)
    elif not isinstance(scene, Lobby):
        raise TypeError(f"unknown scene {scene!r}")
    return frame


def _draw_hud(frame: np.ndarray, hud: Hud) -> None:
    height, width = frame.shape[:2]
    icon_px = max(1, round(ICON_NATIVE_SIZE * height / NATIVE_HEIGHT))
    centre_x = round(ICON_CENTRE_X / NATIVE_WIDTH * width)
    for icon, counter in ICONS.items():
        rect = _pixel_rect(counter, width, height)
        cy = rect[1] + rect[3] // 2
        tile = cv2.resize(load_crop(icon, True), (icon_px, icon_px), interpolation=cv2.INTER_AREA)
        _paste(frame, tile, centre_x - icon_px // 2, cy - icon_px // 2)
        value = getattr(hud, counter)
        if hud.corrupt:
            x, y, w, h = rect
            noise = np.random.default_rng(y).integers(0, 256, (h, w, 3), dtype=np.uint8)
            frame[y : y + h, x : x + w] = noise
        elif value is not None:
            _draw_text(frame, _digits(value), rect, height=rect[3], align="right", color=WHITE)


def _draw_large_banner(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    banner = load_crop("victory_royale_orange", True)
    scale = height / NATIVE_HEIGHT * BANNER_CROP_SCALE * LARGE_BANNER_ZOOM
    size = (round(banner.shape[1] * scale), round(banner.shape[0] * scale))
    left, top = NATIVE_RECTS["results_banner"][:2]
    x = round((left + LARGE_BANNER_OFFSET[0]) / NATIVE_WIDTH * width)
    y = round((top + LARGE_BANNER_OFFSET[1]) / NATIVE_HEIGHT * height)
    _paste(frame, cv2.resize(banner, size, interpolation=cv2.INTER_LINEAR), x, y)


def _draw_results(frame: np.ndarray, results: Results) -> None:
    height, width = frame.shape[:2]
    if results.style == "orange_large":
        _draw_large_banner(frame)
        return
    if results.style == "placed":
        hash_mask, rect_name, color = load_crop("hash_white"), "elimination_placement", WHITE
    else:
        hash_mask, rect_name, color = load_crop("hash"), "placement", GOLD
        if results.placement == 1:
            banner_name = "victory_royale_orange" if results.style == "orange" else "victory_royale"
            banner = load_crop(banner_name, True)
            x, y, _, _ = _pixel_rect("results_banner", width, height)
            scale = height / NATIVE_HEIGHT * BANNER_CROP_SCALE
            size = (round(banner.shape[1] * scale), round(banner.shape[0] * scale))
            _paste(frame, cv2.resize(banner, size, interpolation=cv2.INTER_LINEAR), x, y)
    glyphs = [hash_mask, *_digits(results.placement)]
    rect = _pixel_rect(rect_name, width, height)
    glyph_height = round(hash_mask.shape[0] * height / NATIVE_HEIGHT)
    _draw_text(frame, glyphs, rect, height=glyph_height, align="left", color=color)
