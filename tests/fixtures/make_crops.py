"""Re-cut the committed HUD crops from the author's own Recordings.

Run by hand from the repository root when a crop needs refreshing; CI never
runs it. Sources are local and gitignored; point the script at them with
VICTORIOUS_HUD_RECORDING (a native 2560x1440 Recording showing the in-match HUD)
and VICTORIOUS_WIN_RECORDING (one showing the Victory Royale results screen):

    uv run python -m tests.fixtures.make_crops
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic.encode import decode_frame
from tests.synthetic.render import (
    BANNER_CROP_SCALE,
    DIGIT_HEIGHT_NATIVE,
    ICON_CENTRE_X,
    ICON_NATIVE_SIZE,
    ICONS,
    NATIVE_HEIGHT,
    NATIVE_RECTS,
    NATIVE_WIDTH,
)
from victorious.screens import WHITE_HIGH, WHITE_LOW

HERE = Path(__file__).parent
CROPS = HERE / "crops"
REPO = HERE.parent.parent
REFERENCES = REPO / "src" / "victorious" / "data"

HUD_RECORDING = Path(
    os.environ.get("VICTORIOUS_HUD_RECORDING", "/mnt/windows/Videos/2026-10-09_11-59-53.mkv")
)
WIN_RECORDING = Path(
    os.environ.get(
        "VICTORIOUS_WIN_RECORDING", str(REPO / "assets" / "videos" / "test-video-last-minute.mkv")
    )
)

SPECTATING_RECORDING = Path(
    os.environ.get(
        "VICTORIOUS_SPECTATING_RECORDING",
        str(REPO / "assets" / "recordings" / "2026-10-08_11-11-58.mkv"),
    )
)
SPECTATING_TIMESTAMP = 3540
SPECTATING_BOX = (1100, 0, 1460, 80)
SPECTATING_MIN_AREA = 20

COUNTER_NUMBERS_X = (2060, 2178)
ICON_HALF = ICON_NATIVE_SIZE // 2
ROW_HALF_HEIGHT = 17

HASH_BOX = (918, 182, 996, 292)
ORANGE_BANNER_TIMESTAMP = 2386
ORANGE_BANNER_BOX = (1158, 92, 256, 122)
PLACED_TIMESTAMP = 3630
PLACED_BOX = (1280, 158, 1600, 302)
PLACED_HASH_MIN_HEIGHT = 60


@dataclass(frozen=True)
class DigitSource:
    timestamp: float
    row: str
    text: str


DIGIT_SOURCES = (
    DigitSource(135, "personal_eliminations", "0"),
    DigitSource(135, "players_left", "99"),
    DigitSource(510, "personal_eliminations", "5"),
    DigitSource(510, "team_eliminations", "10"),
    DigitSource(510, "players_left", "44"),
    DigitSource(535, "personal_eliminations", "7"),
    DigitSource(535, "team_eliminations", "12"),
    DigitSource(535, "players_left", "38"),
    DigitSource(410, "team_eliminations", "6"),
)
ICON_SOURCE_TIMESTAMP = 135
BANNER_TIMESTAMP = 38


def extract_frame(video: Path, timestamp: float) -> np.ndarray:
    frame = decode_frame(video, timestamp)
    assert frame.shape[:2] == (NATIVE_HEIGHT, NATIVE_WIDTH)
    return frame


def row_centre_y(row: str) -> int:
    _, y, _, h = NATIVE_RECTS[row]
    return y + h // 2


def row_glyphs(frame: np.ndarray, row: str, text: str) -> list[np.ndarray]:
    cy = row_centre_y(row)
    x0, x1 = COUNTER_NUMBERS_X
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    strip = gray[cy - ROW_HALF_HEIGHT : cy + ROW_HALF_HEIGHT, x0:x1]
    _, binary = cv2.threshold(strip, 200, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = sorted(
        (int(s[0]), int(s[1]), int(s[2]), int(s[3]))
        for s in stats[1:]
        if s[cv2.CC_STAT_AREA] >= 30 and s[cv2.CC_STAT_HEIGHT] >= 15
    )
    assert len(boxes) == len(text), (row, text, boxes)
    top = min(b[1] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    soft = np.clip((strip.astype(np.float32) - 90.0) / 150.0, 0.0, 1.0)
    glyphs = [(soft[top:bottom, x : x + w] * 255).astype(np.uint8) for x, _, w, _ in boxes]
    return [
        cv2.resize(g, (g.shape[1], DIGIT_HEIGHT_NATIVE), interpolation=cv2.INTER_AREA)
        for g in glyphs
    ]


def cut_digits() -> dict[str, np.ndarray]:
    glyphs: dict[str, np.ndarray] = {}
    for source in DIGIT_SOURCES:
        frame = extract_frame(HUD_RECORDING, source.timestamp)
        for char, glyph in zip(
            source.text, row_glyphs(frame, source.row, source.text), strict=True
        ):
            glyphs.setdefault(char, glyph)
    assert sorted(glyphs) == list("0123456789"), sorted(glyphs)
    return glyphs


def cut_icons() -> dict[str, np.ndarray]:
    frame = extract_frame(HUD_RECORDING, ICON_SOURCE_TIMESTAMP)
    icons: dict[str, np.ndarray] = {}
    for icon, row in ICONS.items():
        cy = row_centre_y(row)
        icons[icon] = frame[
            cy - ICON_HALF : cy + ICON_HALF, ICON_CENTRE_X - ICON_HALF : ICON_CENTRE_X + ICON_HALF
        ]
    return icons


def cut_win() -> dict[str, np.ndarray]:
    frame = extract_frame(WIN_RECORDING, BANNER_TIMESTAMP)
    x, y, w, h = NATIVE_RECTS["results_banner"]
    banner = cv2.resize(
        frame[y : y + h, x : x + w],
        None,
        fx=1 / BANNER_CROP_SCALE,
        fy=1 / BANNER_CROP_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    banner = (banner // 8) * 8

    hx0, hy0, hx1, hy1 = HASH_BOX
    hsv = cv2.cvtColor(frame[hy0:hy1, hx0:hx1], cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, (20, 90, 200), (34, 255, 255))
    _, _, stats, _ = cv2.connectedComponentsWithStats(gold)
    largest = max(stats[1:], key=lambda s: int(s[cv2.CC_STAT_AREA]))
    x, y, w, h = (int(largest[i]) for i in range(4))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(gold[y : y + h, x : x + w], cv2.MORPH_OPEN, kernel)
    ys, xs = np.nonzero(opened)
    hash_mask = cv2.GaussianBlur(
        opened[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], (3, 3), 0
    )
    return {"victory_royale": banner, "hash": hash_mask}


def cut_variants() -> dict[str, np.ndarray]:
    """The orange Victory Royale banner and the white "#" of the "YOU PLACED #N" screen."""
    frame = extract_frame(HUD_RECORDING, ORANGE_BANNER_TIMESTAMP)
    x, y, w, h = ORANGE_BANNER_BOX
    orange = cv2.resize(
        frame[y : y + h, x : x + w],
        None,
        fx=1 / BANNER_CROP_SCALE,
        fy=1 / BANNER_CROP_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    frame = extract_frame(HUD_RECORDING, PLACED_TIMESTAMP)
    x0, y0, x1, y1 = PLACED_BOX
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    glyphs = sorted(
        (int(s[cv2.CC_STAT_LEFT]), i)
        for i, s in enumerate(stats)
        if i > 0 and s[cv2.CC_STAT_HEIGHT] >= PLACED_HASH_MIN_HEIGHT
    )
    _, label = glyphs[0]
    left, top, width, height = (int(stats[label][k]) for k in range(4))
    isolated = np.where(labels[top : top + height, left : left + width] == label, 255, 0)
    hash_mask = cv2.GaussianBlur(isolated.astype(np.uint8), (3, 3), 0)
    return {"victory_royale_orange": (orange // 8) * 8, "hash_white": hash_mask}


def cut_spectating() -> dict[str, np.ndarray]:
    """The white "SPECTATING" label an eliminated player sees while watching a teammate."""
    frame = extract_frame(SPECTATING_RECORDING, SPECTATING_TIMESTAMP)
    x0, y0, x1, y1 = SPECTATING_BOX
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)
    _, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    letters = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] > SPECTATING_MIN_AREA]
    left = min(int(s[cv2.CC_STAT_LEFT]) for s in letters)
    top = min(int(s[cv2.CC_STAT_TOP]) for s in letters)
    right = max(int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]) for s in letters)
    bottom = max(int(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]) for s in letters)
    return {"spectating": cv2.GaussianBlur(white[top:bottom, left:right], (3, 3), 0)}


def main() -> None:
    sources = (HUD_RECORDING, WIN_RECORDING, SPECTATING_RECORDING)
    missing = [p for p in sources if not p.is_file()]
    if missing:
        sys.exit(
            "missing source Recording(s): "
            + ", ".join(str(p) for p in missing)
            + "\nset VICTORIOUS_HUD_RECORDING / VICTORIOUS_WIN_RECORDING / "
            "VICTORIOUS_SPECTATING_RECORDING to local files"
        )
    CROPS.mkdir(parents=True, exist_ok=True)
    written = {
        **{f"digit_{c}": g for c, g in cut_digits().items()},
        **cut_icons(),
        **cut_win(),
        **cut_variants(),
        **cut_spectating(),
    }
    for name, image in written.items():
        path = (CROPS if name.startswith("digit_") else REFERENCES) / f"{name}.png"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"{path.name:22s} {image.shape[1]}x{image.shape[0]} {path.stat().st_size} B")


if __name__ == "__main__":
    main()
