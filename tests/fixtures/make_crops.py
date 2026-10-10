"""Re-cut the committed HUD crops from the author's own Recordings.

Run by hand from the repository root when a crop needs refreshing; CI never
runs it. Sources are local and gitignored; point the script at them with
VICTORIOUS_HUD_RECORDING (a native 2560x1440 Recording showing the in-match HUD)
and VICTORIOUS_WIN_RECORDING (one showing the Victory Royale results screen):

    uv run python -m tests.fixtures.make_crops
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic.render import (
    BANNER_CROP_SCALE,
    DIGIT_HEIGHT_NATIVE,
    ICON_CENTRE_X,
    ICON_NATIVE_SIZE,
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
HERE = Path(__file__).parent
CROPS = HERE / "crops"
REPO = HERE.parent.parent

HUD_RECORDING = Path(
    os.environ.get("VICTORIOUS_HUD_RECORDING", "/mnt/windows/Videos/2026-10-09_11-59-53.mkv")
)
WIN_RECORDING = Path(
    os.environ.get(
        "VICTORIOUS_WIN_RECORDING", str(REPO / "assets" / "videos" / "test-video-last-minute.mkv")
    )
)

COUNTER_NUMBERS_X = (2060, 2178)
ICON_HALF = ICON_NATIVE_SIZE // 2
ROW_CENTRE_Y = {"personal_eliminations": 61, "team_eliminations": 102, "players_left": 151}
ROW_HALF_HEIGHT = 17
DIGIT_HEIGHT = DIGIT_HEIGHT_NATIVE

BANNER_BOX = (1105, 110, 1485, 300)
HASH_BOX = (918, 182, 996, 292)


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
    result = subprocess.run(
        [
            FFMPEG, "-v", "error", "-ss", str(timestamp), "-i", str(video),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True,
        check=True,
    )  # fmt: skip
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    assert frame.shape[:2] == (NATIVE_HEIGHT, NATIVE_WIDTH)
    return np.asarray(frame)


def row_glyphs(frame: np.ndarray, row: str, text: str) -> list[np.ndarray]:
    cy = ROW_CENTRE_Y[row]
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
    return [cv2.resize(g, (g.shape[1], DIGIT_HEIGHT), interpolation=cv2.INTER_AREA) for g in glyphs]


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
    names = {"personal_eliminations": "icon_skull", "team_eliminations": "icon_squad",
             "players_left": "icon_person"}  # fmt: skip
    return {
        names[row]: frame[
            cy - ICON_HALF : cy + ICON_HALF, ICON_CENTRE_X - ICON_HALF : ICON_CENTRE_X + ICON_HALF
        ]
        for row, cy in ROW_CENTRE_Y.items()
    }


def cut_win() -> dict[str, np.ndarray]:
    frame = extract_frame(WIN_RECORDING, BANNER_TIMESTAMP)
    x0, y0, x1, y1 = BANNER_BOX
    banner = cv2.resize(
        frame[y0:y1, x0:x1],
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


def main() -> None:
    missing = [p for p in (HUD_RECORDING, WIN_RECORDING) if not p.is_file()]
    if missing:
        sys.exit(
            "missing source Recording(s): "
            + ", ".join(str(p) for p in missing)
            + "\nset VICTORIOUS_HUD_RECORDING / VICTORIOUS_WIN_RECORDING to local files"
        )
    CROPS.mkdir(parents=True, exist_ok=True)
    written: dict[str, np.ndarray] = {}
    written.update({f"digit_{c}": g for c, g in cut_digits().items()})
    written.update(cut_icons())
    written.update(cut_win())
    for name, image in written.items():
        path = CROPS / f"{name}.png"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"{path.name:22s} {image.shape[1]}x{image.shape[0]} {path.stat().st_size} B")


if __name__ == "__main__":
    main()
