"""Locating the results-screen placement where a real Win draws it."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from victorious.detect import find_results
from victorious.references import load_reference

CROPS = Path(__file__).parent / "fixtures" / "crops"
NATIVE = (1440, 2560)
GOLD = (60, 190, 250)
BANNER_AT = (1160, 40)
BANNER_SCALE = 4
# Gold masks of the "#1" of a real 2560x1440 Win: (file, x, y of the mask in the frame, "#1" box).
# There the "1" is much taller than the "#" and too solid for OCR to read.
REAL_WINS = (
    ("win_hash1_516.png", 900, 180, (924, 208, 1064, 296)),
    ("win_hash1_530.png", 990, 60, (1002, 82, 1112, 172)),
)


def win_frame(name: str, x: int, y: int) -> np.ndarray:
    mask = cv2.imread(str(CROPS / name), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    frame = np.full((*NATIVE, 3), (90, 30, 40), np.uint8)
    painted = frame[y : y + mask.shape[0], x : x + mask.shape[1]]
    painted[mask > 0] = GOLD
    banner = load_reference("victory_royale", True)
    size = (banner.shape[1] * BANNER_SCALE, banner.shape[0] * BANNER_SCALE)
    left, top = BANNER_AT
    frame[top : top + size[1], left : left + size[0]] = cv2.resize(banner, size)
    return frame


@pytest.mark.usefixtures("require_tesseract")
@pytest.mark.parametrize(("name", "x", "y", "box"), REAL_WINS)
def test_the_gold_placement_of_a_real_win_is_found_around_its_taller_digit(
    name: str, x: int, y: int, box: tuple[int, int, int, int]
) -> None:
    detection = find_results(win_frame(name, x, y))

    assert "placement" in detection.found, detection.missing
    rect = detection.found["placement"].rect
    left, top = rect.x * NATIVE[1], rect.y * NATIVE[0]
    right, bottom = left + rect.w * NATIVE[1], top + rect.h * NATIVE[0]
    assert left <= box[0]
    assert top <= box[1]
    assert right >= box[2]
    assert bottom >= box[3]
