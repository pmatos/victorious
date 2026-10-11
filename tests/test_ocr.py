"""Behaviour of reading placement digits from a binary mask."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from victorious.ocr import OcrError, glyph_boxes, read_digits

CROPS = Path(__file__).parent / "fixtures" / "crops"
COUNTER_MASKS = sorted(CROPS.glob("counter_*.png"))


def _digits_mask(text: str) -> np.ndarray:
    mask = np.zeros((120, 60 * len(text) + 80), np.uint8)
    cv2.putText(mask, text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 8)
    return mask


def test_an_empty_mask_reads_nothing_and_does_not_crash() -> None:
    assert read_digits(np.zeros((20, 0), np.uint8)) is None
    assert read_digits(np.zeros((0, 20), np.uint8)) is None


@pytest.mark.usefixtures("require_tesseract")
def test_a_stray_speck_beside_the_digits_is_not_a_glyph() -> None:
    mask = _digits_mask("7")
    mask[100:106, mask.shape[1] - 10 : mask.shape[1] - 4] = 255

    reading = read_digits(mask)

    assert reading is not None
    assert reading.text == "7"


def test_missing_tesseract_is_an_ocr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(OcrError, match="tesseract"):
        read_digits(_digits_mask("7"))


def test_malformed_tesseract_output_is_skipped_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    header = "\t".join(["level"] * 12)
    broken = "\t".join(["5", "1", "1", "1", "1", "1", "0", "0", "5", "5", "n/a", "7"])
    output = f"{header}\n{broken}\n"

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 0, output.encode(), b"")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert read_digits(_digits_mask("7")) is None


@pytest.mark.usefixtures("require_tesseract")
@pytest.mark.parametrize("path", COUNTER_MASKS, ids=lambda path: path.stem)
def test_real_counter_masks_are_read_as_what_the_hud_shows(path: Path) -> None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None

    reading = read_digits(mask)

    assert reading is not None
    assert reading.text == path.stem.split("_")[1]


def test_glyph_boxes_of_an_empty_mask_is_empty_rather_than_a_crash() -> None:
    assert glyph_boxes(np.zeros((10, 0), np.uint8)) == []


def test_a_tall_blob_beyond_the_height_cap_does_not_hide_the_digits() -> None:
    mask = np.zeros((40, 60), np.uint8)
    mask[5:17, 10:18] = 255
    mask[5:17, 20:28] = 255
    mask[0:34, 40:46] = 255

    assert glyph_boxes(mask) == [(40, 0, 6, 34)]
    assert glyph_boxes(mask, max_height=20) == [(10, 5, 8, 12), (20, 5, 8, 12)]
