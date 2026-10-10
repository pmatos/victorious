"""Behaviour of reading placement digits from a binary mask."""

import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
import pytest

from victorious.ocr import OcrError, read_digits


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
