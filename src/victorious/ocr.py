"""Reading digits from a binary mask with Tesseract, one glyph at a time.

Line mode confuses adjacent identical digits (11 reads as 1), so each connected
component is recognised on its own.
"""

import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

GLYPH_HEIGHT = 48
GLYPH_PADDING = 16
MIN_GLYPH_AREA = 12


class OcrError(Exception):
    """Tesseract is missing or failed."""


@dataclass(frozen=True)
class DigitReading:
    text: str
    confidence: float  # 0..1, the weakest glyph


def _tesseract() -> str:
    path = shutil.which("tesseract")
    if path is None:
        raise OcrError("tesseract not found on PATH; install tesseract-ocr")
    return path


def _glyph_images(mask: np.ndarray) -> list[np.ndarray]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [tuple(int(v) for v in stats[i]) for i in range(1, count)]
    boxes = [b for b in boxes if b[4] >= MIN_GLYPH_AREA]
    images: list[np.ndarray] = []
    for x, y, w, h, _ in sorted(boxes):
        glyph = mask[y : y + h, x : x + w]
        scale = GLYPH_HEIGHT / h
        glyph = cv2.resize(
            glyph, (max(1, round(w * scale)), GLYPH_HEIGHT), interpolation=cv2.INTER_CUBIC
        )
        glyph = cv2.copyMakeBorder(
            255 - glyph, *(GLYPH_PADDING,) * 4, cv2.BORDER_CONSTANT, value=255
        )
        images.append(glyph)
    return images


def _read_glyph(image: np.ndarray, tesseract: str) -> tuple[str, float] | None:
    encoded = cv2.imencode(".png", image)[1].tobytes()
    command = [
        tesseract, "stdin", "stdout", "--psm", "10",
        "-c", "tessedit_char_whitelist=0123456789", "tsv",
    ]  # fmt: skip
    result = subprocess.run(command, input=encoded, capture_output=True, check=False)
    if result.returncode != 0:
        raise OcrError(f"tesseract failed: {result.stderr.decode(errors='replace').strip()}")
    for line in result.stdout.decode(errors="replace").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) == 12 and fields[11].strip().isdigit() and float(fields[10]) >= 0:
            return fields[11].strip(), float(fields[10]) / 100
    return None


def read_digits(mask: np.ndarray) -> DigitReading | None:
    """The digits in `mask` (white glyphs on black), or None when nothing readable is there."""
    tesseract = _tesseract()
    glyphs = _glyph_images(mask)
    if not glyphs:
        return None
    readings = [_read_glyph(image, tesseract) for image in glyphs]
    if any(reading is None for reading in readings):
        return None
    found = [reading for reading in readings if reading is not None]
    return DigitReading(
        text="".join(text for text, _ in found),
        confidence=min(confidence for _, confidence in found),
    )
