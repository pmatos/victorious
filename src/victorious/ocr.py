"""Reading digits from a binary mask with Tesseract, one glyph at a time.

Line mode confuses adjacent identical digits (11 reads as 1), so each connected
component is recognised on its own. Tesseract's fast models drop some glyphs
(a 0 under a digit whitelist, a 4 at one size), so every glyph is read with a few
independent settings and the digits vote by confidence.
"""

import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

GLYPH_PADDING = 16
MIN_GLYPH_AREA = 12
MIN_GLYPH_HEIGHT_RATIO = 0.5
LOOKALIKES = {"O": "0", "o": "0"}


@dataclass(frozen=True)
class _Pass:
    page_segmentation_mode: int
    glyph_height: int
    whitelist: bool


PASSES = (_Pass(8, 96, True), _Pass(8, 80, True), _Pass(10, 64, False))


class OcrError(Exception):
    """Tesseract is missing or failed."""


@dataclass(frozen=True)
class DigitReading:
    text: str
    confidence: float  # 0..1, the weakest glyph


def find_tesseract() -> str:
    path = shutil.which("tesseract")
    if path is None:
        raise OcrError("tesseract not found on PATH; install tesseract-ocr")
    return path


def _glyph_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, area in (stats[i] for i in range(1, count))
        if area >= MIN_GLYPH_AREA
    ]
    if not boxes:
        return []
    tallest = max(h for _, _, _, h in boxes)
    return sorted(box for box in boxes if box[3] >= MIN_GLYPH_HEIGHT_RATIO * tallest)


def _glyph_image(glyph: np.ndarray, height: int) -> np.ndarray:
    scale = height / glyph.shape[0]
    resized = cv2.resize(
        glyph, (max(1, round(glyph.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC
    )
    return cv2.copyMakeBorder(255 - resized, *(GLYPH_PADDING,) * 4, cv2.BORDER_CONSTANT, value=255)


def _recognise(image: np.ndarray, tesseract: str, settings: _Pass) -> tuple[str, float] | None:
    encoded = cv2.imencode(".png", image)[1].tobytes()
    command = [tesseract, "stdin", "stdout", "--psm", str(settings.page_segmentation_mode)]
    if settings.whitelist:
        command += ["-c", "tessedit_char_whitelist=0123456789"]
    result = subprocess.run([*command, "tsv"], input=encoded, capture_output=True, check=False)
    if result.returncode != 0:
        raise OcrError(f"tesseract failed: {result.stderr.decode(errors='replace').strip()}")
    for line in result.stdout.decode(errors="replace").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) != 12:
            continue
        glyph = fields[11].strip()
        if len(glyph) != 1:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        digit = LOOKALIKES.get(glyph, glyph)
        if confidence >= 0 and digit.isdigit():
            return digit, confidence / 100
    return None


def _read_glyph(glyph: np.ndarray, tesseract: str) -> tuple[str, float] | None:
    votes: dict[str, float] = {}
    for settings in PASSES:
        reading = _recognise(_glyph_image(glyph, settings.glyph_height), tesseract, settings)
        if reading is not None:
            votes[reading[0]] = votes.get(reading[0], 0.0) + reading[1]
    if not votes:
        return None
    digit, total = max(votes.items(), key=lambda item: item[1])
    return digit, total / len(PASSES)


def read_digits(mask: np.ndarray) -> DigitReading | None:
    """The digits in `mask` (white glyphs on black), or None when nothing readable is there."""
    if mask.size == 0:
        return None
    tesseract = find_tesseract()
    boxes = _glyph_boxes(mask)
    if not boxes:
        return None
    readings = [_read_glyph(mask[y : y + h, x : x + w], tesseract) for x, y, w, h in boxes]
    found = [reading for reading in readings if reading is not None]
    if len(found) != len(readings):
        return None
    return DigitReading(
        text="".join(text for text, _ in found),
        confidence=min(confidence for _, confidence in found),
    )
