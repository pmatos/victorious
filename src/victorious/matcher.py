"""Internal detector: multi-scale template matching for reference images.

Used for reference-image matching such as the Victory Royale banner.
"""

import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SCALES: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def _require_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero: {value}")


def parse_scales(scales_str: str) -> list[float]:
    """Parse a comma-separated string of scale factors into a list of floats.

    Empty entries (e.g. trailing commas) are ignored. Raises ValueError if a
    non-empty entry is not a valid float, or is not finite and greater than zero.
    """
    scales = [float(part) for part in scales_str.split(",") if part.strip()]
    for scale in scales:
        _require_positive_finite(scale, "scale")
    return scales


def frame_index_to_timestamp(frame_index: int, fps: float) -> float:
    """Convert a zero-based frame index to a timestamp in seconds.

    Raises ValueError if ``fps`` is not a finite number greater than zero.
    """
    _require_positive_finite(fps, "fps")
    return frame_index / fps


def extract_detection_times(matches: Iterable[tuple[bool, float]]) -> list[float]:
    """Reduce per-frame match results to the start timestamp of each run.

    Given ``(is_match, timestamp)`` pairs in chronological order, emit the
    timestamp only when a match run begins (a False->True transition, or the
    first frame if it already matches). Consecutive matches collapse to a single
    start; a run only restarts after at least one non-matching frame.
    """
    detection_times: list[float] = []
    currently_detected = False
    for is_match, timestamp in matches:
        if is_match and not currently_detected:
            detection_times.append(timestamp)
        currently_detected = is_match
    return detection_times


def load_template(image_path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    template = cv2.imread(str(image_path), flag)
    if template is None:
        raise ValueError(f"Could not load image: {image_path}")
    return template


def match_template_in_frame_multiscale(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float,
    scales: Sequence[float] | None = None,
) -> tuple[bool, float, float]:
    """
    Multi-scale template matching.
    Returns: (is_match, best_confidence, best_scale)

    Raises cv2.error if OpenCV rejected every scale that was attempted (for
    example a colour frame matched against a grayscale template), since that is
    a misuse rather than a genuine absence of a match.
    """
    if scales is None:
        scales = DEFAULT_SCALES

    template_h, template_w = template.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    best_confidence = 0.0
    best_scale = 1.0
    attempted = 0
    errors: list[cv2.error] = []

    for scale in scales:
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)

        if new_w > frame_w or new_h > frame_h:
            logger.debug(
                "Skipping scale %.2fx - template too large (%dx%d > %dx%d)",
                scale,
                new_w,
                new_h,
                frame_w,
                frame_h,
            )
            continue

        if new_w < 10 or new_h < 10:
            logger.debug("Skipping scale %.2fx - template too small (%dx%d)", scale, new_w, new_h)
            continue

        attempted += 1
        try:
            scaled_template = cv2.resize(template, (new_w, new_h))

            result = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            logger.debug("Scale %.2fx: confidence %.4f", scale, max_val)

            if max_val > best_confidence:
                best_confidence = max_val
                best_scale = scale

                if max_val > 0.95:
                    logger.debug(
                        "Early termination at scale %.2fx with confidence %.4f", scale, max_val
                    )
                    break

        except cv2.error as e:
            logger.debug("OpenCV error at scale %.2fx: %s", scale, e)
            errors.append(e)

    if errors and len(errors) == attempted:
        raise errors[-1]

    is_match = best_confidence >= threshold
    logger.debug(
        "Multi-scale result: best_confidence=%.4f, best_scale=%.2fx, is_match=%s",
        best_confidence,
        best_scale,
        is_match,
    )

    return is_match, best_confidence, best_scale
