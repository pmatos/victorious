"""Internal detector: multi-scale template matching for reference images.

Used for reference-image matching such as the Victory Royale banner.
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SCALES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def parse_scales(scales_str: str) -> list[float]:
    """Parse a comma-separated string of scale factors into a list of floats.

    Empty entries (e.g. trailing commas) are ignored. Raises ValueError if a
    non-empty entry is not a valid float.
    """
    return [float(part) for part in scales_str.split(",") if part.strip()]


def frame_index_to_timestamp(frame_index: int, fps: float) -> float:
    """Convert a zero-based frame index to a timestamp in seconds."""
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
            currently_detected = True
        elif not is_match and currently_detected:
            currently_detected = False
    return detection_times


def load_template(image_path: Path, grayscale: bool = False) -> np.ndarray:
    if grayscale:
        template = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        template = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"Could not load image: {image_path}")
    return template


def match_template_in_frame_multiscale(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float,
    scales: list[float] | None = None,
) -> tuple[bool, float, float]:
    """
    Multi-scale template matching.
    Returns: (is_match, best_confidence, best_scale)
    """
    if scales is None:
        scales = DEFAULT_SCALES

    template_h, template_w = template.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    best_confidence = 0.0
    best_scale = 1.0

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
            continue

    is_match = best_confidence >= threshold
    logger.debug(
        "Multi-scale result: best_confidence=%.4f, best_scale=%.2fx, is_match=%s",
        best_confidence,
        best_scale,
        is_match,
    )

    return is_match, best_confidence, best_scale


def match_template_in_frame(frame: np.ndarray, template: np.ndarray, threshold: float) -> bool:
    """
    Backward compatibility wrapper for single-scale template matching.
    """
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= threshold
