"""Template-matching tests using synthetic in-memory frames (no binary assets)."""

import cv2
import numpy as np

from main import match_template_in_frame, match_template_in_frame_multiscale


def make_template() -> np.ndarray:
    """A 40x40 grayscale template with enough variance for TM_CCOEFF_NORMED."""
    template = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(template, (5, 5), (34, 34), 255, -1)
    cv2.rectangle(template, (15, 15), (24, 24), 0, -1)
    return template


def frame_with_template_at(row: int, col: int) -> np.ndarray:
    template = make_template()
    frame = np.zeros((300, 300), dtype=np.uint8)
    frame[row : row + 40, col : col + 40] = template
    return frame


def test_multiscale_detects_embedded_template() -> None:
    template = make_template()
    frame = frame_with_template_at(100, 120)

    is_match, confidence, best_scale = match_template_in_frame_multiscale(
        frame, template, threshold=0.8, scales=[1.0]
    )

    assert is_match
    assert confidence > 0.99
    assert best_scale == 1.0


def test_multiscale_rejects_absent_template() -> None:
    template = make_template()
    other = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(other, (150, 150), 50, (255,), -1)

    is_match, confidence, _ = match_template_in_frame_multiscale(
        other, template, threshold=0.8, scales=[1.0]
    )

    assert not is_match
    assert confidence < 0.8


def test_single_scale_detects_embedded_template() -> None:
    template = make_template()
    frame = frame_with_template_at(60, 200)

    assert match_template_in_frame(frame, template, threshold=0.8)


def test_multiscale_skips_scales_larger_than_frame() -> None:
    template = make_template()
    small_frame = np.zeros((30, 30), dtype=np.uint8)

    # Every requested scale exceeds the frame, so nothing can match.
    is_match, confidence, _ = match_template_in_frame_multiscale(
        small_frame, template, threshold=0.8, scales=[1.0, 1.2]
    )

    assert not is_match
    assert confidence == 0.0
