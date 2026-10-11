"""The annotated preview of a setup: each reference frame with its rectangles and labels."""

from collections.abc import Sequence

import cv2
import numpy as np

from victorious.setup import Panel

MAX_PANEL_WIDTH = 1600
KIND_COLORS = {"hud": (80, 220, 80), "results": (40, 160, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
LABEL_PADDING = 3
BORDER = 2


def _panel_image(panel: Panel) -> np.ndarray:
    height, width = panel.frame.shape[:2]
    if width > MAX_PANEL_WIDTH:
        size = (MAX_PANEL_WIDTH, max(1, round(height * MAX_PANEL_WIDTH / width)))
        image = cv2.resize(panel.frame, size, interpolation=cv2.INTER_AREA)
    else:
        image = panel.frame.copy()
    color = KIND_COLORS[panel.kind]
    side, rows = image.shape[1], image.shape[0]
    for name, found in panel.found.items():
        rect = found.rect
        x0, y0 = round(rect.x * side), round(rect.y * rows)
        x1, y1 = round((rect.x + rect.w) * side), round((rect.y + rect.h) * rows)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, BORDER)
        (text_width, text_height), baseline = cv2.getTextSize(name, FONT, FONT_SCALE, 1)
        strip_height = text_height + baseline + 2 * LABEL_PADDING
        strip_width = text_width + 2 * LABEL_PADDING
        if y1 - y0 < 2 * strip_height and x0 - BORDER >= strip_width:
            left = x0 - BORDER - strip_width
            top = max(0, min(rows - strip_height, (y0 + y1 - strip_height) // 2))
        else:
            left = max(0, min(x0, side - strip_width))
            top = y0 - BORDER - strip_height
            if top < 0:
                top = min(rows - strip_height, y1 + BORDER)
        cv2.rectangle(image, (left, top), (left + strip_width, top + strip_height), color, -1)
        cv2.putText(
            image, name, (left + LABEL_PADDING, top + LABEL_PADDING + text_height),
            FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA,
        )  # fmt: skip
    caption = f"{panel.kind} @ {panel.second} s"
    cv2.putText(image, caption, (8, 20), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, caption, (8, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_preview(panels: Sequence[Panel]) -> np.ndarray:
    """The panels stacked top to bottom in the order given, each at most 1600 px wide."""
    return np.vstack([_panel_image(panel) for panel in panels])
