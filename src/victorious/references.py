"""Reference images of the Fortnite HUD that ship with the package."""

from functools import cache
from importlib.resources import files

import cv2
import numpy as np

NATIVE_HEIGHT = 1440
BANNER_CROP_SCALE = 4.0

_CROP_SCALE = {"victory_royale": BANNER_CROP_SCALE}


@cache
def load_reference(name: str, color: bool) -> np.ndarray:
    data = files("victorious").joinpath("data", f"{name}.png").read_bytes()
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if image is None:
        raise FileNotFoundError(f"missing reference image {name}.png in the victorious package")
    return np.asarray(image)


def reference_scale(name: str, frame_height: int) -> float:
    """How much to resize reference `name` for a frame of `frame_height` pixels."""
    return _CROP_SCALE.get(name, 1.0) * frame_height / NATIVE_HEIGHT
