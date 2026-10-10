"""A Layout: where each HUD element sits on screen, in resolution-independent coordinates."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

EPSILON = 1e-9
ASPECT_RATIO_TOLERANCE = 0.02


class LayoutError(Exception):
    """The Layout file cannot be read or is not a valid Layout."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Rect(_Model):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    w: float = Field(allow_inf_nan=False)
    h: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _inside_the_frame(self) -> "Rect":
        for origin, extent, axis in ((self.x, self.w, "x/w"), (self.y, self.h, "y/h")):
            if origin < 0 or extent <= 0 or origin + extent > 1 + EPSILON:
                raise ValueError(f"{axis} must describe a box inside the frame, within 0..1")
        return self


class HudScreen(_Model):
    hud_anchor: Rect
    personal_eliminations: Rect | None = None
    team_eliminations: Rect | None = None
    players_left: Rect | None = None


class ResultsScreen(_Model):
    results_banner: Rect
    placement: Rect


class Screens(_Model):
    hud: HudScreen
    results: ResultsScreen


class Layout(_Model):
    aspect_ratio: str
    screens: Screens

    @field_validator("aspect_ratio")
    @classmethod
    def _is_a_ratio(cls, value: str) -> str:
        if not re.fullmatch(r"[1-9]\d*:[1-9]\d*", value):
            raise ValueError("must look like 16:9")
        return value

    def require_aspect_ratio(self, width: int, height: int) -> None:
        wide, high = (int(part) for part in self.aspect_ratio.split(":"))
        expected = wide / high
        if abs(width / height - expected) > ASPECT_RATIO_TOLERANCE * expected:
            raise LayoutError(
                f"Layout is for aspect ratio {self.aspect_ratio} "
                f"but the Recording is {width}x{height}"
            )


def load_layout(path: Path) -> Layout:
    try:
        document = yaml.safe_load(path.read_text())
    except OSError as error:
        raise LayoutError(f"cannot read Layout {path}: {error.strerror or error}") from error
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        raise LayoutError(f"Layout {path} is not valid YAML: {error}") from error
    try:
        return Layout.model_validate(document)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or '(document)'}: {issue['msg']}"
            for issue in error.errors()
        )
        raise LayoutError(f"invalid Layout {path}: {problems}") from error
