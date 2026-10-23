"""A Layout: where each HUD element sits on screen, in resolution-independent coordinates."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

EPSILON = 1e-9
ASPECT_RATIO_TOLERANCE = 0.02
LAYOUT_SUFFIXES = (".yaml", ".yml")
HudDesign = Literal["icon_column", "icon_pair", "counter_row"]
DEFAULT_HUD_DESIGN: HudDesign = "icon_column"
# Designs whose counters are one column of icons at fixed rectangles: the icon column has three
# rows, the icon pair two (no team eliminations row).
COLUMN_DESIGNS: tuple[HudDesign, ...] = ("icon_column", "icon_pair")
COMMON_ASPECT_RATIOS = ((16, 9), (16, 10), (21, 9), (32, 9), (4, 3), (5, 4), (3, 2))


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
    elimination_placement: Rect | None = None


class Screens(_Model):
    hud: HudScreen
    results: ResultsScreen


class Layout(_Model):
    aspect_ratio: str
    hud_design: HudDesign = DEFAULT_HUD_DESIGN
    screens: Screens

    @field_validator("aspect_ratio")
    @classmethod
    def _is_a_ratio(cls, value: str) -> str:
        if not re.fullmatch(r"[1-9]\d*:[1-9]\d*", value):
            raise ValueError("must look like 16:9")
        return value

    def fits(self, width: int, height: int) -> bool:
        wide, high = (int(part) for part in self.aspect_ratio.split(":"))
        expected = wide / high
        return abs(width / height - expected) <= ASPECT_RATIO_TOLERANCE * expected


def aspect_ratio_label(width: int, height: int) -> str:
    """A label such as 16:9 that `Layout.fits(width, height)` accepts."""
    actual = width / height
    nearest = min(COMMON_ASPECT_RATIOS, key=lambda ratio: abs(actual - ratio[0] / ratio[1]))
    wide, high = nearest
    if abs(actual - wide / high) <= ASPECT_RATIO_TOLERANCE * wide / high:
        return f"{wide}:{high}"
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def dump_layout(layout: Layout) -> str:
    """The Layout as hand-editable YAML, each rectangle on one line."""
    document = layout.model_dump(exclude_none=True)
    return yaml.safe_dump(document, sort_keys=False, default_flow_style=None)


@dataclass(frozen=True)
class LoadedLayout:
    path: Path
    layout: Layout


def load_layout(path: Path) -> Layout:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
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


def load_layouts(path: Path) -> list[LoadedLayout]:
    """The Layout at `path`, or every `*.yaml` / `*.yml` Layout directly inside it by name."""
    if not path.is_dir():
        return [LoadedLayout(path, load_layout(path))]
    files = sorted(
        (entry for entry in path.iterdir() if entry.suffix in LAYOUT_SUFFIXES and entry.is_file()),
        key=lambda entry: entry.name,
    )
    if not files:
        raise LayoutError(f"no Layout files (*.yaml, *.yml) in {path}")
    return [LoadedLayout(file, load_layout(file)) for file in files]
