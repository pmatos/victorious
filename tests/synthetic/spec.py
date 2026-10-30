"""Declarative description of what is on screen when in a synthetic Recording."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Literal

ResultsStyle = Literal["classic", "orange", "orange_large", "placed", "placed_shrunk"]
HudDesign = Literal["icon_column", "icon_pair", "counter_row"]


@dataclass(frozen=True)
class Lobby:
    """Anything that is not a Match: no HUD, no results."""


@dataclass(frozen=True)
class Hud:
    """The in-match HUD; a counter set to None is not drawn while the HUD anchors stay.

    `corrupt` replaces the three counter regions with noise, the anchors stay. `design` is the
    icon column at the top right, the icon pair (the same column without its team eliminations
    row, so `team_eliminations` is not drawn) or the counter row under the minimap, whose icons
    move with the width of the numbers.
    """

    personal_eliminations: int | None = 0
    team_eliminations: int | None = 0
    players_left: int | None = 100
    corrupt: bool = False
    design: HudDesign = "icon_column"

    def __post_init__(self) -> None:
        for counter in fields(self):
            if counter.name in ("corrupt", "design"):
                continue
            value = getattr(self, counter.name)
            if value is not None and not 0 <= value <= 999:
                raise ValueError(f"{counter.name} must be between 0 and 999, got {value}")


@dataclass(frozen=True)
class Results:
    """The results screen; placement 1 is a Win and carries the Victory Royale banner.

    `style` is "classic" (blue banner, gold "#N"), "orange" (the orange banner skin, a Win),
    "orange_large" (the same banner while it is zoomed in and lower, before it settles; the
    "#1" is part of its art), "placed" (white "YOU PLACED #N" shown once eliminated) or
    "placed_shrunk" (the same text at 0.8x, as it is drawn part-way through its size animation).
    """

    placement: int
    style: ResultsStyle = "classic"

    def __post_init__(self) -> None:
        if not 1 <= self.placement <= 100:
            raise ValueError(f"placement must be between 1 and 100, got {self.placement}")
        if self.style.startswith("orange") and self.placement != 1:
            raise ValueError("the orange banner is a Win, so its placement must be 1")


@dataclass(frozen=True)
class Spectating:
    """The player is out and watching a teammate: no HUD counters, the white "SPECTATING" label."""


Scene = Lobby | Hud | Results | Spectating


@dataclass(frozen=True)
class Shot:
    scene: Scene
    start: float
    end: float


@dataclass(frozen=True)
class AudioClip:
    path: Path
    at: float = 0.0


@dataclass(frozen=True)
class RecordingSpec:
    """Shots must be sorted and non-overlapping; gaps between them show the Lobby."""

    duration: float
    shots: Sequence[Shot] = ()
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    audio: Sequence[AudioClip] = ()
    keyframe_interval: float = 4.0
    crf: int = 18

    def __post_init__(self) -> None:
        object.__setattr__(self, "shots", tuple(self.shots))
        object.__setattr__(self, "audio", tuple(self.audio))
        for name in ("duration", "fps", "keyframe_interval"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {value}")
        if round(self.duration * self.fps) < 1:
            raise ValueError(
                f"duration {self.duration}s at {self.fps} fps yields no frames; "
                "make it at least one frame long"
            )
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError(
                f"width and height must be positive and even, got {self.width}x{self.height}"
            )
        previous_end = 0.0
        for index, shot in enumerate(self.shots):
            label = f"shot {index} ({type(shot.scene).__name__} {shot.start}-{shot.end})"
            if not shot.start < shot.end:
                raise ValueError(f"{label}: start must be before end")
            if shot.start < previous_end:
                raise ValueError(f"{label}: overlaps or precedes the previous shot")
            if shot.end > self.duration:
                raise ValueError(f"{label}: ends after the Recording duration {self.duration}")
            previous_end = shot.end
        for clip in self.audio:
            if not isfinite(clip.at) or clip.at < 0:
                raise ValueError(
                    f"audio clip {clip.path.name}: offset must be finite and non-negative"
                )
            if clip.at >= self.duration:
                raise ValueError(
                    f"audio clip {clip.path.name}: starts at {clip.at}s, "
                    f"after the Recording duration {self.duration}"
                )
            if not clip.path.is_file():
                raise ValueError(f"audio clip {clip.path}: file does not exist")


def match_shots(
    start: float,
    hud_steps: Sequence[tuple[float, int | None, int | None, int | None]],
    results_at: float,
    results_duration: float = 5.0,
    placement: int = 1,
    results_style: ResultsStyle = "classic",
    hud_design: HudDesign = "icon_column",
) -> tuple[Shot, ...]:
    """Expand counter steps into consecutive HUD shots followed by one results shot.

    Each step is (seconds after `start`, personal eliminations, team eliminations, players left)
    and holds until the next step; the last one holds until `results_at`.
    """
    if not hud_steps:
        raise ValueError("a Match needs at least one HUD step")
    boundaries = [start + step[0] for step in hud_steps] + [results_at]
    shots = [
        Shot(Hud(personal, team, left, design=hud_design), boundaries[i], boundaries[i + 1])
        for i, (_, personal, team, left) in enumerate(hud_steps)
    ]
    shots.append(Shot(Results(placement, results_style), results_at, results_at + results_duration))
    return tuple(shots)
