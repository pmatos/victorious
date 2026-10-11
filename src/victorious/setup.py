"""Setup: finding every HUD element in a Recording and assembling them into a Layout."""

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from victorious.counters import COUNTER_ELEMENTS, best_reading
from victorious.detect import Found, find_hud, find_results
from victorious.layout import (
    HudScreen,
    Layout,
    ResultsScreen,
    Screens,
    aspect_ratio_label,
)
from victorious.screens import (
    HUD_THRESHOLD,
    RESULTS_THRESHOLD,
    SCALE_SWEEP,
    counter_masks,
    hud_score,
    results_reading,
)
from victorious.video import RecordingError, VideoInfo, frame_at, probe

AT_KINDS = ("hud", "results")
HUD_ELEMENTS = ("hud_anchor", *COUNTER_ELEMENTS)
REQUIRED = (*HUD_ELEMENTS, "results_banner", "placement")
OPTIONAL_HINT = (
    "pass a second --at results= on a screen that shows YOU PLACED #N to find it, or use --manual"
)
_CLOCK = re.compile(r"\d+(:[0-5]?\d){1,2}(\.\d+)?|\d+(\.\d+)?")


class TimeError(ValueError):
    """An `--at` value that is not a time."""


def parse_time(text: str) -> int:
    """Whole seconds from "754", "754.5", "12:34" or "1:02:03"."""
    text = text.strip()
    if not _CLOCK.fullmatch(text):
        raise TimeError(f"not a time: {text!r} (use seconds, MM:SS or H:MM:SS)")
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    if not math.isfinite(seconds):
        raise TimeError(f"not a time: {text!r}")
    return int(seconds)


def parse_at(text: str) -> tuple[str, int]:
    """One `--at KIND=TIME` argument."""
    kind, separator, time = text.partition("=")
    if not separator:
        raise TimeError(f"expected KIND=TIME, e.g. hud=754, got {text!r}")
    if kind not in AT_KINDS:
        raise TimeError(f"unknown screen kind {kind!r}; use one of {', '.join(AT_KINDS)}")
    return kind, parse_time(time)


@dataclass(frozen=True)
class Panel:
    """One second the user pointed at, with the neighbouring frames that back its readings."""

    kind: str
    second: int
    frames: tuple[np.ndarray, ...]
    found: dict[str, Found] = field(default_factory=dict)

    @property
    def frame(self) -> np.ndarray:
        return self.frames[0]

    def keeping(self, names: Collection[str]) -> "Panel":
        kept = {name: found for name, found in self.found.items() if name in names}
        return Panel(self.kind, self.second, self.frames, kept)


@dataclass(frozen=True)
class SetupResult:
    layout: Layout | None
    panels: list[Panel]
    missing: dict[str, str]
    warnings: list[str]

    @property
    def found(self) -> int:
        return sum(len(panel.found) for panel in self.panels)


def _scale_note(found: Found) -> str:
    low, high = min(SCALE_SWEEP), max(SCALE_SWEEP)
    if low <= found.scale_factor <= high:
        return ""
    return (
        f"; it is {found.scale_factor:.2f}x the reference size and analyse only reads "
        f"{low:.1f}x to {high:.1f}x"
    )


def _read_back(frame: np.ndarray, element: str, layout: Layout) -> tuple[float, bool]:
    """How well `analyse` reads `element` from `frame`, and whether that is good enough."""
    if element == "hud_anchor":
        score = hud_score(frame, layout)
        return score, score >= HUD_THRESHOLD
    if element in COUNTER_ELEMENTS:
        rect = getattr(layout.screens.hud, element)
        reading = best_reading(counter_masks(frame, rect))
        return (reading.confidence, True) if reading is not None else (0.0, False)
    reading_of = results_reading(frame, layout, read_placement=False)
    score = {
        "results_banner": reading_of.banner,
        "placement": reading_of.placement_score,
        "elimination_placement": reading_of.elimination_placement_score,
    }[element]
    return score, score >= RESULTS_THRESHOLD


def _self_check(layout: Layout, panels: Sequence[Panel]) -> dict[str, str]:
    """Reads each element back as `analyse` will; the elements that fail, with why."""
    problems: dict[str, str] = {}
    for panel in panels:
        for element, found in panel.found.items():
            outcomes = [_read_back(frame, element, layout) for frame in panel.frames]
            if not any(ok for _, ok in outcomes):
                best = max(score for score, _ in outcomes)
                problems[element] = (
                    f"the Layout does not read back at {panel.kind}={panel.second} "
                    f"(score {best:.2f}){_scale_note(found)}"
                )
    return problems


def _build_layout(width: int, height: int, found: dict[str, Found]) -> Layout | None:
    if any(name not in found for name in REQUIRED):
        return None
    elimination_placement = found.get("elimination_placement")
    hud = HudScreen(
        hud_anchor=found["hud_anchor"].rect,
        personal_eliminations=found["personal_eliminations"].rect,
        team_eliminations=found["team_eliminations"].rect,
        players_left=found["players_left"].rect,
    )
    results = ResultsScreen(
        results_banner=found["results_banner"].rect,
        placement=found["placement"].rect,
        elimination_placement=elimination_placement.rect if elimination_placement else None,
    )
    label = aspect_ratio_label(width, height)
    return Layout(aspect_ratio=label, screens=Screens(hud=hud, results=results))


def _best_per_element(panels: Sequence[Panel]) -> dict[str, tuple[Found, int]]:
    best: dict[str, tuple[Found, int]] = {}
    for index, panel in enumerate(panels):
        for name, found in panel.found.items():
            if name not in best or found.score > best[name][0].score:
                best[name] = (found, index)
    return best


def _assemble(
    width: int, height: int, candidates: list[Panel], details: dict[str, str]
) -> SetupResult:
    best = _best_per_element(candidates)
    panels = [
        panel.keeping({name for name, (_, index) in best.items() if index == k})
        for k, panel in enumerate(candidates)
    ]
    found = {name: pair[0] for name, pair in best.items()}
    missing = {name: details.get(name, "not found") for name in REQUIRED if name not in found}
    warnings: list[str] = []
    layout = _build_layout(width, height, found)
    if layout is not None:
        problems = _self_check(layout, panels)
        if "elimination_placement" in problems:
            found.pop("elimination_placement")
            details["elimination_placement"] = problems.pop("elimination_placement")
            panels = [panel.keeping(found) for panel in panels]
            layout = _build_layout(width, height, found)
        if problems:
            missing.update(problems)
            layout = None
    if layout is not None and "elimination_placement" not in found:
        reason = details.get("elimination_placement", "not found")
        warnings.append(f"elimination_placement: {reason}; {OPTIONAL_HINT}")
    return SetupResult(layout, panels, missing, warnings)


def _neighbour_frames(recording: Path, info: VideoInfo, second: int) -> list[np.ndarray]:
    """The frames either side of `second`; a neighbour that cannot be decoded is skipped."""
    frames: list[np.ndarray] = []
    for neighbour in (second - 1, second + 1):
        if not 0 <= neighbour < info.duration:
            continue
        try:
            frame = frame_at(recording, info, neighbour)
        except RecordingError:
            continue
        if frame is not None:
            frames.append(frame)
    return frames


def run_setup(recording: Path, at: Sequence[tuple[str, int]]) -> SetupResult:
    """Locate every HUD element on the seconds given; see `SetupResult.layout`."""
    info = probe(recording)
    candidates: list[Panel] = []
    details: dict[str, str] = {}
    for kind, second in at:
        if second >= info.duration:
            raise RecordingError(
                f"--at {kind}={second} is past the end of {recording} ({info.duration:.1f} s long)"
            )
        reference = frame_at(recording, info, second)
        if reference is None:
            raise RecordingError(f"cannot read a frame at {second} s of {recording}")
        readable = (reference, *_neighbour_frames(recording, info, second))
        found: dict[str, Found] = {}
        for frame in readable:
            detection = find_hud(frame) if kind == "hud" else find_results(frame)
            for name, hit in detection.found.items():
                if name not in found or hit.score > found[name].score:
                    found[name] = hit
            for name, reason in detection.missing.items():
                details.setdefault(name, f"{reason} at {kind}={second}")
        candidates.append(Panel(kind, second, readable, found))
    return _assemble(info.width, info.height, candidates, details)
