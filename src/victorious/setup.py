"""Setup: finding every HUD element in a Recording and assembling them into a Layout."""

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from victorious.counters import COUNTER_ELEMENTS, best_reading
from victorious.detect import Found, find_hud, find_results
from victorious.layout import (
    DEFAULT_HUD_DESIGN,
    HudDesign,
    HudScreen,
    Layout,
    Rect,
    ResultsScreen,
    Screens,
    aspect_ratio_label,
)
from victorious.screens import (
    DESIGNS,
    HUD_THRESHOLD,
    RESULTS_THRESHOLD,
    ResultsReading,
    hud_reading,
    hud_score,
    results_reading,
    scale_range,
)
from victorious.video import RecordingError, VideoInfo, frame_at, probe

AT_KINDS = ("hud", "results")
HUD_ELEMENTS = ("hud_anchor", *COUNTER_ELEMENTS)
RESULTS_ELEMENTS = ("results_banner", "placement", "elimination_placement")
ELEMENTS_OF = {"hud": HUD_ELEMENTS, "results": RESULTS_ELEMENTS}
REQUIRED = (*HUD_ELEMENTS, "results_banner", "placement")
OPTIONAL_HINT = (
    "pass a second --at results= on a screen that shows YOU PLACED #N to find it, or use --manual"
)
ROUNDING = 1e-4
SCALE_SLACK = 1e-3
NEIGHBOUR_REACH = 8
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
    design: HudDesign | None = None

    @property
    def frame(self) -> np.ndarray:
        return self.frames[0]

    def keeping(self, names: Collection[str]) -> "Panel":
        kept = {name: found for name, found in self.found.items() if name in names}
        return Panel(self.kind, self.second, self.frames, kept, self.design)


@dataclass(frozen=True)
class SetupResult:
    layout: Layout | None
    panels: list[Panel]
    missing: dict[str, str]
    warnings: list[str]

    @property
    def found(self) -> int:
        return sum(len(panel.found) for panel in self.panels)


def _in_scale(element: str, found: Found) -> bool:
    low, high = scale_range(element)
    return low - SCALE_SLACK <= found.scale_factor <= high + SCALE_SLACK


def _better(element: str, found: Found, current: Found | None) -> bool:
    """Whether `found` beats `current`: a size `analyse` reads first, then the score."""
    if current is None:
        return True
    return (_in_scale(element, found), found.score) > (_in_scale(element, current), current.score)


def _scale_note(element: str, found: Found) -> str:
    if _in_scale(element, found):
        return ""
    low, high = scale_range(element)
    return (
        f"; it is {found.scale_factor:.2f}x the reference size and analyse only reads "
        f"{low:.1f}x to {high:.1f}x"
    )


def _read_back(
    frame: np.ndarray, elements: Collection[str], layout: Layout
) -> dict[str, tuple[float, bool]]:
    """How well `analyse` reads each of `elements` from `frame`, and whether that is enough."""
    outcomes: dict[str, tuple[float, bool]] = {}
    results: ResultsReading | None = None
    hud = hud_reading(frame, layout)
    for element in elements:
        if element == "hud_anchor":
            outcomes[element] = hud.score, hud.score >= HUD_THRESHOLD
        elif element in COUNTER_ELEMENTS:
            reading = best_reading(hud.counter_masks(frame, element))
            outcomes[element] = (reading.confidence, True) if reading is not None else (0.0, False)
        else:
            results = results or results_reading(frame, layout, read_placement=False)
            score = {
                "results_banner": results.banner,
                "placement": results.placement_score,
                "elimination_placement": results.elimination_placement_score,
            }[element]
            outcomes[element] = score, score >= RESULTS_THRESHOLD
    return outcomes


def _self_check(layout: Layout, panels: Sequence[Panel]) -> dict[str, str]:
    """Reads each element back as `analyse` will; the elements that fail, with why."""
    problems: dict[str, str] = {}
    for panel in panels:
        per_frame = [_read_back(frame, panel.found, layout) for frame in panel.frames]
        for element, found in panel.found.items():
            outcomes = [frame_outcomes[element] for frame_outcomes in per_frame]
            if not any(ok for _, ok in outcomes):
                best = max(score for score, _ in outcomes)
                problems[element] = (
                    f"the Layout does not read back at {panel.kind}={panel.second} "
                    f"(score {best:.2f}){_scale_note(element, found)}"
                )
    return problems


def _build_layout(
    width: int, height: int, found: dict[str, Found], design: HudDesign = DEFAULT_HUD_DESIGN
) -> Layout | None:
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
    return Layout(aspect_ratio=label, hud_design=design, screens=Screens(hud=hud, results=results))


def _best_per_element(panels: Sequence[Panel]) -> dict[str, tuple[Found, int]]:
    best: dict[str, tuple[Found, int]] = {}
    for index, panel in enumerate(panels):
        for name, found in panel.found.items():
            if _better(name, found, best[name][0] if name in best else None):
                best[name] = (found, index)
    return best


def _assemble(
    width: int,
    height: int,
    candidates: list[Panel],
    details: dict[str, str],
    design: HudDesign,
) -> SetupResult:
    best = _best_per_element(candidates)
    panels = [
        panel.keeping({name for name, (_, index) in best.items() if index == k})
        for k, panel in enumerate(candidates)
    ]
    found = {name: pair[0] for name, pair in best.items()}
    missing = {name: details.get(name, "not found") for name in REQUIRED if name not in found}
    warnings: list[str] = []
    layout = _build_layout(width, height, found, design)
    if layout is not None:
        problems = _self_check(layout, panels)
        if "elimination_placement" in problems:
            found.pop("elimination_placement")
            details["elimination_placement"] = problems.pop("elimination_placement")
            panels = [panel.keeping(found) for panel in panels]
            layout = _build_layout(width, height, found, design)
        if problems:
            missing.update(problems)
            layout = None
    if layout is not None and "elimination_placement" not in found:
        reason = details.get("elimination_placement", "not found")
        warnings.append(f"elimination_placement: {reason}; {OPTIONAL_HINT}")
    return SetupResult(layout, panels, missing, warnings)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and abs(value) < 1e6


def _snapped(origin: float, extent: float) -> tuple[float, float]:
    """Both rounded to 4 decimals; the extent is kept in the frame if only rounding left it out."""
    start, length = round(origin, 4), round(extent, 4)
    if start + length > 1 and origin + extent <= 1 + ROUNDING:
        length = round(1 - start, 4)
    return start, length


def _submitted_rects(
    panels: Sequence[Panel], elements: dict[str, Any]
) -> tuple[list[dict[str, Found]], list[str]]:
    placed: list[dict[str, Found]] = [{} for _ in panels]
    problems: list[str] = []
    for name, entry in elements.items():
        if not isinstance(entry, dict):
            problems.append(f"{name}: expected an object with panel, x, y, w and h")
            continue
        index = entry.get("panel")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(panels):
            problems.append(f"{name}: panel must be one of 0..{len(panels) - 1}, got {index!r}")
            continue
        kind = panels[index].kind
        if not any(name in names for names in ELEMENTS_OF.values()):
            problems.append(f"{name}: not a HUD element")
            continue
        if name not in ELEMENTS_OF[kind]:
            problems.append(f"{name}: is not on a {kind} screen, panel {index} is a {kind} frame")
            continue
        if not all(_is_number(entry.get(key)) for key in "xywh"):
            problems.append(f"{name}: x, y, w and h must be finite numbers")
            continue
        left, width = _snapped(float(entry["x"]), float(entry["w"]))
        top, height = _snapped(float(entry["y"]), float(entry["h"]))
        try:
            rect = Rect(x=left, y=top, w=width, h=height)
        except ValidationError as error:
            reasons = "; ".join(
                issue["msg"].removeprefix("Value error, ") for issue in error.errors()
            )
            problems.append(f"{name}: {reasons}")
            continue
        placed[index][name] = Found(rect, 1.0, 1.0)
    return placed, problems


def _drawn_design(
    width: int, height: int, found: dict[str, Found], panels: Sequence[Panel]
) -> HudDesign:
    """The HUD design whose icons are under the anchor drawn by hand, the default one unless
    another is clearly on screen."""
    frames = [frame for panel in panels if panel.kind == "hud" for frame in panel.frames]

    def anchor_score(design: HudDesign) -> float:
        layout = _build_layout(width, height, found, design)
        return max((hud_score(frame, layout) for frame in frames), default=0.0) if layout else 0.0

    scores = {design: anchor_score(design) for design in DESIGNS}
    best = max(scores, key=lambda design: scores[design])
    return best if scores[best] >= HUD_THRESHOLD else DEFAULT_HUD_DESIGN


def manual_result(panels: Sequence[Panel], submitted: object) -> SetupResult | list[str]:
    """The Layout drawn by hand on `panels`, or every reason it cannot be one.

    `submitted` is the decoded JSON of the page: `{"elements": {name: {panel, x, y, w, h}}}`.
    The Layout is returned even when a rectangle does not read back; those become warnings.
    """
    elements = submitted.get("elements") if isinstance(submitted, dict) else None
    if not isinstance(elements, dict):
        return ["elements: expected an object of element name to rectangle"]
    placed, problems = _submitted_rects(panels, elements)
    found = {name: hit for names in placed for name, hit in names.items()}
    problems.extend(f"{name}: required, not placed" for name in REQUIRED if name not in found)
    height, width = panels[0].frame.shape[:2]
    design = _drawn_design(width, height, found, panels) if not problems else DEFAULT_HUD_DESIGN
    layout = None if problems else _build_layout(width, height, found, design)
    if layout is None:
        return problems
    drawn = [
        Panel(
            panel.kind, panel.second, panel.frames, names, design if panel.kind == "hud" else None
        )
        for panel, names in zip(panels, placed, strict=True)
    ]
    warnings = [f"{name}: {reason}" for name, reason in _self_check(layout, drawn).items()]
    return SetupResult(layout, drawn, {}, warnings)


def _neighbour_frames(
    recording: Path, info: VideoInfo, second: int, offset: int
) -> list[np.ndarray]:
    """The frames `offset` s either side of `second`; one that cannot be decoded is skipped."""
    frames: list[np.ndarray] = []
    for neighbour in (second - offset, second + offset):
        if not 0 <= neighbour < info.duration:
            continue
        try:
            frame = frame_at(recording, info, neighbour)
        except RecordingError:
            continue
        if frame is not None:
            frames.append(frame)
    return frames


def _detect_panel(
    recording: Path,
    info: VideoInfo,
    kind: str,
    second: int,
    details: dict[str, str],
    design: HudDesign | None,
) -> Panel:
    """Reads `second` and its neighbours; on a results screen, reaches further out while nothing
    is found or an element is at a size `analyse` does not read, since the text animates.

    A HUD panel settles on the design of the first frame that shows its icons; `design` is the
    one an earlier panel settled on, which this one then has to match.
    """
    reference = frame_at(recording, info, second)
    if reference is None:
        raise RecordingError(f"cannot read a frame at {second} s of {recording}")
    frames = [reference]
    found: dict[str, Found] = {}
    design = _merge(found, kind, second, reference, details, design)
    reach = NEIGHBOUR_REACH if kind == "results" else 1
    for offset in range(1, reach + 1):
        if offset > 1 and found and all(_in_scale(name, hit) for name, hit in found.items()):
            break
        for frame in _neighbour_frames(recording, info, second, offset):
            before = dict(found)
            design = _merge(found, kind, second, frame, details, design)
            if found != before or offset == 1:
                frames.append(frame)
    return Panel(kind, second, tuple(frames), found, design)


def _merge(
    found: dict[str, Found],
    kind: str,
    second: int,
    frame: np.ndarray,
    details: dict[str, str],
    design: HudDesign | None,
) -> HudDesign | None:
    """Keeps the better detections of `frame`; the HUD design, once a frame shows one."""
    detection = find_hud(frame, design) if kind == "hud" else find_results(frame)
    for name, hit in detection.found.items():
        if _better(name, hit, found.get(name)):
            found[name] = hit
    for name, reason in detection.missing.items():
        details.setdefault(name, f"{reason} at {kind}={second}")
    if kind == "hud" and design is None and "hud_anchor" in detection.found:
        return detection.design
    return design


def run_setup(recording: Path, at: Sequence[tuple[str, int]]) -> SetupResult:
    """Locate every HUD element on the seconds given; see `SetupResult.layout`."""
    info = probe(recording)
    candidates: list[Panel] = []
    details: dict[str, str] = {}
    design: HudDesign | None = None
    for kind, second in at:
        if second >= info.duration:
            raise RecordingError(
                f"--at {kind}={second} is past the end of {recording} ({info.duration:.1f} s long)"
            )
        panel = _detect_panel(recording, info, kind, second, details, design)
        design = panel.design or design
        candidates.append(panel)
    return _assemble(info.width, info.height, candidates, details, design or DEFAULT_HUD_DESIGN)
