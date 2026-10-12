"""The labels file: the verdicts of a review, and how a later run is compared against them."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from victorious.schema import AnalysisOutput, Match
from victorious.versioned import load_versioned

LABELS_VERSION = "1.0"
MIN_OVERLAP = 0.5
DEFAULT_TOLERANCE = 2.0
DURATION_TOLERANCE = 1.0
VALUE_NAMES = ("win", "placement", "personal_eliminations")

Verdict = Literal["confirmed", "rejected"]
DifferenceKind = Literal["missing", "changed", "shifted", "reappeared", "unlabelled"]


class LabelsError(Exception):
    """The labels file cannot be used."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelledMatch(_Model):
    start: float = Field(ge=0, description="Seconds from the start of the Recording.")
    end: float = Field(ge=0, description="Seconds from the start of the Recording.")
    verdict: Verdict
    win: bool
    placement: int | None
    personal_eliminations: int | None


class LabelsRecording(_Model):
    path: str
    duration: float = Field(ge=0)


class LabelsAnalysis(_Model):
    schema_version: str
    layout: str | None


class Labels(_Model):
    labels_version: Literal["1.0"] = LABELS_VERSION  # type: ignore[assignment]
    recording: LabelsRecording
    analysis: LabelsAnalysis
    matches: list[LabelledMatch]


@dataclass(frozen=True)
class Difference:
    kind: DifferenceKind
    start: float
    end: float
    detail: str


def labels_from(output: AnalysisOutput, verdicts: Mapping[int, Verdict]) -> Labels:
    """The labels of `output` with the verdict of each Match, by its `index`."""
    return Labels(
        recording=LabelsRecording(path=output.recording.path, duration=output.recording.duration),
        analysis=LabelsAnalysis(
            schema_version=output.schema_version,
            layout=output.layout.path if output.layout is not None else None,
        ),
        matches=[
            LabelledMatch(
                start=match.start,
                end=match.end,
                verdict=verdicts[match.index],
                win=match.win,
                placement=match.placement,
                personal_eliminations=match.personal_eliminations,
            )
            for match in output.matches
        ],
    )


def load_labels(path: Path) -> Labels:
    return load_versioned(
        path,
        Labels,
        noun="labels file",
        key="labels_version",
        version=LABELS_VERSION,
        remedy="review the analysis again",
        error=LabelsError,
    )


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Intersection over union of two spans of seconds."""
    union = max(first[1], second[1]) - min(first[0], second[0])
    if union <= 0:
        return 1.0 if first[0] == second[0] else 0.0
    shared = min(first[1], second[1]) - max(first[0], second[0])
    return max(0.0, shared) / union


def _pairs(labels: Sequence[LabelledMatch], matches: Sequence[Match]) -> dict[int, int]:
    """Label position to Match position: the best overlaps first, each used once."""
    candidates = sorted(
        (-_overlap((label.start, label.end), (match.start, match.end)), a, b)
        for a, label in enumerate(labels)
        for b, match in enumerate(matches)
    )
    taken_matches: set[int] = set()
    paired: dict[int, int] = {}
    for negative_overlap, a, b in candidates:
        if -negative_overlap < MIN_OVERLAP:
            break
        if a not in paired and b not in taken_matches:
            taken_matches.add(b)
            paired[a] = b
    return paired


def _values(item: LabelledMatch | Match) -> tuple[bool, int | None, int | None]:
    return item.win, item.placement, item.personal_eliminations


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def _compare_confirmed(label: LabelledMatch, match: Match, tolerance: float) -> list[Difference]:
    found: list[Difference] = []
    for name, expected, actual in zip(VALUE_NAMES, _values(label), _values(match), strict=True):
        if expected != actual:
            found.append(
                Difference(
                    "changed",
                    label.start,
                    label.end,
                    f"{name}: expected {expected}, found {actual}",
                )
            )
    moved = [
        f"{name} {_clock(was)} -> {_clock(now)}"
        for name, was, now in (("start", label.start, match.start), ("end", label.end, match.end))
        if abs(was - now) > tolerance
    ]
    if moved:
        found.append(Difference("shifted", label.start, label.end, ", ".join(moved)))
    return found


def _compare_rejected(label: LabelledMatch, match: Match) -> list[Difference]:
    if _values(label) == _values(match):
        return [
            Difference(
                "reappeared", label.start, label.end, "rejected before, and read the same again"
            )
        ]
    detail = (
        f"previously rejected as win={label.win} placement={label.placement} "
        f"personal_eliminations={label.personal_eliminations}, now win={match.win} "
        f"placement={match.placement} personal_eliminations={match.personal_eliminations}, "
        "re-review"
    )
    return [Difference("changed", label.start, label.end, detail)]


def compare(
    labels: Labels, output: AnalysisOutput, tolerance: float = DEFAULT_TOLERANCE
) -> list[Difference]:
    """How the Matches of `output` differ from `labels`, in time order.

    A Match and a label are the same Match when their spans overlap by at least half, not when
    their indices agree: indices shift when a Match appears or goes away.
    """
    _require_same_recording(labels, output)
    paired = _pairs(labels.matches, output.matches)
    found: list[Difference] = []
    for position, label in enumerate(labels.matches):
        if position not in paired:
            if label.verdict == "confirmed":
                found.append(
                    Difference("missing", label.start, label.end, "confirmed, and not found")
                )
            continue
        match = output.matches[paired[position]]
        if label.verdict == "confirmed":
            found.extend(_compare_confirmed(label, match, tolerance))
        else:
            found.extend(_compare_rejected(label, match))
    used = set(paired.values())
    found.extend(
        Difference("unlabelled", match.start, match.end, "found, and never reviewed")
        for position, match in enumerate(output.matches)
        if position not in used
    )
    return sorted(found, key=lambda difference: (difference.start, difference.end))


def _require_same_recording(labels: Labels, output: AnalysisOutput) -> None:
    named, found = Path(labels.recording.path).name, Path(output.recording.path).name
    if (
        named != found
        or abs(labels.recording.duration - output.recording.duration) > DURATION_TOLERANCE
    ):
        raise LabelsError(
            f"the labels are for Recording {named} ({labels.recording.duration:.0f} s), but the "
            f"analysis is of {found} ({output.recording.duration:.0f} s)"
        )


def format_differences(differences: Sequence[Difference], matches: int) -> str:
    if not differences:
        return f"no differences ({matches} matches)\n"
    lines = [
        f"{difference.kind:<10}  {_clock(difference.start)}-{_clock(difference.end)}  "
        f"{difference.detail}"
        for difference in differences
    ]
    count = len(differences)
    lines.append(f"{count} difference{'s' if count != 1 else ''} ({matches} matches)")
    return "\n".join(lines) + "\n"
