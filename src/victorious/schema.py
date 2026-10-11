"""The structure of `analyse` output, and its exported JSON Schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from victorious.layout import Rect

SCHEMA_VERSION = "1.2"

CounterElement = Literal["personal_eliminations", "team_eliminations", "players_left"]
EventKind = Literal[
    "match_start",
    "win",
    "final_placement",
    "match_end",
    "personal_eliminations",
    "team_eliminations",
    "players_left",
    "unreadable",
]
EventSource = Literal["frame", "speech"]
ObservedElement = Literal[
    "hud_anchor",
    "personal_eliminations",
    "team_eliminations",
    "players_left",
    "results_banner",
    "placement",
    "elimination_placement",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Event(_Model):
    kind: EventKind
    time: float = Field(ge=0, description="Seconds from the start of the Recording.")
    source: EventSource
    confidence: float = Field(ge=0, le=1)
    value: int | None = Field(
        description="The placement for final_placement, the new count for a counter, else null."
    )
    end: float | None = Field(
        ge=0, description="Seconds from the start of the Recording the span ends, for unreadable."
    )
    element: CounterElement | None = Field(description="Which counter, for unreadable only.")


def frame_event(
    kind: EventKind,
    time: float,
    confidence: float,
    value: int | None = None,
    end: float | None = None,
    element: CounterElement | None = None,
) -> Event:
    return Event(
        kind=kind,
        time=time,
        source="frame",
        confidence=min(1.0, max(0.0, confidence)),
        value=value,
        end=end,
        element=element,
    )


class Match(_Model):
    index: int = Field(ge=0)
    start: float = Field(ge=0, description="Seconds from the start of the Recording.")
    end: float = Field(ge=0, description="Seconds from the start of the Recording.")
    win: bool
    placement: int | None = Field(ge=1, le=100)
    personal_eliminations: int | None = Field(
        ge=0, description="Personal eliminations at the end of the Match, null when not read."
    )
    events: list[Event]


class RecordingInfo(_Model):
    path: str
    duration: float = Field(ge=0, description="Seconds.")


class ObservationRecord(_Model):
    """One line of the `--observations` file: a HUD element read at one sampled second."""

    second: int = Field(ge=0, description="Whole seconds from the start of the Recording.")
    element: ObservedElement
    value: int | None = Field(description="The digits read, null when none were read.")
    confidence: float = Field(ge=0, le=1, description="OCR confidence of the digits, else 0.")
    score: float | None = Field(description="Template match score; null for counters.")
    region: Rect


class ChosenLayout(_Model):
    path: str = Field(description="The Layout file used, as given on the command line.")
    score: float = Field(
        ge=0, le=1, description="Mean match of its HUD anchor over the in-match frames sampled."
    )


class AnalysisOutput(_Model):
    schema_version: Literal["1.2"] = SCHEMA_VERSION  # type: ignore[assignment]
    recording: RecordingInfo
    layout: ChosenLayout
    matches: list[Match]


def json_schema() -> dict[str, Any]:
    document = AnalysisOutput.model_json_schema(mode="serialization")
    document["schema_version"] = SCHEMA_VERSION
    return document
