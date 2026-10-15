"""The structure of `analyse` output, and its exported JSON Schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from victorious.layout import Rect

SCHEMA_VERSION = "1.7"

CounterElement = Literal["personal_eliminations", "team_eliminations", "players_left"]
EventKind = Literal[
    "match_start",
    "win",
    "final_placement",
    "eliminated",
    "match_end",
    "personal_eliminations",
    "team_eliminations",
    "players_left",
    "unreadable",
    "transcript",
]
EventSource = Literal["frame", "speech"]
SpeechLanguage = Literal["en", "pt"]
Category = Literal["personal", "game_related"]
ObservedElement = Literal[
    "hud_anchor",
    "personal_eliminations",
    "team_eliminations",
    "players_left",
    "results_banner",
    "placement",
    "elimination_placement",
    "spectating",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Event(_Model):
    kind: EventKind
    time: float = Field(ge=0, description="Seconds from the start of the Recording.")
    source: EventSource
    confidence: float = Field(ge=0, le=1)
    value: int | None = Field(
        description=(
            "The placement for final_placement, the new count for a counter, the placement on "
            "the YOU PLACED screen for eliminated (null for one detected while the squad plays "
            "on, even if the squad is later out too), else null."
        )
    )
    end: float | None = Field(
        ge=0,
        description=(
            "Seconds from the start of the Recording the span ends, for unreadable and transcript."
        ),
    )
    element: CounterElement | None = Field(description="Which counter, for unreadable only.")
    text: str | None = Field(description="What was said, for transcript only.")
    language: SpeechLanguage | None = Field(
        description="The language spoken, English or Portuguese, for transcript only."
    )
    speaker: str | None = Field(
        description=(
            "Who spoke, for transcript only: the name of the enrolled voice it matched, else "
            "unknown_N, numbered from 1 in order of first appearance within the Recording."
        )
    )
    category: Category | None = Field(
        description=(
            "Whether what was said is personal or about the game, for transcript only; null "
            "when it was not judged (no API key, or the categorization failed)."
        )
    )
    category_confidence: float | None = Field(
        ge=0,
        le=1,
        description="The categorization model's confidence in `category`, null when it is null.",
    )


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))


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
        confidence=_unit(confidence),
        value=value,
        end=end,
        element=element,
        text=None,
        language=None,
        speaker=None,
        category=None,
        category_confidence=None,
    )


def transcript_event(
    text: str,
    language: SpeechLanguage,
    start: float,
    end: float,
    confidence: float,
    speaker: str | None = None,
) -> Event:
    return Event(
        kind="transcript",
        time=start,
        source="speech",
        confidence=_unit(confidence),
        value=None,
        end=end,
        element=None,
        text=text,
        language=language,
        speaker=speaker,
        category=None,
        category_confidence=None,
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
    started_at: str | None = Field(
        description=(
            "ISO 8601 time the Recording started: the local time in its file name (no offset), "
            "else the container's creation time (with its UTC offset), else null."
        )
    )


class ObservationRecord(_Model):
    """One line of the `--observations` file: a HUD element read at one sampled second."""

    second: int = Field(ge=0, description="Whole seconds from the start of the Recording.")
    element: ObservedElement
    value: int | None = Field(description="The digits read, null when none were read.")
    confidence: float = Field(ge=0, le=1, description="OCR confidence of the digits, else 0.")
    score: float | None = Field(description="Template match score; null for counters.")
    region: Rect


class MaskedAudioRecord(_Model):
    """One line of the `--observations` file: a span of audio withheld from transcription."""

    element: Literal["music_mask"]
    start: float = Field(ge=0, description="Seconds from the start of the Recording.")
    end: float = Field(ge=0, description="Seconds from the start of the Recording.")
    music: float = Field(ge=0, le=1, description="Mean per-second music score over the span.")
    speech: float = Field(ge=0, le=1, description="Mean per-second speech score over the span.")


class ChosenLayout(_Model):
    path: str = Field(description="The Layout file used, as given on the command line.")
    score: float = Field(
        ge=0, le=1, description="Mean match of its HUD anchor over the in-match frames sampled."
    )


class AnalysisWarning(_Model):
    code: Literal["not_fortnite"]
    message: str


class AnalysisOutput(_Model):
    schema_version: Literal["1.7"] = SCHEMA_VERSION  # type: ignore[assignment]
    recording: RecordingInfo
    layout: ChosenLayout | None = Field(
        description="The Layout used, null when no in-match HUD was found to choose one."
    )
    warnings: list[AnalysisWarning]
    matches: list[Match]


def json_schema() -> dict[str, Any]:
    document = AnalysisOutput.model_json_schema(mode="serialization")
    document["schema_version"] = SCHEMA_VERSION
    return document
