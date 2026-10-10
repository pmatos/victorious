"""The structure of `analyse` output, and its exported JSON Schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

EventKind = Literal["match_start", "win", "final_placement", "match_end"]
EventSource = Literal["frame", "speech"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Event(_Model):
    kind: EventKind
    time: float = Field(ge=0, description="Seconds from the start of the Recording.")
    source: EventSource
    confidence: float = Field(ge=0, le=1)
    value: int | None = Field(description="The placement for final_placement, otherwise null.")


class Match(_Model):
    index: int = Field(ge=0)
    start: float = Field(ge=0, description="Seconds from the start of the Recording.")
    end: float = Field(ge=0, description="Seconds from the start of the Recording.")
    win: bool
    placement: int | None = Field(ge=1, le=100)
    events: list[Event]


class RecordingInfo(_Model):
    path: str
    duration: float = Field(ge=0, description="Seconds.")


class AnalysisOutput(_Model):
    schema_version: Literal["1.0"] = SCHEMA_VERSION  # type: ignore[assignment]
    recording: RecordingInfo
    matches: list[Match]


def json_schema() -> dict[str, Any]:
    document = AnalysisOutput.model_json_schema(mode="serialization")
    document["schema_version"] = SCHEMA_VERSION
    return document
