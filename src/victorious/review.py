"""Reviewing the Matches of an analysis: load it, show each Match, save the verdicts."""

import json
from pathlib import Path

from pydantic import ValidationError

from victorious.schema import SCHEMA_VERSION, AnalysisOutput


class ReviewError(Exception):
    """The analysis cannot be reviewed."""


def load_analysis(path: Path) -> AnalysisOutput:
    try:
        document = json.loads(path.read_text())
    except OSError as error:
        raise ReviewError(f"cannot read analysis {path}: {error.strerror or error}") from error
    except ValueError as error:
        raise ReviewError(f"analysis {path} is not JSON: {error}") from error
    version = document.get("schema_version") if isinstance(document, dict) else None
    if version != SCHEMA_VERSION:
        raise ReviewError(
            f"analysis {path} has schema_version {version!r}, this victorious reads "
            f"{SCHEMA_VERSION!r}; run analyse again"
        )
    try:
        return AnalysisOutput.model_validate(document)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
            for problem in error.errors()
        )
        raise ReviewError(f"analysis {path} is not valid: {problems}") from error
