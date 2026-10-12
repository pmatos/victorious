"""Reading a JSON document that carries a version and validating it against a model."""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError


def load_versioned[Model: BaseModel](
    path: Path,
    model: type[Model],
    *,
    noun: str,
    key: str,
    version: str,
    remedy: str,
    error: type[Exception],
) -> Model:
    """The document at `path` as `model`; every failure is raised as `error` naming `noun`."""
    try:
        document = json.loads(path.read_text())
    except OSError as cause:
        raise error(f"cannot read {noun} {path}: {cause.strerror or cause}") from cause
    except (ValueError, RecursionError) as cause:
        raise error(f"{noun} {path} is not JSON: {cause}") from cause
    found = document.get(key) if isinstance(document, dict) else None
    if found != version:
        raise error(
            f"{noun} {path} has {key} {found!r}, this victorious reads {version!r}; {remedy}"
        )
    try:
        return model.model_validate(document)
    except ValidationError as cause:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
            for problem in cause.errors()
        )
        raise error(f"{noun} {path} is not valid: {problems}") from cause
