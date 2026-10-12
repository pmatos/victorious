"""When a Recording started, from its file name or its container metadata."""

import re
from datetime import datetime
from pathlib import Path

_FILE_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})-(\d{2})-(\d{2})")


def _from_file_name(path: Path) -> str | None:
    found = _FILE_NAME.search(path.stem)
    if found is None:
        return None
    year, month, day, hour, minute, second = (int(part) for part in found.groups())
    try:
        return datetime(year, month, day, hour, minute, second).isoformat()
    except ValueError:
        return None


def _from_container(creation_time: str | None) -> str | None:
    if not creation_time:
        return None
    try:
        return datetime.fromisoformat(creation_time).isoformat()
    except ValueError:
        return None


def started_at(path: Path, creation_time: str | None) -> str | None:
    """ISO 8601 start of the Recording, or None when neither source gives one.

    OBS names files after the start in local time, so the name (naive local time) wins over the
    container's `creation_time`, which keeps its UTC offset and can be a re-mux time.
    """
    return _from_file_name(path) or _from_container(creation_time)
