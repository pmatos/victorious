"""Reading a Recording: its properties and one frame per second."""

import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAIL_TOLERANCE = 2


class RecordingError(Exception):
    """The Recording cannot be read."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    creation_time: str | None = None


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RecordingError(f"{name} not found on PATH; install ffmpeg")
    return path


def probe(path: Path) -> VideoInfo:
    if not path.is_file():
        raise RecordingError(f"Recording not found: {path}")
    result = subprocess.run(
        [
            _tool("ffprobe"), "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    if result.returncode != 0:
        raise RecordingError(f"cannot read Recording {path}: {result.stderr.strip()}")
    try:
        document = json.loads(result.stdout)
        stream = next(s for s in document["streams"] if s.get("codec_type") == "video")
        duration = float(stream.get("duration") or document["format"]["duration"])
        tags = {str(k).lower(): v for k, v in (document["format"].get("tags") or {}).items()}
        created = tags.get("creation_time")
        return VideoInfo(
            int(stream["width"]),
            int(stream["height"]),
            duration,
            created if isinstance(created, str) else None,
        )
    except (StopIteration, KeyError, ValueError, TypeError) as error:
        raise RecordingError(f"cannot read Recording {path}: no video stream") from error


def _decode(path: Path, info: VideoInfo, command: list[str]) -> Iterator[np.ndarray]:
    frame_bytes = info.width * info.height * 3
    shape = (info.height, info.width, 3)
    with (
        tempfile.TemporaryFile() as errors,
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process,
    ):
        stream = process.stdout
        if stream is None:
            raise RecordingError("ffmpeg produced no output stream")
        truncated = False
        try:
            while data := stream.read(frame_bytes):
                if len(data) < frame_bytes:
                    truncated = True
                    break
                yield np.frombuffer(data, np.uint8).reshape(shape)
        except BaseException:
            process.kill()
            raise
        status = process.wait()
        if truncated or status != 0:
            errors.seek(0)
            detail = errors.read().decode(errors="replace").strip() or "truncated frame"
            raise RecordingError(f"cannot decode Recording {path}: {detail}")


def sample_chunk(
    path: Path, info: VideoInfo, start: int, length: int | None, threads: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (whole seconds from the start of the Recording, BGR frame), one per second.

    Covers [start, start + length), or to the end of the Recording when `length` is None. The
    boundaries are whole seconds, so the frames are the ones a single decode would sample.
    """
    seek = ["-ss", str(start), "-t", str(length)] if length is not None else ["-ss", str(start)]
    command = [
        _tool("ffmpeg"), "-v", "error", "-nostdin", "-threads", str(max(1, threads)), *seek,
        "-i", str(path), "-an", "-sn", "-vf", "fps=1:round=down",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    return ((start + k, frame) for k, frame in enumerate(_decode(path, info, command)))


def sample_window(
    path: Path, info: VideoInfo, start: float, end: float, rate: int
) -> Iterator[tuple[float, np.ndarray]]:
    """The frames of [start, end) at `rate` per second with the start of their slot.

    Like `sample_chunk`, each frame is the last one of its slot, so the frame of the slot
    starting at `t` shows the screen just before `t + 1 / rate`.
    """
    command = [
        _tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{start:g}", "-t", f"{end - start:g}",
        "-i", str(path), "-an", "-sn", "-vf", f"fps={rate}:round=down",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    frames = _decode(path, info, command)
    return ((start + k / rate, frame) for k, frame in enumerate(frames))


def frame_at(path: Path, info: VideoInfo, second: int) -> np.ndarray | None:
    """The frame `sample_chunk` yields for `second`, or None when the Recording ends before."""
    for _, frame in sample_chunk(path, info, second, 1, threads=1):
        return frame
    return None


def require_complete(path: Path, info: VideoInfo) -> None:
    """Raises when the Recording ends well before the duration its header claims."""
    last = max(0, math.floor(info.duration) - TAIL_TOLERANCE)
    if frame_at(path, info, last) is None:
        raise RecordingError(
            f"cannot read Recording {path}: it is truncated (no frame at {last} s, "
            f"its header claims {info.duration:.0f} s)"
        )
