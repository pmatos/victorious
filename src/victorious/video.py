"""Reading a Recording: its properties and one frame per second."""

import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class RecordingError(Exception):
    """The Recording cannot be read."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float


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
        return VideoInfo(int(stream["width"]), int(stream["height"]), duration)
    except (StopIteration, KeyError, ValueError, TypeError) as error:
        raise RecordingError(f"cannot read Recording {path}: no video stream") from error


def sample_frames(path: Path, info: VideoInfo) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (whole seconds from the start, BGR frame), one per second."""
    frame_bytes = info.width * info.height * 3
    command = [
        _tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path),
        "-an", "-sn", "-vf", "fps=1:round=down", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    shape = (info.height, info.width, 3)
    with (
        tempfile.TemporaryFile() as errors,
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process,
    ):
        stream = process.stdout
        if stream is None:
            raise RecordingError("ffmpeg produced no output stream")
        second = 0
        truncated = False
        try:
            while data := stream.read(frame_bytes):
                if len(data) < frame_bytes:
                    truncated = True
                    break
                yield second, np.frombuffer(data, np.uint8).reshape(shape)
                second += 1
        except BaseException:
            process.kill()
            raise
        status = process.wait()
        if truncated or status != 0:
            errors.seek(0)
            detail = errors.read().decode(errors="replace").strip() or "truncated frame"
            raise RecordingError(f"cannot decode Recording {path}: {detail}")
