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

from victorious import hwaccel
from victorious.hwaccel import HwDecoder

TAIL_TOLERANCE = 2
AUDIO_RATE = 16000
PROBE_AT = (0.4, 0.6)


class RecordingError(Exception):
    """The Recording cannot be read."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    creation_time: str | None = None
    audio_streams: int = 0
    codec: str = ""
    pix_fmt: str = ""


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
            sum(s.get("codec_type") == "audio" for s in document["streams"]),
            str(stream.get("codec_name") or ""),
            str(stream.get("pix_fmt") or ""),
        )
    except (StopIteration, KeyError, ValueError, TypeError) as error:
        raise RecordingError(f"cannot read Recording {path}: no video stream") from error


def _decode(
    path: Path, info: VideoInfo, command: list[str], env: dict[str, str] | None = None
) -> Iterator[np.ndarray]:
    frame_bytes = info.width * info.height * 3
    shape = (info.height, info.width, 3)
    with (
        tempfile.TemporaryFile() as errors,
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, env=env) as process,
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


def _seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _command(
    path: Path,
    start: float,
    length: float | None,
    rate: int,
    threads: int,
    hardware: HwDecoder | None,
) -> list[str]:
    """ffmpeg's frames of [start, start + length) at `rate` per second, as raw BGR."""
    seek = ["-ss", _seconds(start)] + (["-t", _seconds(length)] if length is not None else [])
    base = [_tool("ffmpeg"), "-v", "error", "-nostdin"]
    if hardware is None:
        pool = ["-threads", str(threads)] if threads > 0 else []
        return [
            *base, *pool, *seek, "-i", str(path), "-an", "-sn", "-vf", f"fps={rate}:round=down",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]  # fmt: skip
    return [
        *base, *hwaccel.input_options(hardware), *seek, "-i", str(path), "-an", "-sn",
        "-vf", f"fps={rate}:round=down,{hwaccel.DOWNLOAD_FILTER}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip


def _first_frame(
    path: Path, info: VideoInfo, at: int, hardware: HwDecoder | None
) -> np.ndarray | None:
    command = _command(path, at, 1, 1, 1, hardware)
    env = hwaccel.environment(hardware) if hardware is not None else None
    for frame in _decode(path, info, command, env):
        return frame
    return None


def _last_line(error: Exception) -> str:
    lines = str(error).splitlines()
    return lines[-1] if lines else "decode failed"


def _hardware_agrees(path: Path, info: VideoInfo, hardware: HwDecoder) -> str | None:
    """None when `hardware` decodes frames of the Recording byte for byte like software."""
    if info.pix_fmt != "yuv420p":
        return f"pixel format {info.pix_fmt or 'unknown'} is not 8-bit 4:2:0"
    try:
        for fraction in PROBE_AT:
            at = math.floor(info.duration * fraction)
            hard = _first_frame(path, info, at, hardware)
            soft = _first_frame(path, info, at, None) if hard is not None else None
            if soft is None or hard is None:
                return f"no frame at {at} s"
            if not np.array_equal(soft, hard):
                return "its frames differ from software decode"
    except RecordingError as error:
        return _last_line(error)
    return None


def resolve_hardware(path: Path, info: VideoInfo) -> HwDecoder | None:
    """The VAAPI device that decodes this kind of Recording exactly like software, if any.

    Probed on the first call of a process (two real frames of the Recording against software)
    and remembered by codec and size; a worker process gets the parent's answer instead.
    """
    key = (info.codec, info.pix_fmt, str(info.width), str(info.height))
    return hwaccel.resolve(key, lambda decoder: _hardware_agrees(path, info, decoder))


def _frames(
    path: Path,
    info: VideoInfo,
    start: float,
    length: float | None,
    rate: int,
    threads: int,
    allow_hardware: bool = True,
) -> Iterator[np.ndarray]:
    """Frames at `rate` per second, on the GPU when it works and in software when it stops.

    A hardware failure part-way is not retried from the start: the frames already given are
    kept and software resumes at the slot after the last of them.
    """
    done = 0
    hardware = resolve_hardware(path, info) if allow_hardware else None
    if hardware is not None:
        command = _command(path, start, length, rate, threads, hardware)
        try:
            for frame in _decode(path, info, command, hwaccel.environment(hardware)):
                done += 1
                yield frame
        except RecordingError as error:
            hwaccel.disable()
            hwaccel.note(f"hardware decode failed ({_last_line(error)})")
        else:
            return
    remaining = None if length is None else length - done / rate
    if remaining is not None and remaining <= 0:
        return
    command = _command(path, start + done / rate, remaining, rate, threads, None)
    yield from _decode(path, info, command)


def sample_chunk(
    path: Path,
    info: VideoInfo,
    start: int,
    length: int | None,
    threads: int,
    hardware: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (whole seconds from the start of the Recording, BGR frame), one per second.

    Covers [start, start + length), or to the end of the Recording when `length` is None. The
    boundaries are whole seconds, so the frames are the ones a single decode would sample.
    """
    frames = _frames(path, info, start, length, 1, max(1, threads), hardware)
    return ((start + k, frame) for k, frame in enumerate(frames))


def sample_window(
    path: Path, info: VideoInfo, start: float, end: float, rate: int
) -> Iterator[tuple[float, np.ndarray]]:
    """The frames of [start, end) at `rate` per second with the start of their slot.

    Like `sample_chunk`, each frame is the last one of its slot, so the frame of the slot
    starting at `t` shows the screen just before `t + 1 / rate`.
    """
    frames = _frames(path, info, start, end - start, rate, 0)
    return ((start + k / rate, frame) for k, frame in enumerate(frames))


def sample_audio(path: Path, start: float, end: float, stream: int) -> np.ndarray:
    """Mono 16 kHz samples of [start, end) of audio stream `stream`, as float32 in [-1, 1]."""
    command = [
        _tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{start:g}", "-t", f"{end - start:g}",
        "-i", str(path), "-map", f"0:a:{stream}", "-vn", "-sn", "-ac", "1",
        "-ar", str(AUDIO_RATE), "-f", "f32le", "-",
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RecordingError(f"cannot decode the audio of Recording {path}: {detail}")
    return np.frombuffer(result.stdout, dtype="<f4", count=len(result.stdout) // 4)


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
