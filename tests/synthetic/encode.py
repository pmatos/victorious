"""Turn a RecordingSpec into a video file with ffmpeg."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic.render import render_scene
from tests.synthetic.spec import Lobby, RecordingSpec, Scene

SAMPLE_RATE = 48000


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH; the synthetic Recording helper needs it")
    return path


def _timeline(spec: RecordingSpec) -> list[tuple[Scene, int]]:
    """Consecutive (scene, frame count) runs covering the whole Recording."""
    total = round(spec.duration * spec.fps)
    runs: list[tuple[Scene, int]] = []
    cursor = 0
    for shot in spec.shots:
        first = round(shot.start * spec.fps)
        last = min(round(shot.end * spec.fps), total)
        if first > cursor:
            runs.append((Lobby(), first - cursor))
        if last > first:
            runs.append((shot.scene, last - first))
        cursor = max(cursor, last)
    if cursor < total:
        runs.append((Lobby(), total - cursor))
    return runs


def _concat_file(path: Path) -> str:
    return "file '" + str(path).replace("'", "'\\''") + "'"


def _write_frames(spec: RecordingSpec, workdir: Path) -> Path:
    listing = workdir / "frames.txt"
    lines: list[str] = []
    rendered: dict[Scene, Path] = {}
    last_image: Path | None = None
    for index, (scene, frames) in enumerate(_timeline(spec)):
        image = rendered.get(scene)
        if image is None:
            image = workdir / f"scene_{index}.png"
            cv2.imwrite(str(image), render_scene(scene, spec.width, spec.height))
            rendered[scene] = image
        lines.append(_concat_file(image))
        lines.append(f"duration {frames / spec.fps!r}")
        last_image = image
    if last_image is not None:
        lines.append(_concat_file(last_image))
    listing.write_text("\n".join(lines) + "\n")
    return listing


def _audio_args(spec: RecordingSpec) -> list[str]:
    inputs = ["-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo"]
    for clip in spec.audio:
        inputs += ["-i", str(clip.path)]
    if not spec.audio:
        return [*inputs, "-map", "0:v", "-map", "1:a"]
    chains = [
        f"[{index + 2}:a]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo,"
        f"adelay={round(clip.at * 1000)}|{round(clip.at * 1000)}[c{index}]"
        for index, clip in enumerate(spec.audio)
    ]
    mixed = "".join(f"[c{index}]" for index in range(len(spec.audio)))
    graph = (
        ";".join(chains)
        + f";[1:a]{mixed}amix=inputs={len(spec.audio) + 1}:duration=first:normalize=0[a]"
    )
    return [*inputs, "-filter_complex", graph, "-map", "0:v", "-map", "[a]"]


def build_recording(spec: RecordingSpec, out_path: Path) -> Path:
    """Encode `spec` to `out_path` (mkv) and return it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="victorious-frames-") as scratch:
        listing = _write_frames(spec, Path(scratch))
        keyframes = str(max(1, round(spec.keyframe_interval * spec.fps)))
        command = [
            _ffmpeg(), "-v", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            *_audio_args(spec),
            "-vf", f"fps={spec.fps!r},format=yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(spec.crf),
            "-g", keyframes, "-keyint_min", keyframes, "-sc_threshold", "0",
            "-c:a", "aac", "-ar", str(SAMPLE_RATE), "-ac", "2",
            "-max_interleave_delta", "0",
            "-t", f"{spec.duration!r}", str(out_path),
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")
    return out_path


def decode_frame(path: Path, seconds: float) -> np.ndarray:
    """The BGR frame ffmpeg decodes from `path` at `seconds`."""
    result = subprocess.run(
        [
            _ffmpeg(), "-v", "error", "-ss", str(seconds), "-i", str(path),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True,
        check=True,
    )  # fmt: skip
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"no frame decoded from {path} at {seconds}s")
    return np.asarray(frame)
