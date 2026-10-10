"""Behaviour of the synthetic Recording helper, observed through ffprobe and decoded frames."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from tests.synthetic import (
    ELEMENT_RECTS,
    AudioClip,
    Hud,
    Lobby,
    RecordingSpec,
    Results,
    Shot,
    build_recording,
    match_shots,
)

pytestmark = pytest.mark.usefixtures("require_ffmpeg")

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
TESTS_DIR = Path(__file__).parent
CROPS = TESTS_DIR / "fixtures" / "crops"
WIDTH, HEIGHT = 1280, 720
NATIVE_HEIGHT = 1440
BANNER_CROP_DOWNSCALE = 4
NATIVE_WIDTH = 2560
NATIVE_FPS = 5.0


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            FFPROBE, "-v", "error", "-count_frames", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )  # fmt: skip
    info: dict[str, Any] = json.loads(result.stdout)
    return info


def frame_at(path: Path, seconds: float) -> np.ndarray:
    result = subprocess.run(
        [
            FFMPEG, "-v", "error", "-ss", str(seconds), "-i", str(path),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True,
        check=True,
    )  # fmt: skip
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    return frame


def region(frame: np.ndarray, element: str, margin: float = 0.0) -> np.ndarray:
    x, y, w, h = ELEMENT_RECTS[element]
    height, width = frame.shape[:2]
    pad_x, pad_y = margin * w * width, margin * h * height
    x0, x1 = int(x * width - pad_x), int((x + w) * width + pad_x)
    y0, y1 = int(y * height - pad_y), int((y + h) * height + pad_y)
    return frame[max(y0, 0) : y1, max(x0, 0) : x1]


def load_crop(name: str, color: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(CROPS / f"{name}.png"), flag)
    assert image is not None
    return image


def hud_scale(frame: np.ndarray) -> float:
    return float(frame.shape[0]) / NATIVE_HEIGHT


def confidence(image: np.ndarray, crop_name: str, scale: float, color: bool = False) -> float:
    template = load_crop(crop_name, color)
    template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if not color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        return 0.0
    scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return float(scores.max())


def hud_anchor_confidence(frame: np.ndarray) -> float:
    scores = [
        confidence(region(frame, "hud_anchor", margin=0.1), name, hud_scale(frame), color=True)
        for name in ("icon_skull", "icon_squad", "icon_person")
    ]
    return min(scores)


def banner_confidence(frame: np.ndarray) -> float:
    return confidence(
        region(frame, "results_banner", margin=0.2),
        "victory_royale",
        BANNER_CROP_DOWNSCALE * hud_scale(frame),
        color=True,
    )


def hash_confidence(frame: np.ndarray) -> float:
    return confidence(region(frame, "placement", margin=0.2), "hash", hud_scale(frame))


def placement_digit_scale(frame: np.ndarray) -> float:
    hash_height = int(load_crop("hash").shape[0])
    digit_height = int(load_crop("digit_7").shape[0])
    return hud_scale(frame) * hash_height / digit_height


def read_digits(frame: np.ndarray, element: str, scale: float | None = None) -> str:
    scale = hud_scale(frame) if scale is None else scale
    image = region(frame, element, margin=0.5)
    found = [
        (_best_x(image, digit, scale), digit)
        for digit in "0123456789"
        if confidence(image, f"digit_{digit}", scale) > 0.9
    ]
    return "".join(digit for _, digit in sorted(found))


def _best_x(image: np.ndarray, digit: str, scale: float) -> int:
    template = cv2.resize(
        load_crop(f"digit_{digit}"), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    return int(np.unravel_index(int(scores.argmax()), scores.shape)[1])


def sine_clip(path: Path, seconds: float) -> Path:
    subprocess.run(
        [
            FFMPEG, "-v", "error", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def silences(path: Path) -> list[tuple[float, float | None]]:
    result = subprocess.run(
        [
            FFMPEG, "-v", "info", "-i", str(path), "-af", "silencedetect=noise=-40dB:d=0.3",
            "-vn", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )  # fmt: skip
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", result.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
    return [(s, ends[i] if i < len(ends) else None) for i, s in enumerate(starts)]


def test_recording_has_requested_duration_and_frame_rate(tmp_path: Path) -> None:
    spec = RecordingSpec(duration=10.0, shots=(Shot(Hud(), 2.0, 8.0),), fps=30.0)

    recording = build_recording(spec, tmp_path / "out.mkv")

    info = ffprobe(recording)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert video["r_frame_rate"] == "30/1"
    assert int(video["nb_read_frames"]) == 300
    assert (video["width"], video["height"]) == (WIDTH, HEIGHT)
    assert float(info["format"]["duration"]) == pytest.approx(10.0, abs=0.1)
    assert (audio["codec_name"], int(audio["sample_rate"]), audio["channels"]) == (
        "aac",
        48000,
        2,
    )


def test_clip_is_mixed_in_at_requested_offset(tmp_path: Path) -> None:
    clip = sine_clip(tmp_path / "tone.wav", seconds=2.0)
    spec = RecordingSpec(duration=8.0, shots=(), audio=(AudioClip(clip, at=3.0),))

    recording = build_recording(spec, tmp_path / "out.mkv")

    quiet = silences(recording)
    assert quiet[0][0] == pytest.approx(0.0, abs=0.1)
    assert quiet[0][1] == pytest.approx(3.0, abs=0.15)
    assert quiet[1][0] == pytest.approx(5.0, abs=0.15)


def test_results_screen_shows_victory_banner_for_a_win_and_placement_otherwise(
    tmp_path: Path,
) -> None:
    spec = RecordingSpec(
        duration=8.0,
        shots=(
            Shot(Hud(), 2.0, 4.0),
            Shot(Results(placement=1), 4.0, 6.0),
            Shot(Results(placement=7), 6.0, 8.0),
        ),
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        fps=NATIVE_FPS,
    )

    recording = build_recording(spec, tmp_path / "out.mkv")

    lobby, hud, win, placed = (frame_at(recording, t) for t in (1.0, 3.0, 5.0, 7.0))
    assert banner_confidence(win) > 0.8
    for frame in (lobby, hud, placed):
        assert banner_confidence(frame) < 0.5
    assert hash_confidence(placed) > 0.8
    assert read_digits(placed, "placement", scale=placement_digit_scale(placed)) == "7"
    assert hash_confidence(hud) < 0.5
    assert hash_confidence(lobby) < 0.5


def test_hud_counters_show_requested_values_and_none_hides_a_counter(tmp_path: Path) -> None:
    spec = RecordingSpec(
        duration=6.0,
        shots=(
            Shot(Hud(personal_eliminations=3, team_eliminations=14, players_left=82), 0.0, 3.0),
            Shot(Hud(personal_eliminations=None, team_eliminations=5, players_left=67), 3.0, 6.0),
        ),
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        fps=NATIVE_FPS,
    )

    recording = build_recording(spec, tmp_path / "out.mkv")

    first, second = frame_at(recording, 1.5), frame_at(recording, 4.5)
    assert read_digits(first, "personal_eliminations") == "3"
    assert read_digits(first, "team_eliminations") == "14"
    assert read_digits(first, "players_left") == "82"
    assert read_digits(second, "personal_eliminations") == ""
    assert read_digits(second, "team_eliminations") == "5"
    assert read_digits(second, "players_left") == "67"
    assert hud_anchor_confidence(second) > 0.8


def test_one_match_with_results_screen(tmp_path: Path) -> None:
    shots = match_shots(
        start=5.0,
        hud_steps=[(0.0, 0, 0, 98), (10.0, 2, 3, 47), (15.0, 2, 4, 25)],
        results_at=25.0,
        results_duration=5.0,
        placement=1,
    )
    spec = RecordingSpec(
        duration=35.0, shots=shots, width=NATIVE_WIDTH, height=NATIVE_HEIGHT, fps=NATIVE_FPS
    )

    recording = build_recording(spec, tmp_path / "out.mkv")

    assert hud_anchor_confidence(frame_at(recording, 2.0)) < 0.5
    assert hud_anchor_confidence(frame_at(recording, 32.0)) < 0.5
    assert banner_confidence(frame_at(recording, 27.0)) > 0.8
    readings = {t: read_digits(frame_at(recording, t), "players_left") for t in (7.0, 17.0, 22.0)}
    assert readings == {7.0: "98", 17.0: "47", 22.0: "25"}
    assert hud_anchor_confidence(frame_at(recording, 7.0)) > 0.8


@pytest.mark.parametrize(
    "shots",
    [
        (Shot(Hud(), 0.0, 5.0), Shot(Results(placement=1), 4.0, 8.0)),
        (Shot(Hud(), 8.0, 12.0),),
        (Shot(Hud(), 5.0, 5.0),),
        (Shot(Hud(), 5.0, 6.0), Shot(Lobby(), 1.0, 2.0)),
    ],
    ids=["overlap", "past-end", "empty", "unsorted"],
)
def test_overlapping_or_out_of_range_shots_are_rejected(shots: tuple[Shot, ...]) -> None:
    with pytest.raises(ValueError, match="shot"):
        RecordingSpec(duration=10.0, shots=shots)


def test_committed_crops_are_small_and_no_video_is_committed() -> None:
    crops = sorted(CROPS.glob("*.png"))

    assert crops
    assert all(crop.stat().st_size < 8 * 1024 for crop in crops)
    assert sum(crop.stat().st_size for crop in crops) < 128 * 1024
    videos = [p for p in TESTS_DIR.rglob("*") if p.suffix in {".mkv", ".mp4", ".webm"}]
    assert videos == []
