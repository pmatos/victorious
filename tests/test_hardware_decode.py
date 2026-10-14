"""Frame sampling with and without VAAPI: the same frames, and software as the fallback."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests.analyse_helpers import FfmpegShim, analyse_text, is_selection_seek, prepare, small
from tests.synthetic import RecordingSpec, build_recording, match_shots, write_layout
from victorious import hwaccel, parallel, video
from victorious.hwaccel import HwDecoder
from victorious.video import VideoInfo, frame_at, probe, sample_chunk, sample_window

pytestmark = pytest.mark.usefixtures("require_ffmpeg")

BOGUS = "/dev/dri/renderD999"


def a_spec() -> RecordingSpec:
    shots = match_shots(2, ((0.0, 0, 0, 99), (6.0, 1, 0, 95)), results_at=16, placement=1)
    return small(22, shots)


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_recording(a_spec(), tmp_path_factory.mktemp("hw") / "recording.mkv")


@pytest.fixture(autouse=True)
def fresh_state() -> Iterator[None]:
    hwaccel.reset()
    yield
    hwaccel.reset()


def working_decoder(path: Path) -> HwDecoder | None:
    hwaccel.configure("vaapi")
    return video.resolve_hardware(path, probe(path))


def frames_of(path: Path, info: VideoInfo, start: int = 0, length: int | None = 10) -> list[bytes]:
    return [frame.tobytes() for _, frame in sample_chunk(path, info, start, length, threads=1)]


def requires_vaapi(path: Path) -> HwDecoder:
    decoder = working_decoder(path)
    if decoder is None:
        pytest.skip("no VAAPI device decodes this codec here")
    hwaccel.reset()
    return decoder


def test_off_never_asks_ffmpeg_for_vaapi(
    recording: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = FfmpegShim(tmp_path / "shim")
    monkeypatch.setenv("PATH", shim.env["PATH"])
    hwaccel.configure("off")

    frames_of(recording, probe(recording))

    assert shim.lines()
    assert not [line for line in shim.lines() if "vaapi" in line]


def test_a_missing_render_device_falls_back_to_the_same_frames_with_one_note(
    recording: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hwaccel.configure("off")
    software = frames_of(recording, probe(recording))
    capsys.readouterr()
    hwaccel.reset()
    monkeypatch.setenv(hwaccel.DEVICE_VARIABLE, BOGUS)
    hwaccel.configure("vaapi")
    info = probe(recording)

    first = frames_of(recording, info)
    second = frames_of(recording, info, 5, 5)
    window = [f for _, f in sample_window(recording, info, 3, 5, 4)]

    assert first == software
    assert second == software[5:]
    assert len(window) == 8
    notes = [line for line in capsys.readouterr().err.splitlines() if "hardware decode" in line]
    assert len(notes) == 1
    assert "software decode" in notes[0]


def test_auto_without_any_render_device_is_silent(
    recording: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(hwaccel, "DRI", tmp_path / "no-dri")
    hwaccel.configure("auto")

    assert len(frames_of(recording, probe(recording))) == 10
    assert capsys.readouterr().err == ""


def test_the_probe_runs_once_per_process(
    recording: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = FfmpegShim(tmp_path / "shim")
    monkeypatch.setenv("PATH", shim.env["PATH"])
    monkeypatch.setenv(hwaccel.DEVICE_VARIABLE, BOGUS)
    hwaccel.configure("vaapi")
    info = probe(recording)

    frames_of(recording, info)
    probes = shim.decodes(matching="vaapi")
    frames_of(recording, info)
    frame_at(recording, info, 3)

    assert probes > 0
    assert shim.decodes(matching="vaapi") == probes


def test_hardware_frames_equal_software_frames(recording: Path) -> None:
    requires_vaapi(recording)
    hwaccel.configure("off")
    info = probe(recording)
    software = frames_of(recording, info, 0, None)
    soft_window = [f.tobytes() for _, f in sample_window(recording, info, 3.5, 8.5, 4)]
    soft_one = frame_at(recording, info, 9)
    hwaccel.reset()
    hwaccel.configure("vaapi")

    assert video.resolve_hardware(recording, info) is not None
    assert frames_of(recording, info, 0, None) == software
    assert [f.tobytes() for _, f in sample_window(recording, info, 3.5, 8.5, 4)] == soft_window
    hard_one = frame_at(recording, info, 9)
    assert soft_one is not None
    assert hard_one is not None
    assert np.array_equal(soft_one, hard_one)


def test_a_hardware_failure_mid_chunk_resumes_in_software_with_one_note(
    recording: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hwaccel.configure("off")
    info = probe(recording)
    software = frames_of(recording, info, 2, 8)
    capsys.readouterr()
    hwaccel.reset()
    decoder = HwDecoder("/dev/dri/renderD128", None)
    monkeypatch.setattr(hwaccel, "candidates", lambda *_: [decoder])
    monkeypatch.setattr(video, "_hardware_agrees", lambda *_: None)
    three = tmp_path / "three-frames.raw"
    three.write_bytes(b"".join(software[:3]))
    real = video._command

    def failing(*args: object, **kwargs: object) -> list[str]:
        command = real(*args, **kwargs)  # type: ignore[arg-type]
        if "vaapi" not in " ".join(command):
            return command
        script = (
            f"import sys; sys.stdout.buffer.write(open({str(three)!r}, 'rb').read()); sys.exit(1)"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(video, "_command", failing)
    hwaccel.configure("vaapi")

    got = frames_of(recording, info, 2, 8)

    assert got == software
    notes = [line for line in capsys.readouterr().err.splitlines() if "hardware decode" in line]
    assert len(notes) == 1


def test_analyse_output_is_identical_whatever_the_hardware_mode(
    tmp_path: Path,
) -> None:
    recording, layout = prepare(tmp_path, a_spec())
    off = analyse_text(recording, layout, "--no-cache", "--hwaccel", "off")
    assert off.returncode == 0, off.stderr
    bogus = analyse_text(
        recording,
        layout,
        "--no-cache",
        "--hwaccel",
        "vaapi",
        env={hwaccel.DEVICE_VARIABLE: BOGUS},
    )

    assert bogus.returncode == 0, bogus.stderr
    assert bogus.stdout == off.stdout
    assert json.loads(off.stdout)["matches"]
    assert sum("hardware decode" in line for line in bogus.stderr.splitlines()) == 1
    assert "hardware decode" not in off.stderr


def chunk_lines(shim: FfmpegShim) -> list[str]:
    return [line for line in shim.lines() if "-vf fps=1:" in line and not is_selection_seek(line)]


@pytest.mark.parametrize("slots", [0, 1])
def test_early_chunks_use_a_free_gpu_slot_else_the_cpu_and_the_last_ones_wait_for_the_gpu(
    recording: Path, tmp_path: Path, slots: int
) -> None:
    requires_vaapi(recording)
    layout = write_layout(tmp_path / "layout.yaml")
    shim = FfmpegShim(tmp_path / "shim")
    off = analyse_text(recording, layout, "--no-cache", "--hwaccel", "off")
    hardware = analyse_text(
        recording,
        layout,
        "--no-cache",
        "--hwaccel",
        "vaapi",
        "--jobs",
        "3",
        "--chunk-seconds",
        "5",
        env={**shim.env, parallel.SLOTS_VARIABLE: str(slots)},
    )

    assert off.returncode == 0, off.stderr
    assert hardware.returncode == 0, hardware.stderr
    assert hardware.stdout == off.stdout
    assert "hardware decode" not in hardware.stderr
    chunks = chunk_lines(shim)
    assert len(chunks) == 5
    on_gpu = [line for line in chunks if "vaapi" in line]
    if slots == 0:
        assert on_gpu == []
    else:
        assert len(on_gpu) >= 3
