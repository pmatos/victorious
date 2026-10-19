"""Frame sampling with and without hardware decode: the same frames, software as the fallback."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests.analyse_helpers import FfmpegShim, analyse_text, is_selection_seek, prepare, small
from tests.synthetic import RecordingSpec, build_recording, match_shots, write_layout
from victorious import cli, hwaccel, parallel, video
from victorious.hwaccel import HwDecoder
from victorious.layout import load_layouts
from victorious.video import VideoInfo, frame_at, probe, sample_chunk, sample_window

pytestmark = pytest.mark.usefixtures("require_ffmpeg")

BOGUS = "/dev/dri/renderD999"
BOGUS_GPU = "99"
# Each hardware mode, the variable that points it at a device, and a device that does not exist.
MODES = [
    ("vaapi", hwaccel.DEVICE_VARIABLE, BOGUS),
    ("cuda", hwaccel.CUDA_DEVICE_VARIABLE, BOGUS_GPU),
]


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


def working_decoder(path: Path, mode: str) -> HwDecoder | None:
    hwaccel.configure(mode)
    return video.resolve_hardware(path, probe(path))


def frames_of(path: Path, info: VideoInfo, start: int = 0, length: int | None = 10) -> list[bytes]:
    return [frame.tobytes() for _, frame in sample_chunk(path, info, start, length, threads=1)]


def requires_hardware(path: Path, mode: str = "vaapi") -> HwDecoder:
    decoder = working_decoder(path, mode)
    if decoder is None:
        pytest.skip(f"no {mode} device decodes this codec here")
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


@pytest.mark.parametrize(("mode", "variable", "missing"), MODES)
def test_a_missing_device_falls_back_to_the_same_frames_with_one_note(
    recording: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    variable: str,
    missing: str,
) -> None:
    hwaccel.configure("off")
    software = frames_of(recording, probe(recording))
    capsys.readouterr()
    hwaccel.reset()
    monkeypatch.setenv(variable, missing)
    hwaccel.configure(mode)
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
    monkeypatch.setattr(hwaccel, "nvidia_present", lambda *_: False)
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


@pytest.mark.parametrize("mode", ["vaapi", "cuda"])
def test_hardware_frames_equal_software_frames(recording: Path, mode: str) -> None:
    requires_hardware(recording, mode)
    hwaccel.configure("off")
    info = probe(recording)
    software = frames_of(recording, info, 0, None)
    soft_window = [f.tobytes() for _, f in sample_window(recording, info, 3.5, 8.5, 4)]
    soft_one = frame_at(recording, info, 9)
    hwaccel.reset()
    hwaccel.configure(mode)

    assert video.resolve_hardware(recording, info) is not None
    assert frames_of(recording, info, 0, None) == software
    assert [f.tobytes() for _, f in sample_window(recording, info, 3.5, 8.5, 4)] == soft_window
    hard_one = frame_at(recording, info, 9)
    assert soft_one is not None
    assert hard_one is not None
    assert np.array_equal(soft_one, hard_one)


def scripted_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: list[bytes], outcomes: list[int]
) -> list[str]:
    """A GPU whose n-th use writes `outcomes[n]` of `frames`, failing unless it wrote them all.

    The last outcome repeats; the returned list logs each use.
    """
    hwaccel.reset()
    decoder = HwDecoder("/dev/dri/renderD128", None)
    monkeypatch.setattr(hwaccel, "candidates", lambda *_: [decoder])
    monkeypatch.setattr(video, "_hardware_agrees", lambda *_: None)
    used: list[str] = []
    real = video._command

    def command(*args: object, **kwargs: object) -> list[str]:
        built = real(*args, **kwargs)  # type: ignore[arg-type]
        if "vaapi" not in " ".join(built):
            return built
        count = outcomes[min(len(used), len(outcomes) - 1)]
        used.append(" ".join(built))
        raw = tmp_path / f"gpu-{len(used)}.raw"
        raw.write_bytes(b"".join(frames[:count]))
        status = 0 if count >= len(frames) else 1
        script = (
            f"import sys; sys.stdout.buffer.write(open({str(raw)!r}, 'rb').read());"
            f" sys.exit({status})"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(video, "_command", command)
    hwaccel.configure("vaapi")
    return used


def software_frames(recording: Path, info: VideoInfo) -> list[bytes]:
    hwaccel.configure("off")
    frames = frames_of(recording, info, 2, 8)
    hwaccel.reset()
    return frames


def test_a_hardware_failure_mid_chunk_resumes_in_software_with_one_note(
    recording: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = probe(recording)
    software = software_frames(recording, info)
    capsys.readouterr()
    scripted_gpu(monkeypatch, tmp_path, software, [3])

    got = frames_of(recording, info, 2, 8)

    assert got == software
    notes = [line for line in capsys.readouterr().err.splitlines() if "hardware decode" in line]
    assert len(notes) == 1


def test_a_failed_chunk_costs_the_gpu_that_chunk_only_until_the_failures_repeat(
    recording: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = probe(recording)
    software = software_frames(recording, info)
    used = scripted_gpu(monkeypatch, tmp_path, software, [3])

    runs = [frames_of(recording, info, 2, 8) for _ in range(hwaccel.FAILURE_LIMIT + 2)]

    assert all(run == software for run in runs)
    assert len(used) == hwaccel.FAILURE_LIMIT


def test_a_clean_hardware_chunk_between_failures_keeps_the_gpu_on(
    recording: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = probe(recording)
    software = software_frames(recording, info)
    failing = [3] * (hwaccel.FAILURE_LIMIT - 1)
    outcomes = [*failing, len(software), *failing]
    used = scripted_gpu(monkeypatch, tmp_path, software, outcomes)

    for _ in outcomes:
        assert frames_of(recording, info, 2, 8) == software

    assert not hwaccel.is_disabled()
    assert len(used) == len(outcomes)


def test_the_decode_note_says_how_many_chunks_the_gpu_took() -> None:
    note = cli.decode_note({"gpu": 31, "cpu": 50, "resumed": 2})

    assert note == (
        "victorious analyse: decode: 31 of 83 chunks on the GPU, 50 on the CPU, "
        "2 resumed on the CPU after a GPU failure"
    )
    assert "hardware decode" not in note


def test_the_decode_note_is_silent_when_the_gpu_was_not_used() -> None:
    assert cli.decode_note({"cpu": 83}) is None
    assert cli.decode_note({}) is None


def test_the_chunks_of_a_run_are_counted_by_how_they_were_decoded(
    recording: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = probe(recording)
    layout = load_layouts(write_layout(tmp_path / "layout.yaml"))[0].layout
    hwaccel.configure("off")
    _, chunks = parallel.extract_seconds(recording, info, layout, jobs=1, chunk_seconds=4)
    assert chunks == {"cpu": 6}
    scripted_gpu(monkeypatch, tmp_path, [b"\0"], [0])

    _, chunks = parallel.extract_seconds(recording, info, layout, jobs=1, chunk_seconds=4)

    assert chunks == {"resumed": hwaccel.FAILURE_LIMIT, "cpu": 6 - hwaccel.FAILURE_LIMIT}


@pytest.mark.parametrize(
    ("frames_from_gpu", "mode", "kind"),
    [(8, "vaapi", "gpu"), (3, "vaapi", "resumed"), (8, "off", "cpu")],
)
def test_a_chunk_reports_how_it_was_decoded(
    recording: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames_from_gpu: int,
    mode: str,
    kind: str,
) -> None:
    info = probe(recording)
    software = software_frames(recording, info)
    scripted_gpu(monkeypatch, tmp_path, software, [frames_from_gpu])
    hwaccel.configure(mode)
    route = video.Route()

    got = [f.tobytes() for _, f in sample_chunk(recording, info, 2, 8, threads=1, route=route)]

    assert got == software
    assert route.kind == kind


def test_a_decoder_that_dies_silently_is_reported_with_its_exit_status(
    recording: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hwaccel.configure("off")
    info = probe(recording)
    script = "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
    monkeypatch.setattr(video, "_command", lambda *_: [sys.executable, "-c", script])

    with pytest.raises(video.RecordingError, match="SIGSEGV"):
        frames_of(recording, info, 2, 8)

    monkeypatch.setattr(
        video, "_command", lambda *_: [sys.executable, "-c", "import sys; sys.exit(3)"]
    )
    with pytest.raises(video.RecordingError, match="status 3"):
        frames_of(recording, info, 2, 8)


@pytest.mark.parametrize(("mode", "variable", "missing"), MODES)
def test_analyse_output_is_identical_whatever_the_hardware_mode(
    tmp_path: Path, mode: str, variable: str, missing: str
) -> None:
    recording, layout = prepare(tmp_path, a_spec())
    off = analyse_text(recording, layout, "--no-cache", "--hwaccel", "off")
    assert off.returncode == 0, off.stderr
    bogus = analyse_text(
        recording,
        layout,
        "--no-cache",
        "--hwaccel",
        mode,
        env={variable: missing},
    )

    assert bogus.returncode == 0, bogus.stderr
    assert bogus.stdout == off.stdout
    assert json.loads(off.stdout)["matches"]
    assert sum("hardware decode" in line for line in bogus.stderr.splitlines()) == 1
    assert "hardware decode" not in off.stderr


def chunk_lines(shim: FfmpegShim) -> list[str]:
    return [line for line in shim.lines() if "-vf fps=1:" in line and not is_selection_seek(line)]


@pytest.mark.parametrize("mode", ["vaapi", "cuda"])
@pytest.mark.parametrize("slots", [0, 1])
def test_early_chunks_use_a_free_gpu_slot_else_the_cpu_and_the_last_ones_wait_for_the_gpu(
    recording: Path, tmp_path: Path, slots: int, mode: str
) -> None:
    requires_hardware(recording, mode)
    layout = write_layout(tmp_path / "layout.yaml")
    shim = FfmpegShim(tmp_path / "shim")
    off = analyse_text(recording, layout, "--no-cache", "--hwaccel", "off")
    hardware = analyse_text(
        recording,
        layout,
        "--no-cache",
        "--hwaccel",
        mode,
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
    on_gpu = [line for line in chunks if f"-hwaccel {mode}" in line]
    if slots == 0:
        assert on_gpu == []
        assert "decode:" not in hardware.stderr
    else:
        assert len(on_gpu) >= 3
        assert f"decode: {len(on_gpu)} of 5 chunks on the GPU" in hardware.stderr


def test_no_gpu_slots_means_software_for_every_decode(recording: Path, tmp_path: Path) -> None:
    layout = write_layout(tmp_path / "layout.yaml")
    shim = FfmpegShim(tmp_path / "shim")

    result = analyse_text(
        recording,
        layout,
        "--no-cache",
        "--hwaccel",
        "vaapi",
        env={**shim.env, hwaccel.DEVICE_VARIABLE: BOGUS, parallel.SLOTS_VARIABLE: "0"},
    )

    assert result.returncode == 0, result.stderr
    assert shim.lines()
    assert not [line for line in shim.lines() if "vaapi" in line]
    assert "hardware decode" not in result.stderr
