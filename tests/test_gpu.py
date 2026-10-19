"""Which device the speech models run on: CUDA when an NVIDIA GPU is usable, else the CPU."""

from types import SimpleNamespace
from typing import Any

from victorious import gpu


def _torch(available: bool) -> Any:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: available),
        device=lambda name: SimpleNamespace(type=name),
    )


def test_auto_picks_cuda_when_ctranslate2_sees_a_gpu() -> None:
    assert gpu.whisper_device("auto", lambda: 1) == gpu.CUDA


def test_auto_falls_back_to_the_cpu_without_a_gpu() -> None:
    assert gpu.whisper_device("auto", lambda: 0) == gpu.CPU


def test_cpu_mode_never_looks_for_a_gpu() -> None:
    def fail() -> int:
        raise AssertionError("probed")

    assert gpu.whisper_device("cpu", fail) == gpu.CPU


def test_cuda_mode_insists_on_the_gpu_even_when_none_is_seen() -> None:
    assert gpu.whisper_device(gpu.mode_from({gpu.MODE_VARIABLE: "CUDA"}), lambda: 0) == gpu.CUDA


def test_an_unknown_mode_is_auto() -> None:
    assert gpu.mode_from({gpu.MODE_VARIABLE: "metal"}) == "auto"


def test_diarization_follows_the_same_rule() -> None:
    assert gpu.torch_device(_torch(True), "auto").type == "cuda"
    assert gpu.torch_device(_torch(False), "auto").type == "cpu"
    assert gpu.torch_device(_torch(True), "cpu").type == "cpu"
    assert gpu.torch_device(_torch(False), "cuda").type == "cuda"
