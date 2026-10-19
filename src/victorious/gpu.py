"""Where the speech models run: an NVIDIA GPU through CUDA when one is usable, else the CPU."""

import functools
import importlib.util
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from victorious.errors import SpeechError

MODES = ("auto", "cpu", "cuda")
MODE_VARIABLE = "VICTORIOUS_DEVICE"


@dataclass(frozen=True)
class Device:
    """A device for Whisper and the diarization pipeline, and the Whisper precision on it."""

    name: str
    compute_type: str


CPU = Device("cpu", "int8")
CUDA = Device("cuda", "float16")


def mode_from(env: Mapping[str, str]) -> str:
    mode = env.get(MODE_VARIABLE, "auto").lower()
    return mode if mode in MODES else "auto"


def torch_libraries() -> Path | None:
    """The DLL folder of a CUDA build of torch (cuBLAS, cuDNN), found without importing torch."""
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    folder = Path(spec.origin).parent / "lib"
    return folder if folder.is_dir() else None


@functools.cache
def expose_cuda_libraries() -> None:
    """Lets CTranslate2 load cuBLAS and cuDNN from the CUDA build of torch on Windows.

    CTranslate2 ships only part of cuDNN and none of cuBLAS; torch ships both, and Windows looks
    for a DLL's dependencies only in the folders it was told about.
    """
    if sys.platform != "win32":
        return
    folder = torch_libraries()
    if folder is not None:
        os.add_dll_directory(str(folder))
        os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"


def _cuda_devices() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def wants_cuda(mode: str, gpu_seen: Callable[[], bool]) -> bool:
    """`cuda` insists on the GPU even when none is seen, so that loading fails with the reason;
    `auto` uses it when one is seen; `cpu` never looks."""
    return mode == "cuda" or (mode == "auto" and gpu_seen())


def whisper_device(mode: str, cuda_devices: Callable[[], int] = _cuda_devices) -> Device:
    """CUDA when the mode allows it and CTranslate2 sees an NVIDIA GPU, else the CPU."""
    return CUDA if wants_cuda(mode, lambda: cuda_devices() > 0) else CPU


def torch_device(torch: Any, mode: str) -> Any:
    """The torch device for the diarization pipeline, by the same rule as Whisper's."""
    return torch.device("cuda" if wants_cuda(mode, torch.cuda.is_available) else "cpu")


def fall_back(mode: str, model: str, error: Exception, message: str) -> None:
    """A GPU that failed `model`: an error in `cuda` mode, else `message` noted for the CPU run."""
    if mode == "cuda":
        raise SpeechError(
            f"cannot run {model} on CUDA: {error}; unset {MODE_VARIABLE} to fall back to the CPU"
        ) from error
    note(message)


def note(message: str) -> None:
    print(f"victorious: {message}", file=sys.stderr)
