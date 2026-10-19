"""Optional hardware decode, NVIDIA's (CUDA) or VAAPI: which device to try, and the ffmpeg flags."""

import os
import shutil
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from victorious.gpu import wants_cuda

DRI = Path("/dev/dri")
DRM = Path("/sys/class/drm")
MODES = ("auto", "off", "vaapi", "cuda")
MODE_VARIABLE = "VICTORIOUS_HWACCEL"
DEVICE_VARIABLE = "VICTORIOUS_VAAPI_DEVICE"
CUDA_DEVICE_VARIABLE = "VICTORIOUS_CUDA_DEVICE"
DRIVER_VARIABLE = "LIBVA_DRIVER_NAME"
NVIDIA = "0x10de"

_DRIVERS = {"0x1002": ["radeonsi"], "0x8086": ["iHD", "i965"]}

# Software decodes 8-bit 4:2:0 into yuv420p before the BGR conversion; converting the nv12 that
# hwdownload yields (from VAAPI and CUDA alike) straight to BGR takes another swscale path and
# shifts pixels by up to 60.
DOWNLOAD_FILTER = "hwdownload,format=nv12,format=yuv420p"


@dataclass(frozen=True)
class HwDecoder:
    """A device and how to drive it: a VAAPI render device and the libva driver to load for it
    (None: let libva choose), or with `api` "cuda" the index of an NVIDIA GPU (no driver)."""

    device: str
    driver: str | None = None
    api: Literal["vaapi", "cuda"] = "vaapi"


def mode_from(env: Mapping[str, str]) -> str:
    mode = env.get(MODE_VARIABLE, "auto")
    return mode if mode in MODES else "auto"


def vendor_drivers(vendor: str | None) -> list[str]:
    return list(_DRIVERS.get(vendor or "", []))


def render_devices(directory: Path | None = None) -> list[Path]:
    try:
        return sorted((directory or DRI).glob("renderD*"))
    except OSError:
        return []


def _vendor(device: Path, drm: Path | None) -> str | None:
    try:
        return ((drm or DRM) / device.name / "device" / "vendor").read_text().strip().lower()
    except OSError:
        return None


def candidates(
    env: Mapping[str, str] | None = None,
    directory: Path | None = None,
    drm: Path | None = None,
) -> list[HwDecoder]:
    """What to probe, in order.

    The user's own LIBVA_DRIVER_NAME is tried first, as it is; the vendor's drivers follow it, so a
    variable left over from another GPU (iHD on an AMD machine) does not switch decode off.
    """
    env = os.environ if env is None else env
    explicit = env.get(DEVICE_VARIABLE)
    devices = [Path(explicit)] if explicit else render_devices(directory)
    found: list[HwDecoder] = []
    for device in devices:
        guessed = vendor_drivers(_vendor(device, drm))
        drivers = [None, *guessed] if DRIVER_VARIABLE in env else [*guessed, None]
        found.extend(HwDecoder(str(device), driver) for driver in drivers)
    return found


def nvidia_present(directory: Path | None = None, drm: Path | None = None) -> bool:
    """An NVIDIA driver is installed: `nvidia-smi` is on PATH, or a render device is NVIDIA's."""
    if shutil.which("nvidia-smi") is not None:
        return True
    return any(_vendor(device, drm) == NVIDIA for device in render_devices(directory))


def decoders(mode: str, env: Mapping[str, str] | None = None) -> list[HwDecoder]:
    """What `mode` probes, in order: in `auto` the NVIDIA GPU first (when there is one or
    VICTORIOUS_CUDA_DEVICE names it), then the VAAPI candidates.

    `cuda` always probes the GPU, so a machine without one gets the note that says why.
    """
    env = os.environ if env is None else env
    explicit = env.get(CUDA_DEVICE_VARIABLE)
    seen = wants_cuda(mode, lambda: bool(explicit) or nvidia_present())
    cuda = [HwDecoder(explicit or "0", api="cuda")] if seen else []
    return [*cuda, *([] if mode == "cuda" else candidates(env))]


def environment(decoder: HwDecoder, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of the ffmpeg subprocess; the process's own is never touched.

    CUDA numbers GPUs fastest first unless told otherwise; PCI bus order makes the index the one
    `nvidia-smi` shows.
    """
    env = dict(os.environ if base is None else base)
    if decoder.api == "cuda":
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if decoder.driver is not None:
        env[DRIVER_VARIABLE] = decoder.driver
    return env


def input_options(decoder: HwDecoder) -> list[str]:
    if decoder.api == "cuda":
        return [
            "-hwaccel", "cuda", "-hwaccel_device", decoder.device,
            "-hwaccel_output_format", "cuda",
        ]  # fmt: skip
    return [
        "-init_hw_device", f"vaapi=va:{decoder.device}", "-hwaccel", "vaapi",
        "-hwaccel_device", "va", "-hwaccel_output_format", "vaapi",
    ]  # fmt: skip


_lock = threading.Lock()
_resolved: dict[tuple[str, ...], HwDecoder | None] = {}
_mode: str | None = None
_noted: Any = threading.Event()
_failures = 0
_dropped: set[HwDecoder] = set()
FAILURE_LIMIT = 3


@dataclass(frozen=True)
class Shared:
    """What a worker process needs to behave like its parent: no second probe, one note."""

    mode: str | None
    resolved: dict[tuple[str, ...], HwDecoder | None]
    noted: Any


def configure(mode: str | None) -> None:
    """Forces a mode for this process; None goes back to VICTORIOUS_HWACCEL, else auto."""
    global _mode
    _mode = mode


def current_mode() -> str:
    return _mode if _mode is not None else mode_from(os.environ)


def reset() -> None:
    global _mode, _noted, _failures
    with _lock:
        _resolved.clear()
        _mode = None
        _failures = 0
        _dropped.clear()
        _noted = threading.Event()


def record_failure(decoder: HwDecoder) -> None:
    """`decoder` failed mid-stream; `FAILURE_LIMIT` in a row drop it for this process.

    The next `resolve` probes the candidates after it (VAAPI after CUDA in `auto`) and switches
    hardware decode off when none is left.
    """
    global _failures
    with _lock:
        _failures += 1
        if _failures >= FAILURE_LIMIT:
            _failures = 0
            _dropped.add(decoder)
            for key in [key for key, chosen in _resolved.items() if chosen == decoder]:
                del _resolved[key]


def record_success() -> None:
    global _failures
    with _lock:
        _failures = 0


def is_disabled() -> bool:
    """A decoder was dropped and nothing has replaced it: this process decodes in software."""
    return bool(_dropped) and not any(_resolved.values())


def share(noted: Any) -> Shared:
    """The state to hand to workers; `noted` is an Event they all see (a multiprocessing one)."""
    global _noted
    _noted = noted
    return Shared(_mode, dict(_resolved), noted)


def adopt(shared: Shared) -> None:
    global _mode, _noted
    _mode = shared.mode
    _resolved.update(shared.resolved)
    _noted = shared.noted


def note(message: str, outcome: str = "using software decode") -> None:
    """One stderr line per run, however many threads or workers hit the failure."""
    if _noted.is_set():
        return
    _noted.set()
    print(f"victorious: {message}; {outcome}", file=sys.stderr)


def resolve(key: tuple[str, ...], check: Callable[[HwDecoder], str | None]) -> HwDecoder | None:
    """The first candidate whose `check` passes (it returns None) or None for software decode.

    Decided once per process for a given `key`; `off` never probes, and `auto` is silent on a
    machine without a render device or an NVIDIA GPU.
    """
    mode = current_mode()
    if mode == "off":
        return None
    key = (mode, *key)
    with _lock:
        if key in _resolved:
            return _resolved[key]
        tried = [candidate for candidate in decoders(mode) if candidate not in _dropped]
        problems: dict[str, str] = {}
        chosen = None
        for candidate in tried:
            problem = check(candidate)
            if problem is None:
                chosen = candidate
                break
            problems.setdefault(candidate.api, problem)
        _resolved[key] = chosen
    if chosen is None and (tried or mode != "auto"):
        # The first problem of each kind of GPU, so a VAAPI fix is not hidden behind CUDA's.
        reasons = "; ".join(problems.values()) or "no render device"
        note(f"hardware decode unavailable ({reasons})")
    return chosen
