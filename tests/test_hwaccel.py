"""Optional VAAPI hardware decode: which device and driver, and software as the fallback."""

from pathlib import Path

from victorious import hwaccel
from victorious.hwaccel import HwDecoder

AMD = "0x1002"
INTEL = "0x8086"
NVIDIA = "0x10de"


def fake_drm(root: Path, devices: dict[str, str]) -> Path:
    """A /sys/class/drm look-alike: renderD128 -> its PCI vendor id."""
    for name, vendor in devices.items():
        directory = root / name / "device"
        directory.mkdir(parents=True)
        (directory / "vendor").write_text(f"{vendor}\n")
    return root


def test_amd_gpus_load_radeonsi() -> None:
    assert hwaccel.vendor_drivers(AMD) == ["radeonsi"]


def test_intel_gpus_try_iHD_then_i965() -> None:
    assert hwaccel.vendor_drivers(INTEL) == ["iHD", "i965"]


def test_other_vendors_get_no_driver_guess() -> None:
    assert hwaccel.vendor_drivers(NVIDIA) == []
    assert hwaccel.vendor_drivers(None) == []


def test_render_devices_are_listed_in_order(tmp_path: Path) -> None:
    dri = tmp_path / "dri"
    dri.mkdir()
    for name in ("renderD129", "card1", "renderD128", "by-path"):
        (dri / name).touch()

    assert hwaccel.render_devices(dri) == [dri / "renderD128", dri / "renderD129"]


def test_a_machine_without_a_dri_directory_has_no_render_devices(tmp_path: Path) -> None:
    assert hwaccel.render_devices(tmp_path / "missing") == []


def test_the_user_picks_the_render_device_with_an_environment_variable(tmp_path: Path) -> None:
    drm = fake_drm(tmp_path, {"renderD128": AMD})
    env = {"VICTORIOUS_VAAPI_DEVICE": "/dev/dri/renderD129"}

    candidates = hwaccel.candidates(env, tmp_path / "dri", drm)

    assert [c.device for c in candidates] == ["/dev/dri/renderD129"]


def test_candidates_pair_each_device_with_its_vendor_drivers_then_the_libva_default(
    tmp_path: Path,
) -> None:
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").touch()
    drm = fake_drm(tmp_path / "drm", {"renderD128": INTEL})

    candidates = hwaccel.candidates({}, dri, drm)

    assert candidates == [
        HwDecoder(str(dri / "renderD128"), "iHD"),
        HwDecoder(str(dri / "renderD128"), "i965"),
        HwDecoder(str(dri / "renderD128"), None),
    ]


def test_an_explicit_libva_driver_is_tried_first_and_the_vendors_follow(tmp_path: Path) -> None:
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").touch()
    drm = fake_drm(tmp_path / "drm", {"renderD128": AMD})

    candidates = hwaccel.candidates({"LIBVA_DRIVER_NAME": "custom"}, dri, drm)

    assert candidates == [
        HwDecoder(str(dri / "renderD128"), None),
        HwDecoder(str(dri / "renderD128"), "radeonsi"),
    ]


def test_no_render_device_means_no_candidate(tmp_path: Path) -> None:
    assert hwaccel.candidates({}, tmp_path / "dri", tmp_path / "drm") == []


def test_the_driver_goes_into_the_ffmpeg_environment_only_when_chosen() -> None:
    assert hwaccel.environment(HwDecoder("/dev/dri/renderD128", "radeonsi"), {"A": "1"}) == {
        "A": "1",
        "LIBVA_DRIVER_NAME": "radeonsi",
    }
    assert hwaccel.environment(HwDecoder("/dev/dri/renderD128", None), {"A": "1"}) == {"A": "1"}


def test_the_mode_comes_from_the_environment_and_defaults_to_auto() -> None:
    assert hwaccel.mode_from({}) == "auto"
    assert hwaccel.mode_from({"VICTORIOUS_HWACCEL": "off"}) == "off"
    assert hwaccel.mode_from({"VICTORIOUS_HWACCEL": "vaapi"}) == "vaapi"
    assert hwaccel.mode_from({"VICTORIOUS_HWACCEL": "nonsense"}) == "auto"


def test_one_failure_does_not_switch_hardware_decode_off() -> None:
    hwaccel.reset()

    hwaccel.record_failure()

    assert not hwaccel.is_disabled()


def test_consecutive_failures_switch_it_off_and_a_success_resets_the_count() -> None:
    hwaccel.reset()
    for _ in range(hwaccel.FAILURE_LIMIT - 1):
        hwaccel.record_failure()
    hwaccel.record_success()
    for _ in range(hwaccel.FAILURE_LIMIT - 1):
        hwaccel.record_failure()
    assert not hwaccel.is_disabled()

    hwaccel.record_failure()

    assert hwaccel.is_disabled()
    hwaccel.reset()
