import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from victorious import hwaccel

REAL_HF_HOME = os.environ.get("HF_HOME") or str(
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
)


os.environ["VICTORIOUS_NO_SPEECH"] = "1"


E2E_MODULES = frozenset(
    {
        "test_bisect",
        "test_counter_row",
        "test_eliminated",
        "test_input_robustness",
        "test_layout_selection",
        "test_observation_cache",
        "test_observations_sidecar",
        "test_parallel_decode",
        "test_setup",
        "test_setup_manual",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Marks every test that turns speech on (so loads the real models) as `slow_models`, and
    every test of a module in `E2E_MODULES` (each runs the whole pipeline on built Recordings) as
    `e2e`."""
    for item in items:
        if "speech_on" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow_models)
        if item.path.stem in E2E_MODULES:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        pytest.fail(
            f"{', '.join(missing)} not found on PATH: the synthetic Recording tests need ffmpeg "
            "(the product's fast path needs it too). Install it, e.g. `apt install ffmpeg`.",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def require_tesseract() -> None:
    if shutil.which("tesseract") is None:
        pytest.fail(
            "tesseract not found on PATH: reading placement digits needs it. "
            "Install it, e.g. `apt install tesseract-ocr`.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def isolated_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("xdg-cache")))


@pytest.fixture(autouse=True)
def speech_is_off_and_models_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps `analyse` model-free by default, and the model cache out of the isolated cache.

    Module-scoped fixtures run before this one, so the variable is also set on import above.
    """
    monkeypatch.setenv("VICTORIOUS_NO_SPEECH", "1")
    monkeypatch.setenv("HF_HOME", REAL_HF_HOME)


@pytest.fixture(autouse=True)
def decoding_is_software(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keeps tests on software decode; the hardware tests ask for VAAPI themselves."""
    monkeypatch.setenv("VICTORIOUS_HWACCEL", "off")
    hwaccel.reset()
    yield
    hwaccel.reset()


@pytest.fixture(autouse=True)
def jev_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's real TypeSafe key must never reach a test."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("VICTORIOUS_JEV_URL", raising=False)
