import shutil

import pytest


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
