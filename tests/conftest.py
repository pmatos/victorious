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
