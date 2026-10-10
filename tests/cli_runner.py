import subprocess
import sys
from pathlib import Path


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "victorious", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
