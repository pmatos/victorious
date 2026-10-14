import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PATHS_FILE = ROOT / ".github" / "slow-models-paths.txt"
SCRIPT = ROOT / ".github" / "scripts" / "slow-models-needed.sh"


def listed() -> list[str]:
    return PATHS_FILE.read_text().splitlines()


def status(*changed: str) -> int:
    result = subprocess.run(
        [str(SCRIPT)], input="".join(f"{name}\n" for name in changed), text=True
    )
    return result.returncode


def needed(*changed: str) -> bool:
    return status(*changed) == 0


def collected(*arguments: str) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            *arguments,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"},
    )
    return [line for line in result.stdout.splitlines() if "::" in line]


def test_every_listed_path_exists() -> None:
    assert listed()
    assert [path for path in listed() if not (ROOT / path).exists()] == []


def test_no_listed_path_is_blank() -> None:
    assert all(line.strip() == line and line for line in listed())


def test_every_module_that_turns_speech_on_is_listed() -> None:
    modules = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_*.py")
        if re.search(r"^def speech_on\(|delenv\(.VICTORIOUS_NO_SPEECH.\)", path.read_text(), re.M)
    ]
    assert modules
    assert [module for module in modules if module not in listed()] == []


@pytest.mark.parametrize(
    "changed",
    [
        "src/victorious/speech.py",
        "src/victorious/speakers.py",
        "src/victorious/music.py",
        "tests/fixtures/speech/en.opus",
        "tests/conftest.py",
        "uv.lock",
    ],
)
def test_a_change_to_a_listed_path_needs_the_model_tests(changed: str) -> None:
    assert needed("README.md", changed)


@pytest.mark.parametrize(
    "changed",
    ["src/victorious/screens.py", "tests/test_setup.py", "docs/uv.lock", "docs/adr/0004.md"],
)
def test_a_change_elsewhere_does_not(changed: str) -> None:
    assert not needed(changed)


def test_a_change_elsewhere_exits_one_so_a_broken_script_is_not_mistaken_for_it() -> None:
    assert status("src/victorious/screens.py") == 1
    assert status() == 1


def test_tests_that_load_models_are_marked_and_the_rest_are_not() -> None:
    marked = collected("-m", "slow_models", "tests/test_speech.py", "tests/test_screens.py")
    unmarked = collected("-m", "not slow_models", "tests/test_speech.py", "tests/test_screens.py")
    assert marked
    assert all(item.startswith("tests/test_speech.py::") for item in marked)
    assert any(item.startswith("tests/test_screens.py::") for item in unmarked)
    assert not any(item.startswith("tests/test_speech.py::") for item in unmarked)
