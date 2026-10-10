"""Behaviour of the `victorious` command line, observed through a real process."""

from importlib.metadata import entry_points
from pathlib import Path

import pytest

from tests.cli_runner import run_cli

SUBCOMMANDS = ["setup", "analyse", "review", "schema"]
STUBS = ["setup", "review"]


def test_help_lists_the_four_subcommands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    for name in SUBCOMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("command", STUBS)
def test_stub_subcommand_reports_not_implemented_on_stderr(command: str) -> None:
    result = run_cli(command)

    assert result.returncode == 1
    assert "not implemented" in result.stderr
    assert command in result.stderr
    assert result.stdout == ""


def test_stub_accepts_arguments_and_writes_no_output_file(tmp_path: Path) -> None:
    result = run_cli("setup", "recording.mkv", "-o", "layout.yaml", cwd=tmp_path)

    assert result.returncode == 1
    assert "not implemented" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_missing_subcommand_is_a_usage_error() -> None:
    result = run_cli()

    assert result.returncode == 2
    assert result.stdout == ""


def test_unknown_top_level_option_is_a_usage_error() -> None:
    result = run_cli("--bogus", "schema")

    assert result.returncode == 2
    assert "--bogus" in result.stderr


def test_unknown_subcommand_is_a_usage_error() -> None:
    result = run_cli("frobnicate")

    assert result.returncode == 2
    assert "frobnicate" in result.stderr


def test_console_script_entry_point_is_installed() -> None:
    scripts = entry_points(group="console_scripts", name="victorious")

    assert len(scripts) == 1
    assert callable(next(iter(scripts)).load())
