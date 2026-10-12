"""Behaviour of the `victorious` command line, observed through a real process."""

from importlib.metadata import entry_points
from pathlib import Path

import pytest

from tests.cli_runner import run_cli

SUBCOMMANDS = ["setup", "analyse", "review", "schema"]


def test_help_lists_the_four_subcommands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    for name in SUBCOMMANDS:
        assert name in result.stdout


def test_setup_help_describes_the_manual_editor() -> None:
    result = run_cli("setup", "--help")

    assert result.returncode == 0
    for option in ("--manual", "--no-browser", "--port"):
        assert option in result.stdout
    assert "not implemented" not in result.stdout


def test_review_help_describes_the_review_and_the_comparison() -> None:
    result = run_cli("review", "--help")

    assert result.returncode == 0
    for option in ("--against", "--tolerance", "--output", "--recording", "--no-browser", "--port"):
        assert option in result.stdout
    assert "not implemented" not in result.stdout


def test_review_needs_an_analysis_file() -> None:
    result = run_cli("review")

    assert result.returncode == 2
    assert "analysis" in result.stderr


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


@pytest.mark.parametrize("option", [["--no-browser"], ["--port", "8080"]])
def test_manual_only_options_are_refused_without_manual(option: list[str], tmp_path: Path) -> None:
    result = run_cli(
        "setup", "rec.mkv", "--at", "hud=1", "--at", "results=2", "-o", str(tmp_path / "l.yaml"),
        *option,
    )  # fmt: skip

    assert result.returncode == 2
    assert "--manual" in result.stderr
