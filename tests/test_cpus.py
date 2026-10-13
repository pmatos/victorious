"""How many threads CPU-bound libraries should use: one per physical core."""

import os
from pathlib import Path

import pytest

from victorious.cpus import physical_cores


@pytest.fixture(autouse=True)
def sixteen_logical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 16)


def test_with_hyper_threading_on_there_is_one_core_per_two_logical_cpus(tmp_path: Path) -> None:
    flag = tmp_path / "active"
    flag.write_text("1\n")

    assert physical_cores(flag) == 8


def test_with_hyper_threading_off_every_logical_cpu_is_a_core(tmp_path: Path) -> None:
    flag = tmp_path / "active"
    flag.write_text("0\n")

    assert physical_cores(flag) == 16


def test_without_the_flag_every_logical_cpu_counts(tmp_path: Path) -> None:
    assert physical_cores(tmp_path / "missing") == 16


def test_one_logical_cpu_is_one_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 1)
    flag = tmp_path / "active"
    flag.write_text("1\n")

    assert physical_cores(flag) == 1
