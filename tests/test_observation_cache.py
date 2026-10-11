"""Observations are cached by Recording and Layout, so a rerun skips decoding."""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.analyse_helpers import FfmpegShim, analyse_text, fast, prepare
from tests.synthetic import Hud, RecordingSpec, Shot, layout_document, match_shots, write_layout

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")


def with_a_jump(placement: int = 3) -> RecordingSpec:
    steps = ((0.0, 0, 0, 99), (8.3, 2, 0, 99), (14.0, 2, 1, 98))
    return fast(30, match_shots(2, steps, results_at=22, placement=placement))


def cached_entries(root: Path) -> list[Path]:
    return sorted(root.rglob("observations.json"))


def xdg_root() -> Path:
    return Path(os.environ["XDG_CACHE_HOME"]) / "victorious"


def test_second_run_skips_decoding(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())
    shim = FfmpegShim(tmp_path / "shim")

    first = analyse_text(recording, layout, env=shim.env)
    cold_decodes = shim.decodes(chunks_only=True)
    shim.reset()
    second = analyse_text(recording, layout, env=shim.env)

    assert first.returncode == 0, first.stderr
    assert cold_decodes >= 2  # the chunk and the window that times the jump
    assert second.returncode == 0, second.stderr
    assert shim.decodes() == 0
    assert second.stdout == first.stdout
    assert '"personal_eliminations"' in second.stdout


def test_changing_the_layout_invalidates_the_cache(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())
    shim = FfmpegShim(tmp_path / "shim")
    analyse_text(recording, layout, env=shim.env)

    document = layout_document()
    document["screens"]["hud"]["players_left"]["x"] += 0.001  # type: ignore[index]
    moved = write_layout(tmp_path / "moved.yaml", document)
    shim.reset()
    result = analyse_text(recording, moved, env=shim.env)

    assert result.returncode == 0, result.stderr
    assert shim.decodes() > 0
    assert len(cached_entries(xdg_root())) == 2


def test_rewriting_the_layout_file_without_changing_it_keeps_the_cache(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())
    shim = FfmpegShim(tmp_path / "shim")
    analyse_text(recording, layout, env=shim.env)

    document: dict[str, Any] = layout_document()
    reordered = {"screens": dict(reversed(document["screens"].items())), "aspect_ratio": "16:9"}
    rewritten = tmp_path / "rewritten.yaml"
    rewritten.write_text("# the same Layout\n" + yaml.safe_dump(reordered, sort_keys=False))
    shim.reset()
    result = analyse_text(recording, rewritten, env=shim.env)

    assert result.returncode == 0, result.stderr
    assert shim.decodes() == 0


def test_a_different_recording_at_the_same_path_is_not_served_from_cache(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump(placement=3))
    before = analyse_text(recording, layout)
    assert before.returncode == 0, before.stderr
    assert '"placement": 3' in before.stdout

    recording, layout = prepare(tmp_path, with_a_jump(placement=9))
    after = analyse_text(recording, layout)

    assert after.returncode == 0, after.stderr
    assert '"placement": 9' in after.stdout


def test_corrupt_cache_entry_is_recomputed(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())
    first = analyse_text(recording, layout)
    (entry,) = cached_entries(xdg_root())
    entry.write_text(entry.read_text()[:100])
    shim = FfmpegShim(tmp_path / "shim")

    second = analyse_text(recording, layout, env=shim.env)

    assert second.returncode == 0, second.stderr
    assert second.stdout == first.stdout
    assert shim.decodes() > 0


def test_no_cache_always_decodes_and_writes_nothing(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())
    shim = FfmpegShim(tmp_path / "shim")

    analyse_text(recording, layout, "--no-cache", env=shim.env)
    first = shim.decodes(chunks_only=True)
    analyse_text(recording, layout, "--no-cache", env=shim.env)

    assert shim.decodes(chunks_only=True) == 2 * first
    assert first > 0
    assert not xdg_root().exists()


def test_cache_dir_option_replaces_the_default_location(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())
    chosen = tmp_path / "chosen"
    shim = FfmpegShim(tmp_path / "shim")

    analyse_text(recording, layout, "--cache-dir", str(chosen), env=shim.env)
    shim.reset()
    again = analyse_text(recording, layout, "--cache-dir", str(chosen), env=shim.env)

    assert again.returncode == 0, again.stderr
    assert len(cached_entries(chosen)) == 1
    assert shim.decodes() == 0
    assert not xdg_root().exists()


def test_default_cache_location_follows_xdg_cache_home(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, with_a_jump())

    result = analyse_text(recording, layout)

    assert result.returncode == 0, result.stderr
    assert len(cached_entries(xdg_root())) == 1


def test_failed_run_leaves_no_cache_entry(tmp_path: Path) -> None:
    shots = (Shot(Hud(), 2, 30),)
    recording, layout = prepare(tmp_path, fast(35, shots))
    shim = FfmpegShim(tmp_path / "shim", fail_when="-ss 20 -t 10")

    result = analyse_text(recording, layout, "--jobs", "2", "--chunk-seconds", "10", env=shim.env)

    assert result.returncode == 1
    assert cached_entries(xdg_root()) == []
