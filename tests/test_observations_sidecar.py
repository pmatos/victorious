"""`--observations` dumps every raw reading as JSON lines, one per HUD element per second."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.analyse_helpers import FfmpegShim, analyse_text, prepare, small
from tests.synthetic import ELEMENT_RECTS, Hud, Shot, layout_document, match_shots
from victorious.schema import ObservationRecord

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

DURATION = 30
ORDER = [
    "hud_anchor",
    "personal_eliminations",
    "team_eliminations",
    "players_left",
    "results_banner",
    "placement",
    "elimination_placement",
]
STEPS = ((0.0, 0, 0, 99), (10.0, 1, 0, 98))


def parse(path: Path) -> list[ObservationRecord]:
    return [ObservationRecord.model_validate_json(line) for line in path.read_text().splitlines()]


def at(records: list[ObservationRecord], second: int, element: str) -> ObservationRecord:
    (found,) = (r for r in records if r.second == second and r.element == element)
    return found


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory: pytest.TempPathFactory) -> list[ObservationRecord]:
    directory = tmp_path_factory.mktemp("sidecar")
    spec = small(DURATION, match_shots(2, STEPS, results_at=20, placement=5))
    recording, layout = prepare(directory, spec)
    destination = directory / "observations.jsonl"
    result = analyse_text(
        recording,
        layout,
        "--no-cache",
        "--observations",
        str(destination),
        "-o",
        str(directory / "out.json"),
    )
    assert result.returncode == 0, result.stderr
    return parse(destination)


def test_one_record_per_element_per_second(sidecar: list[ObservationRecord]) -> None:
    assert [r.element for r in sidecar] == ORDER * DURATION
    assert [r.second for r in sidecar] == [s for s in range(DURATION) for _ in ORDER]
    for record in sidecar:
        x, y, w, h = ELEMENT_RECTS[record.element]
        assert (record.region.x, record.region.y) == pytest.approx((x, y))
        assert (record.region.w, record.region.h) == pytest.approx((w, h))


def test_counter_values_and_scores_match_what_is_on_screen(
    sidecar: list[ObservationRecord],
) -> None:
    assert at(sidecar, 5, "hud_anchor").score is not None
    assert at(sidecar, 5, "hud_anchor").score >= 0.7  # type: ignore[operator]
    assert at(sidecar, 0, "hud_anchor").score < 0.7  # type: ignore[operator]

    during = {e: at(sidecar, 15, e) for e in ORDER[1:4]}
    assert [during[e].value for e in ORDER[1:4]] == [1, 0, 98]
    assert all(record.confidence > 0.5 for record in during.values())
    assert all(record.score is None for record in during.values())

    lobby = at(sidecar, 0, "personal_eliminations")
    assert (lobby.value, lobby.confidence) == (None, 0.0)


def test_placement_digits_are_read_on_the_results_screen(sidecar: list[ObservationRecord]) -> None:
    results = at(sidecar, 22, "placement")
    assert results.value == 5
    assert results.score is not None
    assert results.score >= 0.7
    assert results.confidence > 0.5
    assert at(sidecar, 22, "elimination_placement").value is None
    assert at(sidecar, 15, "placement").value is None
    assert at(sidecar, 22, "results_banner").value is None


def test_layout_without_optional_elements_has_fewer_records(tmp_path: Path) -> None:
    document = layout_document()
    document["screens"]["hud"].pop("team_eliminations")  # type: ignore[index]
    document["screens"]["results"].pop("elimination_placement")  # type: ignore[index]
    recording, layout = prepare(tmp_path, small(10, (Shot(Hud(), 2, 8),)), document)
    destination = tmp_path / "observations.jsonl"

    result = analyse_text(recording, layout, "--observations", str(destination))

    assert result.returncode == 0, result.stderr
    records = parse(destination)
    wanted = [e for e in ORDER if e not in {"team_eliminations", "elimination_placement"}]
    assert [r.element for r in records] == wanted * 10


def test_sidecar_is_identical_from_cache_and_from_a_fresh_decode(tmp_path: Path) -> None:
    spec = small(DURATION, match_shots(2, STEPS, results_at=20, placement=5))
    recording, layout = prepare(tmp_path, spec)
    fresh, warm, uncached = (tmp_path / f"{n}.jsonl" for n in ("fresh", "warm", "uncached"))
    shim = FfmpegShim(tmp_path / "shim")

    analyse_text(recording, layout, "--observations", str(fresh), env=shim.env)
    shim.reset()
    analyse_text(recording, layout, "--observations", str(warm), env=shim.env)
    decodes = shim.decodes()
    analyse_text(recording, layout, "--no-cache", "--observations", str(uncached))

    assert decodes == 0
    assert fresh.read_text() == warm.read_text() == uncached.read_text()


def test_sidecar_is_not_written_when_the_analysis_fails(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, small(DURATION, (Shot(Hud(), 2, 25),)))
    shim = FfmpegShim(tmp_path / "shim", fail_when="-ss 10 ")
    destination = tmp_path / "observations.jsonl"

    result = analyse_text(
        recording,
        layout,
        "--jobs",
        "2",
        "--chunk-seconds",
        "10",
        "--observations",
        str(destination),
        env=shim.env,
    )

    assert result.returncode == 1
    assert not destination.exists()
    assert list(tmp_path.glob(".observations.jsonl*")) == []


def test_missing_sidecar_directory_is_a_clean_error(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, small(5, ()))
    destination = tmp_path / "nowhere" / "observations.jsonl"

    result = analyse_text(recording, layout, "--observations", str(destination))

    assert result.returncode == 1
    assert "nowhere" in result.stderr
    assert result.stdout == ""


def test_sidecar_and_output_naming_the_same_file_is_a_clean_error(tmp_path: Path) -> None:
    recording, layout = prepare(tmp_path, small(5, ()))
    shared = tmp_path / "same.json"

    result = analyse_text(recording, layout, "--observations", str(shared), "-o", str(shared))

    assert result.returncode == 1
    assert "same file" in result.stderr
    assert not shared.exists()


def test_sidecar_and_output_naming_one_file_through_a_symlink_is_a_clean_error(
    tmp_path: Path,
) -> None:
    recording, layout = prepare(tmp_path, small(5, ()))
    shared = tmp_path / "same.json"
    linked = tmp_path / "link.json"
    linked.symlink_to(shared)

    result = analyse_text(recording, layout, "--observations", str(shared), "-o", str(linked))

    assert result.returncode == 1
    assert "same file" in result.stderr
    assert not shared.exists()


def test_the_analysis_output_is_unchanged_by_asking_for_observations(tmp_path: Path) -> None:
    spec = small(DURATION, match_shots(2, STEPS, results_at=20, placement=5))
    recording, layout = prepare(tmp_path, spec)

    plain = analyse_text(recording, layout, "--no-cache")
    asked = analyse_text(
        recording, layout, "--no-cache", "--observations", str(tmp_path / "o.jsonl")
    )

    assert plain.returncode == asked.returncode == 0
    assert asked.stdout == plain.stdout
    document: dict[str, Any] = json.loads(asked.stdout)
    assert document["matches"][0]["placement"] == 5
