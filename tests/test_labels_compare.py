"""`victorious review ANALYSIS --against LABELS` reports how a run differs from a labels file."""

import json
import re
from pathlib import Path

import pytest

from tests.cli_runner import run_cli
from tests.labels_helpers import RECORDING, analysis, label, match, write_analysis, write_labels
from victorious.labels import LabelsError, compare, load_labels
from victorious.schema import AnalysisOutput, Match

THREE = (
    match(0, 0, 134, win=True, placement=1, eliminations=4),
    match(1, 200, 340, placement=6, eliminations=1),
    match(2, 400, 520, placement=12, eliminations=0),
)
THREE_LABELS = (
    label(0, 134, win=True, placement=1, eliminations=4),
    label(200, 340, placement=6, eliminations=1),
    label(400, 520, placement=12, eliminations=0),
)


def against(
    tmp_path: Path,
    matches: tuple[Match, ...],
    labels: tuple[dict[str, object], ...],
    *options: str,
) -> tuple[int, list[str], str]:
    path = write_analysis(tmp_path, analysis(*matches))
    labels_path = write_labels(tmp_path, *labels)
    result = run_cli("review", str(path), "--against", str(labels_path), *options)
    assert "Traceback" not in result.stderr
    return result.returncode, result.stdout.splitlines(), result.stderr


def kinds(lines: list[str]) -> list[str]:
    return [line.split()[0] for line in lines if re.match(r"[a-z]+\s", line)]


def test_a_run_that_matches_the_labels_is_clean(tmp_path: Path) -> None:
    code, lines, stderr = against(tmp_path, THREE, THREE_LABELS)

    assert (code, lines) == (0, ["no differences (3 matches)"])
    assert stderr == ""


def test_a_changed_reading_names_the_field_what_was_expected_and_what_was_found(
    tmp_path: Path,
) -> None:
    found = (match(0, 0, 134, win=False, placement=3, eliminations=2), *THREE[1:])

    code, lines, _ = against(tmp_path, found, THREE_LABELS)

    assert code == 1
    assert kinds(lines) == ["changed", "changed", "changed"]
    text = "\n".join(lines)
    assert "0:00:00-0:02:14" in text
    assert re.search(r"win.*expected True.*found False", text)
    assert re.search(r"placement.*expected 1.*found 3", text)
    assert re.search(r"personal_eliminations.*expected 4.*found 2", text)
    assert "3 differences" in lines[-1]


def test_a_confirmed_match_that_is_no_longer_found_is_missing(tmp_path: Path) -> None:
    code, lines, _ = against(tmp_path, THREE[:2], THREE_LABELS)

    assert code == 1
    assert kinds(lines) == ["missing"]
    assert "0:06:40-0:08:40" in lines[0]


def test_a_match_that_was_never_labelled_is_unlabelled(tmp_path: Path) -> None:
    code, lines, _ = against(tmp_path, THREE, THREE_LABELS[:2])

    assert code == 1
    assert kinds(lines) == ["unlabelled"]
    assert "0:06:40-0:08:40" in lines[0]


def test_matches_pair_by_time_not_by_index(tmp_path: Path) -> None:
    inserted = (
        match(0, 0, 134, win=True, placement=1, eliminations=4),
        match(1, 150, 190, placement=40, eliminations=0),
        match(2, 200, 340, placement=6, eliminations=1),
        match(3, 400, 520, placement=12, eliminations=0),
    )

    code, lines, _ = against(tmp_path, inserted, THREE_LABELS)

    assert code == 1
    assert kinds(lines) == ["unlabelled"]
    assert "0:02:30-0:03:10" in lines[0]


def test_boundaries_within_the_tolerance_are_clean_and_beyond_it_are_shifted(
    tmp_path: Path,
) -> None:
    near = (match(0, 1, 135, win=True, placement=1, eliminations=4), *THREE[1:])
    far = (match(0, 0, 140, win=True, placement=1, eliminations=4), *THREE[1:])

    assert against(tmp_path, near, THREE_LABELS)[0] == 0
    code, lines, _ = against(tmp_path, far, THREE_LABELS)
    assert code == 1
    assert kinds(lines) == ["shifted"]
    assert "end" in lines[0]
    assert against(tmp_path, far, THREE_LABELS, "--tolerance", "10")[0] == 0
    assert against(tmp_path, near, THREE_LABELS, "--tolerance", "0")[0] == 1


def test_a_rejected_match_that_is_gone_is_clean(tmp_path: Path) -> None:
    labels = (*THREE_LABELS[:2], label(400, 520, "rejected", placement=12, eliminations=0))

    code, lines, _ = against(tmp_path, THREE[:2], labels)

    assert (code, lines) == (0, ["no differences (2 matches)"])


def test_a_rejected_match_that_comes_back_the_same_has_reappeared(tmp_path: Path) -> None:
    labels = (*THREE_LABELS[:2], label(400, 520, "rejected", placement=12, eliminations=0))

    code, lines, _ = against(tmp_path, THREE, labels)

    assert code == 1
    assert kinds(lines) == ["reappeared"]
    assert "0:06:40-0:08:40" in lines[0]


def test_a_rejected_match_that_is_now_read_differently_asks_for_a_re_review(
    tmp_path: Path,
) -> None:
    labels = (*THREE_LABELS[:2], label(400, 520, "rejected", placement=99, eliminations=0))

    code, lines, _ = against(tmp_path, THREE, labels)

    assert code == 1
    assert kinds(lines) == ["changed"]
    assert "previously rejected" in lines[0]
    assert "re-review" in lines[0]


def test_a_match_split_in_two_is_a_missing_match_and_two_unlabelled(tmp_path: Path) -> None:
    split = (
        match(0, 0, 134, win=True, placement=1, eliminations=4),
        match(1, 200, 250, placement=6, eliminations=1),
        match(2, 290, 340, placement=6, eliminations=1),
        match(3, 400, 520, placement=12, eliminations=0),
    )

    code, lines, _ = against(tmp_path, split, THREE_LABELS)

    assert code == 1
    assert sorted(kinds(lines)) == ["missing", "unlabelled", "unlabelled"]


def test_an_analysis_with_no_matches_against_no_labels_is_clean(tmp_path: Path) -> None:
    code, lines, _ = against(tmp_path, (), ())

    assert (code, lines) == (0, ["no differences (0 matches)"])


def test_labels_for_another_recording_are_refused_naming_both(tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE))
    labels = write_labels(tmp_path, *THREE_LABELS, path="/videos/another session.mkv")

    result = run_cli("review", str(path), "--against", str(labels))

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("victorious review:")
    assert "another session.mkv" in result.stderr
    assert Path(RECORDING).name in result.stderr


def test_the_same_recording_name_in_another_folder_is_still_the_recording(
    tmp_path: Path,
) -> None:
    path = write_analysis(tmp_path, analysis(*THREE, path="relative/2026-03-01 session.mkv"))
    labels = write_labels(tmp_path, *THREE_LABELS)

    result = run_cli("review", str(path), "--against", str(labels))

    assert result.returncode == 0, result.stderr


def test_a_recording_whose_length_differs_is_another_recording(tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE, duration=3590.0))
    labels = write_labels(tmp_path, *THREE_LABELS)

    result = run_cli("review", str(path), "--against", str(labels))

    assert result.returncode == 1
    assert "3590" in result.stderr
    assert "3600" in result.stderr


def test_a_corrupt_labels_file_is_an_error_naming_the_file(tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE))
    labels = tmp_path / "labels.json"
    labels.write_text('{"labels_version": "1.0", "matches": [')

    result = run_cli("review", str(path), "--against", str(labels))

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert str(labels) in result.stderr


def test_an_unknown_labels_version_is_refused(tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE))
    labels = write_labels(tmp_path, *THREE_LABELS, labels_version="2.0")

    result = run_cli("review", str(path), "--against", str(labels))

    assert result.returncode == 1
    assert "labels_version" in result.stderr
    assert "2.0" in result.stderr


def test_a_missing_labels_file_is_an_error(tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE))

    result = run_cli("review", str(path), "--against", str(tmp_path / "nope.json"))

    assert result.returncode == 1
    assert "nope.json" in result.stderr


def test_an_analysis_of_another_schema_version_says_to_run_analyse_again(
    tmp_path: Path,
) -> None:
    document = json.loads(write_analysis(tmp_path, analysis(*THREE)).read_text())
    document["schema_version"] = "1.4"
    path = tmp_path / "old.json"
    path.write_text(json.dumps(document))
    labels = write_labels(tmp_path, *THREE_LABELS)

    result = run_cli("review", str(path), "--against", str(labels))

    assert result.returncode == 1
    assert "1.4" in result.stderr
    assert "analyse" in result.stderr
    assert "Traceback" not in result.stderr


def test_an_analysis_that_is_not_json_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    path.write_text("not json")
    labels = write_labels(tmp_path, *THREE_LABELS)

    result = run_cli("review", str(path), "--against", str(labels))

    assert result.returncode == 1
    assert str(path) in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "option", [["-o", "x.json"], ["--recording", "r.mkv"], ["--no-browser"], ["--port", "0"]]
)
def test_serving_options_are_refused_with_against(option: list[str], tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE))
    labels = write_labels(tmp_path, *THREE_LABELS)

    result = run_cli("review", str(path), "--against", str(labels), *option)

    assert result.returncode == 2
    assert "--against" in result.stderr


def test_a_tolerance_needs_against(tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE))

    result = run_cli("review", str(path), "--tolerance", "3", "--no-browser")

    assert result.returncode == 2
    assert "--against" in result.stderr


def test_a_negative_tolerance_is_a_usage_error(tmp_path: Path) -> None:
    path = write_analysis(tmp_path, analysis(*THREE))
    labels = write_labels(tmp_path, *THREE_LABELS)

    result = run_cli("review", str(path), "--against", str(labels), "--tolerance", "-1")

    assert result.returncode == 2


def test_the_python_api_reports_the_same_differences(tmp_path: Path) -> None:
    found = (match(0, 0, 134, win=True, placement=2, eliminations=4), THREE[1])
    path = write_analysis(tmp_path, analysis(*found))
    labels = write_labels(tmp_path, *THREE_LABELS)

    differences = compare(load_labels(labels), AnalysisOutput.model_validate_json(path.read_text()))

    assert sorted(difference.kind for difference in differences) == ["changed", "missing"]
    with pytest.raises(LabelsError, match="nope"):
        load_labels(tmp_path / "nope.json")
