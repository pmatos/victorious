"""`victorious review` serves the detected Matches and saves the verdicts as a labels file."""

import json
import shutil
import socket
import struct
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import pytest

from tests.analyse_helpers import scenario_spec
from tests.cli_runner import run_cli
from tests.manual_server import RunningCli
from tests.synthetic import build_recording, decode_frame, write_layout

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

Starter = Callable[..., RunningCli]
EVERYTHING = {"0": "confirmed", "1": "rejected", "2": "confirmed"}
RESULTS_SETTLE_LIMIT = 5


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("review-recording")
    return build_recording(scenario_spec(), directory / "recording.mkv")


@pytest.fixture(scope="module")
def analysed(recording: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("review-analysis")
    layout = write_layout(directory / "layout.yaml")
    path = directory / "analysis.json"
    result = run_cli(
        "analyse", str(recording), "--layout", str(layout), "--no-cache", "-o", str(path)
    )
    assert result.returncode == 0, result.stderr
    return path


@pytest.fixture
def workdir(tmp_path: Path, analysed: Path) -> Path:
    shutil.copy(analysed, tmp_path / "analysis.json")
    return tmp_path


@pytest.fixture
def start(workdir: Path) -> Iterator[Starter]:
    started: list[RunningCli] = []

    def launch(
        *extra: str,
        analysis: str = "analysis.json",
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> RunningCli:
        options = extra if "--no-browser" in extra else ("--no-browser", *extra)
        running = RunningCli("review", str(workdir / analysis), *options, env=env, cwd=cwd)
        started.append(running)
        return running

    yield launch
    for running in started:
        running.close()


def edited(
    workdir: Path, change: Callable[[dict[str, Any]], None], name: str = "edited.json"
) -> str:
    document = json.loads((workdir / "analysis.json").read_text())
    change(document)
    (workdir / name).write_text(json.dumps(document))
    return name


def fetch_image(running: RunningCli, route: str) -> np.ndarray:
    status, body, content_type = running.request(route)
    assert (status, content_type) == (200, "image/jpeg"), route
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return np.asarray(image)


def last_value(match: dict[str, Any], kind: str) -> int | None:
    values = [event["value"] for event in match["events"] if event["kind"] == kind]
    return values[-1] if values else None


def test_the_page_lists_every_match_with_its_two_frames_and_the_values_read(
    recording: Path, analysed: Path, start: Starter
) -> None:
    running = start()
    matches = json.loads(analysed.read_text())["matches"]

    status, page, content_type = running.request("")
    state = running.get_json("state.json")

    assert urlparse(running.url()).hostname == "127.0.0.1"
    assert (status, content_type.split(";")[0]) == (200, "text/html")
    assert b"<html" in page.lower()
    assert state["recording"] == "recording.mkv"
    assert len(state["matches"]) == len(matches) == 3
    for card, match in zip(state["matches"], matches, strict=True):
        assert card["index"] == match["index"]
        assert (card["start"], card["end"]) == (match["start"], match["end"])
        assert card["win"] is match["win"]
        assert card["placement"] == match["placement"]
        assert card["personal_eliminations"] == match["personal_eliminations"]
        assert card["team_eliminations"] == last_value(match, "team_eliminations")
        assert card["players_left"] == last_value(match, "players_left")
        assert card["has_results"] is True
    first = state["matches"][0]
    assert (first["win"], first["placement"], first["personal_eliminations"]) == (True, 1, 1)
    assert (first["team_eliminations"], first["players_left"]) == (1, 90)
    running.interrupt()


def test_each_frame_is_the_frame_of_its_second_and_the_two_differ(
    recording: Path, start: Starter
) -> None:
    running = start()

    state = running.get_json("state.json")

    for card in state["matches"]:
        begun, finished = card["start_frame"], card["results_frame"]
        assert begun["second"] == card["start"]
        assert card["end"] <= finished["second"] <= card["end"] + RESULTS_SETTLE_LIMIT
        images = []
        for frame in (begun, finished):
            image = fetch_image(running, frame["image"])
            height, width = image.shape[:2]
            assert (width, height) == (1600, 900)
            expected = cv2.resize(
                decode_frame(recording, frame["second"]),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            assert np.abs(image.astype(int) - expected.astype(int)).mean() < 3, frame
            images.append(image)
        assert np.abs(images[0].astype(int) - images[1].astype(int)).mean() > 0.1
    running.interrupt()


def test_a_match_without_a_results_screen_is_flagged(workdir: Path, start: Starter) -> None:
    def drop_results(document: dict[str, Any]) -> None:
        for event in document["matches"][1]["events"]:
            if event["kind"] == "match_end":
                event["confidence"] = 0.0

    running = start(analysis=edited(workdir, drop_results))

    cards = running.get_json("state.json")["matches"]

    assert [card["has_results"] for card in cards] == [True, False, True]
    running.interrupt()


def test_only_the_announced_address_and_known_routes_are_served(start: Starter) -> None:
    running = start()
    base = running.url()
    origin = base[: base.index("/", len("http://"))] + "/"
    card = running.get_json("state.json")["matches"][0]

    for path in ("", "state.json", card["start_frame"]["image"]):
        assert running.request(path, base=origin)[0] == 404
        assert running.request(path, base=origin + "not-the-token/")[0] == 404
    assert running.request("frame/99-start.jpg")[0] == 404
    assert running.request("frame/" + "9" * 5000 + "-start.jpg")[0] == 404
    assert running.request("layout", b"{}")[0] == 404
    assert running.request("state.json", b"{}")[0] == 404
    assert running.alive()
    running.interrupt()


def test_confirmed_and_rejected_matches_are_saved_to_a_labels_file(
    analysed: Path, workdir: Path, start: Starter
) -> None:
    running = start()
    running.url()

    status, reply = running.post_json("labels", {"verdicts": EVERYTHING})

    assert status == 200, reply
    assert running.wait() == 0
    saved = json.loads((workdir / "analysis.labels.json").read_text())
    matches = json.loads(analysed.read_text())["matches"]
    assert saved["labels_version"] == "1.0"
    assert saved["recording"]["path"] == json.loads(analysed.read_text())["recording"]["path"]
    assert saved["analysis"]["schema_version"] == "1.7"
    assert [item["verdict"] for item in saved["matches"]] == ["confirmed", "rejected", "confirmed"]
    for item, match in zip(saved["matches"], matches, strict=True):
        assert {key: item[key] for key in ("start", "end", "win", "placement")} == {
            key: match[key] for key in ("start", "end", "win", "placement")
        }
        assert item["personal_eliminations"] == match["personal_eliminations"]


def test_the_server_exits_after_saving_and_leaves_only_the_labels(
    workdir: Path, start: Starter
) -> None:
    running = start()
    port = urlparse(running.url()).port
    assert port is not None

    status, _ = running.post_json("labels", {"verdicts": EVERYTHING})

    assert status == 200
    assert running.wait() == 0
    with (
        pytest.raises(ConnectionRefusedError),
        socket.create_connection(("127.0.0.1", port), timeout=2),
    ):
        pass
    assert str(workdir / "analysis.labels.json") in running.stderr_text
    assert "2 confirmed, 1 rejected" in running.stderr_text
    assert sorted(path.name for path in workdir.iterdir()) == [
        "analysis.json",
        "analysis.labels.json",
    ]


def test_a_save_that_is_incomplete_or_invalid_is_refused_and_the_server_keeps_running(
    workdir: Path, start: Starter
) -> None:
    running = start()
    running.url()
    refused: list[tuple[str, object, str]] = [
        ("incomplete", {"verdicts": {"0": "confirmed", "1": "rejected"}}, "2"),
        ("unknown index", {"verdicts": {**EVERYTHING, "7": "confirmed"}}, "7"),
        ("bad verdict", {"verdicts": {**EVERYTHING, "1": "maybe"}}, "maybe"),
        ("not an object", {"verdicts": []}, "verdicts"),
        ("no verdicts", {}, "verdicts"),
        ("not a document", [], "verdicts"),
    ]

    for label, document, named in refused:
        status, reply = running.post_json("labels", document)
        assert status == 400, label
        assert any(named in error for error in reply["errors"]), (label, reply)
        assert running.alive()
        assert [path.name for path in workdir.iterdir()] == ["analysis.json"], label

    status, reply = running.post_json("labels", {"verdicts": EVERYTHING})
    assert status == 200, reply
    assert running.wait() == 0


def test_a_body_the_server_cannot_parse_is_answered_not_dropped(start: Starter) -> None:
    running = start()
    running.url()

    nested = running.request("labels", b"[" * 60_000)

    assert nested[0] == 400
    assert running.alive()
    running.interrupt()


def test_ctrl_c_cancels_without_writing(workdir: Path, start: Starter) -> None:
    running = start()
    running.url()

    running.interrupt()

    assert running.wait() == 1
    assert "cancelled" in running.stderr_text
    assert [path.name for path in workdir.iterdir()] == ["analysis.json"]


def test_the_labels_go_where_output_says(workdir: Path, start: Starter) -> None:
    (workdir / "elsewhere").mkdir()
    running = start("-o", str(workdir / "elsewhere" / "mine.json"))
    running.url()

    status, _ = running.post_json("labels", {"verdicts": EVERYTHING})

    assert status == 200
    assert running.wait() == 0
    assert (workdir / "elsewhere" / "mine.json").exists()
    assert not (workdir / "analysis.labels.json").exists()


def test_an_existing_labels_file_is_replaced(workdir: Path, start: Starter) -> None:
    target = workdir / "analysis.labels.json"
    target.write_text("old")
    running = start()
    running.url()

    status, _ = running.post_json("labels", {"verdicts": EVERYTHING})

    assert status == 200
    assert running.wait() == 0
    assert json.loads(target.read_text())["labels_version"] == "1.0"


def test_a_missing_output_directory_is_reported_before_serving(
    workdir: Path, start: Starter
) -> None:
    running = start("-o", str(workdir / "nowhere" / "labels.json"))

    assert running.wait(timeout=60) == 1
    assert "nowhere" in running.stderr_text
    assert "http://" not in running.stderr_text


def test_output_must_not_be_the_analysis_or_the_recording(
    workdir: Path, recording: Path, start: Starter
) -> None:
    for protected in (workdir / "analysis.json", recording):
        running = start("-o", str(protected))

        assert running.wait(timeout=60) == 1
        assert "http://" not in running.stderr_text
    assert json.loads((workdir / "analysis.json").read_text())["schema_version"] == "1.7"


def test_a_recording_path_given_overrides_the_one_in_the_analysis(
    recording: Path, workdir: Path, start: Starter
) -> None:
    def lose(document: dict[str, Any]) -> None:
        document["recording"]["path"] = "/nowhere/recording.mkv"

    running = start("--recording", str(recording), analysis=edited(workdir, lose))

    cards = running.get_json("state.json")["matches"]

    assert len(cards) == 3
    running.interrupt()


def test_a_missing_recording_is_reported_before_serving(workdir: Path, start: Starter) -> None:
    def lose(document: dict[str, Any]) -> None:
        document["recording"]["path"] = "/nowhere/recording.mkv"

    running = start(analysis=edited(workdir, lose))

    assert running.wait(timeout=60) == 1
    assert "/nowhere/recording.mkv" in running.stderr_text
    assert "--recording" in running.stderr_text
    assert "Traceback" not in running.stderr_text
    assert "http://" not in running.stderr_text


def test_a_recording_of_another_length_than_the_analysed_one_is_refused(
    workdir: Path, start: Starter
) -> None:
    def lengthen(document: dict[str, Any]) -> None:
        document["recording"]["duration"] += 60

    running = start(analysis=edited(workdir, lengthen))

    assert running.wait(timeout=60) == 1
    assert "--recording" in running.stderr_text
    assert "Traceback" not in running.stderr_text
    assert "http://" not in running.stderr_text


def test_a_relative_recording_path_is_read_from_the_working_directory(
    recording: Path, workdir: Path, start: Starter
) -> None:
    shutil.copy(recording, workdir / "recording.mkv")
    name = edited(workdir, lambda document: document["recording"].update(path="recording.mkv"))

    running = start(analysis=name, cwd=workdir)

    assert len(running.get_json("state.json")["matches"]) == 3
    running.interrupt()


def test_an_analysis_with_no_matches_is_served_and_saves_an_empty_labels_file(
    workdir: Path, start: Starter
) -> None:
    def empty(document: dict[str, Any]) -> None:
        document["matches"] = []
        document["layout"] = None

    running = start(analysis=edited(workdir, empty))

    state = running.get_json("state.json")
    status, reply = running.post_json("labels", {"verdicts": {}})

    assert state["matches"] == []
    assert status == 200, reply
    assert running.wait() == 0
    saved = json.loads((workdir / "edited.labels.json").read_text())
    assert saved["matches"] == []


def test_an_analysis_of_another_schema_version_is_refused_without_a_traceback(
    workdir: Path, start: Starter
) -> None:
    running = start(
        analysis=edited(workdir, lambda document: document.update(schema_version="1.4"))
    )

    assert running.wait(timeout=60) == 1
    assert "1.4" in running.stderr_text
    assert "analyse" in running.stderr_text
    assert "Traceback" not in running.stderr_text


def test_a_failed_write_is_reported_to_the_page_and_the_server_keeps_running(
    workdir: Path, start: Starter
) -> None:
    running = start()
    running.url()
    (workdir / "analysis.labels.json").mkdir()

    status, reply = running.post_json("labels", {"verdicts": EVERYTHING})

    assert status == 500
    assert reply["errors"]
    assert running.alive()
    (workdir / "analysis.labels.json").rmdir()
    status, reply = running.post_json("labels", {"verdicts": EVERYTHING})
    assert status == 200, reply
    assert running.wait() == 0


def test_a_save_survives_the_page_going_away_before_the_answer(
    workdir: Path, start: Starter
) -> None:
    running = start()
    parsed = urlparse(running.url())
    body = json.dumps({"verdicts": EVERYTHING}).encode()
    request = (
        f"POST {parsed.path}labels HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body

    with socket.create_connection(("127.0.0.1", parsed.port or 0), timeout=10) as connection:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        connection.sendall(request)

    assert running.wait(timeout=60) == 0
    assert (workdir / "analysis.labels.json").exists()
    assert "cancelled" not in running.stderr_text


def test_a_fixed_port_can_be_requested_and_a_taken_one_is_reported(
    workdir: Path, start: Starter
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    running = start("--port", str(port))
    assert urlparse(running.url()).port == port
    running.interrupt()

    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        taken = holder.getsockname()[1]
        blocked = start("--port", str(taken))

        assert blocked.wait(timeout=90) == 1
    assert str(taken) in blocked.stderr_text or "in use" in blocked.stderr_text


def test_what_was_saved_compares_clean_against_the_same_analysis_and_not_a_changed_one(
    workdir: Path, start: Starter
) -> None:
    running = start()
    running.url()
    status, _ = running.post_json("labels", {"verdicts": {**EVERYTHING, "1": "confirmed"}})
    assert status == 200
    assert running.wait() == 0
    labels = workdir / "analysis.labels.json"

    same = run_cli("review", str(workdir / "analysis.json"), "--against", str(labels))
    assert (same.returncode, same.stdout) == (0, "no differences (3 matches)\n")

    def worse(document: dict[str, Any]) -> None:
        document["matches"][0].update(win=False, placement=3)

    changed = run_cli("review", str(workdir / edited(workdir, worse)), "--against", str(labels))
    assert changed.returncode == 1
    assert "changed" in changed.stdout
    assert "placement: expected 1, found 3" in changed.stdout
