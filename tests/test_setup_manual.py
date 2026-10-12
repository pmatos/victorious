"""`victorious setup --manual` serves a local page and saves the Layout drawn on it."""

import json
import socket
import stat
import struct
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import jsonschema
import numpy as np
import pytest

from tests.analyse_helpers import (
    EXPECTED_MATCHES,
    HUD_AT,
    LOBBY_HUD_AT,
    LOBBY_RESULTS_AT,
    PLACED_RESULTS_AT,
    WIN_RESULTS_AT,
    analyse_facts,
    intersection_over_union,
    scenario_spec,
)
from tests.cli_runner import run_cli
from tests.manual_server import RunningCli, wait_until
from tests.synthetic import ELEMENT_RECTS, build_recording, decode_frame, write_layout
from victorious.layout import load_layout
from victorious.schema import json_schema

pytestmark = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract")

AT = (f"hud={HUD_AT}", f"results={WIN_RESULTS_AT}", f"results={PLACED_RESULTS_AT}")
LOBBY_AT = (f"hud={LOBBY_HUD_AT}", f"results={LOBBY_RESULTS_AT}")
HUD_ELEMENTS = ("hud_anchor", "personal_eliminations", "team_eliminations", "players_left")
ELIMINATION = "elimination_placement"

Starter = Callable[..., RunningCli]


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("manual-recording")
    return build_recording(scenario_spec(), directory / "recording.mkv")


@pytest.fixture
def start(tmp_path: Path) -> Iterator[Starter]:
    started: list[RunningCli] = []

    def launch(
        recording: Path,
        *at: str,
        layout: str = "layout.yaml",
        extra: tuple[str, ...] = ("--no-browser",),
        env: dict[str, str] | None = None,
    ) -> RunningCli:
        options = [part for item in at for part in ("--at", item)]
        command = [
            "setup",
            str(recording),
            *options,
            "--manual",
            "-o",
            str(tmp_path / layout),
            *extra,
        ]
        running = RunningCli(*command, env=env)
        started.append(running)
        return running

    yield launch
    for running in started:
        running.close()


def drawn(
    panels: dict[str, int], rects: dict[str, tuple[float, float, float, float]] | None = None
) -> dict[str, Any]:
    """The body the page posts: every element of `ELEMENT_RECTS` on the panel given."""
    chosen = rects or ELEMENT_RECTS
    return {
        "elements": {
            name: {"panel": panels[name], "x": x, "y": y, "w": w, "h": h}
            for name, (x, y, w, h) in chosen.items()
        }
    }


THREE_PANELS = {**dict.fromkeys(HUD_ELEMENTS, 0), "results_banner": 1, "placement": 1}
THREE_PANELS[ELIMINATION] = 2
TWO_PANELS = {**dict.fromkeys(HUD_ELEMENTS, 0), "results_banner": 1, "placement": 1}
TWO_PANELS[ELIMINATION] = 1


def saved_rects(layout_path: Path) -> dict[str, dict[str, float]]:
    screens = load_layout(layout_path).model_dump(exclude_none=True)["screens"]
    return {name: rect for screen in screens.values() for name, rect in screen.items()}


def rounded(rects: dict[str, tuple[float, float, float, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {key: round(value, 4) for key, value in zip("xywh", rect, strict=True)}
        for name, rect in rects.items()
    }


def test_the_page_shows_the_frame_of_each_requested_second(recording: Path, start: Starter) -> None:
    running = start(recording, *AT)

    url = running.url()
    status, page, content_type = running.request("")
    state = running.get_json("state.json")

    assert urlparse(url).hostname == "127.0.0.1"
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"<html" in page.lower()
    assert [(p["kind"], p["second"]) for p in state["panels"]] == [
        ("hud", HUD_AT),
        ("results", WIN_RESULTS_AT),
        ("results", PLACED_RESULTS_AT),
    ]
    assert state["aspect_ratio"] == "16:9"
    for index, panel in enumerate(state["panels"]):
        status, body, content_type = running.request(panel["image"])
        assert (status, content_type) == (200, "image/png")
        image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        height, width = image.shape[:2]
        assert (width, height) == (panel["width"], panel["height"])
        assert width / height == pytest.approx(16 / 9, abs=0.01)
        expected = cv2.resize(
            decode_frame(recording, panel["second"]), (width, height), interpolation=cv2.INTER_AREA
        )
        assert np.abs(image.astype(int) - expected.astype(int)).mean() < 3, index
    running.interrupt()


def test_only_the_announced_address_is_served(recording: Path, start: Starter) -> None:
    running = start(recording, *AT)
    base = running.url()
    origin = base[: base.index("/", len("http://"))] + "/"

    for path in ("", "state.json"):
        assert running.request(path, base=origin)[0] == 404
        assert running.request(path, base=origin + "not-the-token/")[0] == 404
    assert running.request("frame/99.png")[0] == 404
    assert running.request("layout")[0] in {404, 405}
    running.interrupt()


def test_auto_detected_rectangles_are_prefilled_and_editable(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *AT)

    state = running.get_json("state.json")

    kinds = [panel["kind"] for panel in state["panels"]]
    assert set(state["rects"]) == set(ELEMENT_RECTS)
    for name, rect in state["rects"].items():
        assert kinds[rect["panel"]] == ("hud" if name in HUD_ELEMENTS else "results"), name
        assert intersection_over_union(rect, ELEMENT_RECTS[name]) >= 0.5, name
    assert state["notes"] == {}

    edited = {name: dict(rect) for name, rect in state["rects"].items()}
    edited["players_left"] = {"panel": 0, "x": 0.8, "y": 0.1, "w": 0.05, "h": 0.03}
    status, reply = running.post_json("layout", {"elements": edited})

    assert status == 200, reply
    assert running.wait() == 0
    assert saved_rects(tmp_path / "layout.yaml")["players_left"] == {
        "x": 0.8, "y": 0.1, "w": 0.05, "h": 0.03,
    }  # fmt: skip


def test_prefill_names_what_auto_detect_could_not_find(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *LOBBY_AT)

    state = running.get_json("state.json")

    assert state["rects"] == {}
    for name in (*HUD_ELEMENTS, "results_banner", "placement"):
        assert name in state["notes"], name
    assert state["required"] == [*HUD_ELEMENTS, "results_banner", "placement"]
    assert [panel["elements"] for panel in state["panels"]] == [
        list(HUD_ELEMENTS),
        ["results_banner", "placement", ELIMINATION],
    ]
    assert running.alive()
    assert list(tmp_path.iterdir()) == []
    running.interrupt()


def test_drawn_rectangles_are_saved_in_normalized_coordinates(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *LOBBY_AT)
    running.url()

    status, reply = running.post_json("layout", drawn(TWO_PANELS))

    assert status == 200, reply
    assert running.wait() == 0
    layout = load_layout(tmp_path / "layout.yaml")
    assert layout.aspect_ratio == "16:9"
    assert saved_rects(tmp_path / "layout.yaml") == rounded(ELEMENT_RECTS)


def test_the_same_drawing_gives_the_same_file_at_another_resolution(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    large = build_recording(scenario_spec(2560, 1440), tmp_path / "large.mkv")
    first = start(recording, *LOBBY_AT, layout="small.yaml")
    second = start(large, *LOBBY_AT, layout="large.yaml")

    for running in (first, second):
        status, reply = running.post_json("layout", drawn(TWO_PANELS))
        assert status == 200, reply
        assert running.wait() == 0

    assert (tmp_path / "small.yaml").read_text() == (tmp_path / "large.yaml").read_text()


def test_a_drawn_layout_is_accepted_by_analyse(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *LOBBY_AT)
    status, reply = running.post_json("layout", drawn(TWO_PANELS))
    assert status == 200, reply
    assert running.wait() == 0
    hand_written = write_layout(tmp_path / "hand.yaml")

    from_manual = analyse_facts(recording, tmp_path / "layout.yaml")

    assert from_manual == analyse_facts(recording, hand_written)
    assert [fact[:5] for fact in from_manual] == EXPECTED_MATCHES
    result = run_cli("analyse", str(recording), "--layout", str(tmp_path / "layout.yaml"))
    jsonschema.validate(json.loads(result.stdout), json_schema())


def test_the_server_exits_after_saving(recording: Path, start: Starter, tmp_path: Path) -> None:
    running = start(recording, *AT)
    url = running.url()
    port = urlparse(url).port
    assert port is not None

    status, _ = running.post_json("layout", drawn(THREE_PANELS))

    assert status == 200
    assert running.wait() == 0
    with (
        pytest.raises(ConnectionRefusedError),
        socket.create_connection(("127.0.0.1", port), timeout=2),
    ):
        pass
    assert str(tmp_path / "layout.yaml") in running.stderr_text
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "layout.preview.png",
        "layout.yaml",
    ]
    assert cv2.imread(str(tmp_path / "layout.preview.png")) is not None


def test_a_save_that_is_incomplete_or_invalid_is_refused_and_the_server_keeps_running(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *AT)
    running.url()
    complete = drawn(THREE_PANELS)["elements"]
    without_counter = {k: v for k, v in complete.items() if k != "players_left"}
    refused: list[tuple[str, dict[str, Any], str]] = [
        ("missing", {"elements": without_counter}, "players_left"),
        ("unknown", {"elements": {**complete, "minimap": complete["placement"]}}, "minimap"),
        (
            "wrong kind",
            {"elements": {**complete, "placement": {**complete["placement"], "panel": 0}}},
            "placement",
        ),
        (
            "outside",
            {"elements": {**complete, "hud_anchor": {**complete["hud_anchor"], "x": 0.99}}},
            "hud_anchor",
        ),
        (
            "no such panel",
            {"elements": {**complete, "hud_anchor": {**complete["hud_anchor"], "panel": 7}}},
            "hud_anchor",
        ),
        ("not an object", {"elements": []}, "elements"),
    ]

    for label, document, named in refused:
        status, reply = running.post_json("layout", document)
        assert status == 400, label
        assert any(named in error for error in reply["errors"]), (label, reply)
        assert running.alive()
        assert list(tmp_path.iterdir()) == [], label

    status, reply = running.post_json("layout", {"elements": complete})
    assert status == 200, reply
    assert running.wait() == 0
    assert (tmp_path / "layout.yaml").exists()


def test_a_layout_that_does_not_read_back_is_saved_with_a_warning(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *LOBBY_AT)
    running.url()

    status, reply = running.post_json("layout", drawn(TWO_PANELS))

    assert status == 200, reply
    assert running.wait() == 0
    assert reply["warnings"]
    named = " ".join(reply["warnings"])
    for name in ("hud_anchor", "players_left", "results_banner"):
        assert name in named
    for warning in reply["warnings"]:
        assert warning in running.stderr_text
    assert (tmp_path / "layout.yaml").exists()


def test_a_layout_drawn_on_the_hud_reads_back_without_warnings(
    recording: Path, start: Starter
) -> None:
    running = start(recording, *AT)

    status, reply = running.post_json("layout", drawn(THREE_PANELS))

    assert (status, reply) == (200, {"warnings": []})
    assert running.wait() == 0


def test_ctrl_c_cancels_without_writing(recording: Path, start: Starter, tmp_path: Path) -> None:
    running = start(recording, *AT)
    running.url()

    running.interrupt()

    assert running.wait() == 1
    assert "cancelled" in running.stderr_text
    assert list(tmp_path.iterdir()) == []


def test_the_browser_is_opened_on_the_announced_url(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    opened = tmp_path / "opened.txt"
    script = tmp_path / "open-browser"
    script.write_text(f'#!/bin/sh\nprintf %s "$1" > {opened}\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    running = start(recording, *AT, extra=(), env={"BROWSER": str(script)})

    url = running.url()

    assert wait_until(opened.exists)
    assert opened.read_text() == url
    running.interrupt()


def test_a_failed_write_is_reported_to_the_page_and_the_server_keeps_running(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *AT)
    running.url()
    (tmp_path / "layout.yaml").mkdir()

    status, reply = running.post_json("layout", drawn(THREE_PANELS))

    assert status == 500
    assert reply["errors"]
    assert running.alive()
    (tmp_path / "layout.yaml").rmdir()
    status, reply = running.post_json("layout", drawn(THREE_PANELS))
    assert status == 200, reply
    assert running.wait() == 0
    assert load_layout(tmp_path / "layout.yaml")


def test_a_fixed_port_can_be_requested(recording: Path, start: Starter) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    running = start(recording, *AT, extra=("--no-browser", "--port", str(port)))

    assert urlparse(running.url()).port == port
    running.interrupt()


def test_a_port_that_is_taken_is_reported_before_serving(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        running = start(recording, *AT, extra=("--no-browser", "--port", str(port)))

        assert running.wait(timeout=90) == 1
    assert str(port) in running.stderr_text or "in use" in running.stderr_text
    assert list(tmp_path.iterdir()) == []


def test_the_environment_leaves_no_browser_process_when_none_is_wanted(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    marker = tmp_path / "opened.txt"
    script = tmp_path / "open-browser"
    script.write_text(f"#!/bin/sh\ntouch {marker}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    running = start(recording, *AT, env={"BROWSER": str(script)})

    running.url()
    time.sleep(0.5)

    assert not marker.exists()
    running.interrupt()


def test_a_rectangle_flush_with_the_frame_edge_survives_rounding(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *AT)
    running.url()
    document = drawn(THREE_PANELS)
    document["elements"]["placement"].update(x=0.84996, w=0.15006)

    status, reply = running.post_json("layout", document)

    assert status == 200, reply
    assert running.wait() == 0
    placement = saved_rects(tmp_path / "layout.yaml")["placement"]
    assert placement["x"] + placement["w"] == pytest.approx(1.0, abs=1e-9)


def test_requests_the_server_cannot_parse_are_answered_not_dropped(
    recording: Path, start: Starter
) -> None:
    running = start(recording, *AT)
    running.url()

    nested = running.request("layout", b"[" * 60_000)
    huge_frame = running.request("frame/" + "9" * 5000 + ".png")

    assert nested[0] == 400
    assert huge_frame[0] == 404
    assert running.alive()
    running.interrupt()


def test_a_save_survives_the_page_going_away_before_the_answer(
    recording: Path, start: Starter, tmp_path: Path
) -> None:
    running = start(recording, *AT)
    url = running.url()
    parsed = urlparse(url)
    body = json.dumps(drawn(THREE_PANELS)).encode()
    request = (
        f"POST {parsed.path}layout HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body

    with socket.create_connection(("127.0.0.1", parsed.port or 0), timeout=10) as connection:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        connection.sendall(request)

    assert running.wait(timeout=60) == 0
    assert load_layout(tmp_path / "layout.yaml")
    assert "cancelled" not in running.stderr_text
