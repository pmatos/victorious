"""`analyse` judges each transcript segment personal or game-related through Jev."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema
import pytest

from tests.analyse_helpers import STEADY, analyse_text, small
from tests.jev_stub import JevStub, Recorded, Reply, Responder, always, choice
from tests.synthetic import AudioClip, build_recording, match_shots, write_layout
from victorious import categorize
from victorious.schema import Event, Match, frame_event, json_schema, transcript_event

end_to_end = pytest.mark.usefixtures("require_ffmpeg", "require_tesseract", "speech_on")

SPEECH = Path(__file__).parent / "fixtures" / "speech"
ENGLISH = SPEECH / "en.opus"
PORTUGUESE = SPEECH / "pt.opus"
KEY = "test-key"


@pytest.fixture
def speech_on(speech_is_off_and_models_are_kept: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTORIOUS_NO_SPEECH")
    monkeypatch.setenv("HF_TOKEN", "test-token")


def recording(tmp_path: Path) -> Path:
    shots = (
        *match_shots(5, STEADY, results_at=25, placement=1),
        *match_shots(65, STEADY, results_at=85, placement=6),
    )
    spec = replace(
        small(100, shots), audio=(AudioClip(ENGLISH, at=8), AudioClip(PORTUGUESE, at=70))
    )
    return build_recording(spec, tmp_path / "recording.mkv")


def analyse_with(
    tmp_path: Path, *options: str, env: dict[str, str]
) -> tuple[int, dict[str, Any] | None, str]:
    layout = write_layout(tmp_path / "layout.yaml")
    result = analyse_text(recording(tmp_path), layout, "--no-cache", *options, env=env)
    document = json.loads(result.stdout) if result.returncode == 0 else None
    if document is not None:
        jsonschema.Draft202012Validator(json_schema()).validate(document)
    return result.returncode, document, result.stderr


def events(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for match in document["matches"] for event in match["events"]]


def transcripts(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in events(document) if event["kind"] == "transcript"]


def others(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in events(document) if event["kind"] != "transcript"]


def configured(stub: JevStub, key: str = KEY) -> dict[str, str]:
    return {"TYPESAFE_API_KEY": key, "VICTORIOUS_JEV_URL": stub.url}


def verdict(text: str) -> Reply:
    if "players" in text.lower():
        return choice("game_related", 0.82)
    return choice("personal", 0.64)


class Run:
    def __init__(self, document: dict[str, Any], stub: JevStub, stderr: str) -> None:
        self.document = document
        self.requests = stub.requests
        self.stderr = stderr


@pytest.fixture(scope="module")
def judged(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """One full run against a stubbed Jev, shared by the tests that only read its output."""
    tmp_path = tmp_path_factory.mktemp("judged")
    layout = write_layout(tmp_path / "layout.yaml")
    with JevStub(verdict) as stub:
        env = {
            "VICTORIOUS_NO_SPEECH": "0",
            "HF_TOKEN": "test-token",
            "TYPESAFE_API_KEY": KEY,
            "VICTORIOUS_JEV_URL": stub.url,
        }
        result = analyse_text(recording(tmp_path), layout, "--no-cache", env=env)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    jsonschema.Draft202012Validator(json_schema()).validate(document)
    return Run(document, stub, result.stderr)


@end_to_end
def test_segments_get_the_returned_category_and_confidence(judged: Run) -> None:
    english, portuguese = transcripts(judged.document)

    assert (english["category"], english["category_confidence"]) == ("game_related", 0.82)
    assert (portuguese["category"], portuguese["category_confidence"]) == ("personal", 0.64)
    assert english["confidence"] != english["category_confidence"]
    for event in others(judged.document):
        assert event["category"] is None
        assert event["category_confidence"] is None
    assert {r.path for r in judged.requests} == {"/v1/systemone"}
    assert {r.headers["authorization"] for r in judged.requests} == {f"Bearer {KEY}"}
    assert "categories 2 of 2" in judged.stderr


@end_to_end
def test_only_the_transcript_text_is_sent(judged: Run) -> None:
    spoken = {event["text"] for event in transcripts(judged.document)}

    assert {r.body["state"] for r in judged.requests} == spoken
    for request in judged.requests:
        assert set(request.body) == {"state", "model", "questions"}
        assert isinstance(request.body["state"], str)
        [question] = request.body["questions"].values()
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"personal", "game_related"}
        assert request.size < 4096


@end_to_end
def test_without_the_key_the_run_stops_before_any_work(tmp_path: Path) -> None:
    with JevStub(verdict) as stub:
        code, document, stderr = analyse_with(
            tmp_path, env={**configured(stub), "TYPESAFE_API_KEY": "  "}
        )

    assert code == 1
    assert document is None
    assert stub.requests == []
    assert "TYPESAFE_API_KEY" in stderr
    assert "--no-categories" in stderr
    assert "Traceback" not in stderr


@end_to_end
def test_no_categories_runs_without_the_key_and_leaves_categories_null(tmp_path: Path) -> None:
    with JevStub(verdict) as stub:
        code, document, stderr = analyse_with(
            tmp_path,
            "--no-categories",
            env={**configured(stub), "TYPESAFE_API_KEY": "  "},
        )

    assert code == 0
    assert document is not None
    spoken = transcripts(document)
    assert len(spoken) == 2
    assert [(e["category"], e["category_confidence"]) for e in spoken] == [(None, None)] * 2
    assert stub.requests == []
    assert "TYPESAFE_API_KEY" not in stderr


@end_to_end
def test_a_jev_failure_nulls_only_the_affected_segment(tmp_path: Path) -> None:
    def respond(text: str) -> Reply:
        if "players" in text.lower():
            return choice("game_related", 0.82)
        return 500, {"error": "boom"}

    with JevStub(respond) as stub:
        code, document, stderr = analyse_with(tmp_path, env=configured(stub))

    assert code == 0
    assert document is not None
    english, portuguese = transcripts(document)
    assert (english["category"], english["category_confidence"]) == ("game_related", 0.82)
    assert (portuguese["category"], portuguese["category_confidence"]) == (None, None)
    assert "categories 1 of 2" in stderr
    assert "HTTP 500" in stderr
    assert "Traceback" not in stderr


@end_to_end
def test_an_unreachable_jev_does_not_abort_the_run(tmp_path: Path) -> None:
    with JevStub() as stub:
        url = stub.url
    code, document, stderr = analyse_with(
        tmp_path, env={"TYPESAFE_API_KEY": KEY, "VICTORIOUS_JEV_URL": url}
    )

    assert code == 0
    assert document is not None
    spoken = transcripts(document)
    assert len(spoken) == 2
    assert [(e["category"], e["category_confidence"]) for e in spoken] == [(None, None)] * 2
    assert "Traceback" not in stderr


@end_to_end
def test_no_speech_sends_nothing(tmp_path: Path) -> None:
    with JevStub() as stub:
        code, document, _ = analyse_with(tmp_path, "--no-speech", env=configured(stub))

    assert code == 0
    assert document is not None
    assert transcripts(document) == []
    assert stub.requests == []


@end_to_end
def test_no_speech_without_a_key_does_not_mention_the_key(tmp_path: Path) -> None:
    layout = write_layout(tmp_path / "layout.yaml")
    program = "import sys; from victorious.cli import main; print(main(sys.argv[1:]))"

    result = subprocess.run(
        [
            sys.executable, "-c", program, "analyse", str(recording(tmp_path)),
            "--layout", str(layout), "--no-cache", "--no-speech",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "TYPESAFE_API_KEY": ""},
        check=False,
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    assert "TYPESAFE_API_KEY" not in result.stderr


def speech_match(*events: Event) -> Match:
    return Match(
        index=0,
        start=0.0,
        end=100.0,
        win=False,
        placement=None,
        personal_eliminations=None,
        events=list(events),
    )


def said(text: str, at: float = 1.0) -> Event:
    return transcript_event(text, "en", at, at + 2.0, 0.5)


def categorized(
    matches: list[Match], responder: Responder, **client: Any
) -> tuple[categorize.CategorizeResult, list[Recorded]]:
    with JevStub(responder) as stub:
        jev = categorize.JevClient("key", url=stub.url, **client)
        result = categorize.categorize_matches(matches, jev)
    return result, stub.requests


def categories(result: categorize.CategorizeResult) -> list[Any]:
    return [
        (event.category, event.category_confidence)
        for match in result.matches
        for event in match.events
        if event.kind == "transcript"
    ]


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(categorize, "BACKOFF", 0.0)


def test_identical_texts_are_asked_once() -> None:
    matches = [speech_match(said("Go left"), said("Go left", 5.0)), speech_match(said("Go left"))]

    result, requests = categorized(matches, always("game_related", 0.7))

    assert len(requests) == 1
    assert categories(result) == [("game_related", 0.7)] * 3


def test_events_other_than_transcripts_and_their_order_are_untouched() -> None:
    frames = [
        frame_event("match_start", 0.5, 0.9),
        frame_event("personal_eliminations", 2.0, 0.9, value=1),
    ]
    original = [frames[0], said("Hi", 1.0), frames[1], said("Nice", 3.0)]

    result, _ = categorized([speech_match(*original)], always("personal", 0.6))

    [match] = result.matches
    assert [e.kind for e in match.events] == [e.kind for e in original]
    assert [e.time for e in match.events] == [e.time for e in original]
    assert match.events[0] == frames[0]
    assert match.events[2] == frames[1]
    assert [e.text for e in match.events] == [e.text for e in original]
    assert all(e.confidence == o.confidence for e, o in zip(match.events, original, strict=True))
    assert result.categorized == 2
    assert result.first_failure is None


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_stops_further_requests(status: int) -> None:
    matches = [speech_match(*(said(f"line {i}", i) for i in range(6)))]

    result, requests = categorized(matches, lambda _t: (status, {"error": "no"}), workers=1)

    assert len(requests) == 1
    assert categories(result) == [(None, None)] * 6
    assert result.first_failure is not None
    assert "API key" in result.first_failure


def test_a_rejected_request_is_not_retried_and_does_not_affect_other_segments() -> None:
    def respond(text: str) -> Reply:
        return (422, {"detail": "bad"}) if text == "bad" else choice("personal", 0.9)

    matches = [speech_match(said("bad"), said("fine", 5.0))]

    result, requests = categorized(matches, respond, workers=1)

    assert len(requests) == 2
    assert categories(result) == [(None, None), ("personal", 0.9)]
    assert result.first_failure == "HTTP 422"


@pytest.mark.parametrize(
    "reply",
    [
        b"not json",
        b"{}",
        {"answers": {}},
        {"answers": {"category": {"type": "choice", "choice": "sports", "confidence": 0.9}}},
        {"answers": {"category": {"type": "choice", "choice": "personal", "confidence": 1.5}}},
        {"answers": {"category": {"type": "choice", "choice": "personal"}}},
    ],
)
def test_an_unusable_reply_leaves_the_category_null(reply: Any) -> None:
    result, _ = categorized([speech_match(said("Hello"))], lambda _t: (200, reply))

    assert categories(result) == [(None, None)]
    assert result.first_failure is not None
    assert result.first_failure.startswith("unexpected reply from Jev")


@pytest.mark.usefixtures("no_backoff")
def test_a_rate_limited_request_is_retried() -> None:
    calls: list[str] = []

    def respond(text: str) -> Reply:
        calls.append(text)
        return (429, {}) if len(calls) == 1 else choice("game_related", 0.8)

    result, requests = categorized([speech_match(said("Go left"))], respond)

    assert len(requests) == 2
    assert categories(result) == [("game_related", 0.8)]


@pytest.mark.usefixtures("no_backoff")
def test_a_server_that_keeps_failing_is_tried_three_times_then_given_up() -> None:
    result, requests = categorized([speech_match(said("Go left"))], lambda _t: (503, {}))

    assert len(requests) == 3
    assert categories(result) == [(None, None)]
    assert result.first_failure == "HTTP 503"


@pytest.mark.usefixtures("no_backoff")
def test_five_failures_in_a_row_cut_the_run_off() -> None:
    matches = [speech_match(*(said(f"line {i}", i) for i in range(12)))]

    result, requests = categorized(matches, lambda _t: (500, {}), workers=1)

    assert len(requests) == 5 * 3
    assert categories(result) == [(None, None)] * 12
    assert result.first_failure == "HTTP 500"


@pytest.mark.usefixtures("no_backoff")
def test_a_success_between_failures_keeps_the_run_going() -> None:
    def respond(text: str) -> Reply:
        return (500, {}) if int(text.split()[-1]) % 3 else choice("personal", 0.9)

    matches = [speech_match(*(said(f"line {i}", i) for i in range(12)))]

    result, _ = categorized(matches, respond, workers=1)

    assert result.categorized == 4


@pytest.mark.parametrize(
    "url",
    ["not a url", "file:///etc/passwd", "http://example.com/v1", "https:///v1", "http://"],
)
def test_a_url_that_is_not_http_leaves_the_categories_null(url: str) -> None:
    matches = [speech_match(said("Go left"), said("Nice", 5.0))]

    result = categorize.categorize_matches(matches, categorize.JevClient("key", url=url))

    assert categories(result) == [(None, None)] * 2
    assert result.first_failure is not None
    assert "https" in result.first_failure


@pytest.mark.usefixtures("no_backoff")
def test_five_unusable_replies_in_a_row_cut_the_run_off() -> None:
    matches = [speech_match(*(said(f"line {i}", i) for i in range(12)))]

    result, requests = categorized(matches, lambda _t: (404, {}), workers=1)

    assert len(requests) == 5
    assert categories(result) == [(None, None)] * 12
    assert result.first_failure == "HTTP 404"


def test_a_redirect_is_not_followed_so_the_key_stays_put() -> None:
    with JevStub() as elsewhere:
        target = elsewhere.url

        result, requests = categorized(
            [speech_match(said("Go left"))],
            lambda _t: (302, {}, {"Location": target}),
            workers=1,
        )

    assert len(requests) == 1
    assert elsewhere.requests == []
    assert categories(result) == [(None, None)]
    assert result.first_failure == "HTTP 302"


def test_retry_after_is_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("victorious.categorize.time", SimpleNamespace(sleep=slept.append))
    calls: list[str] = []

    def respond(text: str) -> Reply:
        calls.append(text)
        if len(calls) == 1:
            return 429, {}, {"Retry-After": "3"}
        return choice("personal", 0.7)

    result, _ = categorized([speech_match(said("Go left"))], respond)

    assert slept == [3.0]
    assert categories(result) == [("personal", 0.7)]


def test_an_exception_outside_the_client_counts_toward_the_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(self: categorize.JevClient, text: str) -> categorize.Judgment:
        raise ValueError("bad header")

    monkeypatch.setattr(categorize.JevClient, "_ask", explode)
    matches = [speech_match(*(said(f"line {i}", i) for i in range(8)))]

    client = categorize.JevClient("key", workers=1)
    result = categorize.categorize_matches(matches, client)

    assert categories(result) == [(None, None)] * 8
    assert result.first_failure == "ValueError: bad header"
    assert client._halted is not None


def test_no_client_leaves_every_event_as_it_was() -> None:
    matches = [speech_match(said("Go left"))]

    result = categorize.categorize_matches(matches, None)

    assert result.matches == matches
    assert result.categorized == 0


def test_the_client_is_built_from_the_environment() -> None:
    assert categorize.jev_from_env({}) is None
    assert categorize.jev_from_env({"TYPESAFE_API_KEY": " "}) is None
    default = categorize.jev_from_env({"TYPESAFE_API_KEY": "k"})
    assert default is not None
    assert default.url == "https://api.typesafe.ai/v1/systemone"
    override = categorize.jev_from_env(
        {"TYPESAFE_API_KEY": "k", "VICTORIOUS_JEV_URL": "http://localhost:9/x"}
    )
    assert override is not None
    assert override.url == "http://localhost:9/x"
    padded = categorize.jev_from_env(
        {"TYPESAFE_API_KEY": "k", "VICTORIOUS_JEV_URL": " https://a/b "}
    )
    assert padded is not None
    assert padded.url == "https://a/b"
