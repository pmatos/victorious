"""Judging each transcript segment personal or game-related with TypeSafe's Jev.

Only the segment text is sent. Any failure leaves the category `null`; nothing here aborts a run.
"""

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from victorious.schema import Category, Match

KEY_VARIABLE = "TYPESAFE_API_KEY"
URL_VARIABLE = "VICTORIOUS_JEV_URL"
DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
QUESTION = "category"
TIMEOUT = 30.0
WORKERS = 8
ATTEMPTS = 3
BACKOFF = 0.5
MAX_CONSECUTIVE_FAILURES = 5
MAX_RETRY_AFTER = 10.0
RETRYABLE_STATUSES = frozenset({429, 529})
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
REJECTED_STATUSES = frozenset({401, 403})

INSTRUCTIONS = (
    "A Fortnite player said this while playing, or while talking with friends over voice chat. "
    "Is it about the game being played or about something else?"
)
CRITERIA: dict[str, str] = {
    "game_related": (
        "About the match in progress: callouts, enemies and where they are, loot, building, "
        "health and shields, the zone, strategy, positions, eliminations, who to help or revive."
    ),
    "personal": (
        "Not about the match: chit-chat, life outside the game, jokes, greetings, reactions "
        "unrelated to what is happening in the game."
    ),
}


@dataclass(frozen=True)
class Judgment:
    category: Category
    confidence: float


class JevError(Exception):
    """A segment could not be judged."""


class _Stopped(JevError):
    """The run was cut off by an earlier failure, so this segment was not asked about."""


class _Transient(JevError):
    """Jev could not be reached or kept failing after the retries."""


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"]
    choice: Category
    confidence: float = Field(ge=0, le=1)


class _Reply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answers: dict[str, _Answer]


def _request(text: str) -> dict[str, Any]:
    return {
        "state": text,
        "model": MODEL,
        "questions": {
            QUESTION: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}
        },
    }


def _parse(raw: bytes) -> Judgment:
    try:
        reply = _Reply.model_validate_json(raw)
    except ValidationError as error:
        raise JevError(f"unexpected reply from Jev: {error.errors()[0]['msg']}") from error
    answer = reply.answers.get(QUESTION)
    if answer is None:
        raise JevError("unexpected reply from Jev: no answer to the question")
    return Judgment(answer.choice, answer.confidence)


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirects)


def _usable_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if not parts.hostname:
        return False
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK_HOSTS)


def _retry_after(header: str | None) -> float:
    try:
        return min(max(float(header or 0), 0.0), MAX_RETRY_AFTER)
    except ValueError:
        return 0.0


class JevClient:
    def __init__(
        self,
        api_key: str,
        url: str = DEFAULT_URL,
        timeout: float = TIMEOUT,
        workers: int = WORKERS,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.timeout = timeout
        self.workers = workers
        self._lock = threading.Lock()
        self._halted: str | None = None
        self._consecutive = 0

    def judge(self, text: str) -> Judgment:
        """Raises JevError when the text cannot be judged."""
        with self._lock:
            halted = self._halted
        if halted is not None:
            raise _Stopped(halted)
        try:
            judgment = self._ask(text)
        except JevError:
            self._note(failed=True)
            raise
        except Exception as error:
            self._note(failed=True)
            raise JevError(f"{type(error).__name__}: {error}") from error
        self._note(failed=False)
        return judgment

    def _note(self, *, failed: bool) -> None:
        with self._lock:
            self._consecutive = self._consecutive + 1 if failed else 0
            if self._consecutive >= MAX_CONSECUTIVE_FAILURES and self._halted is None:
                self._halted = (
                    f"Jev failed {MAX_CONSECUTIVE_FAILURES} times in a row, so it was not asked "
                    "about the remaining segments"
                )

    def _halt(self, reason: str) -> None:
        with self._lock:
            if self._halted is None:
                self._halted = reason

    def _ask(self, text: str) -> Judgment:
        if not _usable_url(self.url):
            reason = f"the Jev URL must be https (http only to localhost), got {self.url!r}"
            self._halt(reason)
            raise JevError(reason)
        body = json.dumps(_request(text)).encode()
        last: JevError = _Transient("no attempt was made")
        retry_after = 0.0
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(max(BACKOFF * 2 ** (attempt - 1), retry_after))
            request = urllib.request.Request(  # noqa: S310
                self.url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with _OPENER.open(request, timeout=self.timeout) as response:
                    return _parse(response.read())
            except urllib.error.HTTPError as error:
                retry_after = _retry_after(error.headers.get("Retry-After"))
                error.close()
                if error.code in REJECTED_STATUSES:
                    reason = f"Jev rejected the API key (HTTP {error.code})"
                    self._halt(f"{reason}, so it was not asked about the remaining segments")
                    raise JevError(reason) from error
                last = _Transient(f"HTTP {error.code}")
                if error.code not in RETRYABLE_STATUSES and error.code < 500:
                    raise JevError(f"HTTP {error.code}") from error
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as e:
                detail = str(e.reason) if isinstance(e, urllib.error.URLError) else str(e)
                last = _Transient(f"{type(e).__name__}: {detail}")
        raise last


def jev_from_env(environ: Mapping[str, str]) -> JevClient | None:
    key = environ.get(KEY_VARIABLE, "").strip()
    if not key:
        return None
    return JevClient(key, url=environ.get(URL_VARIABLE, "").strip() or DEFAULT_URL)


@dataclass(frozen=True)
class CategorizeResult:
    matches: list[Match]
    categorized: int = 0
    first_failure: str | None = None


def categorize_matches(matches: list[Match], client: JevClient | None) -> CategorizeResult:
    """The Matches with each transcript Event judged; every other Event is left as it is."""
    texts: list[str] = []
    for match in matches:
        for event in match.events:
            if event.kind == "transcript" and event.text is not None:
                texts.append(event.text)
    if client is None or not texts:
        return CategorizeResult(matches)

    distinct = list(dict.fromkeys(texts))
    judgments: dict[str, Judgment] = {}
    first_failure: JevError | None = None

    def ask(text: str) -> Judgment | JevError:
        try:
            return client.judge(text)
        except JevError as error:
            return error

    with ThreadPoolExecutor(max_workers=min(client.workers, len(distinct))) as pool:
        for text, outcome in zip(distinct, pool.map(ask, distinct), strict=True):
            if isinstance(outcome, Judgment):
                judgments[text] = outcome
            elif first_failure is None or (
                isinstance(first_failure, _Stopped) and not isinstance(outcome, _Stopped)
            ):
                first_failure = outcome

    categorized = 0
    rewritten: list[Match] = []
    for match in matches:
        events = []
        for event in match.events:
            judgment = judgments.get(event.text or "") if event.kind == "transcript" else None
            if judgment is not None:
                categorized += 1
                event = event.model_copy(
                    update={
                        "category": judgment.category,
                        "category_confidence": judgment.confidence,
                    }
                )
            events.append(event)
        rewritten.append(match.model_copy(update={"events": events}))
    return CategorizeResult(rewritten, categorized, str(first_failure) if first_failure else None)
