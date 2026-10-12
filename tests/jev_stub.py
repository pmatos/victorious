"""A local HTTP server standing in for TypeSafe's Jev API, the only boundary tests stub."""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

Reply = tuple[int, Any]
Responder = Callable[[str], Reply]


@dataclass(frozen=True)
class Recorded:
    path: str
    headers: dict[str, str]
    body: Any
    size: int


def choice(option: str, confidence: float) -> Reply:
    other = {"personal": "game_related", "game_related": "personal"}[option]
    return 200, {
        "model": "jev-1.13.0",
        "answers": {
            "category": {
                "type": "choice",
                "choice": option,
                "probabilities": {option: confidence, other: 1 - confidence},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def always(option: str, confidence: float = 0.9) -> Responder:
    return lambda _text: choice(option, confidence)


class JevStub:
    def __init__(self, respond: Responder | None = None) -> None:
        self.respond: Responder = respond or always("personal")
        self.requests: list[Recorded] = []
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                with stub._lock:
                    stub.requests.append(
                        Recorded(
                            self.path,
                            {key.lower(): value for key, value in self.headers.items()},
                            body,
                            len(raw),
                        )
                    )
                text = body.get("state") if isinstance(body, dict) else None
                status, payload = stub.respond(text if isinstance(text, str) else "")
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1/systemone"

    def texts(self) -> list[str]:
        return [r.body["state"] for r in self.requests]

    def __enter__(self) -> "JevStub":
        self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
