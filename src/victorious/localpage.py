"""A page served from 127.0.0.1 under a random token path until one save succeeds."""

import contextlib
import json
import os
import secrets
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "127.0.0.1"
MAX_BODY = 64 * 1024

Resource = tuple[bytes, str]
Save = Callable[[object], tuple[HTTPStatus, dict[str, Any]]]


class _Saves:
    """Runs `save` one at a time and refuses a second save once one has succeeded."""

    def __init__(self, save: Save) -> None:
        self._save = save
        self._lock = threading.Lock()
        self._done = False

    def submit(self, submitted: object) -> tuple[HTTPStatus, dict[str, Any]]:
        with self._lock:
            if self._done:
                return HTTPStatus.CONFLICT, {"errors": ["already saved"]}
            try:
                status, document = self._save(submitted)
            except OSError as error:
                return HTTPStatus.INTERNAL_SERVER_ERROR, {"errors": [str(error)]}
            self._done = status == HTTPStatus.OK
            return status, document

    def done(self) -> bool:
        """Whether a save succeeded, waiting for one that is still being written."""
        with self._lock:
            return self._done


def _handler(
    resources: Mapping[str, Resource], post_route: str, saves: _Saves, prefix: str
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _route(self) -> str | None:
            path = self.path.split("?", 1)[0]
            return path[len(prefix) :] if path.startswith(prefix) else None

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _json(self, status: HTTPStatus, document: object) -> None:
            self._send(status, json.dumps(document).encode(), "application/json")

        def do_GET(self) -> None:
            route = self._route()
            found = resources.get(route) if route is not None else None
            if found is None:
                self._json(HTTPStatus.NOT_FOUND, {"errors": ["not found"]})
            else:
                self._send(HTTPStatus.OK, *found)

        def do_POST(self) -> None:
            if self._route() != post_route:
                self._json(HTTPStatus.NOT_FOUND, {"errors": ["not found"]})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(HTTPStatus.LENGTH_REQUIRED, {"errors": ["Content-Length is required"]})
                return
            if not 0 <= length <= MAX_BODY:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"errors": ["body too large"]})
                return
            try:
                submitted = json.loads(self.rfile.read(length))
            except (ValueError, RecursionError):
                self._json(HTTPStatus.BAD_REQUEST, {"errors": ["the body is not JSON"]})
                return
            status, document = saves.submit(submitted)
            with contextlib.suppress(ConnectionError):
                self._json(status, document)
            if status == HTTPStatus.OK:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    return Handler


def serve_until_saved(
    resources: Mapping[str, Resource],
    post_route: str,
    save: Save,
    *,
    port: int,
    open_browser: bool,
    announce: Callable[[str], None],
) -> bool:
    """Serve `resources` (route to body and content type) until `save` answers 200 to a POST.

    `save` runs under a lock and must have written everything before it returns 200. Returns
    whether a save succeeded, False when interrupted before.
    """
    saves = _Saves(save)
    prefix = f"/{secrets.token_urlsafe(16)}/"
    try:
        server = ThreadingHTTPServer((HOST, port), _handler(resources, post_route, saves, prefix))
    except OSError as error:
        raise OSError(f"cannot listen on {HOST}:{port}: {error.strerror or error}") from error
    server.daemon_threads = True
    url = f"http://{HOST}:{server.server_address[1]}{prefix}"
    try:
        with contextlib.suppress(KeyboardInterrupt):
            announce(url)
            if open_browser and _has_display():
                threading.Thread(target=_open, args=(url,), daemon=True).start()
            server.serve_forever()
    finally:
        server.server_close()
    return saves.done()


def _has_display() -> bool:
    """False where `webbrowser` could only fall back to a text browser on this terminal."""
    if sys.platform != "linux":
        return True
    return any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER"))


def _open(url: str) -> None:
    with contextlib.suppress(webbrowser.Error, OSError):
        webbrowser.open(url)
