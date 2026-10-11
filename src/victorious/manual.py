"""The local page of `setup --manual`: draw HUD rectangles on the frames and save a Layout."""

import contextlib
import json
import re
import secrets
import threading
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any

import cv2

from victorious.layout import aspect_ratio_label
from victorious.preview import scaled_frame
from victorious.setup import ELEMENTS_OF, REQUIRED, SetupResult, manual_result

HOST = "127.0.0.1"
MAX_BODY = 64 * 1024
_FRAME = re.compile(r"frame/(\d+)\.png")


class _Session:
    """What the page needs, encoded once, and the Layout once it has been saved."""

    def __init__(self, result: SetupResult, commit: Callable[[SetupResult], None]) -> None:
        self.panels = result.panels
        self.commit = commit
        self.lock = threading.Lock()
        self.saved: SetupResult | None = None
        self.page = files("victorious").joinpath("data", "manual.html").read_bytes()
        images = [scaled_frame(panel.frame) for panel in self.panels]
        self.images = [cv2.imencode(".png", image)[1].tobytes() for image in images]
        height, width = self.panels[0].frame.shape[:2]
        self.state = json.dumps(
            {
                "aspect_ratio": aspect_ratio_label(width, height),
                "panels": [
                    {
                        "kind": panel.kind,
                        "second": panel.second,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "image": f"frame/{index}.png",
                        "elements": list(ELEMENTS_OF[panel.kind]),
                    }
                    for index, (panel, image) in enumerate(zip(self.panels, images, strict=True))
                ],
                "required": list(REQUIRED),
                "rects": {
                    name: {"panel": index, **found.rect.model_dump()}
                    for index, panel in enumerate(self.panels)
                    for name, found in panel.found.items()
                },
                "notes": result.missing,
            }
        ).encode()

    def save(self, submitted: object) -> tuple[HTTPStatus, dict[str, Any]]:
        with self.lock:
            if self.saved is not None:
                return HTTPStatus.CONFLICT, {"errors": ["the Layout was already saved"]}
            result = manual_result(self.panels, submitted)
            if isinstance(result, list):
                return HTTPStatus.BAD_REQUEST, {"errors": result}
            try:
                self.commit(result)
            except (OSError, cv2.error) as error:
                return HTTPStatus.INTERNAL_SERVER_ERROR, {"errors": [str(error)]}
            self.saved = result
            return HTTPStatus.OK, {"warnings": result.warnings}


def _handler(
    session: _Session, prefix: str, finish: Callable[[], None]
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
            frame = _FRAME.fullmatch(route) if route is not None else None
            if route == "":
                self._send(HTTPStatus.OK, session.page, "text/html; charset=utf-8")
            elif route == "state.json":
                self._send(HTTPStatus.OK, session.state, "application/json")
            elif frame is not None and int(frame.group(1)) < len(session.images):
                self._send(HTTPStatus.OK, session.images[int(frame.group(1))], "image/png")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"errors": ["not found"]})

        def do_POST(self) -> None:
            if self._route() != "layout":
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
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"errors": ["the body is not JSON"]})
                return
            status, document = session.save(submitted)
            self._json(status, document)
            if status == HTTPStatus.OK:
                finish()

    return Handler


def serve_manual(
    result: SetupResult,
    *,
    port: int,
    open_browser: bool,
    announce: Callable[[str], None],
    commit: Callable[[SetupResult], None],
) -> SetupResult | None:
    """Serve the page until a Layout is saved (returned) or the user interrupts (`None`)."""
    session = _Session(result, commit)
    prefix = f"/{secrets.token_urlsafe(16)}/"
    holder: list[ThreadingHTTPServer] = []

    def finish() -> None:
        threading.Thread(target=holder[0].shutdown, daemon=True).start()

    try:
        server = ThreadingHTTPServer((HOST, port), _handler(session, prefix, finish))
    except OSError as error:
        raise OSError(f"cannot listen on {HOST}:{port}: {error.strerror or error}") from error
    holder.append(server)
    server.daemon_threads = True
    url = f"http://{HOST}:{server.server_address[1]}{prefix}"
    try:
        announce(url)
        if open_browser:
            threading.Thread(target=_open, args=(url,), daemon=True).start()
        server.serve_forever()
    except KeyboardInterrupt:
        return None
    finally:
        server.server_close()
    return session.saved


def _open(url: str) -> None:
    with contextlib.suppress(webbrowser.Error, OSError):
        webbrowser.open(url)
