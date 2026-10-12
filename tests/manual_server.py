"""Drive a long-running `victorious setup --manual` process the way its browser page does."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ANNOUNCED = re.compile(r"http://127\.0\.0\.1:\d+/[\w-]+/")


class RunningCli:
    def __init__(
        self, *args: str, env: Mapping[str, str] | None = None, cwd: Path | None = None
    ) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "victorious", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env={**os.environ, **env} if env else None,
        )
        self.lines: list[str] = []
        self._announced: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._url: str | None = None

    def _read(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.lines.append(line)
            found = ANNOUNCED.search(line)
            if found:
                self._announced.put(found.group(0))
        self._announced.put("")

    @property
    def stderr_text(self) -> str:
        self._reader.join(timeout=10)
        return "".join(self.lines)

    def url(self, timeout: float = 90) -> str:
        if self._url is None:
            try:
                announced = self._announced.get(timeout=timeout)
            except queue.Empty:
                self.kill()
                raise AssertionError(
                    "no URL announced; stderr so far:\n" + self.stderr_text
                ) from None
            if not announced:
                raise AssertionError("exited before announcing a URL:\n" + self.stderr_text)
            self._url = announced
        return self._url

    def request(
        self, path: str, body: bytes | None = None, *, base: str | None = None
    ) -> tuple[int, bytes, str]:
        request = urllib.request.Request((base or self.url()) + path, data=body)  # noqa: S310
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return response.status, response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers.get("Content-Type", "")

    def get_json(self, path: str) -> Any:
        status, body, _ = self.request(path)
        assert status == 200, body
        return json.loads(body)

    def post_json(self, path: str, document: object) -> tuple[int, Any]:
        status, body, _ = self.request(path, json.dumps(document).encode())
        return status, json.loads(body)

    def alive(self) -> bool:
        return self.process.poll() is None

    def interrupt(self) -> None:
        self.process.send_signal(2)

    def wait(self, timeout: float = 30) -> int:
        return self.process.wait(timeout=timeout)

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def close(self) -> None:
        self.kill()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._reader.join(timeout=5)


def wait_until(condition: Any, timeout: float = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False
