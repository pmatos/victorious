"""The local page of `setup --manual`: draw HUD rectangles on the frames and save a Layout."""

import json
from collections.abc import Callable
from http import HTTPStatus
from importlib.resources import files
from typing import Any

import cv2

from victorious.layout import aspect_ratio_label
from victorious.localpage import Resource, serve_until_saved
from victorious.ocr import OcrError
from victorious.preview import scaled_frame
from victorious.setup import ELEMENTS_OF, REQUIRED, SetupResult, manual_result


def _resources(result: SetupResult) -> dict[str, Resource]:
    panels = result.panels
    images = [scaled_frame(panel.frame) for panel in panels]
    height, width = panels[0].frame.shape[:2]
    state = {
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
            for index, (panel, image) in enumerate(zip(panels, images, strict=True))
        ],
        "required": list(REQUIRED),
        "rects": {
            name: {"panel": index, **found.rect.model_dump()}
            for index, panel in enumerate(panels)
            for name, found in panel.found.items()
        },
        "notes": result.missing,
    }
    resources: dict[str, Resource] = {
        "": (
            files("victorious").joinpath("data", "manual.html").read_bytes(),
            "text/html; charset=utf-8",
        ),
        "state.json": (json.dumps(state).encode(), "application/json"),
    }
    for index, image in enumerate(images):
        resources[f"frame/{index}.png"] = (cv2.imencode(".png", image)[1].tobytes(), "image/png")
    return resources


def serve_manual(
    result: SetupResult,
    *,
    port: int,
    open_browser: bool,
    announce: Callable[[str], None],
    commit: Callable[[SetupResult], None],
) -> SetupResult | None:
    """Serve the page until a Layout is saved, then return it; `None` if interrupted before."""
    saved: list[SetupResult] = []

    def save(submitted: object) -> tuple[HTTPStatus, dict[str, Any]]:
        try:
            drawn = manual_result(result.panels, submitted)
            if isinstance(drawn, list):
                return HTTPStatus.BAD_REQUEST, {"errors": drawn}
            commit(drawn)
        except (cv2.error, OcrError) as error:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"errors": [str(error)]}
        saved.append(drawn)
        return HTTPStatus.OK, {"warnings": drawn.warnings}

    done = serve_until_saved(
        _resources(result),
        "layout",
        save,
        port=port,
        open_browser=open_browser,
        announce=announce,
    )
    return saved[0] if done and saved else None
