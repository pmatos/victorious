"""The hand-written Layout that matches the synthetic HUD, as the YAML a user would write."""

from pathlib import Path

import yaml

from tests.synthetic.render import ELEMENT_RECTS

HUD_ELEMENTS = ("hud_anchor", "personal_eliminations", "team_eliminations", "players_left")
RESULTS_ELEMENTS = ("results_banner", "placement")


def _rect(name: str) -> dict[str, float]:
    x, y, w, h = ELEMENT_RECTS[name]
    return {"x": x, "y": y, "w": w, "h": h}


def layout_document() -> dict[str, object]:
    return {
        "aspect_ratio": "16:9",
        "screens": {
            "hud": {name: _rect(name) for name in HUD_ELEMENTS},
            "results": {name: _rect(name) for name in RESULTS_ELEMENTS},
        },
    }


def write_layout(path: Path, document: dict[str, object] | None = None) -> Path:
    path.write_text(yaml.safe_dump(document if document is not None else layout_document()))
    return path
