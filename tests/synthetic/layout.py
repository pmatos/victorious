"""The hand-written Layout that matches the synthetic HUD, as the YAML a user would write."""

from pathlib import Path

import yaml

from tests.synthetic.render import ELEMENT_RECTS, PAIR_ELEMENT_RECTS, ROW_ELEMENT_RECTS
from tests.synthetic.spec import HudDesign

HUD_ELEMENTS = ("hud_anchor", "personal_eliminations", "team_eliminations", "players_left")
RESULTS_ELEMENTS = ("results_banner", "placement", "elimination_placement")


RECTS = {
    "icon_column": ELEMENT_RECTS,
    "icon_pair": PAIR_ELEMENT_RECTS,
    "counter_row": ROW_ELEMENT_RECTS,
}


def _rect(name: str, design: HudDesign) -> dict[str, float]:
    x, y, w, h = RECTS[design][name]
    return {"x": x, "y": y, "w": w, "h": h}


def layout_document(design: HudDesign = "icon_column") -> dict[str, object]:
    document: dict[str, object] = {"aspect_ratio": "16:9"}
    if design != "icon_column":
        document["hud_design"] = design
    document["screens"] = {
        "hud": {
            name: _rect(name, design)
            for name in HUD_ELEMENTS
            if not (design == "icon_pair" and name == "team_eliminations")
        },
        "results": {name: _rect(name, design) for name in RESULTS_ELEMENTS},
    }
    return document


def write_layout(path: Path, document: dict[str, object] | None = None) -> Path:
    path.write_text(yaml.safe_dump(document if document is not None else layout_document()))
    return path
