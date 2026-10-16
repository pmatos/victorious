"""A Layout is for one HUD design; the icon column is the default and keeps its cache key."""

from pathlib import Path

import pytest

from tests.synthetic import layout_document, write_layout
from victorious.cache import layout_hash
from victorious.layout import Layout, LayoutError, dump_layout, load_layout

ACCEPTANCE = Path(__file__).parent.parent / "acceptance" / "layouts"
COLUMN_LAYOUT_HASH = "9d5c470d830213505bb65c2d"


def test_a_layout_without_a_design_is_the_icon_column() -> None:
    layout = Layout.model_validate(layout_document())

    assert layout.hud_design == "icon_column"


def test_the_acceptance_icon_column_layout_keeps_the_cache_key_it_had_before_designs() -> None:
    layout = load_layout(ACCEPTANCE / "hud_icons_16x9.yaml")

    assert layout_hash(layout).startswith(COLUMN_LAYOUT_HASH)


def test_the_design_is_part_of_a_layout_identity(tmp_path: Path) -> None:
    column = load_layout(write_layout(tmp_path / "column.yaml", layout_document()))
    row = load_layout(write_layout(tmp_path / "row.yaml", layout_document("counter_row")))

    assert row.hud_design == "counter_row"
    assert layout_hash(row) != layout_hash(column)


def test_a_dumped_layout_loads_back_with_its_design(tmp_path: Path) -> None:
    row = Layout.model_validate(layout_document("counter_row"))
    path = tmp_path / "row.yaml"
    path.write_text(dump_layout(row))

    assert load_layout(path) == row


def test_an_unknown_design_is_rejected(tmp_path: Path) -> None:
    document = layout_document()
    document["hud_design"] = "diagonal"
    path = write_layout(tmp_path / "layout.yaml", document)

    with pytest.raises(LayoutError, match="hud_design"):
        load_layout(path)
