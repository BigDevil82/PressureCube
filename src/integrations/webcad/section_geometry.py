from __future__ import annotations

import json
from functools import lru_cache
from math import isclose
from pathlib import Path

from src.structure.section_library import BoxSection, ChannelSection, HShapeSection

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data/steel_section_geometry.json"


@lru_cache(maxsize=1)
def geometry_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def find_section_geometry(section) -> dict | None:
    """Match numeric dimensions, not the non-unique display name or sorted index."""
    if isinstance(section, BoxSection):
        family = "box"
        dimensions = {"sideLengthA": section.B * 1000, "wallThicknessT": section.t * 1000}
    elif isinstance(section, HShapeSection):
        family = "h"
        dimensions = dict(zip(
            ["height", "widthB", "thicknessT1", "thicknessT2"],
            [section.H * 1000, section.B * 1000, section.tw * 1000, section.tf * 1000],
        ))
    elif isinstance(section, ChannelSection):
        family = "channel"
        # Existing analysis libraries lack flange thickness and fall back to tw.
        dimensions = {"height": section.H * 1000, "legWidthB": section.B * 1000, "thicknessD": section.tw * 1000}
    else:
        return None
    matches = [
        row for row in geometry_catalog()["sections"]
        if row["family"] == family and all(
            isclose(value, row["sectionParams"][key], abs_tol=1e-7, rel_tol=1e-9)
            for key, value in dimensions.items()
        )
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous standard section geometry: {section.name}")
    return matches[0] if matches else None
