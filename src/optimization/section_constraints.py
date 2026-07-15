from __future__ import annotations

from src.structure.enums import BeamType


MIN_FRAME_SECTION_INDEX = 2  # BOX-150x6


def apply_group_section_constraints(
    sections: dict[str, int],
    group_beam_types: dict[str, BeamType],
) -> dict[str, int]:
    """Apply engineering minimums to grouped section indices."""

    constrained = dict(sections)
    for group_id, beam_type in group_beam_types.items():
        if beam_type == BeamType.FRAME:
            constrained[group_id] = max(constrained[group_id], MIN_FRAME_SECTION_INDEX)
    return constrained


def apply_ungrouped_section_constraints(sections: list[int]) -> list[int]:
    """Apply engineering minimums to [frame, main, secondary, brace]."""

    constrained = list(sections)
    constrained[0] = max(constrained[0], MIN_FRAME_SECTION_INDEX)
    return constrained
