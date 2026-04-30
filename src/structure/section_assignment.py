from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.structure.enums import BeamType
from src.structure.section_library import SectionLibrary, SectionType

if TYPE_CHECKING:
    from src.structure.structural_model import StructuralModel


BEAM_SECTION_TYPES = {
    BeamType.FRAME: SectionType.BOX,
    BeamType.MAIN: SectionType.H_MAIN,
    BeamType.SECONDARY: SectionType.H_SEC,
    BeamType.BRACE: SectionType.CHANNEL,
}


@dataclass(frozen=True)
class SectionGroup:
    group_id: str
    beam_type: BeamType
    section_type: SectionType
    section_index: int


@dataclass
class SectionAssignment:
    """Concrete section assignment for a generated structural model.

    This is the actual section contract: groups define allowed section pools,
    each group chooses a section index, and beam_groups optionally assigns
    individual beams to groups.
    """

    groups: dict[str, SectionGroup]
    beam_groups: dict[int, str] = field(default_factory=dict)
    assign_missing_by_geometry: bool = False

    @classmethod
    def by_beam_type(cls, section_indices_by_type: dict[BeamType, int]) -> "SectionAssignment":
        groups = {}
        for beam_type in sorted(section_indices_by_type, key=lambda item: int(item.value)):
            group_id = f"{beam_type.name}-SG1"
            groups[group_id] = SectionGroup(
                group_id=group_id,
                beam_type=beam_type,
                section_type=BEAM_SECTION_TYPES[beam_type],
                section_index=int(section_indices_by_type[beam_type]),
            )
        return cls(groups=groups)

    @classmethod
    def from_group_indices(
        cls,
        group_beam_types: dict[str, BeamType],
        section_indices_by_group: dict[str, int],
        beam_groups: dict[int, str] | None = None,
        assign_missing_by_geometry: bool = False,
    ) -> "SectionAssignment":
        groups = {
            group_id: SectionGroup(
                group_id=group_id,
                beam_type=beam_type,
                section_type=BEAM_SECTION_TYPES[beam_type],
                section_index=int(section_indices_by_group[group_id]),
            )
            for group_id, beam_type in group_beam_types.items()
        }
        return cls(
            groups=groups,
            beam_groups=beam_groups or {},
            assign_missing_by_geometry=assign_missing_by_geometry,
        )

    def apply_to_model(self, model: StructuralModel, section_lib: SectionLibrary) -> None:
        if self.assign_missing_by_geometry:
            self._assign_missing_beam_groups_by_geometry(model)

        for beam_id, beam in model.beams.items():
            if beam.is_virtual:
                continue
            group_id = self.beam_groups.get(beam_id) or self._single_group_for_type(beam.beam_type)
            group = self.groups[group_id]
            beam.section = section_lib.get_section(group.section_type, group.section_index)
            self.beam_groups[beam_id] = group_id  # Ensure every beam has an assigned group

    def _assign_missing_beam_groups_by_geometry(self, model: StructuralModel) -> None:
        from src.structure.section_grouping.geometric import assign_geometric_beam_groups

        n_groups_by_type: dict[BeamType, int] = {}
        for group in self.groups.values():
            n_groups_by_type[group.beam_type] = n_groups_by_type.get(group.beam_type, 0) + 1

        geometric_groups = assign_geometric_beam_groups(model, n_groups_by_type)
        for beam_id, group_id in geometric_groups.items():
            self.beam_groups.setdefault(beam_id, group_id)

    def _single_group_for_type(self, beam_type: BeamType) -> str:
        matches = [group_id for group_id, group in self.groups.items() if group.beam_type == beam_type]
        if len(matches) != 1:
            raise ValueError(f"Beam type {beam_type.name} does not have exactly one section group.")
        return matches[0]


def zero_assignment() -> SectionAssignment:
    return SectionAssignment.by_beam_type({beam_type: 0 for beam_type in BEAM_SECTION_TYPES})
