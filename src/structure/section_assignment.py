from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

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


@dataclass
class SectionAssignment:
    """Concrete section assignment for a generated structural model.

    This is the actual section contract: groups define allowed section pools,
    section_indices chooses a section for each group, and beam_groups optionally
    assigns individual beams to groups.
    """

    groups: dict[str, SectionGroup]
    section_indices: dict[str, int]
    beam_groups: dict[int, str] = field(default_factory=dict)
    random_seed: int | None = None

    @classmethod
    def by_beam_type(cls, section_indices_by_type: dict[BeamType, int]) -> "SectionAssignment":
        groups = {}
        section_indices = {}
        for beam_type in sorted(section_indices_by_type, key=lambda item: int(item.value)):
            group_id = f"{beam_type.name}-SG1"
            groups[group_id] = SectionGroup(
                group_id=group_id,
                beam_type=beam_type,
                section_type=BEAM_SECTION_TYPES[beam_type],
            )
            section_indices[group_id] = int(section_indices_by_type[beam_type])
        return cls(groups=groups, section_indices=section_indices)

    @classmethod
    def from_group_indices(
        cls,
        group_beam_types: dict[str, BeamType],
        section_indices_by_group: dict[str, int],
        beam_groups: dict[int, str] | None = None,
        random_seed: int | None = None,
    ) -> "SectionAssignment":
        groups = {
            group_id: SectionGroup(
                group_id=group_id,
                beam_type=beam_type,
                section_type=BEAM_SECTION_TYPES[beam_type],
            )
            for group_id, beam_type in group_beam_types.items()
        }
        return cls(
            groups=groups,
            section_indices={group_id: int(index) for group_id, index in section_indices_by_group.items()},
            beam_groups=beam_groups or {},
            random_seed=random_seed,
        )

    def apply_to_model(self, model: StructuralModel, section_lib: SectionLibrary) -> None:
        if self.random_seed is not None:
            self._assign_missing_beam_groups(model)

        for beam_id, beam in model.beams.items():
            if beam.is_virtual:
                continue
            group_id = self.beam_groups.get(beam_id) or self._single_group_for_type(beam.beam_type)
            group = self.groups[group_id]
            beam.section = section_lib.get_section(group.section_type, self.section_indices[group_id])
            self.beam_groups[beam_id] = group_id  # Ensure every beam has an assigned group

    def _assign_missing_beam_groups(self, model: StructuralModel) -> None:
        for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
            group_ids = [
                group_id
                for group_id, group in self.groups.items()
                if group.beam_type == beam_type
            ]
            if not group_ids:
                continue

            beam_ids = [
                beam_id
                for beam_id, beam in model.beams.items()
                if not beam.is_virtual and beam.beam_type == beam_type and beam_id not in self.beam_groups
            ]
            if not beam_ids:
                continue

            rng = np.random.default_rng(np.random.SeedSequence([self.random_seed, int(beam_type.value)]))
            rng.shuffle(beam_ids)
            for offset, beam_id in enumerate(beam_ids):
                if offset < len(group_ids):
                    group_id = group_ids[offset]
                else:
                    group_id = group_ids[int(rng.integers(0, len(group_ids)))]
                self.beam_groups[beam_id] = group_id

    def _single_group_for_type(self, beam_type: BeamType) -> str:
        matches = [group_id for group_id, group in self.groups.items() if group.beam_type == beam_type]
        if len(matches) != 1:
            raise ValueError(f"Beam type {beam_type.name} does not have exactly one section group.")
        return matches[0]


def zero_assignment() -> SectionAssignment:
    return SectionAssignment.by_beam_type({beam_type: 0 for beam_type in BEAM_SECTION_TYPES})
