from __future__ import annotations

from dataclasses import dataclass

from src.structure.layout.params import BaseLayoutParameters
from src.structure.section_assignment import SectionAssignment
from src.structure.section_library import SectionLibrary


@dataclass
class StructuralDesign:
    """Complete structural design to be evaluated or optimized.

    Layout parameters describe geometry/topology generation. Section assignment
    describes the actual section used by every beam group. Pressure belongs to
    the analysis case, so it lives with the evaluated design instead of the
    layout parameters.
    """

    params: BaseLayoutParameters
    pressure: float
    section_assignment: SectionAssignment

    @property
    def layout_strategy(self) -> str:
        return self.params.layout_strategy

    def build_model(self, section_lib: SectionLibrary):
        from src.structure.structural_model import StructuralModel

        model = StructuralModel(self.params, section_lib, section_assignment=self.section_assignment)
        model.generate()
        return model
