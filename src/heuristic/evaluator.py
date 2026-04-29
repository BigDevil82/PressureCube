"""
Unified structural evaluator.

The evaluator consumes a complete StructuralDesign. It no longer reconstructs
section assignment from optimization-side rule fragments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.analysis.surrogate_model import AdvancedSurrogateModel
from src.optimization.design_space import compute_mass_fast, get_section_lib
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_design import StructuralDesign


@dataclass
class EvalResult:
    """Result of structural evaluation."""

    mass: float
    u_sigma: float
    u_delta: float
    feasible: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "mass": self.mass,
            "u_sigma": self.u_sigma,
            "u_delta": self.u_delta,
            "feasible": self.feasible,
        }


class StructuralEvaluator:
    """Evaluate StructuralDesign objects with FEA or a trained surrogate."""

    COMBO_STRENGTH = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
    COMBO_SERVICE = {"DL": 1.0, "LL": 1.0, "PL": 1.0}

    def __init__(
        self,
        mode: str = "fea",
        surrogate_path: Optional[str] = None,
        section_lib_path: Optional[str] = None,
        mesh_size: float = 0.5,
    ):
        self.mode = mode
        self.mesh_size = mesh_size
        self.section_lib_path = section_lib_path

        self.surrogate = None
        if mode == "surrogate":
            self.surrogate = AdvancedSurrogateModel.load(surrogate_path)
            print(f"Loaded surrogate model from: {surrogate_path}")

    def evaluate(self, design: StructuralDesign) -> EvalResult:
        if self.mode == "surrogate":
            return self._evaluate_surrogate(design)
        return self._evaluate_fea(design)

    def _evaluate_fea(self, design: StructuralDesign) -> EvalResult:
        section_lib = get_section_lib(design.layout_strategy, self.section_lib_path)
        model = design.build_model(section_lib)
        mass = model.calculate_mass_kg()

        model = MeshGenerator(model, self.mesh_size).generate()

        solver = OpenSeesSolver(model)
        res_strength = solver.run_load_combination(self.COMBO_STRENGTH, pressure_pa=design.pressure)
        res_service = solver.run_load_combination(self.COMBO_SERVICE, pressure_pa=design.pressure)

        processor = ResultProcessor(model)
        strength_check = processor.run_gb50017_check(res_strength)
        disp_check = processor.run_gb50017_check(res_service)

        u_sigma = strength_check.max_stress_ratio
        u_delta = disp_check.max_deflection_ratio

        return EvalResult(
            mass=mass,
            u_sigma=u_sigma,
            u_delta=u_delta,
            feasible=u_sigma <= 1.0 and u_delta <= 1.0,
        )

    def _evaluate_surrogate(self, design: StructuralDesign) -> EvalResult:
        section_lib = get_section_lib(design.layout_strategy, self.section_lib_path)
        mass = compute_mass_fast(design, section_lib)
        u_sigma, u_delta = self.surrogate.predict(design)

        return EvalResult(
            mass=mass,
            u_sigma=u_sigma,
            u_delta=u_delta,
            feasible=u_sigma <= 1.0 and u_delta <= 1.0,
        )

    def evaluate_batch(self, designs: List[StructuralDesign]) -> List[EvalResult]:
        if self.mode != "surrogate":
            return [self._evaluate_fea(design) for design in designs]

        section_libs = {
            design.layout_strategy: get_section_lib(design.layout_strategy, self.section_lib_path)
            for design in designs
        }
        masses = [compute_mass_fast(design, section_libs[design.layout_strategy]) for design in designs]
        sigmas, deltas = self.surrogate.predict_batch(designs)
        return [
            EvalResult(
                mass=mass,
                u_sigma=float(u_sigma),
                u_delta=float(u_delta),
                feasible=bool(u_sigma <= 1.0 and u_delta <= 1.0),
            )
            for mass, u_sigma, u_delta in zip(masses, sigmas, deltas)
        ]

    def get_mass_only(self, design: StructuralDesign) -> float:
        return compute_mass_fast(design, get_section_lib(design.layout_strategy, self.section_lib_path))


def evaluate_design(
    design: StructuralDesign,
    mode: str = "fea",
    surrogate_path: Optional[str] = None,
) -> Tuple[float, float, float, bool]:
    evaluator = StructuralEvaluator(mode=mode, surrogate_path=surrogate_path)
    result = evaluator.evaluate(design)
    return result.mass, result.u_sigma, result.u_delta, result.feasible
