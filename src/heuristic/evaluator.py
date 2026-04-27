"""
Unified Structural Evaluator.

Provides a single interface for evaluating structural designs, supporting:
1. FEA mode: Real finite element analysis using OpenSeesSolver
2. Surrogate mode: Fast ML-based prediction using trained surrogate model

Usage:
    # FEA mode (default)
    evaluator = StructuralEvaluator(mode="fea")
    result = evaluator.evaluate(L=8.0, W=3.6, H=3.6, pressure=30000.0,
                                 main_beam_spacing=1.0, num_secondary_width=4,
                                 num_secondary_height=4, add_support_beam=True,
                                 sections=[10, 15, 5, 12])

    # Surrogate mode
    evaluator = StructuralEvaluator(mode="surrogate",
                                     surrogate_path="results/surrogate_data/model_advanced.pkl")
    result = evaluator.evaluate(...)

Returns:
    {"mass": float, "u_sigma": float, "u_delta": float, "feasible": bool}
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.analysis.surrogate_data import compute_mass_fast
from src.analysis.surrogate_model import AdvancedSurrogateModel
from src.optimization.design_space import (
    DesignConfig,
    build_params_from_config,
    design_config_from_params,
    get_section_lib,
)
from src.optimization.section_grouping import SectionGroupingRuleset, apply_group_sections
from src.structure.layout.params import BaseLayoutParameters
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_model import StructuralModel


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
    """
    Unified structural evaluation interface.

    Supports two modes:
    - "fea": Real finite element analysis (accurate but slow)
    - "surrogate": ML-based prediction (fast but approximate)
    """

    # Standard load combinations
    COMBO_STRENGTH = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
    COMBO_SERVICE = {"DL": 1.0, "LL": 1.0, "PL": 1.0}

    def __init__(
        self,
        mode: str = "fea",
        surrogate_path: Optional[str] = None,
        section_lib_path: Optional[str] = None,
        mesh_size: float = 0.5,
        section_group_rules: SectionGroupingRuleset | str | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            mode: "fea" for finite element, "surrogate" for ML prediction
            surrogate_path: Path to trained surrogate model (required for surrogate mode)
            section_lib_path: Path to section library JSON
            mesh_size: Mesh size for FEA (only used in FEA mode)
        """
        self.mode = mode
        self.mesh_size = mesh_size
        self.section_lib_path = section_lib_path
        self.default_section_group_rules = self._load_section_group_rules(section_group_rules)

        self.surrogate = None
        if mode == "surrogate":
            self._load_surrogate(surrogate_path)

    def _load_section_group_rules(
        self, rules: SectionGroupingRuleset | str | None
    ) -> SectionGroupingRuleset | None:
        if rules is None:
            return None
        if isinstance(rules, SectionGroupingRuleset):
            return rules
        return SectionGroupingRuleset.load_json(rules)

    def _load_surrogate(self, path: str):
        """Load surrogate model from file."""
        self.surrogate = AdvancedSurrogateModel.load(path)
        print(f"Loaded surrogate model from: {path}")

    def evaluate(
        self,
        config: DesignConfig,
        section_group_rules: SectionGroupingRuleset | None = None,
        grouped_section_indices: dict[str, int] | None = None,
    ) -> EvalResult:
        """
        Evaluate a structural design.
        """
        rules = section_group_rules or self.default_section_group_rules
        if self.mode == "surrogate":
            return self._evaluate_surrogate(config, rules, grouped_section_indices)
        return self._evaluate_fea(config, rules, grouped_section_indices)

    def _evaluate_fea(
        self,
        config: DesignConfig,
        section_group_rules: SectionGroupingRuleset | None = None,
        grouped_section_indices: dict[str, int] | None = None,
    ) -> EvalResult:
        """Real FEA evaluation using OpenSeesSolver."""
        params = build_params_from_config(config)
        section_lib = get_section_lib(config.layout_strategy, self.section_lib_path)

        # Build model
        model = StructuralModel(params, section_lib)
        model.generate()
        self._apply_grouped_sections(model, section_lib, section_group_rules, grouped_section_indices)
        mass = model.calculate_mass_kg()

        # Mesh
        mesh_gen = MeshGenerator(model, self.mesh_size)
        model = mesh_gen.generate()

        # Run FEA
        solver = OpenSeesSolver(model)
        res_strength = solver.run_load_combination(self.COMBO_STRENGTH, pressure_pa=config.pressure)
        res_service = solver.run_load_combination(self.COMBO_SERVICE, pressure_pa=config.pressure)

        # Check results
        processor = ResultProcessor(model)
        strength_check = processor.run_gb50017_check(res_strength)
        disp_check = processor.run_gb50017_check(res_service)

        u_sigma = strength_check.max_stress_ratio
        u_delta = disp_check.max_deflection_ratio

        feasible = u_sigma <= 1.0 and u_delta <= 1.0

        return EvalResult(mass=mass, u_sigma=u_sigma, u_delta=u_delta, feasible=feasible)

    def _evaluate_surrogate(
        self,
        config: DesignConfig,
        section_group_rules: SectionGroupingRuleset | None = None,
        grouped_section_indices: dict[str, int] | None = None,
    ) -> EvalResult:
        """Fast surrogate model evaluation."""
        # Compute mass (fast, no FEA)
        section_lib = get_section_lib(config.layout_strategy, self.section_lib_path)
        if section_group_rules and grouped_section_indices:
            params = build_params_from_config(config)
            model = StructuralModel(params, section_lib)
            model.generate()
            self._apply_grouped_sections(
                model,
                section_lib,
                section_group_rules,
                grouped_section_indices,
            )
            mass = model.calculate_mass_kg()
        else:
            mass = compute_mass_fast(config, section_lib)

        # Predict u_sigma and u_delta
        u_sigma, u_delta = self.surrogate.predict(config)

        feasible = u_sigma <= 1.0 and u_delta <= 1.0

        return EvalResult(mass=mass, u_sigma=u_sigma, u_delta=u_delta, feasible=feasible)

    def evaluate_batch(
        self,
        configs: List[DesignConfig],
        section_group_rules: SectionGroupingRuleset | None = None,
        grouped_section_indices: List[dict[str, int]] | None = None,
    ) -> List[EvalResult]:
        """Evaluate a batch of designs.

        Surrogate mode uses one model call for the whole batch. FEA remains
        single-design because OpenSees analyses are externally parallelized by
        the caller.
        """
        rules = section_group_rules or self.default_section_group_rules
        if self.mode != "surrogate":
            assignments = grouped_section_indices or [None] * len(configs)
            return [
                self._evaluate_fea(config, rules, assignment)
                for config, assignment in zip(configs, assignments)
            ]

        if rules and grouped_section_indices:
            return [
                self._evaluate_surrogate(config, rules, assignment)
                for config, assignment in zip(configs, grouped_section_indices)
            ]

        section_libs = {
            config.layout_strategy: get_section_lib(config.layout_strategy, self.section_lib_path)
            for config in configs
        }
        masses = [compute_mass_fast(config, section_libs[config.layout_strategy]) for config in configs]
        sigmas, deltas = self.surrogate.predict_batch(configs)
        return [
            EvalResult(
                mass=mass,
                u_sigma=float(u_sigma),
                u_delta=float(u_delta),
                feasible=bool(u_sigma <= 1.0 and u_delta <= 1.0),
            )
            for mass, u_sigma, u_delta in zip(masses, sigmas, deltas)
        ]

    def _apply_grouped_sections(
        self,
        model: StructuralModel,
        section_lib,
        section_group_rules: SectionGroupingRuleset | None,
        grouped_section_indices: dict[str, int] | None,
    ) -> None:
        if not section_group_rules or not grouped_section_indices:
            return
        apply_group_sections(
            model,
            section_lib,
            section_group_rules,
            grouped_section_indices,
        )

    def evaluate_params(self, params: BaseLayoutParameters, pressure: float) -> EvalResult:
        """
        Evaluate using layout parameters directly.

        Convenience method for integration with existing code.
        """
        return self.evaluate(design_config_from_params(params, pressure))

    def get_mass_only(self, config: DesignConfig) -> float:
        """
        Fast mass calculation without any analysis.

        Useful for quick filtering before full evaluation.
        """
        return compute_mass_fast(config, get_section_lib(config.layout_strategy, self.section_lib_path))


# Standalone function for parallel processing
def evaluate_design(
    config: DesignConfig,
    mode: str = "fea",
    surrogate_path: Optional[str] = None,
    section_group_rules: SectionGroupingRuleset | None = None,
    grouped_section_indices: dict[str, int] | None = None,
) -> Tuple[float, float, float, bool]:
    """
    Standalone evaluation function for parallel processing.

    Returns:
        (mass, u_sigma, u_delta, feasible)
    """
    evaluator = StructuralEvaluator(mode=mode, surrogate_path=surrogate_path)
    result = evaluator.evaluate(config, section_group_rules, grouped_section_indices)
    return result.mass, result.u_sigma, result.u_delta, result.feasible


