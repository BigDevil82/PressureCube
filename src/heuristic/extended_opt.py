"""
Extended Optimization with Topology + Section Variables.

This module implements optimization over both topology and section variables:

Topology variables (discrete, small space = 300 combinations):
- main_beam_spacing: [0.7, 1.2], step 0.1 → 6 values
- num_secondary_width: [2, 6] → 5 values
- num_secondary_height: [2, 6] → 5 values
- add_support_beam: True/False → 2 values

Section variables (discrete, 20^4 = 160,000 combinations):
- frame_section_index: [0, 19]
- main_beam_section_index: [0, 19]
- secondary_beam_section_index: [0, 19]
- brace_section_index: [0, 19]

Total space: 300 × 160,000 = 48,000,000 combinations

Strategies implemented:
1. Joint Bayesian Optimization: All 8 variables together
2. Hierarchical Optimization: Topology first, then sections
3. Hybrid: Enumerate topology, Bayesian for sections

Recommendation: Hierarchical is most efficient for this problem.
"""

import time
from dataclasses import dataclass
from typing import Optional

import optuna
from optuna.samplers import TPESampler

from src.heuristic.evaluator import StructuralEvaluator
from src.structure.section_library import SectionLibrary, SectionType


@dataclass
class ExtendedOptResult:
    """Result of extended optimization."""

    structure_id: str
    geometry: dict

    # Optimal topology
    main_beam_spacing: float
    num_secondary_width: int
    num_secondary_height: int
    add_support_beam: bool

    # Optimal sections
    optimal_sections: list[int]

    # Metrics
    initial_mass: float  # With max sections and default topology
    optimal_mass: float
    mass_reduction_pct: float
    u_sigma: float
    u_delta: float
    is_feasible: bool

    # Efficiency
    n_evaluations: int
    time_seconds: float


# ============================================================================
# STRATEGY 1: Joint Bayesian Optimization (All 8 variables)
# ============================================================================


class JointBayesianOptimizer:
    """
    Bayesian optimization over all 8 variables jointly.

    Pros:
    - Can find global optimum
    - Captures interactions between topology and sections

    Cons:
    - Slower (needs more trials for larger space)
    - May waste trials on bad topologies
    """

    # Topology variable definitions
    SPACING_VALUES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    SEC_WIDTH_VALUES = [2, 3, 4, 5, 6]
    SEC_HEIGHT_VALUES = [2, 3, 4, 5, 6]

    def __init__(
        self,
        n_trials: int = 300,
        constraint_penalty: float = 1000.0,
        seed: Optional[int] = None,
        eval_mode: str = "fea",
        surrogate_path: Optional[str] = None,
    ):
        self.n_trials = n_trials
        self.constraint_penalty = constraint_penalty
        self.seed = seed
        self.eval_mode = eval_mode
        self.surrogate_path = surrogate_path

        # Get section counts from library
        section_lib = SectionLibrary("src/data/steel_sections_large.json")
        sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [section_lib.section_counts[t] for t in sec_types]
        self.max_indices = [count - 1 for count in self.sec_counts]

        # Create evaluator
        self.evaluator = StructuralEvaluator(mode=eval_mode, surrogate_path=surrogate_path)

    def _create_objective(self, L: float, W: float, H: float, pressure: float):
        """Create objective function for joint optimization."""

        def objective(trial: optuna.Trial) -> float:
            # Sample topology variables
            spacing = trial.suggest_categorical("main_beam_spacing", self.SPACING_VALUES)
            sec_w = trial.suggest_int("num_secondary_width", 2, 6)
            sec_h = trial.suggest_int("num_secondary_height", 2, 6)
            add_support = trial.suggest_categorical("add_support_beam", [True, False])

            # Sample section indices
            sections = [trial.suggest_int(f"section_{i}", 0, self.sec_counts[i] - 1) for i in range(4)]

            # Run evaluation
            try:
                result = self.evaluator.evaluate(
                    L=L,
                    W=W,
                    H=H,
                    pressure=pressure,
                    main_beam_spacing=spacing,
                    num_secondary_width=sec_w,
                    num_secondary_height=sec_h,
                    add_support_beam=add_support,
                    sections=sections,
                )
            except Exception as e:
                # Evaluation failed - heavily penalize
                trial.set_user_attr("error", str(e))
                return 1e9

            mass = result.mass
            u_sigma = result.u_sigma
            u_delta = result.u_delta
            if self.eval_mode == "surrogate":
                violation = max(0, u_sigma - 0.95) + max(0, u_delta - 0.95)
            else:
                violation = max(0, u_sigma - 1.0) + max(0, u_delta - 1.0)

            trial.set_user_attr("mass", mass)
            trial.set_user_attr("u_sigma", u_sigma)
            trial.set_user_attr("u_delta", u_delta)
            trial.set_user_attr("feasible", violation == 0)

            # Penalized objective
            if violation > 0:
                return mass * (1 + self.constraint_penalty * violation)
            return mass

        return objective

    def optimize(self, parameters: dict) -> ExtendedOptResult:
        """Run joint optimization."""
        start_time = time.time()

        L = parameters["L"]
        W = parameters["W"]
        H = parameters["H"]
        pressure = parameters.get("pressure", 30000.0)

        # Get initial mass (max sections, default topology)
        initial_result = self.evaluator.evaluate(
            L=L,
            W=W,
            H=H,
            pressure=pressure,
            main_beam_spacing=0.7,
            num_secondary_width=4,
            num_secondary_height=4,
            add_support_beam=True,
            sections=self.max_indices,
        )
        initial_mass = initial_result.mass

        # Create study
        sampler = TPESampler(seed=self.seed, n_startup_trials=20)
        study = optuna.create_study(
            study_name="joint_optimization",
            direction="minimize",
            sampler=sampler,
        )

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            self._create_objective(L, W, H, pressure),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # Get best feasible trial
        best = study.best_trial
        for trial in study.trials:
            if trial.user_attrs.get("feasible", False):
                if trial.value < best.value or not best.user_attrs.get("feasible", False):
                    best = trial

        elapsed = time.time() - start_time

        return ExtendedOptResult(
            structure_id=parameters.get("structure_id", "unknown"),
            geometry={"L": L, "W": W, "H": H},
            main_beam_spacing=best.params["main_beam_spacing"],
            num_secondary_width=best.params["num_secondary_width"],
            num_secondary_height=best.params["num_secondary_height"],
            add_support_beam=best.params["add_support_beam"],
            optimal_sections=[best.params[f"section_{i}"] for i in range(4)],
            initial_mass=initial_mass,
            optimal_mass=best.user_attrs["mass"],
            mass_reduction_pct=(initial_mass - best.user_attrs["mass"]) / initial_mass * 100,
            u_sigma=best.user_attrs["u_sigma"],
            u_delta=best.user_attrs["u_delta"],
            is_feasible=best.user_attrs["feasible"],
            n_evaluations=len(study.trials),
            time_seconds=elapsed,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended Topology + Section Optimization")
    parser.add_argument("--L", type=float, default=8.0)
    parser.add_argument("--W", type=float, default=3.6)
    parser.add_argument("--H", type=float, default=3.6)
    parser.add_argument("--pressure", type=float, default=30000.0)
    parser.add_argument("--mode", type=str, default="fea", choices=["fea", "surrogate"])
    parser.add_argument("--surrogate", type=str, default=None, help="Path to surrogate model")
    args = parser.parse_args()

    parameter = {"L": args.L, "W": args.W, "H": args.H, "pressure": args.pressure}

    opt = JointBayesianOptimizer(n_trials=200, seed=42, eval_mode=args.mode, surrogate_path=args.surrogate)
    result = opt.optimize(parameter)
    print(f"Mass: {result.optimal_mass:.0f}kg, Reduction: {result.mass_reduction_pct:.1f}%")
    from src.heuristic.evaluator import evaluate_design

    eval_result = evaluate_design(
        L=args.L,
        W=args.W,
        H=args.H,
        pressure=args.pressure,
        main_beam_spacing=result.main_beam_spacing,
        num_secondary_width=result.num_secondary_width,
        num_secondary_height=result.num_secondary_height,
        add_support_beam=result.add_support_beam,
        sections=result.optimal_sections,
        mode="fea",
    )
    print("FEA Check - u_sigma: {:.3f}, u_delta: {:.3f}".format(eval_result[1], eval_result[2]))
