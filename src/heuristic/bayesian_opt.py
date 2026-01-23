"""
Bayesian Optimization for Steel Section Optimization.

Why Bayesian Optimization works well for this problem:
1. Expensive black-box function (FEA simulation ~0.5-2s per evaluation)
2. Low-dimensional discrete space (4 variables, 20 choices each)
3. Smooth objective landscape (mass is monotonic with section size)
4. Need for sample efficiency (100-200 evaluations vs DQN's 100,000+)

This implementation uses Optuna with TPE (Tree-structured Parzen Estimator),
which is a form of Bayesian optimization well-suited for discrete/mixed spaces.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.structure.structural_model import StructuralParameters


@dataclass
class OptimizationResult:
    """Result of a single structure optimization."""

    structure_id: str
    geometry: dict  # L, W, H
    initial_sections: list[int]
    optimal_sections: list[int]
    initial_mass: float
    optimal_mass: float
    mass_reduction_pct: float
    u_sigma: float
    u_delta: float
    is_feasible: bool
    n_trials: int
    time_seconds: float


class BayesianSectionOptimizer:
    """
    Bayesian Optimization for steel section selection.

    Uses Optuna's TPE sampler for efficient exploration of the
    discrete 4-dimensional section space.
    """

    def __init__(
        self,
        env_config: Optional[dict] = None,
        n_trials: int = 100,
        constraint_penalty: float = 1000.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            env_config: Configuration for SteelSectionOptEnv
            n_trials: Number of optimization trials (FEA evaluations)
            constraint_penalty: Penalty multiplier for constraint violations
            seed: Random seed for reproducibility
        """

        self.env_config = env_config or {}
        self.n_trials = n_trials
        self.constraint_penalty = constraint_penalty
        self.seed = seed

        # Get section counts from env
        temp_env = SteelSectionOptEnv(self.env_config)
        self.sec_counts = temp_env.sec_counts
        self.max_indices = temp_env.max_indices

    def _create_objective(self, env: SteelSectionOptEnv, initial_mass: float):
        """
        Create the objective function for Optuna.

        Objective: Minimize mass + penalty for constraint violations
        """

        def objective(trial: optuna.Trial) -> float:
            # Sample section indices
            idx = [trial.suggest_int(f"idx_{i}", 0, self.sec_counts[i] - 1) for i in range(4)]

            # Update environment
            env.idx = np.array(idx, dtype=np.int64)
            env.current_params.frame_section_index = idx[0]
            env.current_params.main_beam_section_index = idx[1]
            env.current_params.secondary_beam_section_index = idx[2]
            env.current_params.brace_section_index = idx[3]

            # Run FEA simulation
            out = env._run_simulation_and_collect()

            mass = out["mass"]
            u_sigma = out["u_sigma"]
            u_delta = out["u_delta"]

            # Compute constraint violation
            violation = max(0, u_sigma - 1.0) + max(0, u_delta - 1.0)

            # Penalized objective: minimize mass, but heavily penalize violations
            # Using exterior penalty method
            objective_value = mass + self.constraint_penalty * violation * initial_mass

            # Store additional info for analysis
            trial.set_user_attr("mass", mass)
            trial.set_user_attr("u_sigma", u_sigma)
            trial.set_user_attr("u_delta", u_delta)
            trial.set_user_attr("violation", violation)
            trial.set_user_attr("feasible", violation == 0)

            return objective_value

        return objective

    def optimize(self, geometry: dict) -> OptimizationResult:
        """
        Optimize section selection for a single structure.

        Args:
            geometry: Dict with keys 'L', 'W', 'H', and optionally 'pressure'

        Returns:
            OptimizationResult with optimal sections and metrics
        """
        start_time = time.time()

        # Create environment with specified geometry
        env = SteelSectionOptEnv(self.env_config)

        # Build StructuralParameters
        L = geometry["L"]
        W = geometry["W"]
        H = geometry["H"]
        pressure = geometry.get("pressure", 30000.0)

        env.current_params = StructuralParameters(
            length=L,
            width=W,
            height=H,
            main_beam_spacing=geometry.get("main_beam_spacing", 0.7),
            num_secondary_width=geometry.get("num_secondary_width", 4),
            num_secondary_height=geometry.get("num_secondary_height", 4),
            add_support_beam=geometry.get("add_support_beam", True),
            frame_section_index=int(self.max_indices[0]),
            main_beam_section_index=int(self.max_indices[1]),
            secondary_beam_section_index=int(self.max_indices[2]),
            brace_section_index=int(self.max_indices[3]),
        )
        env.pressure = pressure
        env.idx = self.max_indices.copy()

        # Get initial mass (with max sections)
        initial_out = env._run_simulation_and_collect()
        initial_mass = initial_out["mass"]

        # Create Optuna study
        sampler = TPESampler(seed=self.seed, n_startup_trials=10)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=f"section_opt_L{L}_W{W}_H{H}",
        )

        # Add initial point (max sections) as first trial
        study.enqueue_trial({f"idx_{i}": int(self.max_indices[i]) for i in range(4)})

        # Run optimization (suppress Optuna's verbose output)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            self._create_objective(env, initial_mass),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        # Get best feasible trial
        best_trial = None
        for trial in study.best_trials if hasattr(study, "best_trials") else [study.best_trial]:
            if trial.user_attrs.get("feasible", False):
                best_trial = trial
                break

        # If no feasible solution found, use best trial anyway
        if best_trial is None:
            best_trial = study.best_trial

        # Extract results
        optimal_sections = [best_trial.params[f"idx_{i}"] for i in range(4)]
        optimal_mass = best_trial.user_attrs["mass"]
        u_sigma = best_trial.user_attrs["u_sigma"]
        u_delta = best_trial.user_attrs["u_delta"]
        is_feasible = best_trial.user_attrs["feasible"]

        elapsed = time.time() - start_time

        return OptimizationResult(
            structure_id=geometry.get("structure_id", "unknown"),
            geometry={"L": L, "W": W, "H": H},
            initial_sections=self.max_indices.tolist(),
            optimal_sections=optimal_sections,
            initial_mass=initial_mass,
            optimal_mass=optimal_mass,
            mass_reduction_pct=(initial_mass - optimal_mass) / initial_mass * 100,
            u_sigma=u_sigma,
            u_delta=u_delta,
            is_feasible=is_feasible,
            n_trials=len(study.trials),
            time_seconds=elapsed,
        )

    def optimize_batch(
        self,
        structures: list[dict],
        output_dir: str = "results/bayesian_opt",
    ) -> list[OptimizationResult]:
        """
        Optimize multiple structures.

        Args:
            structures: List of geometry dicts
            output_dir: Directory to save results

        Returns:
            List of OptimizationResult
        """
        os.makedirs(output_dir, exist_ok=True)
        results: List[OptimizationResult] = []

        print(f"Bayesian Optimization: {len(structures)} structures, {self.n_trials} trials each")
        print("=" * 70)

        for i, geom in enumerate(structures):
            sid = geom.get("structure_id", f"struct_{i}")
            print(
                f"[{i+1}/{len(structures)}] {sid}: L={geom['L']:.1f}, W={geom['W']:.1f}, H={geom['H']:.1f} ... ",
                end="",
                flush=True,
            )

            try:
                result = self.optimize(geom)
                results.append(result)

                status = "✓" if result.is_feasible else "⚠"
                print(
                    f"{status} mass={result.optimal_mass:.0f}kg "
                    f"({result.mass_reduction_pct:.1f}% reduction), "
                    f"sections={result.optimal_sections}, "
                    f"time={result.time_seconds:.1f}s"
                )

            except Exception as e:
                print(f"✗ ERROR: {e}")

        # Save results
        results_dict = [
            {
                "structure_id": r.structure_id,
                **r.geometry,
                "initial_sections": r.initial_sections,
                "optimal_sections": r.optimal_sections,
                "initial_mass": r.initial_mass,
                "optimal_mass": r.optimal_mass,
                "mass_reduction_pct": r.mass_reduction_pct,
                "u_sigma": r.u_sigma,
                "u_delta": r.u_delta,
                "is_feasible": r.is_feasible,
                "n_trials": r.n_trials,
                "time_seconds": r.time_seconds,
            }
            for r in results
        ]

        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        # Print summary
        if results:
            feasible = [r for r in results if r.is_feasible]
            print("\n" + "=" * 70)
            print(f"SUMMARY: {len(feasible)}/{len(results)} feasible solutions")
            if feasible:
                reductions = [r.mass_reduction_pct for r in feasible]
                times = [r.time_seconds for r in feasible]
                print(f"Mass reduction: {np.mean(reductions):.1f}% ± {np.std(reductions):.1f}%")
                print(f"Time per structure: {np.mean(times):.1f}s ± {np.std(times):.1f}s")
            print(f"Results saved to: {output_dir}/results.json")

        return results


class ConstrainedBayesianOptimizer(BayesianSectionOptimizer):
    """
    Constrained Bayesian Optimization using Optuna's pruning.

    This version uses a two-stage approach:
    1. First, find feasible region using constraint satisfaction
    2. Then, minimize mass within feasible region
    """

    def _create_objective(self, env: SteelSectionOptEnv, initial_mass: float):
        """
        Create objective with early pruning for infeasible solutions.
        """

        def objective(trial: optuna.Trial) -> float:
            idx = [trial.suggest_int(f"idx_{i}", 0, self.sec_counts[i] - 1) for i in range(4)]

            env.idx = np.array(idx, dtype=np.int64)
            env.current_params.frame_section_index = idx[0]
            env.current_params.main_beam_section_index = idx[1]
            env.current_params.secondary_beam_section_index = idx[2]
            env.current_params.brace_section_index = idx[3]

            out = env._run_simulation_and_collect()

            mass = out["mass"]
            u_sigma = out["u_sigma"]
            u_delta = out["u_delta"]
            violation = max(0, u_sigma - 1.0) + max(0, u_delta - 1.0)

            trial.set_user_attr("mass", mass)
            trial.set_user_attr("u_sigma", u_sigma)
            trial.set_user_attr("u_delta", u_delta)
            trial.set_user_attr("violation", violation)
            trial.set_user_attr("feasible", violation == 0)

            # Two-stage objective:
            # Stage 1: If infeasible, objective = initial_mass + penalty * violation
            # Stage 2: If feasible, objective = mass (direct minimization)
            if violation > 0:
                return initial_mass + self.constraint_penalty * violation
            else:
                return mass

        return objective


def run_single_optimization(
    L: float = 8.0,
    W: float = 3.6,
    H: float = 3.6,
    n_trials: int = 100,
    seed: int = 42,
):
    """
    Run optimization on a single structure with visualization.
    """
    print("=" * 70)
    print("Bayesian Optimization - Single Structure Demo")
    print("=" * 70)

    geometry = {"L": L, "W": W, "H": H, "structure_id": f"demo_L{L}_W{W}_H{H}"}

    optimizer = BayesianSectionOptimizer(
        n_trials=n_trials,
        constraint_penalty=1000.0,
        seed=seed,
    )

    print(f"Geometry: L={L}m, W={W}m, H={H}m")
    print(f"Trials: {n_trials}")
    print("-" * 70)

    result = optimizer.optimize(geometry)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Initial sections: {result.initial_sections}")
    print(f"Optimal sections: {result.optimal_sections}")
    print(f"Initial mass:     {result.initial_mass:.1f} kg")
    print(f"Optimal mass:     {result.optimal_mass:.1f} kg")
    print(f"Mass reduction:   {result.mass_reduction_pct:.1f}%")
    print(f"u_sigma:          {result.u_sigma:.4f} {'✓' if result.u_sigma <= 1 else '✗'}")
    print(f"u_delta:          {result.u_delta:.4f} {'✓' if result.u_delta <= 1 else '✗'}")
    print(f"Feasible:         {'Yes' if result.is_feasible else 'No'}")
    print(f"Time:             {result.time_seconds:.1f}s")
    print(f"FEA evaluations:  {result.n_trials}")

    return result


def compare_with_greedy():
    """
    Compare Bayesian optimization with Greedy baseline.
    """
    from heuristic.baseline import GreedyShrinkBaseline

    print("=" * 70)
    print("COMPARISON: Bayesian Optimization vs Greedy Shrink")
    print("=" * 70)

    # Test structures
    test_geometries = [
        {"L": 8.0, "W": 3.6, "H": 3.6, "structure_id": "test_1"},
        {"L": 10.0, "W": 4.0, "H": 4.0, "structure_id": "test_2"},
        {"L": 6.0, "W": 3.0, "H": 3.0, "structure_id": "test_3"},
    ]

    # Run Greedy
    print("\n--- Greedy Shrink ---")
    greedy = GreedyShrinkBaseline()
    greedy_results = []
    for geom in test_geometries:
        result = greedy.optimize(geom)
        greedy_results.append(result)
        print(
            f"{geom['structure_id']}: mass={result['final_mass']:.0f}kg, "
            f"reduction={result['mass_reduction_pct']:.1f}%, "
            f"sims={result['num_simulations']}, time={result['time_seconds']:.1f}s"
        )

    # Run Bayesian
    print("\n--- Bayesian Optimization (100 trials) ---")
    bayesian = BayesianSectionOptimizer(n_trials=100, seed=42)
    bayesian_results: List[OptimizationResult] = []
    for geom in test_geometries:
        result = bayesian.optimize(geom)
        bayesian_results.append(result)
        print(
            f"{result.structure_id}: mass={result.optimal_mass:.0f}kg, "
            f"reduction={result.mass_reduction_pct:.1f}%, "
            f"sims={result.n_trials}, time={result.time_seconds:.1f}s"
        )

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Method':<20} {'Avg Reduction':<15} {'Avg Sims':<12} {'Avg Time':<10}")
    print("-" * 60)

    greedy_reductions = [r["mass_reduction_pct"] for r in greedy_results]
    greedy_sims = [r["num_simulations"] for r in greedy_results]
    greedy_times = [r["time_seconds"] for r in greedy_results]
    print(
        f"{'Greedy':<20} {np.mean(greedy_reductions):.1f}% ± {np.std(greedy_reductions):.1f}%"
        f"   {np.mean(greedy_sims):.0f}          {np.mean(greedy_times):.1f}s"
    )

    bayesian_reductions = [r.mass_reduction_pct for r in bayesian_results]
    bayesian_sims = [r.n_trials for r in bayesian_results]
    bayesian_times = [r.time_seconds for r in bayesian_results]
    print(
        f"{'Bayesian (100)':<20} {np.mean(bayesian_reductions):.1f}% ± {np.std(bayesian_reductions):.1f}%"
        f"   {np.mean(bayesian_sims):.0f}         {np.mean(bayesian_times):.1f}s"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian Optimization for Steel Section Selection")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "compare", "batch"])
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--L", type=float, default=9.6, help="Structure length")
    parser.add_argument("--W", type=float, default=3.7, help="Structure width")
    parser.add_argument("--H", type=float, default=3.3, help="Structure height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.mode == "single":
        run_single_optimization(L=args.L, W=args.W, H=args.H, n_trials=args.trials, seed=args.seed)
    elif args.mode == "compare":
        compare_with_greedy()
    elif args.mode == "batch":
        # Load test structures if available
        test_file = "results/test_structures.json"
        if os.path.exists(test_file):
            with open(test_file) as f:
                structures = json.load(f)
            optimizer = BayesianSectionOptimizer(n_trials=args.trials, seed=args.seed)
            optimizer.optimize_batch(structures)
        else:
            print(f"Test file not found: {test_file}")
            print("Run: python scripts/generate_test_set.py first")
