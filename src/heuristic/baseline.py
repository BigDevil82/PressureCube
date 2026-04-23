"""
Greedy shrink baseline for steel section optimization.

Sequential reduction strategy: frame → main beam → secondary beam → brace
"""

import csv
import json
import os
import time
from typing import Any

import numpy as np

from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.structure.layout.params import StructuralParameters


class GreedyShrinkBaseline:
    """
    Greedy shrink baseline optimizer.

    Strategy: For each component type (in fixed order), reduce section index
    iteratively until constraints are violated, then move to next component.
    """

    def __init__(self, env_config: dict | None = None):
        """
        Args:
            env_config: Configuration dict for SteelSectionOptEnv
        """
        self.env_config = env_config or {}
        # Get max_indices from a temporary env
        temp_env = SteelSectionOptEnv(self.env_config)
        self.max_indices = temp_env.max_indices.copy()

    def optimize(self, geometry_params: dict) -> dict[str, Any]:
        """
        Run greedy shrink optimization on a single structure.

        Args:
            geometry_params: Dict with keys 'L', 'W', 'H', 'pressure', etc.

        Returns:
            Dict with results: final_sections, mass, utilization ratios, num_simulations
        """
        start_time = time.time()

        # Extract geometry
        L = geometry_params["L"]
        W = geometry_params["W"]
        H = geometry_params["H"]
        pressure = geometry_params.get("pressure", 30000.0)

        # Create new env for this structure to properly initialize StructuralParameters
        env = SteelSectionOptEnv(self.env_config)

        # Build StructuralParameters with correct geometry
        init_sections = self.max_indices.copy()
        current_params = StructuralParameters(
            length=L,
            width=W,
            height=H,
            main_beam_spacing=geometry_params.get("main_beam_spacing", 0.7),
            num_secondary_width=geometry_params.get("num_secondary_width", 4),
            num_secondary_height=geometry_params.get("num_secondary_height", 4),
            add_support_beam=geometry_params.get("add_support_beam", True),
            frame_section_index=int(init_sections[0]),
            main_beam_section_index=int(init_sections[1]),
            secondary_beam_section_index=int(init_sections[2]),
            brace_section_index=int(init_sections[3]),
        )

        env.current_params = current_params
        env.idx = init_sections.copy()
        env.pressure = pressure

        # Run initial simulation to get baseline mass
        initial_metrics = env._run_simulation_and_collect()
        initial_mass = initial_metrics["mass"]

        num_simulations = 1  # Count initial simulation

        # Sequential greedy reduction: frame (0) → main (1) → sec (2) → brace (3)
        idx = self.max_indices.copy()

        for component_idx in range(4):
            while idx[component_idx] > 0:
                # Try reducing this component by 1
                idx[component_idx] -= 1

                # Update env sections
                env.idx = idx.copy()
                env.current_params.frame_section_index = int(idx[0])
                env.current_params.main_beam_section_index = int(idx[1])
                env.current_params.secondary_beam_section_index = int(idx[2])
                env.current_params.brace_section_index = int(idx[3])

                # Run FEA simulation
                metrics = env._run_simulation_and_collect()
                num_simulations += 1

                u_sigma = metrics["u_sigma"]
                u_delta = metrics["u_delta"]

                # Check feasibility
                if u_sigma <= 1.0 and u_delta <= 1.0:
                    # Accept reduction, continue shrinking this component
                    continue
                else:
                    # Violation - revert to last feasible solution
                    idx[component_idx] += 1
                    break

        # Get final metrics with final sections
        env.idx = idx.copy()
        env.current_params.frame_section_index = int(idx[0])
        env.current_params.main_beam_section_index = int(idx[1])
        env.current_params.secondary_beam_section_index = int(idx[2])
        env.current_params.brace_section_index = int(idx[3])

        final_metrics = env._run_simulation_and_collect()
        final_mass = final_metrics["mass"]

        elapsed_time = time.time() - start_time

        # Compute mass reduction
        mass_reduction_pct = (initial_mass - final_mass) / initial_mass * 100.0

        return {
            "final_sections": idx.tolist(),
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "mass_reduction_pct": mass_reduction_pct,
            "u_sigma": final_metrics["u_sigma"],
            "u_delta": final_metrics["u_delta"],
            "num_simulations": num_simulations,
            "time_seconds": elapsed_time,
        }

    def evaluate_batch(self, test_structures: list[dict], output_path: str) -> list[dict]:
        """
        Run greedy shrink on multiple test structures.

        Args:
            test_structures: List of structure configs (from generate_test_set.py)
            output_path: CSV file path to save results

        Returns:
            List of result dicts
        """
        results = []

        print(f"Running greedy shrink baseline on {len(test_structures)} structures...")
        print(f"Output: {output_path}\n")

        for i, structure in enumerate(test_structures):
            print(
                f"[{i+1}/{len(test_structures)}] Structure {structure['structure_id']}: ", end="", flush=True
            )
            print(
                f"L={structure['L']:.1f}, W={structure['W']:.1f}, H={structure['H']:.1f} ... ",
                end="",
                flush=True,
            )

            try:
                result = self.optimize(structure)

                # Merge structure info with results
                full_result = {
                    "structure_id": structure["structure_id"],
                    "L": structure["L"],
                    "W": structure["W"],
                    "H": structure["H"],
                    **result,
                }
                results.append(full_result)

                print(
                    f"✓ mass: {result['final_mass']:.1f} kg ({result['mass_reduction_pct']:.1f}% reduction), "
                    f"sims: {result['num_simulations']}, time: {result['time_seconds']:.1f}s"
                )

            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                # Still record the failure
                results.append(
                    {
                        "structure_id": structure["structure_id"],
                        "L": structure["L"],
                        "W": structure["W"],
                        "H": structure["H"],
                        "error": str(e),
                    }
                )

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            if results:
                fieldnames = list(results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        # Print summary statistics
        if results:
            successes = [r for r in results if "error" not in r]
            if successes:
                print(f"\n{'='*80}")
                print(f"BASELINE RESULTS SUMMARY ({len(successes)}/{len(results)} successful)")
                print(f"{'='*80}")

                mass_reductions = [r["mass_reduction_pct"] for r in successes]
                num_sims = [r["num_simulations"] for r in successes]
                times = [r["time_seconds"] for r in successes]

                print(f"Mass reduction:     {np.mean(mass_reductions):.2f}% ± {np.std(mass_reductions):.2f}%")
                print(f"  Min/Max:          {np.min(mass_reductions):.2f}% / {np.max(mass_reductions):.2f}%")
                print(f"Simulations/struct: {np.mean(num_sims):.1f} ± {np.std(num_sims):.1f}")
                print(f"Time/struct:        {np.mean(times):.1f}s ± {np.std(times):.1f}s")
                print(f"Total time:         {sum(times) / 3600:.2f} hours")
                print(f"\nResults saved to: {output_path}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run greedy shrink baseline")
    parser.add_argument(
        "--test-set",
        type=str,
        default="results/test_structures.json",
        help="Path to test structures JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline/greedy_shrink_results.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    # Load test structures
    with open(args.test_set) as f:
        test_structures = json.load(f)

    # Run baseline
    baseline = GreedyShrinkBaseline()
    baseline.evaluate_batch(test_structures, args.output)
