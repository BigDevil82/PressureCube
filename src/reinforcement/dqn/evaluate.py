"""
Evaluation framework for trained DQN models on steel structure optimization.
"""
import csv
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.reinforcement.dqn.agent import DQNAgent
from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.structure.structural_model import StructuralParameters


class ModelEvaluator:
    """
    Evaluate trained RL agent on test structures and compare against baseline.
    """

    def __init__(self, agent_path: str, env_config: dict | None = None):
        """
        Args:
            agent_path: Path to trained agent checkpoint (.pth file)
            env_config: Configuration dict for SteelSectionOptEnv
        """
        self.env_config = env_config or {}

        # Create temporary env to get dimensions and load agent
        temp_env = SteelSectionOptEnv(self.env_config)
        self.max_indices = temp_env.max_indices.copy()

        # Load trained agent
        self.agent = DQNAgent(obs_dim=temp_env.obs_dim, n_actions=temp_env.n_actions)
        self.agent.load(agent_path)
        print(f"Loaded agent from: {agent_path}")

    def evaluate_structure(self, geometry_params: dict, max_steps: int = 50) -> dict[str, Any]:
        """
        Run trained RL policy on a single structure (greedy, no exploration).

        Args:
            geometry_params: Dict with keys 'L', 'W', 'H', 'pressure', etc.
            max_steps: Maximum number of steps allowed

        Returns:
            Dict with results: final_sections, mass, utilization ratios, num_steps
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

        # Run initial simulation
        initial_metrics = env._run_simulation_and_collect()
        initial_mass = initial_metrics["mass"]

        # Build initial observation
        env.m0 = initial_mass
        env.prev_v = env._violation(initial_metrics["u_sigma"], initial_metrics["u_delta"])
        env.prev_mnorm = 1.0

        obs = env._build_obs(initial_metrics)
        mask = env._get_action_mask()

        # RL episode (greedy policy, epsilon=0)
        step = 0
        ep_return = 0.0
        best_mass = initial_mass
        best_sections = init_sections.copy()
        best_u_sigma = initial_metrics["u_sigma"]
        best_u_delta = initial_metrics["u_delta"]

        terminated = False
        truncated = False

        while not (terminated or truncated) and step < max_steps:
            # Select greedy action (no exploration)
            action = self.agent.act(obs, mask, eps=0.0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]

            ep_return += reward
            step += 1

            # Track best feasible solution
            u_sigma = info["u_sigma"]
            u_delta = info["u_delta"]
            mass = info["mass"]

            if u_sigma <= 1.0 and u_delta <= 1.0:
                if mass < best_mass:
                    best_mass = mass
                    best_sections = env.idx.copy()
                    best_u_sigma = u_sigma
                    best_u_delta = u_delta

        # Final metrics
        final_mass = info["mass"]
        final_u_sigma = info["u_sigma"]
        final_u_delta = info["u_delta"]
        final_sections = env.idx.copy()

        # Use best feasible solution if final state is infeasible
        if final_u_sigma > 1.0 or final_u_delta > 1.0:
            if best_mass < initial_mass:  # Have a feasible solution
                final_mass = best_mass
                final_sections = best_sections
                final_u_sigma = best_u_sigma
                final_u_delta = best_u_delta
            else:  # No feasible solution found
                # Revert to initial max sections
                final_mass = initial_mass
                final_sections = init_sections
                final_u_sigma = initial_metrics["u_sigma"]
                final_u_delta = initial_metrics["u_delta"]

        elapsed_time = time.time() - start_time

        # Compute mass reduction
        mass_reduction_pct = (initial_mass - final_mass) / initial_mass * 100.0

        # Check if final solution is feasible
        is_feasible = (final_u_sigma <= 1.0) and (final_u_delta <= 1.0)

        return {
            "final_sections": final_sections.tolist(),
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "mass_reduction_pct": mass_reduction_pct,
            "u_sigma": final_u_sigma,
            "u_delta": final_u_delta,
            "is_feasible": int(is_feasible),
            "num_steps": step,
            "ep_return": ep_return,
            "time_seconds": elapsed_time,
        }

    def evaluate_batch(self, test_structures: list[dict], output_path: str) -> list[dict]:
        """
        Run RL agent on multiple test structures.

        Args:
            test_structures: List of structure configs (from generate_test_set.py)
            output_path: CSV file path to save results

        Returns:
            List of result dicts
        """
        results = []

        print(f"Evaluating RL agent on {len(test_structures)} structures...")
        print(f"Output: {output_path}\n")

        for i, structure in enumerate(test_structures):
            print(f"[{i+1}/{len(test_structures)}] Structure {structure['structure_id']}: ", end="", flush=True)
            print(f"L={structure['L']:.1f}, W={structure['W']:.1f}, H={structure['H']:.1f} ... ", end="", flush=True)

            try:
                result = self.evaluate_structure(structure)

                # Merge structure info with results
                full_result = {
                    "structure_id": structure["structure_id"],
                    "L": structure["L"],
                    "W": structure["W"],
                    "H": structure["H"],
                    **result,
                }
                results.append(full_result)

                feas_str = "✓" if result["is_feasible"] else "✗"
                print(
                    f"{feas_str} mass: {result['final_mass']:.1f} kg ({result['mass_reduction_pct']:.1f}% reduction), "
                    f"steps: {result['num_steps']}, time: {result['time_seconds']:.1f}s"
                )

            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
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
                print(f"RL EVALUATION SUMMARY ({len(successes)}/{len(results)} successful)")
                print(f"{'='*80}")

                feasible = [r for r in successes if r["is_feasible"]]
                mass_reductions = [r["mass_reduction_pct"] for r in feasible] if feasible else [0]
                num_steps = [r["num_steps"] for r in successes]
                times = [r["time_seconds"] for r in successes]

                print(f"Feasibility rate:   {len(feasible)}/{len(successes)} ({len(feasible)/len(successes)*100:.1f}%)")
                print(f"Mass reduction:     {np.mean(mass_reductions):.2f}% ± {np.std(mass_reductions):.2f}%")
                print(f"  Min/Max:          {np.min(mass_reductions):.2f}% / {np.max(mass_reductions):.2f}%")
                print(f"Steps/struct:       {np.mean(num_steps):.1f} ± {np.std(num_steps):.1f}")
                print(f"Time/struct:        {np.mean(times):.1f}s ± {np.std(times):.1f}s")
                print(f"\nResults saved to: {output_path}")

        return results

    def compare_vs_baseline(
        self,
        rl_results_path: str,
        baseline_results_path: str,
        output_path: str | None = None,
    ) -> dict:
        """
        Statistical comparison between RL and baseline methods.

        Args:
            rl_results_path: Path to RL evaluation results CSV
            baseline_results_path: Path to baseline results CSV
            output_path: Optional path to save comparison report

        Returns:
            Dict with comparison statistics
        """
        # Load results
        rl_df = pd.read_csv(rl_results_path)
        baseline_df = pd.read_csv(baseline_results_path)

        # Merge on structure_id
        merged = pd.merge(
            rl_df,
            baseline_df,
            on="structure_id",
            suffixes=("_rl", "_baseline"),
        )

        # Filter to successful cases
        merged = merged[
            (~merged["final_mass_rl"].isna())
            & (~merged["final_mass_baseline"].isna())
            & (merged.get("is_feasible", 1) == 1)  # Only feasible RL solutions
        ]

        n = len(merged)
        if n == 0:
            print("No valid comparisons found!")
            return {}

        # Compute differences
        mass_diff = merged["final_mass_rl"] - merged["final_mass_baseline"]
        mass_reduction_diff = merged["mass_reduction_pct_rl"] - merged["mass_reduction_pct_baseline"]

        # Statistical tests
        t_stat, p_value = stats.ttest_rel(merged["final_mass_rl"], merged["final_mass_baseline"])
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(mass_diff)

        # Comparison metrics
        rl_better_count = (mass_diff < 0).sum()
        baseline_better_count = (mass_diff > 0).sum()
        tie_count = (mass_diff == 0).sum()

        comparison = {
            "n_structures": n,
            "rl_mean_mass": merged["final_mass_rl"].mean(),
            "baseline_mean_mass": merged["final_mass_baseline"].mean(),
            "rl_mean_reduction_pct": merged["mass_reduction_pct_rl"].mean(),
            "baseline_mean_reduction_pct": merged["mass_reduction_pct_baseline"].mean(),
            "mean_mass_diff": mass_diff.mean(),
            "std_mass_diff": mass_diff.std(),
            "mean_reduction_diff": mass_reduction_diff.mean(),
            "rl_better_count": int(rl_better_count),
            "baseline_better_count": int(baseline_better_count),
            "tie_count": int(tie_count),
            "paired_t_statistic": t_stat,
            "paired_t_p_value": p_value,
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_value": wilcoxon_p,
        }

        # Print comparison
        print(f"\n{'='*80}")
        print(f"RL vs GREEDY BASELINE COMPARISON")
        print(f"{'='*80}")
        print(f"Number of structures: {n}")
        print(f"\nMass (kg):")
        print(f"  RL:       {comparison['rl_mean_mass']:.1f}")
        print(f"  Greedy:   {comparison['baseline_mean_mass']:.1f}")
        print(f"  Diff:     {comparison['mean_mass_diff']:.1f} ± {comparison['std_mass_diff']:.1f}")
        print(f"\nMass reduction (%):")
        print(f"  RL:       {comparison['rl_mean_reduction_pct']:.2f}%")
        print(f"  Greedy:   {comparison['baseline_mean_reduction_pct']:.2f}%")
        print(f"  Diff:     {comparison['mean_reduction_diff']:.2f}%")
        print(f"\nWin/Lose/Tie (RL perspective):")
        print(f"  RL better:       {rl_better_count} ({rl_better_count/n*100:.1f}%)")
        print(f"  Greedy better:   {baseline_better_count} ({baseline_better_count/n*100:.1f}%)")
        print(f"  Tie:             {tie_count} ({tie_count/n*100:.1f}%)")
        print(f"\nStatistical tests:")
        print(f"  Paired t-test:   t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Wilcoxon:        W={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}")

        if p_value < 0.05:
            if comparison["mean_mass_diff"] < 0:
                print(f"\n✓ RL significantly better than greedy (p < 0.05)")
            else:
                print(f"\n✗ Greedy significantly better than RL (p < 0.05)")
        else:
            print(f"\n~ No significant difference (p >= 0.05)")

        # Save comparison report
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to: {output_path}")

        return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument(
        "--agent-path",
        type=str,
        required=True,
        help="Path to trained agent checkpoint",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="results/test_structures.json",
        help="Path to test structures JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rl_eval/evaluation_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Path to baseline results CSV for comparison",
    )

    args = parser.parse_args()

    # Load test structures
    with open(args.test_set) as f:
        test_structures = json.load(f)

    # Run evaluation
    evaluator = ModelEvaluator(args.agent_path)
    evaluator.evaluate_batch(test_structures, args.output)

    # Compare with baseline if provided
    if args.compare_baseline:
        comparison_output = args.output.replace(".csv", "_comparison.json")
        evaluator.compare_vs_baseline(args.output, args.compare_baseline, comparison_output)
