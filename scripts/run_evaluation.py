"""
Run RL model evaluation.

Usage:
    python scripts/run_evaluation.py --agent-path results/dqn/0119_1648/agent_final.pth
    python scripts/run_evaluation.py --agent-path MODEL.pth --test-set TEST.json --output RESULTS.csv
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reinforcement.dqn.evaluate import ModelEvaluator

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument(
        "--agent-path",
        type=str,
        required=True,
        help="Path to trained agent checkpoint (.pth)",
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
    print(f"Loading test structures from: {args.test_set}")
    with open(args.test_set) as f:
        test_structures = json.load(f)

    print(f"Found {len(test_structures)} test structures\n")

    # Run evaluation
    evaluator = ModelEvaluator(args.agent_path)
    evaluator.evaluate_batch(test_structures, args.output)

    # Compare with baseline if provided
    if args.compare_baseline:
        print(f"\nComparing with baseline: {args.compare_baseline}")
        comparison_output = args.output.replace(".csv", "_comparison.json")
        evaluator.compare_vs_baseline(args.output, args.compare_baseline, comparison_output)
