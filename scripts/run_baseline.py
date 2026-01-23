"""
Run greedy shrink baseline evaluation.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --test-set results/test_structures.json --output results/baseline/greedy_results.csv
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heuristic.baseline import GreedyShrinkBaseline

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
    print(f"Loading test structures from: {args.test_set}")
    with open(args.test_set) as f:
        test_structures = json.load(f)

    print(f"Found {len(test_structures)} test structures\n")

    # Run baseline
    baseline = GreedyShrinkBaseline()
    baseline.evaluate_batch(test_structures, args.output)
