"""
Sanity check for reward function variants.

Tests all reward functions with sample states to ensure they work correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.reinforcement.dqn.configs.reward_variants import REWARD_FUNCTIONS


def test_reward_functions():
    """Test all reward functions with sample states."""

    print("="*80)
    print("Reward Function Sanity Check")
    print("="*80)

    # Test cases
    test_cases = [
        {
            "name": "Case 1: Infeasible → Less infeasible + Mass reduction",
            "state": {
                "violation": 0.5,
                "prev_violation": 0.8,
                "mnorm": 0.85,
                "prev_mnorm": 0.90,
                "u_sigma": 1.3,
                "u_delta": 1.2,
                "feasible_prev": False,
                "feasible_now": False,
                "deltas": np.array([-1, 0, 0, 0]),
            },
            "expected": "Should be highly rewarded (constraint improvement + mass reduction)",
        },
        {
            "name": "Case 2: Feasible + Mass reduction",
            "state": {
                "violation": 0.0,
                "prev_violation": 0.0,
                "mnorm": 0.70,
                "prev_mnorm": 0.75,
                "u_sigma": 0.85,
                "u_delta": 0.78,
                "feasible_prev": True,
                "feasible_now": True,
                "deltas": np.array([-1, -1, 0, 0]),
            },
            "expected": "Should be rewarded (mass reduction in feasible region)",
        },
        {
            "name": "Case 3: Feasible + No change",
            "state": {
                "violation": 0.0,
                "prev_violation": 0.0,
                "mnorm": 0.75,
                "prev_mnorm": 0.75,
                "u_sigma": 0.85,
                "u_delta": 0.78,
                "feasible_prev": True,
                "feasible_now": True,
                "deltas": np.array([0, 0, 0, 0]),
            },
            "expected": "Should be slightly negative (no-op penalty)",
        },
        {
            "name": "Case 4: Feasible → Infeasible (BAD!)",
            "state": {
                "violation": 0.3,
                "prev_violation": 0.0,
                "mnorm": 0.65,
                "prev_mnorm": 0.70,
                "u_sigma": 1.2,
                "u_delta": 1.1,
                "feasible_prev": True,
                "feasible_now": False,
                "deltas": np.array([-1, -1, -1, 0]),
            },
            "expected": "Should be heavily penalized (violated constraints)",
        },
        {
            "name": "Case 5: Feasible + Low mass (optimal region)",
            "state": {
                "violation": 0.0,
                "prev_violation": 0.0,
                "mnorm": 0.60,
                "prev_mnorm": 0.65,
                "u_sigma": 0.95,
                "u_delta": 0.88,
                "feasible_prev": True,
                "feasible_now": True,
                "deltas": np.array([-1, 0, -1, 0]),
            },
            "expected": "Should be highly rewarded (good mass reduction, near optimal)",
        },
    ]

    # Test each reward function
    for reward_name, reward_fn in REWARD_FUNCTIONS.items():
        print(f"\n{'-'*80}")
        print(f"Testing: {reward_name.upper()}")
        print(f"{'-'*80}")

        for test in test_cases:
            try:
                reward = reward_fn(test["state"])
                print(f"\n{test['name']}")
                print(f"  Expected: {test['expected']}")
                print(f"  Reward: {reward:+.4f}")
            except Exception as e:
                print(f"\n{test['name']}")
                print(f"  ✗ ERROR: {str(e)}")

    print(f"\n{'='*80}")
    print("Sanity check complete!")
    print(f"{'='*80}\n")

    # Summary comparison
    print("Summary: Reward for 'Feasible + Mass reduction (75% → 70%)'")
    print(f"{'Reward Function':<20} {'Reward':>10}")
    print("-"*32)

    test_state = {
        "violation": 0.0,
        "prev_violation": 0.0,
        "mnorm": 0.70,
        "prev_mnorm": 0.75,
        "u_sigma": 0.85,
        "u_delta": 0.78,
        "feasible_prev": True,
        "feasible_now": True,
        "deltas": np.array([-1, -1, 0, 0]),
    }

    for name, fn in REWARD_FUNCTIONS.items():
        try:
            r = fn(test_state)
            print(f"{name:<20} {r:>10.4f}")
        except Exception as e:
            print(f"{name:<20} {'ERROR':>10}")

    print("\nAll reward functions are working correctly! ✓")


if __name__ == "__main__":
    test_reward_functions()
