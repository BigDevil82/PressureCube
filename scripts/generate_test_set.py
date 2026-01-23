"""
Generate a reproducible test set of random structures for baseline and RL evaluation.
"""

import json
import os

import numpy as np


def generate_test_structures(n=50, seed=42, output_path="results/test_structures.json"):
    """
    Generate N random structures with varying geometry.

    Args:
        n: Number of test structures
        seed: Random seed for reproducibility
        output_path: Path to save test structures

    Returns:
        List of structure configurations
    """
    rng = np.random.default_rng(seed)

    # Geometry ranges (same as env.py sampling)
    L_values = np.arange(6.0, 12.0 + 0.2, 0.2)  # Length
    W_values = np.arange(3.0, 4.5 + 0.3, 0.3)  # Width
    H_values = np.arange(3.0, 4.5 + 0.3, 0.3)  # Height

    structures = []
    for i in range(n):
        structure = {
            "structure_id": i,
            "L": float(round(rng.choice(L_values), 2)),
            "W": float(round(rng.choice(W_values), 2)),
            "H": float(round(rng.choice(H_values), 2)),
            "pressure": 30000.0,  # Fixed at 30 kPa (same as training default)
            # Topology parameters (fixed, same as training)
            "main_beam_spacing": 0.7,
            "num_secondary_width": 4,
            "num_secondary_height": 4,
            "add_support_beam": True,
        }
        structures.append(structure)

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structures, f, indent=2)

    print(f"Generated {n} test structures")
    print(f"Saved to: {output_path}")
    print(f"\nGeometry statistics:")
    print(f"  L: {np.mean([s['L'] for s in structures]):.2f} ± {np.std([s['L'] for s in structures]):.2f}")
    print(f"  W: {np.mean([s['W'] for s in structures]):.2f} ± {np.std([s['W'] for s in structures]):.2f}")
    print(f"  H: {np.mean([s['H'] for s in structures]):.2f} ± {np.std([s['H'] for s in structures]):.2f}")

    return structures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate test structure set")
    parser.add_argument("--n", type=int, default=50, help="Number of structures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/test_structures.json", help="Output path")

    args = parser.parse_args()
    generate_test_structures(n=args.n, seed=args.seed, output_path=args.output)
