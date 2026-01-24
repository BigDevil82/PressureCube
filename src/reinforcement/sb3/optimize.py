"""
Use trained MaskablePPO model to optimize steel sections and topology for new structures.

Extended version: optimizes both section indices and topology parameters.

Usage:
    python -m src.reinforcement.sb3.optimize \
        --model_path results/sb3/0124_1200/final_model.zip \
        --L 8.0 --W 3.6 --H 3.3 \
        --pressure 30000
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.heuristic.evaluator import StructuralEvaluator
from src.reinforcement.sb3.env import SteelSectionOptEnvSB3


@dataclass
class OptimizationResult:
    """Result of structural optimization."""

    sections: List[int]  # [frame, main, secondary, brace]
    topology: Dict  # {spacing, num_secondary_width, num_secondary_height, add_support_beam}
    mass: float
    u_sigma: float
    u_delta: float
    feasible: bool
    steps: int
    trajectory: List[dict] = field(default_factory=list)  # History of all states


def mask_fn(env: SteelSectionOptEnvSB3) -> np.ndarray:
    """Get action mask from environment."""
    return env.action_masks()


def optimize_structure(
    model_path: str,
    L: float,
    W: float,
    H: float,
    pressure: float = 30000.0,
    init_sections: Optional[List[int]] = None,
    init_topology: Optional[Dict] = None,
    eval_mode: str = "fea",
    surrogate_path: Optional[str] = None,
    max_steps: int = 50,
    deterministic: bool = True,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize steel sections and topology for a given structure using trained model.

    Args:
        model_path: Path to trained MaskablePPO model (.zip)
        L, W, H: Structure dimensions (meters)
        pressure: Internal pressure (Pa)
        init_sections: Initial section indices [frame, main, sec, brace] or None for random
        init_topology: Initial topology dict or None for random
        eval_mode: "fea" for real analysis, "surrogate" for fast prediction
        surrogate_path: Path to surrogate model (required if eval_mode="surrogate")
        max_steps: Maximum optimization steps
        deterministic: Use deterministic policy (no exploration)
        verbose: Print progress

    Returns:
        OptimizationResult with optimal sections, topology, and metrics
    """
    # Load model
    model = MaskablePPO.load(model_path)
    if verbose:
        print(f"Loaded model from: {model_path}")

    # Create environment
    env_config = {
        "eval_mode": eval_mode,
        "surrogate_path": surrogate_path,
        "fixed_geometry": (L, W, H),
        "pressure": pressure,
        "T": max_steps,
        "K_no_improve": max_steps,  # Disable early stopping
    }

    env = SteelSectionOptEnvSB3(env_config)
    env = ActionMasker(env, mask_fn)

    # Reset with optional initial state
    reset_options = {}
    if init_sections is not None:
        reset_options["init_sections"] = init_sections
    if init_topology is not None:
        reset_options["init_topology"] = init_topology

    obs, info = env.reset(options=reset_options)

    if verbose:
        print(f"\nStructure: L={L}m, W={W}m, H={H}m, pressure={pressure/1000:.1f}kPa")
        print(f"Initial sections: {env.unwrapped.idx.tolist()}")
        print(f"Initial topology: {info.get('topology', {})}")
        print(
            f"Initial: mass={info['mass']:.1f}kg, u_sigma={info['u_sigma']:.3f}, u_delta={info['u_delta']:.3f}"
        )
        print(f"Initial feasible: {info['feasible']}")
        print("-" * 60)

    # Track trajectory
    trajectory = [
        {
            "step": 0,
            "sections": env.unwrapped.idx.tolist(),
            "topology": info.get("topology", {}),
            "mass": info["mass"],
            "u_sigma": info["u_sigma"],
            "u_delta": info["u_delta"],
            "feasible": info["feasible"],
            "mnorm": info.get("mnorm", info["mass"] / env.unwrapped.m0),
        }
    ]

    # Track best feasible solution
    best_feasible = None
    if info["feasible"]:
        best_feasible = trajectory[0].copy()

    # Optimization loop
    step = 0
    done = False

    while not done and step < max_steps:
        # Get action from model
        action_masks = env.unwrapped.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        # Record state
        state = {
            "step": step,
            "sections": env.unwrapped.idx.tolist(),
            "topology": info.get("topology", {}),
            "mass": info["mass"],
            "u_sigma": info["u_sigma"],
            "u_delta": info["u_delta"],
            "feasible": info["feasible"],
            "mnorm": info.get("mnorm", info["mass"] / env.unwrapped.m0),
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "reward": float(reward),
        }
        trajectory.append(state)

        # Update best feasible
        if info["feasible"]:
            if best_feasible is None or state["mnorm"] < best_feasible["mnorm"]:
                best_feasible = state.copy()

        if verbose:
            section_deltas, topology_deltas, support_toggle = env.unwrapped.decode_action(action)
            topo = list(state["topology"].values())
            print(
                f"Step {step:2d}: sec_delta={section_deltas.tolist()}, topo_delta={topology_deltas.tolist()}, toggle={support_toggle} -> "
                f"sections={state['sections']}, topo={topo}, mass={state['mass']:.1f}kg, "
                f"u_sigma={state['u_sigma']:.3f}, u_delta={state['u_delta']:.3f}, "
                f"feasible={state['feasible']}, reward={reward:.3f}"
            )

    env.close()

    # Determine final result
    if best_feasible is not None:
        final = best_feasible
        if verbose:
            print("-" * 60)
            print(f"Best feasible found at step {final['step']}:")
            print(f"  Sections: {final['sections']}")
            print(f"  Topology: {final.get('topology', {})}")
            print(f"  Mass: {final['mass']:.1f}kg (mnorm={final['mnorm']:.3f})")
            print(f"  u_sigma: {final['u_sigma']:.3f}, u_delta: {final['u_delta']:.3f}")
    else:
        final = trajectory[-1]
        if verbose:
            print("-" * 60)
            print("No feasible solution found. Final state:")
            print(f"  Sections: {final['sections']}")
            print(f"  Topology: {final.get('topology', {})}")
            print(f"  Mass: {final['mass']:.1f}kg")
            print(f"  u_sigma: {final['u_sigma']:.3f}, u_delta: {final['u_delta']:.3f}")

    return OptimizationResult(
        sections=final["sections"],
        topology=final.get("topology", {}),
        mass=final["mass"],
        u_sigma=final["u_sigma"],
        u_delta=final["u_delta"],
        feasible=final["feasible"],
        steps=step,
        trajectory=trajectory,
    )


def validate_with_fea(
    result: OptimizationResult,
    L: float,
    W: float,
    H: float,
    pressure: float = 30000.0,
) -> dict:
    """
    Validate optimization result with real FEA.

    Use this when optimization was done with surrogate model.
    Uses the topology from the optimization result.
    """
    topo = result.topology
    evaluator = StructuralEvaluator(mode="fea")
    result_fea = evaluator.evaluate(
        L=L,
        W=W,
        H=H,
        pressure=pressure,
        main_beam_spacing=topo.get("spacing", 0.7),
        num_secondary_width=topo.get("num_secondary_width", 4),
        num_secondary_height=topo.get("num_secondary_height", 4),
        add_support_beam=topo.get("add_support_beam", True),
        sections=result.sections,
    )

    return {
        "sections": result.sections,
        "topology": result.topology,
        "mass": result_fea.mass,
        "u_sigma": result_fea.u_sigma,
        "u_delta": result_fea.u_delta,
        "feasible": result_fea.feasible,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize steel sections and topology using trained model")

    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.zip)")

    # Structure geometry
    parser.add_argument("--L", type=float, required=True, help="Structure length (m)")
    parser.add_argument("--W", type=float, required=True, help="Structure width (m)")
    parser.add_argument("--H", type=float, required=True, help="Structure height (m)")
    parser.add_argument("--pressure", type=float, default=30000.0, help="Internal pressure (Pa)")

    # Initial sections (optional)
    parser.add_argument("--init_frame", type=int, default=None)
    parser.add_argument("--init_main", type=int, default=None)
    parser.add_argument("--init_sec", type=int, default=None)
    parser.add_argument("--init_brace", type=int, default=None)

    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="fea", choices=["fea", "surrogate"])
    parser.add_argument("--surrogate_path", type=str, default=None)

    # Optimization settings
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument(
        "--validate_fea", action="store_true", help="Validate with FEA after surrogate optimization"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Handle initial sections
    init_sections = None
    if all(x is not None for x in [args.init_frame, args.init_main, args.init_sec, args.init_brace]):
        init_sections = [args.init_frame, args.init_main, args.init_sec, args.init_brace]

    # Run optimization (topology is now optimized by the agent)
    result = optimize_structure(
        model_path=args.model_path,
        L=args.L,
        W=args.W,
        H=args.H,
        pressure=args.pressure,
        init_sections=init_sections,
        eval_mode=args.eval_mode,
        surrogate_path=args.surrogate_path,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
    )

    # Validate with FEA if requested
    if args.validate_fea and args.eval_mode == "surrogate":
        print("\nValidating with FEA...")
        fea_result = validate_with_fea(
            result=result,
            L=args.L,
            W=args.W,
            H=args.H,
            pressure=args.pressure,
        )
        print("FEA validation:")
        print(f"  Mass: {fea_result['mass']:.1f}kg")
        print(f"  u_sigma: {fea_result['u_sigma']:.3f}, u_delta: {fea_result['u_delta']:.3f}")
        print(f"  Feasible: {fea_result['feasible']}")

    # Summary
    print(f"\nOptimization complete in {result.steps} steps")
    print(f"Final sections: {result.sections}")
    print(f"Final topology: {result.topology}")
    print(f"Final mass: {result.mass:.1f}kg")
    print(f"Feasible: {result.feasible}")

    return result


if __name__ == "__main__":
    main()
