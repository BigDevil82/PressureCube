from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.evaluator import evaluate_design
from src.optimization.design_space import get_layout_adapter
from src.optimization.parallel_ga import GAResult, ParallelGAOptimizer
from src.structure.enums import BeamType
from src.structure.section_grouping.discovery import (
    SectionGroupingDiscoveryConfig,
    build_rules_from_random_topology_samples,
)
from src.structure.section_grouping.models import SectionGroupingRuleset

DEFAULT_GROUPS = {
    BeamType.FRAME: 1,
    BeamType.MAIN: 2,
    BeamType.SECONDARY: 2,
    BeamType.BRACE: 1,
}


def discover_rules_for_optimization(
    *,
    geometry: dict,
    sections: list[int],
    layout_strategy: str,
    n_samples: int,
    n_jobs: int,
    mesh_size: float,
    seed: int,
) -> SectionGroupingRuleset:
    print("=" * 70)
    print("SECTION GROUPING DISCOVERY")
    print("=" * 70)
    print(f"Samples: {n_samples}, Workers: {n_jobs}, Mesh size: {mesh_size}")
    print(f"Base sections: {sections}")
    config = SectionGroupingDiscoveryConfig(
        n_groups_by_type=DEFAULT_GROUPS,
        description=f"{layout_strategy} grouping discovered from sampled topology designs",
    )
    rules = build_rules_from_random_topology_samples(
        geometry={"L": geometry["L"], "W": geometry["W"], "H": geometry["H"]},
        pressure=geometry["pressure"],
        sections=sections,
        layout_strategy=layout_strategy,
        discovery_config=config,
        n_samples=n_samples,
        seed=seed,
        mesh_size=mesh_size,
        n_jobs=n_jobs,
    )
    print(f"Discovered groups: {rules.group_ids()}")
    print()
    return rules


def compare_methods(
    geometry: dict,
    n_jobs: int = 4,
    eval_mode: str = "fea",
    surrogate_path: str = None,
    layout_strategy: str = "research",
    grouped: bool = False,
    grouping_samples: int = 12,
    grouping_n_jobs: int | None = None,
    grouping_mesh_size: float = 0.5,
    grouping_sections: list[int] | None = None,
):
    """Compare GA with different configurations."""
    section_group_rules = None
    if grouped:
        section_group_rules = discover_rules_for_optimization(
            geometry=geometry,
            sections=grouping_sections or [2, 8, 7, 14],
            layout_strategy=layout_strategy,
            n_samples=grouping_samples,
            n_jobs=grouping_n_jobs or n_jobs,
            mesh_size=grouping_mesh_size,
            seed=42,
        )

    print("=" * 70)
    print("OPTIMIZATION METHOD COMPARISON")
    print("=" * 70)
    print(f"Geometry: L={geometry['L']}, W={geometry['W']}, H={geometry['H']}")
    print(f"Workers: {n_jobs}, Eval mode: {eval_mode}, Layout: {layout_strategy}, Grouped: {grouped}")
    print()

    results = []
    for name, pop_size, generations in [("GA (32x5)", 32, 5), ("GA (48x4)", 48, 4)]:
        print(f"Method: {name}")
        print("-" * 50)
        opt = ParallelGAOptimizer(
            pop_size=pop_size,
            n_generations=generations,
            n_jobs=n_jobs,
            seed=42,
            eval_mode=eval_mode,
            surrogate_path=surrogate_path,
            layout_strategy=layout_strategy,
            section_group_rules=section_group_rules,
        )
        result = opt.optimize(geometry)
        results.append((name, result))
        print(f"  Mass: {result.optimal_mass:.0f}kg ({result.mass_reduction_pct:.1f}% reduction)")
        print(f"  Feasible: {result.is_feasible}, Evals: {result.n_evaluations}")
        print(f"  Time: {result.time_seconds:.1f}s")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<15} {'Mass(kg)':<10} {'Reduction':<12} {'Evals':<8} {'Time(s)':<10} {'Feasible':<10}")
    print("-" * 70)
    for name, result in results:
        print(
            f"{name:<15} {result.optimal_mass:<10.0f} {result.mass_reduction_pct:<12.1f}% "
            f"{result.n_evaluations:<8} {result.time_seconds:<10.1f} {str(result.is_feasible):<10}"
        )
    return results


def run_ga_optimization(
    L: float = 8.0,
    W: float = 3.6,
    H: float = 3.6,
    pressure: float = 30000.0,
    pop_size: int = 32,
    n_generations: int = 10,
    n_jobs: int = 4,
    seed: int = 42,
    eval_mode: str = "fea",
    surrogate_path: str = None,
    layout_strategy: str = "research",
    grouped: bool = False,
    grouping_samples: int = 12,
    grouping_n_jobs: int | None = None,
    grouping_mesh_size: float = 0.5,
    grouping_sections: list[int] | None = None,
) -> GAResult:
    """Run GA optimization with given parameters."""
    geometry = {"L": L, "W": W, "H": H, "pressure": pressure}
    section_group_rules = None
    if grouped:
        section_group_rules = discover_rules_for_optimization(
            geometry=geometry,
            sections=grouping_sections or [2, 8, 7, 14],
            layout_strategy=layout_strategy,
            n_samples=grouping_samples,
            n_jobs=grouping_n_jobs or n_jobs,
            mesh_size=grouping_mesh_size,
            seed=seed,
        )

    print("=" * 70)
    print("PARALLEL GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 70)
    print(f"INPUT: L={L}, W={W}, H={H}, Pressure={pressure}")
    print(f"Population: {pop_size}, Generations: {n_generations}")
    print(f"Total evaluations: {pop_size * n_generations}")
    print(f"Workers: {n_jobs}, Eval mode: {eval_mode}, Layout: {layout_strategy}, Grouped: {grouped}")
    print()

    opt = ParallelGAOptimizer(
        pop_size=pop_size,
        n_generations=n_generations,
        n_jobs=n_jobs,
        seed=seed,
        eval_mode=eval_mode,
        surrogate_path=surrogate_path,
        layout_strategy=layout_strategy,
        section_group_rules=section_group_rules,
    )
    result = opt.optimize(geometry)

    adapter = get_layout_adapter(layout_strategy)
    design = adapter.build_design(
        geometry={"L": L, "W": W, "H": H},
        pressure=pressure,
        topology=result.optimal_topology,
        section_group_rules=opt.section_scheme.rules,
        section_indices_by_group=result.optimal_section_indices_by_group,
    )
    fea_result = evaluate_design(
        design=design,
        mode="fea",
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Optimal mass: {result.optimal_mass:.0f}kg ({result.mass_reduction_pct:.1f}% reduction)")
    print(f"Topology: {result.optimal_topology}")
    print(f"Grouped sections: {result.optimal_section_indices_by_group}")
    print(f"FEA Feasible Check: {fea_result[3]} (u_sigma={fea_result[1]:.3f}, u_delta={fea_result[2]:.3f})")
    print(f"Evaluations: {result.n_evaluations}, Time: {result.time_seconds:.1f}s")
    return result


def random_structures_opt_test(
    eval_mode: str = "surrogate",
    surrogate_path: str = None,
    layout_strategy: str = "research",
    grouped: bool = False,
    grouping_samples: int = 12,
    grouping_n_jobs: int | None = None,
    grouping_mesh_size: float = 0.5,
    grouping_sections: list[int] | None = None,
):
    """Test optimization on random structures."""
    import random

    for i in range(3):
        L = random.uniform(6.0, 12.0)
        W = random.uniform(3.0, 5.0)
        H = random.uniform(3.0, 5.0)
        pressure = random.uniform(20000.0, 40000.0)

        print("\n" + "=" * 70)
        print(f"Structure {i + 1}: L={L:.1f}, W={W:.1f}, H={H:.1f}, Pressure={pressure:.0f}")
        run_ga_optimization(
            L=L,
            W=W,
            H=H,
            pressure=pressure,
            pop_size=32,
            n_generations=10,
            n_jobs=4,
            seed=42,
            eval_mode=eval_mode,
            surrogate_path=surrogate_path,
            layout_strategy=layout_strategy,
            grouped=grouped,
            grouping_samples=grouping_samples,
            grouping_n_jobs=grouping_n_jobs,
            grouping_mesh_size=grouping_mesh_size,
            grouping_sections=grouping_sections,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Genetic Algorithm Optimization")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "compare", "random_test"])
    parser.add_argument("--pop_size", type=int, default=32, help="Population size")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--L", type=float, default=8.0)
    parser.add_argument("--W", type=float, default=3.6)
    parser.add_argument("--H", type=float, default=3.6)
    parser.add_argument("--pressure", type=float, default=30000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_mode", type=str, default="fea", choices=["fea", "surrogate"])
    parser.add_argument("--surrogate", type=str, default=None, help="Path to surrogate model")
    parser.add_argument(
        "--layout-strategy", type=str, default="research", choices=["research", "engineering"]
    )
    parser.add_argument(
        "--grouped", action="store_true", help="Discover and use section-grouping rules before GA"
    )
    parser.add_argument(
        "--grouping-samples", type=int, default=12, help="Representative designs for rule discovery"
    )
    parser.add_argument("--grouping-n-jobs", type=int, default=None, help="Workers for grouping discovery")
    parser.add_argument(
        "--grouping-mesh-size", type=float, default=0.5, help="FEA mesh size for grouping discovery"
    )
    parser.add_argument("--group-frame", type=int, default=2, help="Base frame section for discovery")
    parser.add_argument("--group-main", type=int, default=8, help="Base main section for discovery")
    parser.add_argument("--group-sec", type=int, default=7, help="Base secondary section for discovery")
    parser.add_argument("--group-brace", type=int, default=14, help="Base brace section for discovery")

    args = parser.parse_args()
    grouping_sections = [args.group_frame, args.group_main, args.group_sec, args.group_brace]

    if args.mode == "run":
        run_ga_optimization(
            L=args.L,
            W=args.W,
            H=args.H,
            pressure=args.pressure,
            pop_size=args.pop_size,
            n_generations=args.generations,
            n_jobs=args.n_jobs,
            seed=args.seed,
            eval_mode=args.eval_mode,
            surrogate_path=args.surrogate,
            layout_strategy=args.layout_strategy,
            grouped=args.grouped,
            grouping_samples=args.grouping_samples,
            grouping_n_jobs=args.grouping_n_jobs,
            grouping_mesh_size=args.grouping_mesh_size,
            grouping_sections=grouping_sections,
        )
    elif args.mode == "compare":
        geometry = {"L": args.L, "W": args.W, "H": args.H, "pressure": args.pressure}
        compare_methods(
            geometry,
            n_jobs=args.n_jobs,
            eval_mode=args.eval_mode,
            surrogate_path=args.surrogate,
            layout_strategy=args.layout_strategy,
            grouped=args.grouped,
            grouping_samples=args.grouping_samples,
            grouping_n_jobs=args.grouping_n_jobs,
            grouping_mesh_size=args.grouping_mesh_size,
            grouping_sections=grouping_sections,
        )
    elif args.mode == "random_test":
        random_structures_opt_test(
            eval_mode=args.eval_mode,
            surrogate_path=args.surrogate,
            layout_strategy=args.layout_strategy,
            grouped=args.grouped,
            grouping_samples=args.grouping_samples,
            grouping_n_jobs=args.grouping_n_jobs,
            grouping_mesh_size=args.grouping_mesh_size,
            grouping_sections=grouping_sections,
        )
