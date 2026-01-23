"""
Parallel Genetic Algorithm for Topology + Section Optimization.

GA is naturally parallelizable - the entire population can be evaluated in parallel.
This implementation uses joblib for process-based parallelism.

Usage:
    python -m src.heuristic.parallel_ga --pop_size 32 --generations 10 --n_jobs 4

    # With surrogate model for fast evaluation:
    python -m src.heuristic.parallel_ga --eval_mode surrogate --surrogate results/surrogate_data/model_advanced.pkl
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from src.heuristic.evaluator import evaluate_design
from src.structure.section_library import SectionLibrary, SectionType


@dataclass
class GAResult:
    """Result of GA optimization."""

    structure_id: str
    geometry: dict

    # Optimal topology
    main_beam_spacing: float
    num_secondary_width: int
    num_secondary_height: int
    add_support_beam: bool

    # Optimal sections
    optimal_sections: list

    # Metrics
    initial_mass: float
    optimal_mass: float
    mass_reduction_pct: float
    u_sigma: float
    u_delta: float
    is_feasible: bool

    # Efficiency
    n_evaluations: int
    n_generations: int
    time_seconds: float


# ============================================================================
# Individual representation
# ============================================================================

# Gene indices
GENE_SPACING = 0  # main_beam_spacing (categorical index)
GENE_SEC_W = 1  # num_secondary_width
GENE_SEC_H = 2  # num_secondary_height
GENE_SUPPORT = 3  # add_support_beam (0 or 1)
GENE_SEC_FRAME = 4  # frame section index
GENE_SEC_MAIN = 5  # main beam section index
GENE_SEC_SECONDARY = 6  # secondary beam section index
GENE_SEC_BRACE = 7  # brace section index

N_GENES = 8

# Gene bounds
SPACING_VALUES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def decode_individual(genes: np.ndarray, sec_counts: List[int]) -> dict:
    """Decode genes to configuration dict."""
    return {
        "spacing": SPACING_VALUES[int(genes[GENE_SPACING]) % len(SPACING_VALUES)],
        "sec_w": int(np.clip(genes[GENE_SEC_W], 2, 6)),
        "sec_h": int(np.clip(genes[GENE_SEC_H], 2, 6)),
        "add_support": bool(genes[GENE_SUPPORT] >= 0.5),
        "sections": [
            int(np.clip(genes[GENE_SEC_FRAME], 0, sec_counts[0] - 1)),
            int(np.clip(genes[GENE_SEC_MAIN], 0, sec_counts[1] - 1)),
            int(np.clip(genes[GENE_SEC_SECONDARY], 0, sec_counts[2] - 1)),
            int(np.clip(genes[GENE_SEC_BRACE], 0, sec_counts[3] - 1)),
        ],
    }


# ============================================================================
# Fitness evaluation (runs in separate process)
# ============================================================================


def evaluate_individual(
    genes: np.ndarray,
    L: float,
    W: float,
    H: float,
    pressure: float,
    sec_counts: List[int],
    eval_mode: str = "fea",
    surrogate_path: Optional[str] = None,
) -> Tuple[float, float, float, float, bool]:
    """
    Evaluate fitness of an individual.

    Returns:
        (mass, u_sigma, u_delta, fitness, feasible)

    Fitness = mass for feasible, mass * (1 + penalty * violation) for infeasible
    Lower fitness is better.
    """
    config = decode_individual(genes, sec_counts)

    mass, u_sigma, u_delta, feasible = evaluate_design(
        L=L,
        W=W,
        H=H,
        pressure=pressure,
        main_beam_spacing=config["spacing"],
        num_secondary_width=config["sec_w"],
        num_secondary_height=config["sec_h"],
        add_support_beam=config["add_support"],
        sections=config["sections"],
        mode=eval_mode,
        surrogate_path=surrogate_path,
    )
    if eval_mode == "surrogate":
        violation = max(0, u_sigma - 0.95) + max(0, u_delta - 0.95)
    else:
        violation = max(0, u_sigma - 1.0) + max(0, u_delta - 1.0)

    # Penalized fitness (lower is better)
    if violation > 0:
        fitness = mass * (1 + 10 * violation)
    else:
        fitness = mass

    return mass, u_sigma, u_delta, fitness, feasible


# ============================================================================
# Genetic operators
# ============================================================================


def create_random_population(
    pop_size: int,
    sec_counts: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create random initial population."""
    population = np.zeros((pop_size, N_GENES), dtype=np.float32)

    for i in range(pop_size):
        population[i, GENE_SPACING] = rng.integers(0, len(SPACING_VALUES))
        population[i, GENE_SEC_W] = rng.integers(2, 7)
        population[i, GENE_SEC_H] = rng.integers(2, 7)
        population[i, GENE_SUPPORT] = rng.integers(0, 2)
        population[i, GENE_SEC_FRAME] = rng.integers(0, sec_counts[0])
        population[i, GENE_SEC_MAIN] = rng.integers(0, sec_counts[1])
        population[i, GENE_SEC_SECONDARY] = rng.integers(0, sec_counts[2])
        population[i, GENE_SEC_BRACE] = rng.integers(0, sec_counts[3])

    return population


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    n_select: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tournament selection - select n_select individuals."""
    pop_size = len(population)
    selected_indices = []

    for _ in range(n_select):
        # Random tournament
        candidates = rng.choice(pop_size, size=tournament_size, replace=False)
        # Select best (lowest fitness)
        winner = candidates[np.argmin(fitness[candidates])]
        selected_indices.append(winner)

    return population[selected_indices].copy()


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover - each gene has 50% chance from either parent."""
    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    mask = rng.random(N_GENES) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    return child1, child2


def mutate(
    individual: np.ndarray,
    mutation_rate: float,
    sec_counts: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutate individual genes."""
    mutated = individual.copy()

    for gene_idx in range(N_GENES):
        if rng.random() < mutation_rate:
            if gene_idx == GENE_SPACING:
                mutated[gene_idx] = rng.integers(0, len(SPACING_VALUES))
            elif gene_idx == GENE_SEC_W:
                mutated[gene_idx] = rng.integers(2, 7)
            elif gene_idx == GENE_SEC_H:
                mutated[gene_idx] = rng.integers(2, 7)
            elif gene_idx == GENE_SUPPORT:
                mutated[gene_idx] = 1 - mutated[gene_idx]  # flip
            elif gene_idx == GENE_SEC_FRAME:
                # Small mutation: ±3
                delta = rng.integers(-3, 4)
                mutated[gene_idx] = np.clip(mutated[gene_idx] + delta, 0, sec_counts[0] - 1)
            elif gene_idx == GENE_SEC_MAIN:
                delta = rng.integers(-3, 4)
                mutated[gene_idx] = np.clip(mutated[gene_idx] + delta, 0, sec_counts[1] - 1)
            elif gene_idx == GENE_SEC_SECONDARY:
                delta = rng.integers(-3, 4)
                mutated[gene_idx] = np.clip(mutated[gene_idx] + delta, 0, sec_counts[2] - 1)
            elif gene_idx == GENE_SEC_BRACE:
                delta = rng.integers(-3, 4)
                mutated[gene_idx] = np.clip(mutated[gene_idx] + delta, 0, sec_counts[3] - 1)

    return mutated


# ============================================================================
# Parallel GA Optimizer
# ============================================================================


class ParallelGAOptimizer:
    """
    Parallel Genetic Algorithm for topology + section optimization.

    Features:
    - Fully parallel population evaluation
    - Elitism (preserve best individuals)
    - Tournament selection
    - Uniform crossover
    - Adaptive mutation rate
    - Supports both FEA and surrogate evaluation modes
    """

    def __init__(
        self,
        pop_size: int = 32,
        n_generations: int = 10,
        n_jobs: int = 4,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        elitism: int = 2,
        seed: Optional[int] = None,
        eval_mode: str = "fea",
        surrogate_path: Optional[str] = None,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.n_jobs = n_jobs
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.seed = seed
        self.eval_mode = eval_mode
        self.surrogate_path = surrogate_path

        # Get section counts from library
        section_lib = SectionLibrary("src/data/steel_sections_large.json")
        sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [section_lib.section_counts[t] for t in sec_types]
        self.max_indices = [count - 1 for count in self.sec_counts]

    def optimize(self, geometry: dict, verbose: bool = True) -> GAResult:
        """Run parallel GA optimization."""
        start_time = time.time()

        L = geometry["L"]
        W = geometry["W"]
        H = geometry["H"]
        pressure = geometry.get("pressure", 30000.0)

        rng = np.random.default_rng(self.seed)

        # Get initial mass (max sections, default topology)
        max_genes = np.array(
            [0, 4, 4, 1] + self.max_indices,
            dtype=np.float32,
        )
        initial_result = evaluate_individual(
            max_genes, L, W, H, pressure, self.sec_counts, self.eval_mode, self.surrogate_path
        )
        initial_mass = initial_result[0]
        print(f"Initial mass (max sections): {initial_mass:.0f}kg")

        # Initialize population
        population = create_random_population(self.pop_size, self.sec_counts, rng)

        # Add max sections individual to ensure feasible start
        population[0] = max_genes

        # Track best
        best_individual = None
        best_result = (1e9, 10.0, 10.0, 1e9, False)
        total_evaluations = 0

        # Evolution loop
        for gen in range(self.n_generations):
            # Parallel evaluation of entire population
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(evaluate_individual)(
                    ind, L, W, H, pressure, self.sec_counts, self.eval_mode, self.surrogate_path
                )
                for ind in population
            )

            total_evaluations += len(population)

            # Extract fitness values
            fitness = np.array([r[3] for r in results])
            feasible_mask = np.array([r[4] for r in results])

            # Update best
            for i, res in enumerate(results):
                if res[4]:  # feasible
                    if not best_result[4] or res[3] < best_result[3]:
                        best_individual = population[i].copy()
                        best_result = res
                elif not best_result[4]:  # both infeasible, prefer lower penalty
                    if res[3] < best_result[3]:
                        best_individual = population[i].copy()
                        best_result = res

            # Statistics
            n_feasible = np.sum(feasible_mask)
            mean_fitness = np.mean(fitness)

            if verbose:
                print(
                    f"  Gen {gen + 1:2d}/{self.n_generations}: "
                    f"best mass={best_result[0]:.0f}kg, "
                    f"feasible={n_feasible}/{self.pop_size}, "
                    f"mean_fit={mean_fitness:.0f}"
                )

            # Last generation - no need to create next population
            if gen == self.n_generations - 1:
                break

            # Create next generation
            next_population = np.zeros_like(population)

            # Elitism - keep best individuals
            elite_indices = np.argsort(fitness)[: self.elitism]
            for i, elite_idx in enumerate(elite_indices):
                next_population[i] = population[elite_idx].copy()

            # Fill rest with offspring
            offspring_idx = self.elitism

            while offspring_idx < self.pop_size:
                # Selection
                parents = tournament_selection(population, fitness, 2, self.tournament_size, rng)

                # Crossover
                child1, child2 = uniform_crossover(parents[0], parents[1], self.crossover_rate, rng)

                # Mutation
                child1 = mutate(child1, self.mutation_rate, self.sec_counts, rng)
                child2 = mutate(child2, self.mutation_rate, self.sec_counts, rng)

                # Add to next population
                next_population[offspring_idx] = child1
                offspring_idx += 1
                if offspring_idx < self.pop_size:
                    next_population[offspring_idx] = child2
                    offspring_idx += 1

            population = next_population

        elapsed = time.time() - start_time

        # Decode best individual
        if best_individual is not None:
            best_config = decode_individual(best_individual, self.sec_counts)
        else:
            best_config = {
                "spacing": 0.7,
                "sec_w": 4,
                "sec_h": 4,
                "add_support": True,
                "sections": self.max_indices,
            }

        return GAResult(
            structure_id=geometry.get("structure_id", "unknown"),
            geometry={"L": L, "W": W, "H": H},
            main_beam_spacing=best_config["spacing"],
            num_secondary_width=best_config["sec_w"],
            num_secondary_height=best_config["sec_h"],
            add_support_beam=best_config["add_support"],
            optimal_sections=best_config["sections"],
            initial_mass=initial_mass,
            optimal_mass=best_result[0],
            mass_reduction_pct=(initial_mass - best_result[0]) / initial_mass * 100,
            u_sigma=best_result[1],
            u_delta=best_result[2],
            is_feasible=best_result[4],
            n_evaluations=total_evaluations,
            n_generations=self.n_generations,
            time_seconds=elapsed,
        )


# ============================================================================
# Comparison with other methods
# ============================================================================


def compare_methods(geometry: dict, n_jobs: int = 4, eval_mode: str = "fea", surrogate_path: str = None):
    """Compare GA with different configurations."""

    print("=" * 70)
    print("OPTIMIZATION METHOD COMPARISON")
    print("=" * 70)
    print(f"Geometry: L={geometry['L']}, W={geometry['W']}, H={geometry['H']}")
    print(f"Workers: {n_jobs}, Eval mode: {eval_mode}")
    print()

    results = []

    # Method 1: Parallel GA (pop=32, gen=5 => 160 evals)
    print("Method 1: Parallel GA (pop=32, gen=5)")
    print("-" * 50)
    opt1 = ParallelGAOptimizer(
        pop_size=32,
        n_generations=5,
        n_jobs=n_jobs,
        seed=42,
        eval_mode=eval_mode,
        surrogate_path=surrogate_path,
    )
    result1 = opt1.optimize(geometry)
    results.append(("GA (32x5)", result1))
    print(f"  Mass: {result1.optimal_mass:.0f}kg ({result1.mass_reduction_pct:.1f}% reduction)")
    print(f"  Feasible: {result1.is_feasible}, Evals: {result1.n_evaluations}")
    print(f"  Time: {result1.time_seconds:.1f}s")
    print()

    # Method 2: Parallel GA (pop=48, gen=4 => 192 evals)
    print("Method 2: Parallel GA (pop=48, gen=4)")
    print("-" * 50)
    opt2 = ParallelGAOptimizer(
        pop_size=48,
        n_generations=4,
        n_jobs=n_jobs,
        seed=42,
        eval_mode=eval_mode,
        surrogate_path=surrogate_path,
    )
    result2 = opt2.optimize(geometry)
    results.append(("GA (48x4)", result2))
    print(f"  Mass: {result2.optimal_mass:.0f}kg ({result2.mass_reduction_pct:.1f}% reduction)")
    print(f"  Feasible: {result2.is_feasible}, Evals: {result2.n_evaluations}")
    print(f"  Time: {result2.time_seconds:.1f}s")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<15} {'Mass(kg)':<10} {'Reduction':<12} {'Evals':<8} {'Time(s)':<10} {'Feasible':<10}")
    print("-" * 70)
    for name, r in results:
        print(
            f"{name:<15} {r.optimal_mass:<10.0f} {r.mass_reduction_pct:<12.1f}% "
            f"{r.n_evaluations:<8} {r.time_seconds:<10.1f} {str(r.is_feasible):<10}"
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
) -> GAResult:
    """Run GA optimization with given parameters."""
    geometry = {"L": L, "W": W, "H": H, "pressure": pressure}

    print("=" * 70)
    print("PARALLEL GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 70)
    print(f"INPUT: L={L}, W={W}, H={H}, Pressure={pressure}")
    print(f"Population: {pop_size}, Generations: {n_generations}")
    print(f"Total evaluations: {pop_size * n_generations}")
    print(f"Workers: {n_jobs}, Eval mode: {eval_mode}")
    print()

    opt = ParallelGAOptimizer(
        pop_size=pop_size,
        n_generations=n_generations,
        n_jobs=n_jobs,
        seed=seed,
        eval_mode=eval_mode,
        surrogate_path=surrogate_path,
    )
    result = opt.optimize(geometry)
    #  Do FEA check
    fea_result = evaluate_design(
        L=L,
        W=W,
        H=H,
        pressure=pressure,
        main_beam_spacing=result.main_beam_spacing,
        num_secondary_width=result.num_secondary_width,
        num_secondary_height=result.num_secondary_height,
        add_support_beam=result.add_support_beam,
        sections=result.optimal_sections,
        mode="fea",
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Optimal mass: {result.optimal_mass:.0f}kg ({result.mass_reduction_pct:.1f}% reduction)")
    print(
        f"Topology: spacing={result.main_beam_spacing}, "
        f"sec_w={result.num_secondary_width}, "
        f"sec_h={result.num_secondary_height}, "
        f"support={result.add_support_beam}"
    )
    print(f"Sections: {result.optimal_sections}")
    print(f"Feasible: {fea_result[3]} (u_sigma={fea_result[1]:.3f}, u_delta={fea_result[2]:.3f})")
    print(f"Evaluations: {result.n_evaluations}, Time: {result.time_seconds:.1f}s")

    return result


def random_structures_opt_test(eval_mode: str = "surrogate", surrogate_path: str = None):
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
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Genetic Algorithm Optimization")
    parser.add_argument(
        "--mode", type=str, default="run", choices=["run", "compare", "random_test"], help="Mode of operation"
    )
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

    args = parser.parse_args()

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
        )
    elif args.mode == "compare":
        geometry = {"L": args.L, "W": args.W, "H": args.H}
        compare_methods(geometry, n_jobs=args.n_jobs, eval_mode=args.eval_mode, surrogate_path=args.surrogate)
    elif args.mode == "random_test":
        random_structures_opt_test(eval_mode=args.eval_mode, surrogate_path=args.surrogate)
