"""
Parallel Genetic Algorithm for Topology + Section Optimization.

GA is naturally parallelizable - the entire population can be evaluated in parallel.
This implementation uses joblib for process-based parallelism.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from src.heuristic.evaluator import StructuralEvaluator, evaluate_design
from src.optimization.design_space import VariableSpec, get_layout_adapter, get_section_lib
from src.optimization.section_grouping import SectionGroupingRuleset, section_counts_for_groups
from src.structure.structural_design import StructuralDesign


@dataclass
class GAResult:
    """Result of GA optimization."""

    structure_id: str
    layout_strategy: str
    geometry: dict
    optimal_topology: dict
    optimal_section_indices_by_group: dict[str, int]
    initial_mass: float
    optimal_mass: float
    mass_reduction_pct: float
    u_sigma: float
    u_delta: float
    is_feasible: bool
    n_evaluations: int
    n_generations: int
    time_seconds: float


@dataclass(frozen=True)
class Candidate:
    design: StructuralDesign
    topology: dict
    section_indices_by_group: dict[str, int]


@dataclass
class SectionScheme:
    """Maps section genes to beam-group section assignments."""

    rules: SectionGroupingRuleset
    group_ids: list[str]
    group_sec_counts: list[int]

    @classmethod
    def by_beam_type(cls, layout_strategy: str, section_lib) -> "SectionScheme":
        rules = SectionGroupingRuleset.by_beam_type(layout_strategy)
        return cls(
            rules=rules,
            group_ids=rules.group_ids(),
            group_sec_counts=section_counts_for_groups(rules, section_lib),
        )

    @classmethod
    def grouped(
        cls,
        rules: SectionGroupingRuleset,
        section_lib,
    ) -> "SectionScheme":
        return cls(
            rules=rules,
            group_ids=rules.group_ids(),
            group_sec_counts=section_counts_for_groups(rules, section_lib),
        )

    @property
    def n_genes(self) -> int:
        return len(self.group_ids)

    @property
    def gene_counts(self) -> list[int]:
        return self.group_sec_counts

    def max_gene_values(self) -> np.ndarray:
        return np.array([count - 1 for count in self.gene_counts], dtype=np.float32)

    def decode(self, genes: np.ndarray) -> dict[str, int]:
        return {
            group_id: int(np.clip(round(genes[offset]), 0, self.group_sec_counts[offset] - 1))
            for offset, group_id in enumerate(self.group_ids)
        }


def _decode_topology_gene(gene_value: float, spec: VariableSpec):
    if spec.kind in {"categorical", "bool"}:
        choices = list(spec.choices or [])
        idx = int(gene_value) % len(choices)
        return choices[idx]
    if spec.kind == "int":
        return int(np.clip(round(gene_value), int(spec.low), int(spec.high)))
    if spec.kind == "float":
        value = float(np.clip(gene_value, float(spec.low), float(spec.high)))
        if spec.step:
            steps = round((value - float(spec.low)) / float(spec.step))
            value = float(spec.low) + steps * float(spec.step)
        return value
    raise ValueError(f"Unsupported variable spec kind: {spec.kind}")


def decode_candidate(
    genes: np.ndarray,
    geometry: dict,
    topology_specs: list[VariableSpec],
    layout_strategy: str,
    section_scheme: SectionScheme,
) -> Candidate:
    """Decode genes to a structural design candidate."""
    topology = {}
    for idx, spec in enumerate(topology_specs):
        topology[spec.name] = _decode_topology_gene(float(genes[idx]), spec)

    section_start = len(topology_specs)
    section_indices_by_group = section_scheme.decode(genes[section_start:])
    adapter = get_layout_adapter(layout_strategy)
    design = adapter.build_design(
        geometry={"L": geometry["L"], "W": geometry["W"], "H": geometry["H"]},
        pressure=geometry.get("pressure", 30000.0),
        topology=topology,
        section_group_rules=section_scheme.rules,
        section_indices_by_group=section_indices_by_group,
    )
    return Candidate(
        design=design,
        topology=topology,
        section_indices_by_group=section_indices_by_group,
    )


def _fitness_from_eval(
    mass: float,
    u_sigma: float,
    u_delta: float,
    feasible: bool,
    eval_mode: str,
) -> tuple[float, float, float, float, bool]:
    if eval_mode == "surrogate":
        violation = max(0, u_sigma - 0.9) + max(0, u_delta - 0.9)
    else:
        violation = max(0, u_sigma - 1.0) + max(0, u_delta - 1.0)

    fitness = mass * (1 + 10 * violation) if violation > 0 else mass
    return mass, u_sigma, u_delta, fitness, feasible


def evaluate_individual(
    genes: np.ndarray,
    geometry: dict,
    topology_specs: list[VariableSpec],
    layout_strategy: str,
    section_scheme: SectionScheme,
    eval_mode: str = "fea",
    surrogate_path: Optional[str] = None,
) -> tuple[float, float, float, float, bool]:
    """
    Evaluate fitness of an individual.

    Returns:
        (mass, u_sigma, u_delta, fitness, feasible)
    """
    candidate = decode_candidate(
        genes,
        geometry,
        topology_specs,
        layout_strategy,
        section_scheme,
    )

    mass, u_sigma, u_delta, feasible = evaluate_design(
        design=candidate.design,
        mode=eval_mode,
        surrogate_path=surrogate_path,
    )

    return _fitness_from_eval(mass, u_sigma, u_delta, feasible, eval_mode)


def create_random_population(
    pop_size: int,
    topology_specs: list[VariableSpec],
    section_gene_counts: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create random initial population."""
    n_section_genes = len(section_gene_counts)
    n_genes = len(topology_specs) + n_section_genes
    population = np.zeros((pop_size, n_genes), dtype=np.float32)

    for i in range(pop_size):
        for gene_idx, spec in enumerate(topology_specs):
            if spec.kind in {"categorical", "bool"}:
                population[i, gene_idx] = rng.integers(0, len(spec.choices or []))
            elif spec.kind == "int":
                population[i, gene_idx] = rng.integers(int(spec.low), int(spec.high) + 1)
            elif spec.kind == "float":
                if spec.step:
                    values = np.arange(
                        float(spec.low), float(spec.high) + 0.5 * float(spec.step), float(spec.step)
                    )
                    population[i, gene_idx] = float(values[rng.integers(0, len(values))])
                else:
                    population[i, gene_idx] = rng.uniform(float(spec.low), float(spec.high))

        section_start = len(topology_specs)
        for offset in range(n_section_genes):
            population[i, section_start + offset] = rng.integers(0, section_gene_counts[offset])

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
        candidates = rng.choice(pop_size, size=tournament_size, replace=False)
        winner = candidates[np.argmin(fitness[candidates])]
        selected_indices.append(winner)
    return population[selected_indices].copy()


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover - each gene has 50% chance from either parent."""
    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    mask = rng.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def mutate(
    individual: np.ndarray,
    mutation_rate: float,
    topology_specs: list[VariableSpec],
    section_gene_counts: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutate individual genes."""
    mutated = individual.copy()

    for gene_idx, spec in enumerate(topology_specs):
        if rng.random() >= mutation_rate:
            continue
        if spec.kind in {"categorical", "bool"}:
            mutated[gene_idx] = rng.integers(0, len(spec.choices or []))
        elif spec.kind == "int":
            delta = rng.integers(-1, 2)
            mutated[gene_idx] = np.clip(mutated[gene_idx] + delta, int(spec.low), int(spec.high))
        elif spec.kind == "float":
            if spec.step:
                step = float(spec.step)
                delta = rng.integers(-2, 3) * step
                mutated[gene_idx] = np.clip(mutated[gene_idx] + delta, float(spec.low), float(spec.high))
            else:
                span = float(spec.high) - float(spec.low)
                mutated[gene_idx] = np.clip(
                    mutated[gene_idx] + rng.normal(0.0, span * 0.1),
                    float(spec.low),
                    float(spec.high),
                )

    section_start = len(topology_specs)
    for offset, count in enumerate(section_gene_counts):
        gene_idx = section_start + offset
        if rng.random() < mutation_rate:
            delta = rng.integers(-3, 4)
            mutated[gene_idx] = np.clip(mutated[gene_idx] + delta, 0, count - 1)

    return mutated


class ParallelGAOptimizer:
    """
    Parallel Genetic Algorithm for topology + section optimization.
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
        layout_strategy: str = "research",
        section_group_rules: SectionGroupingRuleset | str | None = None,
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
        self.layout_strategy = layout_strategy
        self.adapter = get_layout_adapter(layout_strategy)
        self.topology_specs = self.adapter.topology_specs()
        self.section_group_rules = (
            SectionGroupingRuleset.load_json(section_group_rules)
            if isinstance(section_group_rules, str)
            else section_group_rules
        )

        section_lib = get_section_lib(layout_strategy)
        self.section_scheme = (
            SectionScheme.by_beam_type(layout_strategy, section_lib)
            if self.section_group_rules is None
            else SectionScheme.grouped(self.section_group_rules, section_lib)
        )
        self.surrogate_evaluator = (
            StructuralEvaluator(mode="surrogate", surrogate_path=surrogate_path)
            if eval_mode == "surrogate"
            else None
        )

    def _encode_default_individual(self) -> np.ndarray:
        genes = np.zeros(len(self.topology_specs) + self.section_scheme.n_genes, dtype=np.float32)
        defaults = self.adapter.default_topology()
        for idx, spec in enumerate(self.topology_specs):
            value = defaults[spec.name]
            if spec.kind in {"categorical", "bool"}:
                choices = list(spec.choices or [])
                if value in choices:
                    genes[idx] = choices.index(value)
                elif isinstance(value, (int, float)) and choices:
                    nearest = min(
                        range(len(choices)),
                        key=lambda choice_idx: abs(float(choices[choice_idx]) - float(value)),
                    )
                    genes[idx] = nearest
                else:
                    genes[idx] = 0
            else:
                genes[idx] = float(value)
        section_start = len(self.topology_specs)
        genes[section_start:] = self.section_scheme.max_gene_values()
        return genes

    def _evaluate_population_surrogate(
        self,
        population: np.ndarray,
        geometry: dict,
    ) -> list[tuple[float, float, float, float, bool]]:
        candidates = [
            decode_candidate(
                ind,
                geometry,
                self.topology_specs,
                self.layout_strategy,
                self.section_scheme,
            )
            for ind in population
        ]
        eval_results = self.surrogate_evaluator.evaluate_batch(
            [candidate.design for candidate in candidates],
        )
        evals = [(result.mass, result.u_sigma, result.u_delta, result.feasible) for result in eval_results]
        return [
            _fitness_from_eval(mass, u_sigma, u_delta, feasible, self.eval_mode)
            for mass, u_sigma, u_delta, feasible in evals
        ]

    def optimize(self, geometry: dict, verbose: bool = True) -> GAResult:
        """Run parallel GA optimization."""
        start_time = time.time()
        rng = np.random.default_rng(self.seed)

        max_genes = self._encode_default_individual()
        if self.eval_mode == "surrogate":
            initial_result = self._evaluate_population_surrogate(np.array([max_genes]), geometry)[0]
        else:
            initial_result = evaluate_individual(
                max_genes,
                geometry,
                self.topology_specs,
                self.layout_strategy,
                self.section_scheme,
                self.eval_mode,
                self.surrogate_path,
            )
        initial_mass = initial_result[0]
        if verbose:
            print(f"Initial mass (max sections): {initial_mass:.0f}kg")

        population = create_random_population(
            self.pop_size,
            self.topology_specs,
            self.section_scheme.gene_counts,
            rng,
        )
        population[0] = max_genes

        best_individual = None
        best_result = (1e9, 10.0, 10.0, 1e9, False)
        total_evaluations = 0

        for gen in range(self.n_generations):
            if self.eval_mode == "surrogate":
                results = self._evaluate_population_surrogate(population, geometry)
            else:
                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(evaluate_individual)(
                        ind,
                        geometry,
                        self.topology_specs,
                        self.layout_strategy,
                        self.section_scheme,
                        self.eval_mode,
                        self.surrogate_path,
                    )
                    for ind in population
                )
            total_evaluations += len(population)

            fitness = np.array([result[3] for result in results])
            feasible_mask = np.array([result[4] for result in results])

            for idx, result in enumerate(results):
                if result[4]:
                    if not best_result[4] or result[3] < best_result[3]:
                        best_individual = population[idx].copy()
                        best_result = result
                elif not best_result[4] and result[3] < best_result[3]:
                    best_individual = population[idx].copy()
                    best_result = result

            if verbose:
                print(
                    f"  Gen {gen + 1:2d}/{self.n_generations}: "
                    f"best mass={best_result[0]:.0f}kg, "
                    f"feasible={np.sum(feasible_mask)}/{self.pop_size}, "
                    f"mean_fit={np.mean(fitness):.0f}"
                )

            if gen == self.n_generations - 1:
                break

            next_population = np.zeros_like(population)
            elite_indices = np.argsort(fitness)[: self.elitism]
            for idx, elite_idx in enumerate(elite_indices):
                next_population[idx] = population[elite_idx].copy()

            offspring_idx = self.elitism
            while offspring_idx < self.pop_size:
                parents = tournament_selection(population, fitness, 2, self.tournament_size, rng)
                child1, child2 = uniform_crossover(parents[0], parents[1], self.crossover_rate, rng)
                child1 = mutate(
                    child1,
                    self.mutation_rate,
                    self.topology_specs,
                    self.section_scheme.gene_counts,
                    rng,
                )
                child2 = mutate(
                    child2,
                    self.mutation_rate,
                    self.topology_specs,
                    self.section_scheme.gene_counts,
                    rng,
                )
                next_population[offspring_idx] = child1
                offspring_idx += 1
                if offspring_idx < self.pop_size:
                    next_population[offspring_idx] = child2
                    offspring_idx += 1

            population = next_population

        elapsed = time.time() - start_time

        if best_individual is None:
            print("WARNING: No feasible solution found, returning best infeasible candidate.")
            best_candidate = decode_candidate(
                max_genes,
                geometry,
                self.topology_specs,
                self.layout_strategy,
                self.section_scheme,
            )
        else:
            best_candidate = decode_candidate(
                best_individual,
                geometry,
                self.topology_specs,
                self.layout_strategy,
                self.section_scheme,
            )

        return GAResult(
            structure_id=geometry.get("structure_id", "unknown"),
            layout_strategy=self.layout_strategy,
            geometry={"L": geometry["L"], "W": geometry["W"], "H": geometry["H"]},
            optimal_topology=best_candidate.topology,
            optimal_section_indices_by_group=best_candidate.section_indices_by_group,
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


def compare_methods(
    geometry: dict,
    n_jobs: int = 4,
    eval_mode: str = "fea",
    surrogate_path: str = None,
    layout_strategy: str = "research",
    section_group_rules: SectionGroupingRuleset | str | None = None,
):
    """Compare GA with different configurations."""
    print("=" * 70)
    print("OPTIMIZATION METHOD COMPARISON")
    print("=" * 70)
    print(f"Geometry: L={geometry['L']}, W={geometry['W']}, H={geometry['H']}")
    print(f"Workers: {n_jobs}, Eval mode: {eval_mode}, Layout: {layout_strategy}")
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
    section_group_rules: SectionGroupingRuleset | str | None = None,
) -> GAResult:
    """Run GA optimization with given parameters."""
    geometry = {"L": L, "W": W, "H": H, "pressure": pressure}

    print("=" * 70)
    print("PARALLEL GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 70)
    print(f"INPUT: L={L}, W={W}, H={H}, Pressure={pressure}")
    print(f"Population: {pop_size}, Generations: {n_generations}")
    print(f"Total evaluations: {pop_size * n_generations}")
    print(f"Workers: {n_jobs}, Eval mode: {eval_mode}, Layout: {layout_strategy}")
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
    section_group_rules: SectionGroupingRuleset | str | None = None,
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
            section_group_rules=section_group_rules,
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
        "--section-group-rules", type=str, default=None, help="Path to section grouping ruleset JSON"
    )

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
            layout_strategy=args.layout_strategy,
            section_group_rules=args.section_group_rules,
        )
    elif args.mode == "compare":
        geometry = {"L": args.L, "W": args.W, "H": args.H, "pressure": args.pressure}
        compare_methods(
            geometry,
            n_jobs=args.n_jobs,
            eval_mode=args.eval_mode,
            surrogate_path=args.surrogate,
            layout_strategy=args.layout_strategy,
            section_group_rules=args.section_group_rules,
        )
    elif args.mode == "random_test":
        random_structures_opt_test(
            eval_mode=args.eval_mode,
            surrogate_path=args.surrogate,
            layout_strategy=args.layout_strategy,
            section_group_rules=args.section_group_rules,
        )
