"""Reproduce a single-design FEA versus deployed-surrogate comparison."""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.evaluator import StructuralEvaluator
from src.optimization.design_space import get_layout_adapter
from src.structure.section_grouping.models import SectionGroupingRuleset


def measure(evaluator, design, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = evaluator.evaluate(design)
        times.append(time.perf_counter() - start)
    values = {name: float(getattr(result, name)) for name in ("mass", "u_sigma", "u_delta")}
    values["feasible"] = bool(result.feasible)
    return {"result": values, "seconds": times, "median_s": statistics.median(times)}


def main():
    adapter = get_layout_adapter("engineering")
    rules = SectionGroupingRuleset.load_json("results/section_grouping/engineering_rules.json")
    geometry = {"L": 7.2, "W": 3.6, "H": 3.3}
    topology = adapter.default_topology()
    indices = {"FRAME": 2, "MAIN": 8, "SECONDARY": 7, "BRACE": 14}
    sections = {group: indices[beam_type.name] for group, beam_type in rules.group_beam_types().items()}
    design = adapter.build_design(
        geometry=geometry, pressure=20000, topology=topology,
        section_group_rules=rules, section_indices_by_group=sections,
    )
    model_path = "src/data/engineering_group_model.pkl"
    surrogate = StructuralEvaluator(mode="surrogate", surrogate_path=model_path)
    surrogate.evaluate(design)
    predicted = measure(surrogate, design, 30)
    finite_element = measure(StructuralEvaluator(mode="fea", mesh_size=0.5), design, 3)
    data = {
        "date": "2026-10-07", "geometry_m": geometry, "pressure_kpa": 20,
        "topology": topology, "section_indices": sections, "mesh_size_m": 0.5,
        "model_path": model_path, "surrogate": predicted, "fea": finite_element,
        "evaluation_speed_ratio": finite_element["median_s"] / predicted["median_s"],
        "scope": "Single-design warm evaluation; includes model building and mass calculation; excludes model loading, offline training and full GA search.",
    }
    out = Path(__file__).with_name("benchmark.json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
