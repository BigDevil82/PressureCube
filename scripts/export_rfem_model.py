from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.rfem_export import (
    RFEMExportOptions,
    write_rfem_api_script,
    write_rfem_json,
    write_rfem_script_manager_js,
)
from src.structure.enums import BeamType
from src.structure.layout.params import EngineeringLayoutParameters, WindowParameters
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_assignment import SectionAssignment
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel


def build_default_engineering_model(mesh_size: float):
    params = EngineeringLayoutParameters(
        length=8.9,
        width=3.7,
        height=3.3,
        main_beam_spacing=0.95,
        add_support_beam=True,
        support_beam_max_spacing=0.75,
        window=WindowParameters(front_spans=[], back_spans=[], right_spans=[1, 2]),
    )
    assignment = SectionAssignment.by_beam_type(
        {BeamType.FRAME: 2, BeamType.MAIN: 8, BeamType.SECONDARY: 7, BeamType.BRACE: 14}
    )
    model = StructuralModel(params, SectionLibrary("src/data/steel_sections.json"), section_assignment=assignment)
    model.generate()
    return MeshGenerator(model, mesh_size).generate()


def main():
    parser = argparse.ArgumentParser(description="Export the engineering model for RFEM API comparison.")
    parser.add_argument("--mesh-size", type=float, default=0.5)
    parser.add_argument("--pressure-pa", type=float, default=30_000.0)
    parser.add_argument("--json", default="results/rfem/engineering_model.json")
    parser.add_argument("--script", default="results/rfem/create_engineering_model.py")
    parser.add_argument("--script-manager", default="results/rfem/create_engineering_model_script_manager.js")
    args = parser.parse_args()

    model = build_default_engineering_model(args.mesh_size)
    options = RFEMExportOptions(pressure_pa=args.pressure_pa)
    json_path = write_rfem_json(model, args.json, options)
    script_path = write_rfem_api_script(model, args.script, options)
    script_manager_path = write_rfem_script_manager_js(model, args.script_manager, options)

    print(f"RFEM JSON exported: {json_path}")
    print(f"RFEM API script exported: {script_path}")
    print(f"RFEM Script Manager JS exported: {script_manager_path}")
    print(f"Nodes: {len(model.points)}, members: {len(model.beams)}, surfaces: {len(model.plates)}")


if __name__ == "__main__":
    main()
