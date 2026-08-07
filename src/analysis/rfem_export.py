from __future__ import annotations

import json
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.structure.structural_model import StructuralModel
from src.utils.constants import LOAD_DEAD_BOTTOM_KPA, LOAD_LIVE_BOTTOM_KPA


@dataclass
class RFEMExportOptions:
    model_name: str = "pressurized_cabin_check"
    pressure_pa: float = 30_000.0
    steel_material_name: str = "Q355"
    opening_material_name: str = "Glass"
    section_name_map: dict[str, str] = field(default_factory=lambda: {
        "BOX-150x6": "SHS 150/6.0/C",
        "H-150x100.0": "H HT 145x73",
        "H-125x60.0": "H HT 118x58",
        "C-200x75x9.0": "U 5",
    })


def corner_support_node_ids(model: StructuralModel) -> list[int]:
    bottom = _bottom_nodes(model)
    xs = [p.x for p in bottom.values()]
    ys = [p.y for p in bottom.values()]
    corners = [
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    ]
    return sorted({_nearest_node(bottom, x, y) for x, y in corners})


def build_rfem_data(model: StructuralModel, options: RFEMExportOptions | None = None) -> dict[str, Any]:
    options = options or RFEMExportOptions()

    sections, member_section = _collect_sections(model, options.section_name_map)
    line_map, lines = _collect_lines(model)
    members = _collect_members(model, line_map, member_section)
    surfaces = _collect_surfaces(model, line_map)
    surface_loads = _collect_surface_loads(model, options.pressure_pa)

    return {
        "model_name": options.model_name,
        "materials": [
            {
                "no": 1,
                "name": options.steel_material_name,
                "E_MPa": 206000.0,
                "G_MPa": 79000.0,
                "poisson": 0.304,
                "density_kg_m3": 7850.0,
            },
            {
                "no": 2,
                "name": options.opening_material_name,
                "E_MPa": model.params.opening_panel.elastic_modulus_mpa,
                "G_MPa": 28400.0,
                "poisson": model.params.opening_panel.poisson_ratio,
                "density_kg_m3": model.params.opening_panel.density_kg_m3,
            },
        ],
        "thicknesses": [
            {"no": 1, "name": "steel plate", "material": 1, "thickness_m": model.params.plate_thickness},
            {
                "no": 2,
                "name": "opening panel",
                "material": 2,
                "thickness_m": model.params.opening_panel.thickness,
            },
        ],
        "sections": sections,
        "nodes": [
            {"no": nid + 1, "x": p.x, "y": p.y, "z": p.z}
            for nid, p in sorted(model.points.items())
        ],
        "lines": lines,
        "members": members,
        "surfaces": surfaces,
        "supports": [{"no": 1, "nodes": [nid + 1 for nid in corner_support_node_ids(model)]}],
        "load_cases": [
            {"no": 1, "name": "DL"},
            {"no": 2, "name": "LL"},
            {"no": 3, "name": "PL"},
        ],
        "surface_loads": surface_loads,
    }


def write_rfem_json(
    model: StructuralModel,
    path: str | Path,
    options: RFEMExportOptions | None = None,
) -> Path:
    data = build_rfem_data(model, options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def write_rfem_api_script(
    model: StructuralModel,
    path: str | Path,
    options: RFEMExportOptions | None = None,
) -> Path:
    data = build_rfem_data(model, options)
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)
    script = f'''"""Generated RFEM API object-list script.

Run this file in an environment with RFEM 6 and dlubal.api available. The
generated objects follow the official create_object_list pattern:

    objects = build_object_list()
    rfem_app.create_object_list(objects)

Section and material database names may need mapping to the RFEM installation.
"""

from __future__ import annotations

import json
import os

from dlubal.api import rfem


DATA = json.loads(r"""{payload}""")


def build_object_list():
    objects = []
    inf = float("inf")

    for mat in DATA["materials"]:
        objects.append(rfem.structure_core.Material(no=mat["no"], name=mat["name"]))

    for thk in DATA["thicknesses"]:
        objects.append(
            rfem.structure_core.Thickness(
                no=thk["no"],
                name=thk["name"],
                material=thk["material"],
                uniform_thickness_d=thk["thickness_m"],
            )
        )

    for sec in DATA["sections"]:
        objects.append(
            rfem.structure_core.CrossSection(
                no=sec["no"],
                name=sec["rfem_name"],
                material=1,
            )
        )

    for node in DATA["nodes"]:
        objects.append(
            rfem.structure_core.Node(
                no=node["no"],
                coordinate_1=node["x"],
                coordinate_2=node["y"],
                coordinate_3=node["z"],
            )
        )

    for line in DATA["lines"]:
        objects.append(
            rfem.structure_core.Line(
                no=line["no"],
                definition_nodes=line["nodes"],
            )
        )

    for member in DATA["members"]:
        objects.append(
            rfem.structure_core.Member(
                no=member["no"],
                line=member["line"],
                cross_section_start=member["section"],
            )
        )

    for surface in DATA["surfaces"]:
        objects.append(
            rfem.structure_core.Surface(
                no=surface["no"],
                boundary_lines=surface["boundary_lines"],
                thickness=surface["thickness"],
            )
        )

    for support in DATA["supports"]:
        objects.append(
            rfem.types_for_nodes.NodalSupport(
                no=support["no"],
                nodes=support["nodes"],
                spring_x=inf,
                spring_y=inf,
                spring_z=inf,
                rotational_restraint_x=0,
                rotational_restraint_y=0,
                rotational_restraint_z=0,
            )
        )

    objects.append(
        rfem.loading.StaticAnalysisSettings(
            no=1,
            analysis_type=rfem.loading.StaticAnalysisSettings.ANALYSIS_TYPE_GEOMETRICALLY_LINEAR,
        )
    )
    for case in DATA["load_cases"]:
        objects.append(rfem.loading.LoadCase(no=case["no"], name=case["name"], static_analysis_settings=1))

    direction_enum = rfem.loads.SurfaceLoad.LoadDirection
    directions = {{
        "X": direction_enum.LOAD_DIRECTION_GLOBAL_X_OR_USER_DEFINED_U_TRUE_LENGTH,
        "Y": direction_enum.LOAD_DIRECTION_GLOBAL_Y_OR_USER_DEFINED_V_TRUE_LENGTH,
        "Z": direction_enum.LOAD_DIRECTION_GLOBAL_Z_OR_USER_DEFINED_W_TRUE_LENGTH,
    }}
    for load in DATA["surface_loads"]:
        objects.append(
            rfem.loads.SurfaceLoad(
                no=load["no"],
                load_case=load["load_case"],
                surfaces=load["surfaces"],
                load_direction=directions[load["axis"]],
                magnitude=load["magnitude_n_m2"],
            )
        )

    return objects


def create_in_rfem(rfem_app):
    return rfem_app.create_object_list(build_object_list())


def main():
    api_key = os.getenv("DLUBAL_API_KEY")
    api_key_name = os.getenv("DLUBAL_API_KEY_NAME")
    with rfem.Application(api_key_value=api_key, api_key_name=api_key_name) as rfem_app:
        rfem_app.create_model(name=DATA["model_name"])
        rfem_app.delete_all_objects()
        create_in_rfem(rfem_app)
        print(f"Created RFEM model: {{DATA['model_name']}}")


if __name__ == "__main__":
    main()
'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    return path


def _bottom_nodes(model: StructuralModel):
    min_z = min(p.z for p in model.points.values())
    return {nid: p for nid, p in model.points.items() if abs(p.z - min_z) < 1e-6}


def _json_default(value):
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _nearest_node(nodes, x: float, y: float) -> int:
    return min(nodes, key=lambda nid: (nodes[nid].x - x) ** 2 + (nodes[nid].y - y) ** 2)


def _collect_sections(model: StructuralModel, section_name_map: dict[str, str]):
    section_nos: dict[str, int] = {}
    sections = []
    member_section = {}

    for bid, beam in sorted(model.beams.items()):
        if beam.is_virtual:
            continue
        name = beam.section.name
        if name not in section_nos:
            no = len(section_nos) + 1
            section_nos[name] = no
            sections.append(
                {
                    "no": no,
                    "source_name": name,
                    "rfem_name": section_name_map.get(name, name),
                    "area_m2": beam.section.area,
                    "Ix_m4": beam.section.Ix,
                    "Iy_m4": beam.section.Iy,
                    "J_m4": beam.section.Iz,
                }
            )
        member_section[bid] = section_nos[name]

    return sections, member_section


def _collect_lines(model: StructuralModel):
    line_map: dict[tuple[int, int], int] = {}
    lines = []

    def add_line(n1: int, n2: int) -> int:
        key = tuple(sorted((n1, n2)))
        if key not in line_map:
            no = len(line_map) + 1
            line_map[key] = no
            lines.append({"no": no, "nodes": [n1 + 1, n2 + 1]})
        return line_map[key]

    for beam in model.beams.values():
        if not beam.is_virtual:
            add_line(beam.start_node, beam.end_node)

    for plate in model.plates.values():
        nodes = plate.nodes
        for i in range(len(nodes)):
            add_line(nodes[i], nodes[(i + 1) % len(nodes)])

    return line_map, lines


def _collect_members(model: StructuralModel, line_map: dict[tuple[int, int], int], member_section: dict[int, int]):
    members = []
    for bid, beam in sorted(model.beams.items()):
        if beam.is_virtual:
            continue
        key = tuple(sorted((beam.start_node, beam.end_node)))
        members.append(
            {
                "no": bid + 1,
                "line": line_map[key],
                "section": member_section[bid],
                "beam_type": beam.beam_type.name,
                "source_id": bid,
            }
        )
    return members


def _collect_surfaces(model: StructuralModel, line_map: dict[tuple[int, int], int]):
    surfaces = []
    for pid, plate in sorted(model.plates.items()):
        nodes = plate.nodes
        boundary = []
        for i in range(len(nodes)):
            key = tuple(sorted((nodes[i], nodes[(i + 1) % len(nodes)])))
            boundary.append(line_map[key])
        surfaces.append(
            {
                "no": pid,
                "boundary_lines": boundary,
                "thickness": 2 if plate.panel_type == "opening" else 1,
                "face": plate.on_face,
                "panel_type": plate.panel_type,
                "source_id": pid,
            }
        )
    return surfaces


def _collect_surface_loads(model: StructuralModel, pressure_pa: float):
    loads = []
    load_no = 1

    def add(load_case: int, surfaces: list[int], axis: str, value: float):
        nonlocal load_no
        if not surfaces or abs(value) < 1e-12:
            return
        loads.append(
            {
                "no": load_no,
                "load_case": load_case,
                "surfaces": surfaces,
                "axis": axis,
                "magnitude_n_m2": value,
            }
        )
        load_no += 1

    bottom = [pid for pid, p in model.plates.items() if p.on_face == "bottom"]
    add(1, bottom, "Z", -LOAD_DEAD_BOTTOM_KPA * 1000.0)
    add(2, bottom, "Z", -LOAD_LIVE_BOTTOM_KPA * 1000.0)

    pressure_dirs = {
        "top": ("Z", pressure_pa),
        "bottom": ("Z", -pressure_pa),
        "left": ("Y", -pressure_pa),
        "right": ("Y", pressure_pa),
        "front": ("X", pressure_pa),
        "back": ("X", -pressure_pa),
    }
    for face, (axis, value) in pressure_dirs.items():
        surfaces = [pid for pid, p in model.plates.items() if p.on_face == face]
        add(3, surfaces, axis, value)

    return loads
