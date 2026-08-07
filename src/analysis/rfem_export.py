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
        "BOX-150x6": "SHS 150x6",
        "H-150x100.0": "I 150/100/6/10/8/0/H",
        "H-125x60.0": "I 125/60/6/8/8/0/H",
        "C-200x75x9.0": "UTF 200/75/9/11/11/5.5/33/0.100/H",
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
    api_key_name = os.getenv("DLUBAL_API_KEY_NAME", "Leo")
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


def write_rfem_script_manager_js(
    model: StructuralModel,
    path: str | Path,
    options: RFEMExportOptions | None = None,
) -> Path:
    data = build_rfem_data(model, options)
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)
    script = f"""// Generated RFEM Script Manager model creation script.
// Open RFEM 6, create/open an empty model, then paste/run this script in Script Manager.
// It does not use the WebService/gRPC API and therefore does not need an API subscription.

var DATA = {payload};

function clearModel() {{
    var containers = [];
    try {{ if (typeof surface_sets !== "undefined") containers.push(surface_sets); }} catch (err) {{}}
    try {{ if (typeof solid_sets !== "undefined") containers.push(solid_sets); }} catch (err) {{}}
    try {{ if (typeof members !== "undefined") containers.push(members); }} catch (err) {{}}
    try {{ if (typeof surfaces !== "undefined") containers.push(surfaces); }} catch (err) {{}}
    try {{ if (typeof lines !== "undefined") containers.push(lines); }} catch (err) {{}}
    try {{ if (typeof nodes !== "undefined") containers.push(nodes); }} catch (err) {{}}
    try {{ if (typeof nodal_supports !== "undefined") containers.push(nodal_supports); }} catch (err) {{}}
    try {{ if (typeof thicknesses !== "undefined") containers.push(thicknesses); }} catch (err) {{}}
    try {{ if (typeof sections !== "undefined") containers.push(sections); }} catch (err) {{}}
    try {{ if (typeof materials !== "undefined") containers.push(materials); }} catch (err) {{}}
    try {{ if (typeof load_cases !== "undefined") containers.push(load_cases); }} catch (err) {{}}
    try {{ if (typeof actions !== "undefined") containers.push(actions); }} catch (err) {{}}
    try {{ if (typeof load_combinations !== "undefined") containers.push(load_combinations); }} catch (err) {{}}
    try {{ if (typeof result_combinations !== "undefined") containers.push(result_combinations); }} catch (err) {{}}
    try {{ if (typeof static_analysis_settings !== "undefined") containers.push(static_analysis_settings); }} catch (err) {{}}

    for (var i = 0; i < containers.length; ++i) {{
        try {{
            var c = containers[i];
            for (var j = c.count(); j > 0; --j) {{
                c.erase(c.getNthObjectId(j));
            }}
        }} catch (err) {{
        }}
    }}
}}

function createMaterials() {{
    for (var i = 0; i < DATA.materials.length; ++i) {{
        var src = DATA.materials[i];
        var mat = materials.create(src.no);
        mat.name = src.name;

        // Numerical properties are assigned to match the OpenSees comparison model.
        try {{
            mat.user_defined = true;
            mat.elasticity_modulus_global = src.E_MPa * 1000000.0;
            mat.shear_modulus_global = src.G_MPa * 1000000.0;
            mat.poisson_ratio_global = src.poisson;
            mat.mass_density = src.density_kg_m3;
            mat.specific_weight = src.density_kg_m3 * 9.81;
        }} catch (err) {{
            console.log("Material property assignment skipped for material " + src.no + ": " + err);
        }}
    }}
}}

function createThicknesses() {{
    for (var i = 0; i < DATA.thicknesses.length; ++i) {{
        var src = DATA.thicknesses[i];
        var thk = thicknesses.create(src.no);
        thk.name = src.name;
        thk.type = thicknesses.TYPE_UNIFORM;
        thk.material = materials[src.material];
        thk.uniform_thickness = src.thickness_m;
    }}
}}

function createSections() {{
    for (var i = 0; i < DATA.sections.length; ++i) {{
        var src = DATA.sections[i];
        var sec = engine.create_section(src.no, materials[1]);
        sec.name = src.rfem_name;
    }}
}}

function createNodes() {{
    for (var i = 0; i < DATA.nodes.length; ++i) {{
        var src = DATA.nodes[i];
        var node = engine.create_node(src.no);
        node.coordinate_1 = src.x;
        node.coordinate_2 = src.y;
        node.coordinate_3 = src.z;
    }}
}}

function createLines() {{
    for (var i = 0; i < DATA.lines.length; ++i) {{
        var src = DATA.lines[i];
        var line = engine.create_line(src.no, src.nodes);
        line.type = lines.TYPE_POLYLINE;
    }}
}}

function createMemberRotationHelpers() {{
    for (var i = 0; i < DATA.members.length; ++i) {{
        var helper = DATA.members[i].rotation_help_node;
        if (helper === null) {{
            continue;
        }}
        var node = engine.create_node(helper.no);
        node.coordinate_1 = helper.x;
        node.coordinate_2 = helper.y;
        node.coordinate_3 = helper.z;
        node.comment = "member rotation helper";
    }}
}}

function createMembers() {{
    for (var i = 0; i < DATA.members.length; ++i) {{
        var src = DATA.members[i];
        var member = engine.create_member(src.no, src.line);
        member.type = members.TYPE_BEAM;
        member.section_start = sections[src.section];

        if (src.rotation_help_node !== null) {{
            var line = lines[src.line];
            line.rotation_specification_type = lines.COORDINATE_SYSTEM_ROTATION_VIA_HELP_NODE;
            line.rotation_help_node = nodes[src.rotation_help_node.no];
            line.rotation_plane = lines.ROTATION_PLANE_XY;
        }}
    }}
}}

function createSurfaces() {{
    for (var i = 0; i < DATA.surfaces.length; ++i) {{
        var src = DATA.surfaces[i];
        var surface = engine.create_surface(src.no, src.boundary_lines);
        surface.type = surfaces.TYPE_STANDARD;
        surface.thickness = thicknesses[src.thickness];
    }}
}}

function createSupports() {{
    var INF = "inf";
    for (var i = 0; i < DATA.supports.length; ++i) {{
        var src = DATA.supports[i];
        var support = nodal_supports.create(src.no);
        support.nodes = src.nodes;
        support.spring_x = INF;
        support.spring_y = INF;
        support.spring_z = INF;
        support.rotational_restraint_x = 0;
        support.rotational_restraint_y = 0;
        support.rotational_restraint_z = 0;
    }}
}}

function createLoadCases() {{
    var sas = static_analysis_settings.create(1);
    sas.analysis_type = static_analysis_settings.GEOMETRICALLY_LINEAR;

    for (var i = 0; i < DATA.load_cases.length; ++i) {{
        var src = DATA.load_cases[i];
        var lc = load_cases.create(src.no);
        lc.name = src.name;
        lc.analysis_type = load_cases.ANALYSIS_TYPE_STATIC;
        lc.static_analysis_settings = static_analysis_settings[1];
        lc.self_weight_active = false;
    }}
}}

function surfaceLoadDirection(axis) {{
    if (axis === "X") {{
        return surface_loads.LOAD_DIRECTION_GLOBAL_X_OR_USER_DEFINED_U_TRUE;
    }}
    if (axis === "Y") {{
        return surface_loads.LOAD_DIRECTION_GLOBAL_Y_OR_USER_DEFINED_V_TRUE;
    }}
    return surface_loads.LOAD_DIRECTION_GLOBAL_Z_OR_USER_DEFINED_W_TRUE;
}}

function createSurfaceLoads() {{
    for (var i = 0; i < DATA.surface_loads.length; ++i) {{
        var src = DATA.surface_loads[i];
        var load = engine.create_load(src.no, "Surface_Load", load_cases[src.load_case]);
        load.surfaces = src.surfaces;
        load.load_type = surface_loads.LOAD_TYPE_FORCE;
        load.load_distribution = surface_loads.LOAD_DISTRIBUTION_UNIFORM;
        load.load_direction = surfaceLoadDirection(src.axis);
        load.uniform_magnitude = src.magnitude_n_m2;
    }}
}}

clearModel();
createMaterials();
createThicknesses();
createSections();
createNodes();
createLines();
createMemberRotationHelpers();
createMembers();
createSurfaces();
createSupports();
createLoadCases();
createSurfaceLoads();

applyChanges();
console.log("Created RFEM Script Manager model: " + DATA.model_name);
console.log("Nodes: " + DATA.nodes.length + ", members: " + DATA.members.length + ", surfaces: " + DATA.surfaces.length);
"""
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
                "rotation_help_node": _member_rotation_help_node(model, beam),
            }
        )
    return members


def _member_rotation_help_node(model: StructuralModel, beam) -> dict[str, float | int] | None:
    normal = _face_normal(beam.on_face)
    if normal is None:
        return None

    p1 = model.points[beam.start_node]
    p2 = model.points[beam.end_node]
    axis = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
    length = _norm(axis)
    if length < 1e-9:
        return None

    axis = tuple(v / length for v in axis)
    dot = sum(normal[i] * axis[i] for i in range(3))
    vec = tuple(normal[i] - dot * axis[i] for i in range(3))
    vec_len = _norm(vec)
    if vec_len < 1e-9:
        return None

    vec = tuple(v / vec_len for v in vec)
    mid = ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)
    offset = 0.25
    return {
        "no": 900000 + beam.id + 1,
        "x": mid[0] + offset * vec[0],
        "y": mid[1] + offset * vec[1],
        "z": mid[2] + offset * vec[2],
    }


def _face_normal(face: str) -> tuple[float, float, float] | None:
    normals = {
        "top": (0.0, 0.0, 1.0),
        "bottom": (0.0, 0.0, -1.0),
        "left": (0.0, -1.0, 0.0),
        "right": (0.0, 1.0, 0.0),
        "front": (1.0, 0.0, 0.0),
        "back": (-1.0, 0.0, 0.0),
    }
    for part in str(face).split("/"):
        if part in normals:
            return normals[part]
    return None


def _norm(vec: tuple[float, float, float]) -> float:
    return sum(v * v for v in vec) ** 0.5


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
