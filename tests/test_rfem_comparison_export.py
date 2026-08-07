from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.rfem_export import build_rfem_data, write_rfem_script_manager_js
from src.analysis.unit import E_STEEL_MPA, G_STEEL_MPA, NU_STEEL
from src.structure.enums import BeamType
from src.structure.layout.params import EngineeringLayoutParameters, ExtensionParameters, WindowParameters
from src.structure.section_assignment import SectionAssignment
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel


def _engineering_model(length=12.4):
    params = EngineeringLayoutParameters(
        length=length,
        width=3.7,
        height=3.3,
        main_beam_spacing=0.95,
        add_support_beam=True,
        support_beam_max_spacing=0.75,
        extension=ExtensionParameters(active=False),
        window=WindowParameters(active=False, front_spans=[], back_spans=[], right_spans=[]),
    )
    assignment = SectionAssignment.by_beam_type(
        {BeamType.FRAME: 2, BeamType.MAIN: 8, BeamType.SECONDARY: 7, BeamType.BRACE: 14}
    )
    model = StructuralModel(params, SectionLibrary("src/data/steel_sections.json"), section_assignment=assignment)
    model.generate()
    return model


def test_rfem_steel_material_constants_are_used():
    assert E_STEEL_MPA == 206000.0
    assert G_STEEL_MPA == 79000.0
    assert NU_STEEL == 0.304


def test_opensees_supports_are_four_corners_even_for_long_models():
    model = _engineering_model(length=12.4)
    support_ids = OpenSeesSolver(model).support_node_ids
    coords = {(model.points[nid].x, model.points[nid].y, model.points[nid].z) for nid in support_ids}

    assert len(support_ids) == 4
    assert coords == {
        (0.0, 0.0, 0.0),
        (3.7, 0.0, 0.0),
        (3.7, 12.4, 0.0),
        (0.0, 12.4, 0.0),
    }


def test_rfem_export_contains_geometry_sections_supports_and_loads():
    model = _engineering_model(length=4.0)
    data = build_rfem_data(model)

    assert len(data["nodes"]) == len(model.points)
    assert len(data["members"]) == len([b for b in model.beams.values() if not b.is_virtual])
    assert len(data["surfaces"]) == len(model.plates)
    assert data["supports"][0]["nodes"]
    assert len(data["supports"][0]["nodes"]) == 4
    assert {load["load_case"] for load in data["surface_loads"]} == {1, 2, 3}


def test_rfem_export_uses_engineering_section_names():
    model = _engineering_model(length=4.0)
    data = build_rfem_data(model)
    names = {section["rfem_name"] for section in data["sections"]}

    assert data["materials"][0]["name"] == "Q355"
    assert names == {"SHS 150/6.0/C", "H HT 145x73", "H HT 118x58", "U 5"}


def test_rfem_script_manager_export_uses_internal_script_objects(tmp_path):
    model = _engineering_model(length=4.0)
    path = write_rfem_script_manager_js(model, tmp_path / "model.js")
    script = path.read_text(encoding="utf-8")

    assert "engine.create_member" in script
    assert "engine.create_load" in script
    assert "dlubal.api" not in script
    assert "DLUBAL_API_KEY" not in script
