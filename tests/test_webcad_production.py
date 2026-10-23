from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.export_geometry import build_default_engineering_model
from src.integrations.webcad.client import UncertainCallError, WebCADError
from src.integrations.webcad.production import (
    ProductionConfig,
    _subtract,
    build_production_plan,
    overlap_pairs,
    production_request,
)
from src.integrations.webcad.production_export import (
    audit_contacts,
    clear_known_tests,
    export_production_plan,
    write_production_files,
)
from src.integrations.webcad.profiles import prism_params, section_profile
from src.structure.structural_model import StructuralModel
from tests.test_webcad import FakeClient


class ProfileTests(unittest.TestCase):
    def test_analytic_areas_match_actual_native_section_breps(self):
        cases = [
            (
                "h",
                dict(widthB=100, height=150, thicknessT1=6, thicknessT2=10, radius=8),
                2834.938070170254,
            ),
            (
                "box",
                dict(
                    sideLengthA=150,
                    sideLengthB=150,
                    wallThicknessT=6,
                    outerArcRadiusR=9,
                ),
                3394.1946710584693,
            ),
            (
                "channel",
                dict(
                    legWidthB=75,
                    height=200,
                    thicknessD=9,
                    thicknessT=11,
                    radiusR=11,
                    legArcRadiusR1=5.5,
                ),
                3282.7504060802835,
            ),
        ]
        for family, params, area in cases:
            with self.subTest(family=family):
                profile = section_profile(family, params)
                self.assertAlmostEqual(profile["area_mm2"], area, places=6)
                for loop in profile["loops"]:
                    for index, segment in enumerate(loop):
                        self.assertTrue(
                            np.allclose(
                                segment["end"], loop[(index + 1) % len(loop)]["start"]
                            )
                        )

    def test_world_sketch_has_explicit_plane_and_hole(self):
        profile = section_profile(
            "box",
            dict(sideLengthA=150, sideLengthB=150, wallThicknessT=6, outerArcRadiusR=9),
        )
        part = dict(
            start_mm=[10, 20, 30],
            width_axis=[0, 1, 0],
            height_axis=[0, 0, 1],
            axis_vector=[1, 0, 0],
            profile=profile,
            length_mm=200,
        )
        params = prism_params(part)
        self.assertEqual(len(profile["loops"]), 2)
        self.assertEqual(params["lengthEnd"], 200)
        for segment in params["sketch"]:
            self.assertEqual(segment["start"]["x"], 10)
            self.assertEqual(segment["workplane"]["normal"], dict(x=1, y=0, z=0))


class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_default_engineering_model()
        cls.plan = build_production_plan(cls.model)

    def test_full_model_bounds_source_coverage_and_no_omitted_stocks(self):
        plan = self.plan
        self.assertEqual(plan["summary"]["source_segments"], 906)
        self.assertEqual(plan["summary"]["merged_stocks"], 126)
        self.assertEqual(plan["summary"]["parts"], 442)
        self.assertEqual(plan["summary"]["fully_consumed_stocks"], 0)
        self.assertEqual(plan["summary"]["bounds_mm"], [0, -1100, 0, 3700, 8500, 3300])
        ids = [i for s in plan["stocks"] for i in s["source_ids"]]
        self.assertEqual(len(set(ids)), 906)
        self.assertEqual(len(ids), 906)
        self.assertFalse(overlap_pairs(plan["parts"]))

    def test_top_and_bottom_faces_are_flush_for_all_sections(self):
        for part in self.plan["parts"]:
            if part["face"] == "top":
                self.assertEqual(part["bounds_mm"][5], 3300)
            if part["face"] == "bottom":
                self.assertEqual(part["bounds_mm"][2], 0)

    def test_end_grid_adjustment_preserves_near_end_primary_members(self):
        self.assertEqual(
            self.plan["grid_adjustments"],
            [dict(axis="y", nominal_mm=8400.0, placed_mm=8250.0)],
        )
        for stock in self.plan["stocks"]:
            self.assertTrue(stock["part_ids"])

    def test_source_model_unchanged_and_plan_deterministic(self):
        before = [(i, p.coordinates().tolist()) for i, p in self.model.points.items()]
        assignment = deepcopy(self.model.section_assignment)
        self.assertEqual(
            json.dumps(build_production_plan(self.model), sort_keys=True),
            json.dumps(self.plan, sort_keys=True),
        )
        self.assertEqual(
            before,
            [(i, p.coordinates().tolist()) for i, p in self.model.points.items()],
        )
        self.assertEqual(
            assignment.beam_groups, self.model.section_assignment.beam_groups
        )

    def test_a_second_scale_and_nonzero_gap(self):
        params = replace(
            self.model.params,
            length=7.2,
            main_beam_spacing=0.95,
            support_beam_max_spacing=0.55,
        )
        model = StructuralModel(
            params, self.model.section_lib, deepcopy(self.model.section_assignment)
        )
        model.generate()
        plan = build_production_plan(model, ProductionConfig(gap_mm=1))
        self.assertEqual(plan["summary"]["bounds_mm"], [0, -1100, 0, 3700, 7200, 3300])
        self.assertFalse(overlap_pairs(plan["parts"]))
        self.assertEqual(plan["summary"]["fully_consumed_stocks"], 0)

    def test_prism_request_uses_only_validated_fields_without_cut_relations(self):
        request = production_request(self.plan)
        self.assertTrue(request["skipCutRelations"])
        self.assertEqual(len(request["nodeList"]), 442)
        for node in request["nodeList"]:
            self.assertEqual(node["type"], "engine.prism")
            self.assertEqual(
                set(node["params"]), {"id", "sketch", "lengthStart", "lengthEnd"}
            )

    def test_interval_union_and_negative_gap(self):
        self.assertEqual(
            _subtract(0, 100, [(10, 30), (20, 50), (80, 120)]), [(0, 10), (50, 80)]
        )
        with self.assertRaises(ValueError):
            build_production_plan(self.model, ProductionConfig(gap_mm=-1))


class ProductionExportTests(unittest.TestCase):
    def test_existing_export_preserves_plan_and_bom(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            (directory / "export_manifest.json").write_text("{}")
            (directory / "plan.json").write_text("original")
            with self.assertRaises(WebCADError):
                write_production_files({}, directory)
            self.assertEqual((directory / "plan.json").read_text(), "original")

    def client(self):
        client = FakeClient()
        client.describe_type = Mock(
            return_value={
                "createSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        k: {} for k in ["sketch", "lengthStart", "lengthEnd"]
                    },
                }
            }
        )
        return client

    def plan(self):
        profile = section_profile(
            "h", dict(widthB=100, height=150, thicknessT1=6, thicknessT2=10, radius=8)
        )
        return {
            "summary": {},
            "parts": [
                dict(
                    id="production-0001",
                    start_mm=[0, 0, 0],
                    width_axis=[0, 1, 0],
                    height_axis=[0, 0, 1],
                    axis_vector=[1, 0, 0],
                    profile=profile,
                    length_mm=1000,
                )
            ],
        }

    def test_export_audits_actual_saved_document_not_stale_list(self):
        with tempfile.TemporaryDirectory() as td:
            client = self.client()
            client.call_tool = Mock(
                side_effect=[
                    {"properties": {"nodes": []}},
                    {
                        "successCount": 1,
                        "failureCount": 0,
                        "nodeIds": ["production-0001"],
                    },
                    {"properties": {"nodes": [], "hasBackendSave": False}},
                ]
            )
            with (
                patch(
                    "src.integrations.webcad.production_export.audit_production_document",
                    return_value={"verified_parts": 1},
                ),
                patch(
                    "src.integrations.webcad.production_export.audit_contacts",
                    return_value={"connected_components": [1]},
                ),
            ):
                record = export_production_plan(
                    self.plan(), client, Path(td) / "manifest.json", "doc1"
                )
            self.assertEqual(record["status"], "verified")
            self.assertFalse(record["has_backend_save"])
            self.assertEqual(
                [c.args[0] for c in client.call_tool.call_args_list],
                ["cad.document.save", "cad.node.create_batch", "cad.document.save"],
            )

    def test_timeout_blocks_retry_and_preserves_request(self):
        with tempfile.TemporaryDirectory() as td:
            client = self.client()
            client.call_tool = Mock(
                side_effect=[
                    {"properties": {"nodes": []}},
                    UncertainCallError("timeout"),
                ]
            )
            path = Path(td) / "manifest.json"
            with self.assertRaises(UncertainCallError):
                export_production_plan(self.plan(), client, path, "doc1")
            self.assertEqual(json.loads(path.read_text())["status"], "interrupted")
            with self.assertRaises(WebCADError):
                export_production_plan(self.plan(), client, path, "doc1")
            self.assertEqual(client.call_tool.call_count, 2)

    def test_cleanup_never_deletes_unrelated_nodes(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "old.json"
            source.write_text(
                json.dumps({"request": {"nodeList": [{"params": {"id": "known"}}]}})
            )
            client = self.client()
            client.call_tool = Mock(
                return_value={
                    "properties": {
                        "nodes": [
                            {"classKey": "BeamNode", "properties": {"id": "unrelated"}}
                        ]
                    }
                }
            )
            with self.assertRaisesRegex(WebCADError, "non-test"):
                clear_known_tests(client, "doc1", [source], Path(td) / "cleanup.json")
            client.call_tool.assert_called_once_with("cad.document.save")

    @unittest.skipUnless(
        importlib.util.find_spec("gmsh"), "Gmsh required for BREP contact diagnostics"
    )
    def test_contacts_use_real_solids_and_reject_disconnected_assembly(self):
        import gmsh

        with tempfile.TemporaryDirectory() as td:
            registry, nodes, parts = [], [], []
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                for index in range(2):
                    gmsh.clear()
                    gmsh.model.occ.addBox(index, 0, 0, 1, 1, 1)
                    gmsh.model.occ.synchronize()
                    path = Path(td) / "cube.brep"
                    gmsh.write(str(path))
                    key = f"part-{index}"
                    registry.append(dict(key=key, brep=path.read_text()))
                    nodes.append(
                        dict(
                            classKey="PrismNode",
                            properties={"id": key, "shapeVis": {"$shapeRef": key}},
                        )
                    )
                    parts.append(dict(id=key, bounds_mm=[index, 0, 0, index + 1, 1, 1]))
            finally:
                gmsh.finalize()
            doc = {"properties": {"nodes": nodes, "shapeRegistry": registry}}
            plan = {"parts": parts, "config": {"gap_mm": 0}}
            result = audit_contacts(plan, doc)
            self.assertEqual(result["connected_components"], [2])
            self.assertEqual(result["contact_pairs"], 1)
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.model.occ.addBox(3, 0, 0, 1, 1, 1)
                gmsh.model.occ.synchronize()
                gmsh.write(str(path))
                registry[1]["brep"] = path.read_text()
            finally:
                gmsh.finalize()
            # Bounding boxes still claim contact; the actual BREP has a gap.
            with self.assertRaisesRegex(WebCADError, "disconnected"):
                audit_contacts(plan, doc)


if __name__ == "__main__":
    unittest.main()
