import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.check_webcad_cutting import experiment_request, followup_request, measure_document, run_experiment
from src.integrations.webcad import WebCADError
from src.integrations.webcad.client import UncertainCallError
from tests.test_webcad import FakeClient


def document(ids=()):
    return {"properties": {"nodes": [{"classKey": "BeamNode", "properties": {"id": node_id}} for node_id in ids]}}


class CuttingClient(FakeClient):
    def describe_type(self, node_type):
        if node_type == "engine.cutFace":
            return {"createSchema": {"type": "object", "required": ["startPoint", "endPoint"]}}
        return super().describe_type(node_type)


class CuttingTests(unittest.TestCase):
    def test_requests_are_small_unique_and_explicit(self):
        first, second = experiment_request(), followup_request()
        ids = [n["params"]["id"] for n in first["nodeList"] + second["nodeList"]]
        self.assertEqual(len(ids), 19)
        self.assertEqual(len(set(ids)), 19)
        self.assertFalse(first["skipCutRelations"])
        nodes = {n["params"]["id"]: n["params"] for n in first["nodeList"]}
        self.assertEqual(nodes["cut-test-cut-a-secondary"]["cutIdsA"], ["cut-test-cut-a-main"])
        self.assertEqual(nodes["cut-test-cut-b-main"]["cutIdsB"], ["cut-test-cut-b-secondary"])
        self.assertEqual(nodes["cut-test-top-150"]["baseType"], 5)
        self.assertEqual(nodes["cut-test-rotation-90"]["crosssectionRotationAngle"], 90)

    def test_nonempty_saved_document_blocks_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CuttingClient()
            client.call_tool = Mock(return_value=document(["existing"]))
            with self.assertRaisesRegex(WebCADError, "baseline"):
                run_experiment(client, Path(directory) / "record.json", "doc1")
            client.call_tool.assert_called_once_with("cad.document.save")

    def test_measures_saved_document_without_trusting_node_list(self):
        with tempfile.TemporaryDirectory() as directory:
            ids = [n["params"]["id"] for n in experiment_request()["nodeList"]]
            client = CuttingClient()
            client.call_tool = Mock(side_effect=[document(), {"failureCount": 0, "nodeIds": ids}, document(ids)])
            path = Path(directory) / "record.json"
            with patch("scripts.check_webcad_cutting.measure_document", return_value={"verified": True}):
                result = run_experiment(client, path, "doc1")
            self.assertEqual(result["status"], "measured")
            self.assertEqual([c.args[0] for c in client.call_tool.call_args_list],
                             ["cad.document.save", "cad.node.create_batch", "cad.document.save"])
            with self.assertRaises(WebCADError):
                run_experiment(client, path, "doc1")

    def test_timeout_preserves_request_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CuttingClient()
            client.call_tool = Mock(side_effect=[document(), UncertainCallError("timeout")])
            path = Path(directory) / "record.json"
            with self.assertRaises(UncertainCallError):
                run_experiment(client, path, "doc1")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(record["last_stage"], "in_flight")
            self.assertEqual(client.call_tool.call_count, 2)

    @unittest.skipUnless(importlib.util.find_spec("gmsh"), "Gmsh is optional for diagnostic geometry measurements")
    def test_brep_measurement_does_not_double_transform_placed_geometry(self):
        import gmsh

        with tempfile.TemporaryDirectory() as directory:
            shapes = []
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                for name, offset in [("main", 0), ("secondary", .5)]:
                    gmsh.clear()
                    gmsh.model.occ.addBox(offset, 0, 0, 1, 1, 1)
                    gmsh.model.occ.synchronize()
                    path = Path(directory) / f"{name}.brep"
                    gmsh.write(str(path))
                    shapes.append({"key": name, "brep": path.read_text()})
            finally:
                gmsh.finalize()
            data = {"properties": {"nodes": [], "shapeRegistry": shapes}}
            for name, offset in [("main", 0), ("secondary", .5)]:
                data["properties"]["nodes"].append({"classKey": "BeamNode", "properties": {
                    "id": f"cut-test-control-{name}", "shapeVis": {"$shapeRef": name},
                    "transform": {"properties": {"array": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, offset, 0, 0, 1]}},
                }})
            result = measure_document(data)
            self.assertAlmostEqual(result["intersection_volume_mm3"]["control"], .5, places=6)
            metrics = result["geometry"]["cut-test-control-secondary"]
            self.assertAlmostEqual(metrics["volume_mm3"], 1, places=6)
            self.assertAlmostEqual(metrics["bounds_mm"][0], .5, places=6)


if __name__ == "__main__":
    unittest.main()
