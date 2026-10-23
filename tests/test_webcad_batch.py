import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.webcad import trial_plan
from src.integrations.webcad import build_batch_request, export_batch_plan, WebCADError
from src.integrations.webcad.client import UncertainCallError
from src.integrations.webcad.exporter import mark_export_cleared
from tests.test_webcad import FakeClient


class BatchTests(unittest.TestCase):
    def test_mapping_preserves_geometry_without_mutating_plan(self):
        plan = trial_plan()
        before = json.dumps(plan)
        request = build_batch_request(plan, 100)
        self.assertEqual(json.dumps(plan), before)
        node = request["nodeList"][0]
        self.assertEqual(node["params"]["startPoint"], plan["members"][0]["params"]["startPoint"])
        self.assertEqual(node["params"]["sectionParams"], plan["members"][0]["params"]["sectionParams"])
        self.assertEqual(node["params"]["id"], "cabin-documented_trial")
        self.assertEqual(node["params"]["userData"]["originId"], node["params"]["id"])
        self.assertNotIn("shapeReuseKey", node["params"])
        self.assertEqual(request["components"], [])
        self.assertTrue(request["skipCutRelations"])
        self.assertEqual(request["batchSize"], 100)

    def test_nonzero_rotation_pending_and_duplicate_ids_blocked(self):
        plan = trial_plan()
        plan["members"][0]["params"]["crosssectionRotationAngle"] = 90
        with self.assertRaises(ValueError):
            build_batch_request(plan)
        plan = trial_plan()
        plan["pending"] = ["unconfirmed"]
        with self.assertRaises(ValueError):
            build_batch_request(plan)
        plan = trial_plan()
        plan["members"] *= 2
        with self.assertRaises(ValueError):
            build_batch_request(plan)

    def client(self, result=None, after=None):
        client = FakeClient()
        response = {"total": 1, "successCount": 1, "failureCount": 0,
                    "nodeIds": ["cabin-documented_trial"], "failures": []}
        client.call_tool = Mock(side_effect=[[], response if result is None else result,
                                            after if after is not None else [{"nodeId": "cabin-documented_trial"}], {}])
        return client

    def test_single_batch_call_two_lists_then_save(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.client()
            path = Path(directory) / "batch.json"
            record = export_batch_plan(trial_plan(), client, path)
            self.assertEqual(record["status"], "saved")
            self.assertEqual(record["response"]["successCount"], 1)
            self.assertEqual(record["created_nodes"], [{"nodeId": "cabin-documented_trial"}])
            self.assertEqual([c.args[0] for c in client.call_tool.call_args_list],
                             ["cad.node.list", "cad.node.create_batch", "cad.node.list", "cad.document.save"])
            with self.assertRaises(WebCADError):
                export_batch_plan(trial_plan(), client, Path(directory) / "other.json")

    def test_partial_result_does_not_explicitly_save(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(after=[])
            path = Path(directory) / "batch.json"
            with self.assertRaisesRegex(WebCADError, "mismatch"):
                export_batch_plan(trial_plan(), client, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "interrupted")
            self.assertNotIn("cad.document.save", [c.args[0] for c in client.call_tool.call_args_list])

    def test_timeout_is_uncertain_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.client()
            client.call_tool.side_effect = [[], UncertainCallError("timeout")]
            path = Path(directory) / "batch.json"
            with self.assertRaises(UncertainCallError):
                export_batch_plan(trial_plan(), client, path)
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["uncertain"])
            self.assertEqual(client.call_tool.call_count, 2)

    def test_partial_batch_response_is_journaled_and_blocks_explicit_save(self):
        with tempfile.TemporaryDirectory() as directory:
            result = {"total": 1, "successCount": 0, "failureCount": 1, "nodeIds": [], "failures": ["error"]}
            client = self.client(result)
            path = Path(directory) / "batch.json"
            with self.assertRaisesRegex(WebCADError, "complete creation"):
                export_batch_plan(trial_plan(), client, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["response"], result)
            self.assertEqual(client.call_tool.call_count, 2)

    def test_wrong_new_node_id_is_not_accepted_even_when_count_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(after=[{"nodeId": "unrelated"}])
            with self.assertRaisesRegex(WebCADError, "mismatch"):
                export_batch_plan(trial_plan(), client, Path(directory) / "batch.json")

    def test_interrupted_serial_export_can_be_cleared_when_known_nodes_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            record = {"status": "interrupted", "target": {"base_url": "https://example.test", "document_id": "doc1"},
                      "members": [{"status": "created", "node_id": "gone"}, {"status": "failed", "node_id": None}]}
            path.write_text(json.dumps(record), encoding="utf-8")
            client = FakeClient()
            client.call_tool = Mock(return_value=[])
            self.assertEqual(mark_export_cleared(client, path)["status"], "cleared")
            record["members"][-1]["status"] = "uncertain"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(WebCADError):
                mark_export_cleared(client, path)


if __name__ == "__main__":
    unittest.main()
