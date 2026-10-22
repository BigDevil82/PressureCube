import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from scripts.webcad import load_env, trial_plan
from src.integrations.webcad import MappingConfig, WebCADClient, WebCADConfig, WebCADError, build_export_plan, export_plan
from src.integrations.webcad.client import UncertainCallError, encrypt_password
from src.structure.enums import BeamType
from src.structure.structural_model import Beam, Point
from src.structure.section_library import BoxSection, HShapeSection, ChannelSection


def section(cls=BoxSection, name="BOX-test"):
    props = dict(idx=0, name=name, area=0.003, Ix=1e-5, Iy=1e-5, Iz=1e-6)
    if cls == BoxSection:
        return cls(**props, B=.15, t=.006)
    return cls(**props, H=.2, B=.1, tw=.006, tf=.009)


def model(beam_type=BeamType.FRAME, axis="x", reverse=False, sec=None):
    a, b = Point(0, 0, 0, 0), Point(1, 0, 0, 0)
    setattr(b, axis, 2)
    if reverse:
        a, b = b, a
    return SimpleNamespace(
        points={0: a, 1: b},
        beams={0: Beam(0, 0, 1, beam_type, sec or section(), False, "bottom")},
        section_assignment=SimpleNamespace(beam_groups={0: "real-group"}),
    )


def ready_mapping():
    return MappingConfig(
        section_rules={"BOX-test": {"sectionParams": {"outerArcRadiusR": 8}}},
        rotations={"FRAME:bottom:x:+": 90, "FRAME:bottom:z:+": 0},
        horizontal_frames_confirmed=True, world_placement_confirmed=True,
    )


class FakeClient:
    def __init__(self):
        self.config = WebCADConfig("https://example.test", session_id="s1")
        self.calls = []
        self.fail = ""

    def authenticate(self):
        pass

    def select_session(self):
        return {"id": "s1", "documentId": "doc1"}

    def describe_type(self, node_type):
        return {"createSchema": {"type": "object", "required": ["startPoint", "endPoint", "sectionParams"]}}

    def call_tool(self, tool, args=None):
        self.calls.append(tool)
        if tool == self.fail:
            raise UncertainCallError("network timeout")
        return {"nodeId": "remote-1", "properties": {}} if tool == "cad.node.create" else {}


class MappingTests(unittest.TestCase):
    def test_units_group_and_frame_girder(self):
        plan = build_export_plan(model(), ready_mapping())
        self.assertFalse(plan["pending"])
        member = plan["members"][0]
        self.assertEqual(member["group_id"], "real-group")
        self.assertEqual(member["type"], "structure.squareTubeSteelGirder")
        self.assertEqual(member["params"]["endPoint"]["x"], 2000)
        self.assertEqual(member["params"]["sectionParams"]["sideLengthA"], 150)
        self.assertEqual(member["params"]["crosssectionRotationAngle"], 90)

    def test_column_order(self):
        member = build_export_plan(model(axis="z", reverse=True), ready_mapping())["members"][0]
        self.assertTrue(member["type"].endswith("Pillar"))
        self.assertEqual(member["params"]["typeCode"], "15-002")
        self.assertLess(member["params"]["startPoint"]["z"], member["params"]["endPoint"]["z"])

    def test_unknown_dimensions_and_rotation_not_guessed(self):
        plan = build_export_plan(model())
        self.assertTrue(plan["pending"])
        self.assertNotIn("crosssectionRotationAngle", plan["members"][0]["params"])
        self.assertNotIn("outerArcRadiusR", plan["members"][0]["params"]["sectionParams"])

    def test_open_sections_require_confirmation(self):
        for cls, expected in [(HShapeSection, "hShaped"), (ChannelSection, "channel")]:
            with self.subTest(cls=cls):
                plan = build_export_plan(model(BeamType.MAIN, sec=section(cls, "open")))
                self.assertIn(expected, plan["members"][0]["type"])
                self.assertTrue(any("equivalence" in p for p in plan["pending"]))

    def test_virtual_and_mesh(self):
        m = model()
        m.beams[0].is_virtual = True
        self.assertEqual(build_export_plan(m)["members"], [])
        m.beams[0].is_virtual = False
        m.beams[0].parent_id = 1
        with self.assertRaises(ValueError):
            build_export_plan(m)

    def test_invalid_box(self):
        m = model()
        m.beams[0].section.t = .08
        with self.assertRaises(ValueError):
            build_export_plan(m)


class ClientTests(unittest.TestCase):
    def test_login_uses_local_api(self):
        client = WebCADClient(WebCADConfig("https://example.test", username="user", password="test-secret"))
        client._request = Mock(return_value={"code": 200, "data": {"access_token": "test-token"}})
        with patch("src.integrations.webcad.client.encrypt_password", return_value="encrypted"):
            client.authenticate()
        args = client._request.call_args.args
        self.assertEqual(args[:2], ("POST", "/local-api/auth/login"))
        self.assertEqual(args[2]["password"], "encrypted")
        self.assertEqual(client.http.headers["Authorization"], "Bearer test-token")
        client.close()

    def test_ca_bundle_is_used_without_disabling_tls(self):
        with patch.dict("os.environ", {"WEBCAD_BASE_URL": "https://example.test", "WEBCAD_CA_BUNDLE": "trusted-ca.pem"}, clear=True):
            config = WebCADConfig.from_env()
        self.assertEqual(config.verify_tls, "trusted-ca.pem")
        client = WebCADClient(config)
        response = Mock(status_code=200)
        response.json.return_value = {"sessions": []}
        client.http.request = Mock(return_value=response)
        self.assertEqual(client.sessions(), [])
        self.assertEqual(client.http.request.call_args.kwargs["verify"], "trusted-ca.pem")
        client.close()

    def test_multiple_sessions_need_explicit_selection(self):
        client = WebCADClient(WebCADConfig("https://example.test"))
        client.sessions = Mock(return_value=[{"id": "a"}, {"id": "b"}])
        with self.assertRaises(WebCADError):
            client.select_session()
        client.config.session_id = "b"
        self.assertEqual(client.select_session()["id"], "b")
        client.close()

    def test_business_failure_on_http_200(self):
        client = WebCADClient(WebCADConfig("https://example.test", session_id="a"))
        client._request = Mock(return_value={"ok": False, "error": {"code": "INVALID_PARAMS"}})
        with self.assertRaisesRegex(WebCADError, "INVALID_PARAMS"):
            client.call_tool("cad.node.create")
        client.close()

    def test_timeout_does_not_retry(self):
        client = WebCADClient(WebCADConfig("https://example.test"))
        client.http.request = Mock(side_effect=requests.Timeout())
        with self.assertRaises(UncertainCallError):
            client._request("POST", "/test")
        self.assertEqual(client.http.request.call_count, 1)
        client.close()

    def test_login_encryption_roundtrip(self):
        from cryptography.hazmat.primitives import padding, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa, padding as rsa_padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        der = private.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        with patch("src.integrations.webcad.client.LOGIN_PUBLIC_KEY", base64.b64encode(der).decode()):
            wrapped, ciphertext = encrypt_password("test-password").split("|")
        key = base64.b64decode(private.decrypt(base64.b64decode(wrapped), rsa_padding.PKCS1v15()))
        decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
        padded = decryptor.update(base64.b64decode(ciphertext)) + decryptor.finalize()
        unpad = padding.PKCS7(128).unpadder()
        self.assertEqual(unpad.update(padded) + unpad.finalize(), b"test-password")

    def test_no_credentials_in_config_repr(self):
        config = WebCADConfig("https://example.test", password="secret123", access_token="token123")
        self.assertNotIn("secret123", repr(config))
        self.assertNotIn("token123", repr(config))

    def test_env_file_literal(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            path = Path(directory) / ".env.local"
            path.write_text('WEBCAD_PASSWORD="literal#with=sign"\n', encoding="utf-8")
            load_env(path)
            import os
            self.assertEqual(os.environ["WEBCAD_PASSWORD"], "literal#with=sign")

    def test_local_short_login_keys(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"USERNAME": "windows-login"}, clear=True):
            path = Path(directory) / ".env.local"
            path.write_text('username=user\npassword=test-secret\n', encoding="utf-8")
            load_env(path)
            import os
            self.assertEqual(os.environ["WEBCAD_USERNAME"], "user")
            self.assertEqual(os.environ["WEBCAD_PASSWORD"], "test-secret")
            self.assertEqual(os.environ["USERNAME"], "windows-login")


class ExportTests(unittest.TestCase):
    def test_create_save_and_block_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            path = Path(directory) / "manifest.json"
            record = export_plan(trial_plan(), client, path)
            self.assertEqual(record["status"], "saved")
            self.assertEqual(client.calls, ["cad.node.create", "cad.document.save"])
            with self.assertRaises(WebCADError):
                export_plan(trial_plan(), client, Path(directory) / "other.json")

    def test_pending_blocks_before_network(self):
        client = FakeClient()
        client.authenticate = Mock()
        plan = trial_plan()
        plan["members"][0]["pending"] = ["unconfirmed"]
        with self.assertRaises(ValueError):
            export_plan(plan, client, "unused.json")
        client.authenticate.assert_not_called()

    def test_timeout_journal_no_save(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            client.fail = "cad.node.create"
            path = Path(directory) / "manifest.json"
            with self.assertRaises(UncertainCallError):
                export_plan(trial_plan(), client, path)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(record["members"][0]["status"], "uncertain")
            self.assertNotIn("cad.document.save", client.calls)

    def test_new_session_same_document_does_not_allow_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            export_plan(trial_plan(), client, Path(directory) / "first.json")
            client.select_session = Mock(return_value={"id": "new-session", "documentId": "doc1"})
            with self.assertRaises(WebCADError):
                export_plan(trial_plan(), client, Path(directory) / "second.json")

    def test_preflight_schema_rejects_all_before_creating(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            client.describe_type = Mock(return_value={"createSchema": {"type": "object", "required": ["missing"]}})
            with self.assertRaises(Exception):
                export_plan(trial_plan(), client, Path(directory) / "manifest.json")
            self.assertEqual(client.calls, [])
            self.assertFalse(list(Path(directory).glob("*.json")))

    def test_save_failure_retains_created_node(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            client.fail = "cad.document.save"
            path = Path(directory) / "manifest.json"
            with self.assertRaises(UncertainCallError):
                export_plan(trial_plan(), client, path)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["members"][0]["node_id"], "remote-1")
            self.assertEqual(record["members"][0]["status"], "created")


if __name__ == "__main__":
    unittest.main()
