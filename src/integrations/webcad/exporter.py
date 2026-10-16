from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from jsonschema.validators import validator_for

from src.integrations.webcad.client import UncertainCallError, WebCADClient, WebCADError
from src.integrations.webcad.mapping import MappingConfig, build_export_plan


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_members(client: WebCADClient, members: list[dict]) -> dict:
    """Validate every request against live schemas before the first write."""
    schemas = {}
    for member in members:
        node_type = member["type"]
        if node_type not in schemas:
            schemas[node_type] = client.describe_type(node_type)
        schema = schemas[node_type]["createSchema"]
        validator = validator_for(schema)
        validator.check_schema(schema)
        # No defaults, guessed fields, or silent field deletion here.
        validator(schema).validate(member["params"])
    return schemas


def export_plan(
    plan: dict,
    client: WebCADClient,
    manifest_path: str | Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Create serially, journal each attempt, and save only after all succeed.

    A manifest is single-use, including after failures. This deliberately blocks
    blind retries of ambiguous writes and repeated Streamlit button clicks.
    """
    path = Path(manifest_path)
    if plan["pending"] or any(member.get("pending") for member in plan["members"]):
        raise ValueError("Export has unresolved engineering decisions; inspect plan.pending")
    if not plan["members"]:
        raise ValueError("No physical members to export")
    client.authenticate()
    target = client.select_session()
    identity = {"base_url": client.config.base_url.rstrip("/"),
                "session_id": target["id"], "document_id": target.get("documentId")}
    # Scan neighboring manifests as well, not only this chosen filename.
    for previous in path.parent.glob("*.json"):
        data = json.loads(previous.read_text(encoding="utf-8"))
        previous_target = data.get("target", {}) if isinstance(data, dict) else {}
        same_document = bool(identity["document_id"]) and previous_target.get("document_id") == identity["document_id"]
        same_session = previous_target.get("session_id") == identity["session_id"]
        if previous_target.get("base_url") == identity["base_url"] and (same_document or same_session):
            raise WebCADError("This target already has an export manifest; reconcile existing nodes before re-exporting")
    if path.exists():
        raise WebCADError("Manifest already exists; inspect it instead of retrying")
    schemas = validate_members(client, plan["members"])
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    record = {"target": identity, "fingerprint": fingerprint, "status": "creating", "members": [], "schemas": schemas}
    # Exclusive creation also guards same-path simultaneous callers.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    try:
        for index, member in enumerate(plan["members"], 1):
            entry = {**member, "status": "in_flight", "node_id": None}
            record["members"].append(entry)
            write_json(path, record)
            value = client.call_tool("cad.node.create", {"type": member["type"], "params": member["params"]})
            if not isinstance(value, dict) or not value.get("nodeId"):
                raise UncertainCallError("Create response has no nodeId; verify the remote document")
            entry["node_id"] = value["nodeId"]
            entry["status"] = "created"
            entry["properties"] = value.get("properties", {})
            write_json(path, record)
            if on_progress:
                on_progress(index, len(plan["members"]))
        record["status"] = "saving"
        write_json(path, record)
        client.call_tool("cad.document.save")
        record["status"] = "saved"
        write_json(path, record)
    except Exception as exc:
        record["status"] = "interrupted"
        record["error"] = str(exc)
        if record["members"] and record["members"][-1]["status"] == "in_flight":
            record["members"][-1]["status"] = "uncertain" if isinstance(exc, UncertainCallError) else "failed"
        write_json(path, record)
        raise
    return record


def export_to_webcad(model, client: WebCADClient, mapping: MappingConfig, manifest_path, on_progress=None):
    """Streamlit-ready entry; pass GroupedOptimizationResult.model or AppSession.model."""
    return export_plan(build_export_plan(model, mapping), client, manifest_path, on_progress)
