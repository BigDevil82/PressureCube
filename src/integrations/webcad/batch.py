"""Documented WebCAD batch pipeline; no geometry reuse across world positions."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

from src.integrations.webcad.client import UncertainCallError, WebCADClient, WebCADError
from src.integrations.webcad.exporter import check_previous_exports, validate_members, write_json


LABELS = {
    "structure.squareTubeSteelGirder": "方管梁",
    "structure.squareTubeSteelPillar": "方管柱",
    "structure.hShapedSteelGirder": "H 型钢梁",
    "structure.channelSteelGirder": "槽钢梁",
}


def build_batch_request(plan: dict, batch_size: int = 100) -> dict:
    """Translate the reviewed plan without changing its coordinates or sections."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not plan["members"] or plan["pending"] or any(m.get("pending") for m in plan["members"]):
        raise ValueError("Batch export requires members with no pending decisions")
    nodes = []
    ids = set()
    for index, member in enumerate(plan["members"]):
        params = deepcopy(member["params"])
        if params.get("crosssectionRotationAngle", 0) != 0:
            raise ValueError("Platform batch BeamNode rotation is not confirmed; nonzero rotation blocked")
        origin = f"cabin-{member.get('source_id', index + 1)}"
        if origin in ids:
            raise ValueError(f"Duplicate batch origin ID: {origin}")
        ids.add(origin)
        params.pop("shapeReuseKey", None)
        params["id"] = origin
        params["compentName"] = LABELS[member["type"]]
        if member.get("section_name"):
            params["detailName"] = member["section_name"]
        params["userData"] = {"originId": origin}
        for source, dest in [("group_id", "groupId"), ("beam_type", "beamType"),
                             ("face", "face"), ("section_name", "sectionName")]:
            if member.get(source):
                params["userData"][dest] = member[source]
        nodes.append({"type": member["type"], "params": params})
    return {"nodeList": nodes, "components": [], "batchSize": batch_size, "skipCutRelations": True}


def _node_ids(nodes) -> set[str]:
    if not isinstance(nodes, list) or any(not isinstance(n, dict) or not n.get("nodeId") for n in nodes):
        raise WebCADError("Unexpected node list response")
    ids = {n["nodeId"] for n in nodes}
    if len(ids) != len(nodes):
        raise WebCADError("Duplicate node IDs in list response")
    return ids


def export_batch_plan(plan: dict, client: WebCADClient, manifest_path: str | Path,
                      batch_size: int = 100, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Send one batch call, reconcile once, then explicitly save. Never retry writes.

    Preserve the result and verify both the observed batch response fields and
    the created node set. Chunk saves on the platform are not transactional.
    """
    request = build_batch_request(plan, batch_size)
    path = Path(manifest_path)
    client.authenticate()
    target = client.select_session()
    identity = {"base_url": client.config.base_url.rstrip("/"),
                "session_id": target["id"], "document_id": target.get("documentId")}
    check_previous_exports(path, identity)
    schemas = validate_members(client, plan["members"])
    before = _node_ids(client.call_tool("cad.node.list"))
    expected = {node["params"]["id"] for node in request["nodeList"]}
    if before & expected:
        raise WebCADError("Batch origin IDs already exist in the document")
    record = {"target": identity, "status": "in_flight", "method": "cad.node.create_batch",
              "total_members": len(request["nodeList"]), "request": request,
              "members": plan["members"], "schemas": schemas, "before_node_ids": sorted(before),
              "test_only": plan.get("test_only", False), "warnings": plan.get("warnings", []),
              "model_parameters": plan.get("model_parameters", {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    try:
        record["response"] = client.call_tool("cad.node.create_batch", request)
        record["status"] = "returned"
        write_json(path, record)
        result = record["response"]
        # These fields and preservation of params.id were confirmed live on
        # 2026-10-23, not specified by the supplied static JSON example.
        if (not isinstance(result, dict) or result.get("total") != len(expected)
                or result.get("successCount") != len(expected) or result.get("failureCount") != 0
                or result.get("failures") != [] or not isinstance(result.get("nodeIds"), list)
                or len(result["nodeIds"]) != len(expected) or set(result["nodeIds"]) != expected):
            raise WebCADError("Batch response does not confirm complete creation; inspect the saved response")
        nodes = client.call_tool("cad.node.list")
        after = _node_ids(nodes)
        created = after - before
        record["created_nodes"] = [node for node in nodes if node["nodeId"] in created]
        write_json(path, record)
        if not before <= after or created != expected:
            raise WebCADError(f"Batch reconciliation mismatch: expected {len(request['nodeList'])} new nodes, got {len(created)}")
        record["status"] = "saving"
        write_json(path, record)
        record["save_response"] = client.call_tool("cad.document.save")
        record["status"] = "saved"
        write_json(path, record)
        if on_progress:
            on_progress(len(created), len(request["nodeList"]))
    except Exception as exc:
        record["last_stage"] = record["status"]
        record["status"] = "interrupted"
        record["uncertain"] = isinstance(exc, UncertainCallError)
        record["error"] = str(exc)
        write_json(path, record)
        raise
    return record
