"""Export locally trimmed prisms and audit the saved BREP geometry."""

import csv
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from jsonschema.validators import validator_for

from src.integrations.webcad.client import WebCADClient, WebCADError
from src.integrations.webcad.exporter import write_json
from src.integrations.webcad.production import overlap_pairs, production_request


def saved_nodes(document):
    return {
        n["properties"]["id"]: n
        for n in document["properties"]["nodes"]
        if n["classKey"] != "FolderNode"
    }


def write_production_files(plan, directory):
    directory = Path(directory)
    if (directory / "export_manifest.json").exists():
        raise WebCADError(
            "Export record exists: preserve its plan and BOM; use a new output directory"
        )
    write_json(directory / "plan.json", plan)
    write_json(directory / "request.json", production_request(plan))
    fields = [
        "id",
        "stock_id",
        "source_ids",
        "group_id",
        "section_name",
        "face",
        "length_mm",
        "start_mm",
        "end_mm",
        "offset_mm",
        "volume_mm3",
    ]
    with (directory / "cut_list.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for part in plan["parts"]:
            writer.writerow(
                {
                    k: json.dumps(part[k]) if isinstance(part[k], list) else part[k]
                    for k in fields
                }
            )


def audit_production_document(plan, document):
    import gmsh

    nodes = saved_nodes(document)
    if set(nodes) != {p["id"] for p in plan["parts"]}:
        raise WebCADError("Saved document IDs differ from the production plan")
    registry = {r["key"]: r for r in document["properties"].get("shapeRegistry", [])}
    metrics = []
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        with tempfile.TemporaryDirectory(prefix="webcad-production-") as temporary:
            path = Path(temporary) / "part.brep"
            for part in plan["parts"]:
                node = nodes[part["id"]]
                if node["classKey"] != "PrismNode":
                    raise WebCADError("Unexpected production node class")
                props = node["properties"]
                matrix = props["transform"]["properties"]["array"]
                if matrix != [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]:
                    raise WebCADError("Production prisms must use identity transforms")
                path.write_text(
                    registry[props["shapeVis"]["$shapeRef"]]["brep"], encoding="utf-8"
                )
                gmsh.clear()
                entities = gmsh.model.occ.importShapes(str(path))
                solids = [e for e in entities if e[0] == 3]
                if len(solids) != 1:
                    raise WebCADError(
                        f"{part['id']}: expected one connected solid, got {len(solids)}"
                    )
                volume = gmsh.model.occ.getMass(*solids[0])
                bounds = list(gmsh.model.occ.getBoundingBox(*solids[0]))
                if not np.isclose(volume, part["volume_mm3"], rtol=1e-6, atol=0.01):
                    raise WebCADError(
                        f"{part['id']}: BREP volume differs from the analytic profile"
                    )
                if not np.allclose(bounds, part["bounds_mm"], rtol=0, atol=0.002):
                    raise WebCADError(
                        f"{part['id']}: BREP placement differs from the production plan"
                    )
                metrics.append(
                    {"id": part["id"], "volume_mm3": volume, "bounds_mm": bounds}
                )
    finally:
        gmsh.finalize()
    overlaps = overlap_pairs(metrics, tolerance=0.004)
    if overlaps:
        raise WebCADError(f"Saved BREP envelopes overlap: {overlaps[:3]}")
    return {
        "verified_parts": len(metrics),
        "overlap_pairs": 0,
        "all_single_solids": True,
        "max_bounds_error_mm": max(
            abs(a - b)
            for p, m in zip(plan["parts"], metrics)
            for a, b in zip(p["bounds_mm"], m["bounds_mm"])
        ),
        "volume_mm3": sum(m["volume_mm3"] for m in metrics),
        "parts": metrics,
    }


def audit_contacts(plan, document):
    """Verify actual solid-to-solid contact, not merely touching bounding boxes."""
    import gmsh

    nodes = saved_nodes(document)
    registry = {r["key"]: r for r in document["properties"]["shapeRegistry"]}
    parts = plan["parts"]
    low = np.array([p["bounds_mm"][:3] for p in parts])
    high = np.array([p["bounds_mm"][3:] for p in parts])
    neighbors = {p["id"]: set() for p in parts}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    checks = 0
    try:
        with tempfile.TemporaryDirectory(prefix="webcad-contacts-") as temporary:
            path = Path(temporary) / "part.brep"
            solids = {}
            for part in parts:
                ref = nodes[part["id"]]["properties"]["shapeVis"]["$shapeRef"]
                path.write_text(registry[ref]["brep"], encoding="utf-8")
                solids[part["id"]] = next(
                    e for e in gmsh.model.occ.importShapes(str(path)) if e[0] == 3
                )
            for i, part in enumerate(parts):
                depth = np.minimum(high[i], high[i + 1 :]) - np.maximum(
                    low[i], low[i + 1 :]
                )
                for relative in np.flatnonzero(np.all(depth >= -0.002, axis=1)):
                    other = parts[i + 1 + relative]
                    distance, *_ = gmsh.model.occ.getDistance(
                        *solids[part["id"]], *solids[other["id"]]
                    )
                    checks += 1
                    if distance < 0:
                        raise WebCADError(
                            "OpenCASCADE contact distance evaluation failed"
                        )
                    if distance <= 0.002:
                        neighbors[part["id"]].add(other["id"])
                        neighbors[other["id"]].add(part["id"])
    finally:
        gmsh.finalize()
    remaining = set(neighbors)
    components = []
    while remaining:
        pending = [remaining.pop()]
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            new = neighbors[node] & remaining
            remaining -= new
            pending.extend(new)
        components.append(size)
    result = {
        "checked_pairs": checks,
        "contact_pairs": sum(map(len, neighbors.values())) // 2,
        "connected_components": sorted(components, reverse=True),
        "isolated_parts": [key for key, value in neighbors.items() if not value],
    }
    if plan["config"]["gap_mm"] == 0 and len(components) != 1:
        raise WebCADError(f"Production assembly is disconnected: {result}")
    return result


def clear_known_tests(client, document_id, record_paths, output):
    """Delete only IDs in the specified test records, never a prefix wildcard."""
    output = Path(output)
    if output.exists():
        raise WebCADError("Cleanup record exists; inspect it before retrying")
    client.authenticate()
    if client.select_session().get("documentId") != document_id:
        raise WebCADError("Cleanup target document mismatch")
    known = set()
    for path in record_paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        known.update(n["params"]["id"] for n in record["request"]["nodeList"])
    before = client.call_tool("cad.document.save")
    nodes = saved_nodes(before)
    if not set(nodes) <= known:
        raise WebCADError("Document contains non-test nodes: no deletion performed")
    record = {
        "document_id": document_id,
        "status": "deleting",
        "approved_ids": sorted(nodes),
        "deleted": [],
    }
    write_json(output, record)
    for node_id in sorted(nodes):
        client.call_tool("cad.node.delete", {"nodeId": node_id})
        record["deleted"].append(node_id)
        write_json(output, record)
    after = client.call_tool("cad.document.save")
    if saved_nodes(after):
        raise WebCADError("Saved document still contains nodes after test cleanup")
    record["status"] = "cleared"
    write_json(output, record)


def export_production_plan(
    plan, client: WebCADClient, manifest_path, document_id, on_progress=None
):
    path = Path(manifest_path)
    if path.exists():
        raise WebCADError(
            "Production manifest exists; never blindly retry a creation request"
        )
    request = production_request(plan)
    client.authenticate()
    target = client.select_session()
    if target.get("documentId") != document_id:
        raise WebCADError("Production target document mismatch")
    schema = client.describe_type("engine.prism")
    validator = validator_for(schema["createSchema"])(schema["createSchema"])
    for node in request["nodeList"]:
        # The batch pipeline handles params.id; prism's live createSchema omits it.
        validator.validate({k: v for k, v in node["params"].items() if k != "id"})
    before = client.call_tool("cad.document.save")
    if saved_nodes(before):
        raise WebCADError("Production import requires an empty actual saved document")
    record = {
        "status": "in_flight",
        "target": target,
        "request": request,
        "schema": schema,
        "summary": plan["summary"],
        "before_document": before,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    start = perf_counter()
    try:
        record["response"] = client.call_tool("cad.node.create_batch", request)
        record["batch_seconds"] = perf_counter() - start
        record["status"] = "returned"
        write_json(path, record)
        result = record["response"]
        expected = {p["id"] for p in plan["parts"]}
        if (
            result.get("successCount") != len(expected)
            or result.get("failureCount") != 0
            or len(result.get("nodeIds", [])) != len(expected)
            or set(result["nodeIds"]) != expected
        ):
            raise WebCADError(
                "Production batch was not completely successful; inspect its response"
            )
        if on_progress:
            on_progress(len(expected), len(expected))
        record["document"] = client.call_tool("cad.document.save")
        record["status"] = "saved"
        write_json(path, record)
        record["audit"] = audit_production_document(plan, record["document"])
        record["contacts"] = audit_contacts(plan, record["document"])
        record["total_seconds"] = perf_counter() - start
        record["has_backend_save"] = record["document"]["properties"].get(
            "hasBackendSave"
        )
        record["status"] = "verified"
        write_json(path, record)
    except Exception as exc:
        record["last_stage"] = record["status"]
        record["status"] = "interrupted"
        record["error"] = str(exc)
        write_json(path, record)
        raise
    return record
