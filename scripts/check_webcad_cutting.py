"""Small, journaled WebCAD cutting and placement experiment in an empty model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jsonschema.validators import validator_for

from scripts.webcad import load_env
from src.integrations.webcad import WebCADClient, WebCADConfig, WebCADError
from src.integrations.webcad.exporter import write_json


def point(x, y, z):
    return dict(x=x, y=y, z=z)


def beam(name, a, b, height=150, base=0, rotation=0, **relations):
    section = (dict(widthB=100, height=150, thicknessT1=6, thicknessT2=10, radius=8)
               if height == 150 else dict(widthB=200, height=200, thicknessT1=8, thicknessT2=12, radius=13))
    return {"type": "structure.hShapedSteelGirder", "params": {
        "id": f"cut-test-{name}", "name": name, "startPoint": a, "endPoint": b,
        "sectionParams": section, "baseType": base, "crosssectionRotationAngle": rotation,
        "typeCode": "15-001", "componentCode": "15-001-0004", **relations,
    }}


def experiment_request() -> dict:
    nodes = []
    for case, x in [("control", 0), ("cut-a", 3000), ("cut-b", 6000)]:
        main = beam(f"{case}-main", point(x, 0, 1000), point(x + 2000, 0, 1000))
        secondary = beam(f"{case}-secondary", point(x + 1000, -1000, 1000), point(x + 1000, 1000, 1000))
        if case == "cut-a":
            secondary["params"]["cutIdsA"] = [main["params"]["id"]]
        if case == "cut-b":
            main["params"]["cutIdsB"] = [secondary["params"]["id"]]
            nodes.extend([secondary, main])
        else:
            nodes.extend([main, secondary])
    nodes.extend([
        {"type": "engine.cutFace", "params": {
            "id": "cut-test-face", "name": "cut-face-x10000",
            "startPoint": point(10000, -600, 400), "endPoint": point(10000, 600, 1600),
            "normal": point(1, 0, 0), "widthRatio": 1, "isCenter": False,
        }},
        beam("face-beam", point(9000, 0, 1000), point(11000, 0, 1000), faceIds=["cut-test-face"]),
        beam("center-150", point(0, 3000, 3000), point(2000, 3000, 3000)),
        beam("top-150", point(0, 4000, 3000), point(2000, 4000, 3000), base=5),
        beam("top-200", point(0, 5000, 3000), point(2000, 5000, 3000), height=200, base=5),
        beam("rotation-90", point(0, 6000, 3000), point(2000, 6000, 3000), rotation=90),
    ])
    return {"nodeList": nodes, "components": [], "batchSize": 100, "skipCutRelations": False}


def followup_request() -> dict:
    main = beam("both-main", point(12000, 0, 1000), point(14000, 0, 1000), cutIdsB=["cut-test-both-secondary"])
    secondary = beam("both-secondary", point(13000, -1000, 1000), point(13000, 1000, 1000), cutIdsA=["cut-test-both-main"])
    face = {"type": "engine.cutFace", "params": {
        "id": "cut-test-face-both", "name": "cut-face-documented-example",
        "startPoint": point(16000, -600, 400), "endPoint": point(16000, 600, 400),
        "normal": point(0, 0, 1), "cutIds": ["cut-test-face-both-beam"],
    }}
    face_beam = beam("face-both-beam", point(15000, 0, 1000), point(17000, 0, 1000), faceIds=["cut-test-face-both"])
    workplane = {"plane": {"origin": point(0, 0, 0), "normal": point(0, -1, 0), "xvec": point(1, 0, 0)},
                 "planeId": "", "planeName": "cut-test-xz", "planeHeight": 0}
    nodes = [main, secondary, face, face_beam,
             beam("wp-center", point(0, 7000, 3000), point(2000, 7000, 3000), workplane=workplane),
             beam("wp-top", point(0, 8000, 3000), point(2000, 8000, 3000), base=5, workplane=workplane),
             beam("transform", point(0, 0, 0), point(2000, 0, 0),
                  transform=[1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 9000, 3000, 1])]
    return {"nodeList": nodes, "components": [], "batchSize": 100, "skipCutRelations": False}


def document_nodes(document: dict) -> dict:
    """Read the saved document, not the potentially stale CAD Agent list."""
    return {n["properties"]["id"]: n for n in document["properties"]["nodes"]}


def solid_paths(document: dict, directory: Path) -> dict:
    registry = {shape["key"]: shape for shape in document["properties"].get("shapeRegistry", [])}
    paths = {}
    for node_id, node in document_nodes(document).items():
        if not node_id.startswith("cut-test-") or node["classKey"] != "BeamNode":
            continue
        props = node["properties"]
        ref = props["shapeVis"]["$shapeRef"]
        path = directory / f"{node_id}.brep"
        path.write_text(registry[ref]["brep"], encoding="utf-8")
        paths[node_id] = path
    return paths


def measure_document(document: dict) -> dict:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    metrics = {}
    overlaps = {}
    nodes = document_nodes(document)
    try:
        with tempfile.TemporaryDirectory(prefix="webcad-cut-") as directory:
            paths = solid_paths(document, Path(directory))
            for node_id, path in paths.items():
                gmsh.clear()
                # These BeamNode BREP records already contain placed geometry.
                # Do not apply the saved transform a second time.
                entities = gmsh.model.occ.importShapes(str(path))
                volumes = [(dim, tag) for dim, tag in entities if dim == 3]
                boxes = [gmsh.model.occ.getBoundingBox(dim, tag) for dim, tag in volumes]
                metrics[node_id] = {
                    "volume_mm3": sum(gmsh.model.occ.getMass(dim, tag) for dim, tag in volumes),
                    "solid_count": len(volumes),
                    "bounds_mm": [min(b[i] for b in boxes) if i < 3 else max(b[i] for b in boxes) for i in range(6)],
                    "bounds_space": "saved_brep",
                }
                metrics[node_id]["saved_transform"] = nodes[node_id]["properties"]["transform"]["properties"]["array"]
            for case in ["control", "cut-a", "cut-b", "both"]:
                if f"cut-test-{case}-main" not in paths:
                    continue
                gmsh.clear()
                main = gmsh.model.occ.importShapes(str(paths[f"cut-test-{case}-main"]))
                secondary = gmsh.model.occ.importShapes(str(paths[f"cut-test-{case}-secondary"]))
                common, _ = gmsh.model.occ.intersect(main, secondary)
                overlaps[case] = sum(gmsh.model.occ.getMass(dim, tag) for dim, tag in common if dim == 3)
    finally:
        gmsh.finalize()
    return {"geometry": metrics, "intersection_volume_mm3": overlaps}


def run_experiment(client: WebCADClient, path: Path, document_id: str, follow_up=False) -> dict:
    if path.exists():
        raise WebCADError("Experiment record already exists; inspect it instead of retrying")
    client.authenticate()
    target = client.select_session()
    if target.get("documentId") != document_id:
        raise WebCADError("Selected editor does not match the explicitly requested document")
    request = followup_request() if follow_up else experiment_request()
    schemas = {}
    for node in request["nodeList"]:
        kind = node["type"]
        if kind not in schemas:
            schemas[kind] = client.describe_type(kind)
        schema = schemas[kind]["createSchema"]
        validator_for(schema)(schema).validate(node["params"])
    before = client.call_tool("cad.document.save")
    allowed = {n["params"]["id"] for n in experiment_request()["nodeList"]} if follow_up else set()
    existing = {key for key, node in document_nodes(before).items() if node["classKey"] != "FolderNode"}
    if existing != allowed:
        raise WebCADError("Saved document does not match the expected experiment baseline; no nodes were created")
    record = {"status": "in_flight", "target": target, "request": request,
              "schemas": schemas, "before_document": before}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    try:
        record["batch_response"] = client.call_tool("cad.node.create_batch", request)
        record["status"] = "returned"
        write_json(path, record)
        expected = {n["params"]["id"] for n in request["nodeList"]}
        result = record["batch_response"]
        if result.get("failureCount") != 0 or set(result.get("nodeIds", [])) != expected:
            raise WebCADError("Batch did not confirm all test nodes; inspect the response")
        record["document"] = client.call_tool("cad.document.save")
        record["status"] = "saved"
        write_json(path, record)
        actual = {key for key, node in document_nodes(record["document"]).items() if node["classKey"] != "FolderNode"}
        if actual != expected | allowed:
            raise WebCADError("Saved document does not contain exactly the expected test nodes")
        record["measurements"] = measure_document(record["document"])
        record["status"] = "measured"
        write_json(path, record)
    except Exception as exc:
        record["last_stage"] = record["status"]
        record["status"] = "interrupted"
        record["error"] = str(exc)
        write_json(path, record)
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-create", action="store_true")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/webcad/cut_tests")
    parser.add_argument("--ca-bundle", type=Path, required=True)
    parser.add_argument("--follow-up", action="store_true", help="Add reciprocal references and placement tests after the first experiment")
    args = parser.parse_args()
    if not args.confirm_create:
        write_json(args.output_dir / "request_preview.json", followup_request() if args.follow_up else experiment_request())
        print("Preview written; no network calls or model changes")
        return
    load_env(ROOT / ".env.local")
    os.environ.setdefault("WEBCAD_BASE_URL", "https://124.223.165.124:4006")
    config = WebCADConfig.from_env()
    config.verify_tls = str(args.ca_bundle.resolve())
    config.timeout = 120
    client = WebCADClient(config)
    try:
        filename = "followup.json" if args.follow_up else "experiment.json"
        result = run_experiment(client, args.output_dir / filename, args.document_id, args.follow_up)
        print(json.dumps(result["measurements"], ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
