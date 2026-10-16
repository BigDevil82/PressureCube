"""WebCAD HTTP diagnostics and explicit test-member creation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integrations.webcad import MappingConfig, WebCADClient, WebCADConfig, WebCADError, build_export_plan, export_plan
from src.integrations.webcad.exporter import write_json


def load_env(path: Path) -> None:
    """Load literal KEY=VALUE lines; never evaluate shell expressions."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)
    # Accept the short keys already used in the local credential file.
    for short, full in [("username", "WEBCAD_USERNAME"), ("password", "WEBCAD_PASSWORD")]:
        if os.getenv(short):
            os.environ.setdefault(full, os.environ[short])


def trial_plan() -> dict:
    # Exact square-tube example in the supplied task document, not a design result.
    return {"units": "mm", "test_only": True, "pending": [], "members": [{
        "source_id": "documented_trial", "group_id": "trial", "section_name": "trial-100x100x5-R8",
        "type": "structure.squareTubeSteelGirder", "pending": [],
        "params": {
            "startPoint": {"x": 0, "y": 0, "z": 3000},
            "endPoint": {"x": 4000, "y": 0, "z": 3000},
            "sectionParams": {"sideLengthA": 100, "sideLengthB": 100, "wallThicknessT": 5, "outerArcRadiusR": 8},
            "baseType": 0, "crosssectionRotationAngle": 0,
            "typeCode": "15-001", "componentCode": "15-001-0001",
        },
    }]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["probe", "preview", "trial", "export"])
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/webcad")
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--plan", type=Path, help="Reviewed plan from the actual final model; required for export")
    parser.add_argument("--base-url", default="https://124.223.165.124:4006", help="Defaults to the supplied JS example server")
    parser.add_argument("--confirm-create", action="store_true", help="Explicitly create and save in the target editor")
    args = parser.parse_args()
    load_env(args.env_file)
    os.environ.setdefault("WEBCAD_BASE_URL", args.base_url)
    if args.action == "preview":
        from scripts.export_geometry import build_default_engineering_model

        config = MappingConfig(**json.loads(args.mapping.read_text(encoding="utf-8"))) if args.mapping else MappingConfig()
        plan = build_export_plan(build_default_engineering_model(), config)
        write_json(args.output_dir / "preview.json", plan)
        print(f"Preview only (default geometry, not an optimized result): {len(plan['members'])} members, {len(plan['pending'])} pending decisions")
        return
    if args.action in {"trial", "export"} and not args.confirm_create:
        parser.error("Writing to WebCAD requires --confirm-create")
    if args.action == "export" and not args.plan:
        parser.error("Export requires --plan from the actual final model")
    config = WebCADConfig.from_env()
    if config.verify_tls is False:
        print("WARNING: TLS verification is explicitly disabled for this connection")
    client = WebCADClient(config)
    try:
        client.authenticate()
        sessions = client.sessions()
        print(f"Authenticated; {len(sessions)} editor sessions")
        if args.action == "probe":
            write_json(args.output_dir / "sessions.json", {"sessions": sessions})
            client.select_session()
            types = client.call_tool("cad.node.list_types")
            write_json(args.output_dir / "types.json", types)
            for family in ["squareTubeSteel", "gongShapedSteel", "channelSteel"]:
                for kind in ["Girder", "Pillar"]:
                    node_type = f"structure.{family}{kind}"
                    schema = client.describe_type(node_type)
                    write_json(args.output_dir / f"{node_type}.json", schema)
            print("Saved target sessions, supported types and six live schemas; no model changes")
        else:
            plan = trial_plan() if args.action == "trial" else json.loads(args.plan.read_text(encoding="utf-8"))
            manifest = args.output_dir / ("trial_manifest.json" if args.action == "trial" else "export_manifest.json")
            record = export_plan(plan, client, manifest)
            print(f"{record['status']}: {len(record['members'])} members; manifest: {manifest}")
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except (WebCADError, KeyError, ValueError) as exc:
        print(f"WebCAD: {exc}", file=sys.stderr)
        sys.exit(1)
