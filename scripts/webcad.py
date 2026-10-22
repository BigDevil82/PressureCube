"""WebCAD HTTP diagnostics and explicit test-member creation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integrations.webcad import MappingConfig, WebCADClient, WebCADConfig, WebCADError, build_export_plan, export_plan
from src.integrations.webcad.exporter import mark_export_cleared, write_json


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
        # Normalize before touching os.environ: Windows treats username and the
        # system USERNAME as the same key. Never replace/use the OS login name.
        key = {"username": "WEBCAD_USERNAME", "password": "WEBCAD_PASSWORD"}.get(key.strip(), key.strip())
        os.environ.setdefault(key, value)


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


def wait_for_sessions(client: WebCADClient, seconds: float) -> list[dict]:
    deadline = time.monotonic() + seconds
    waiting = False
    while True:
        sessions = client.sessions()
        remaining = deadline - time.monotonic()
        if any(session.get("documentId") for session in sessions) or remaining <= 0:
            return sessions
        if not waiting:
            print(f"Waiting up to {seconds:g} seconds for an editor session...", flush=True)
            waiting = True
        time.sleep(min(3, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["probe", "preview", "trial", "cabin", "export"])
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/webcad")
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--plan", type=Path, help="Reviewed plan from the actual final model; required for export")
    parser.add_argument("--base-url", default="https://124.223.165.124:4006", help="Defaults to the supplied JS example server")
    parser.add_argument("--ca-bundle", type=Path, help="Trusted CA certificate bundle for this connection only")
    parser.add_argument("--confirm-create", action="store_true", help="Explicitly create and save in the target editor")
    parser.add_argument("--cleared-manifest", type=Path, help="Verify user-deleted nodes are absent and preserve the old record as cleared")
    parser.add_argument("--wait-for-editor", type=float, default=0, help="Seconds to wait for a browser editor to connect")
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
    if args.action in {"trial", "cabin", "export"} and not args.confirm_create:
        parser.error("Writing to WebCAD requires --confirm-create")
    if args.action == "export" and not args.plan:
        parser.error("Export requires --plan from the actual final model")
    config = WebCADConfig.from_env()
    if args.ca_bundle:
        config.verify_tls = str(args.ca_bundle.resolve())
    if config.verify_tls is False:
        print("WARNING: TLS verification is explicitly disabled for this connection")
    client = WebCADClient(config)
    try:
        client.authenticate()
        sessions = wait_for_sessions(client, args.wait_for_editor)
        print(f"Authenticated; {len(sessions)} editor sessions", flush=True)
        if args.action == "probe":
            write_json(args.output_dir / "sessions.json", {"sessions": sessions})
            client.select_session()
            types = client.call_tool("cad.node.list_types")
            write_json(args.output_dir / "types.json", types)
            for family in ["squareTubeSteel", "hShapedSteel", "gongShapedSteel", "channelSteel"]:
                for kind in ["Girder", "Pillar"]:
                    node_type = f"structure.{family}{kind}"
                    schema = client.describe_type(node_type)
                    write_json(args.output_dir / f"{node_type}.json", schema)
            print("Saved target sessions, supported types and eight live schemas; no model changes")
        else:
            if args.cleared_manifest:
                mark_export_cleared(client, args.cleared_manifest)
                print("Previous export nodes are absent; retained its manifest as cleared", flush=True)
            if args.action == "cabin":
                from scripts.export_geometry import build_default_engineering_model

                settings = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else {}
                mapping = MappingConfig(**{**settings, "trial_mode": True, "use_box_reference_radius": True})
                model = build_default_engineering_model()
                plan = build_export_plan(model, mapping)
                plan["model_parameters"] = {
                    "length_m": model.params.length, "width_m": model.params.width, "height_m": model.params.height,
                    "main_beam_spacing_m": model.params.main_beam_spacing,
                    "support_beam_max_spacing_m": model.params.support_beam_max_spacing,
                }
                write_json(args.output_dir / "cabin_plan.json", plan)
                print(f"Cabin visual trial: {len(plan['members'])} physical steel members; {len(plan['warnings'])} placement warnings", flush=True)
            else:
                plan = trial_plan() if args.action == "trial" else json.loads(args.plan.read_text(encoding="utf-8"))
            manifest = args.output_dir / ("trial_manifest.json" if args.action == "trial" else "export_manifest.json")
            def progress(done, total):
                if done == 1 or done % 25 == 0 or done == total:
                    print(f"Created {done}/{total}", flush=True)

            record = export_plan(plan, client, manifest, on_progress=progress)
            print(f"{record['status']}: {len(record['members'])} members; manifest: {manifest}")
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except (WebCADError, KeyError, ValueError) as exc:
        print(f"WebCAD: {exc}", file=sys.stderr)
        sys.exit(1)
