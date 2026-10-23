"""Build a local manufacturing-geometry plan and explicitly import it into WebCAD."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_geometry import build_default_engineering_model
from scripts.webcad import load_env
from src.integrations.webcad import WebCADClient, WebCADConfig
from src.integrations.webcad.production import ProductionConfig, build_production_plan
from src.integrations.webcad.production_export import (
    clear_known_tests,
    export_production_plan,
    write_production_files,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-create", action="store_true")
    parser.add_argument(
        "--plan",
        type=Path,
        help="Use a reviewed production plan from the actual optimized model",
    )
    parser.add_argument("--gap-mm", type=float, default=0)
    parser.add_argument("--end-ring-threshold-mm", type=float, default=500)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/webcad/production"
    )
    parser.add_argument("--document-id")
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument(
        "--clear-test-record", "--clear-record", type=Path, action="append", default=[]
    )
    args = parser.parse_args()
    plan = (
        json.loads(args.plan.read_text(encoding="utf-8"))
        if args.plan
        else build_production_plan(
            build_default_engineering_model(),
            ProductionConfig(
                gap_mm=args.gap_mm, end_ring_threshold_mm=args.end_ring_threshold_mm
            ),
        )
    )
    write_production_files(plan, args.output_dir)
    print(json.dumps(plan["summary"], ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_create:
        print("Offline plan only; no platform changes")
        return
    if not args.document_id or not args.ca_bundle:
        parser.error("Creation requires --document-id and --ca-bundle")
    load_env(ROOT / ".env.local")
    os.environ.setdefault("WEBCAD_BASE_URL", "https://124.223.165.124:4006")
    config = WebCADConfig.from_env()
    config.verify_tls = str(args.ca_bundle.resolve())
    config.timeout = 600
    client = WebCADClient(config)
    try:
        if args.clear_test_record:
            clear_known_tests(
                client,
                args.document_id,
                args.clear_test_record,
                args.output_dir / "test_cleanup.json",
            )
            print("Recorded nodes removed; actual saved document is empty", flush=True)
        result = export_production_plan(
            plan,
            client,
            args.output_dir / "export_manifest.json",
            args.document_id,
            on_progress=lambda n, total: print(
                f"Created {n}/{total}; auditing saved BREP...", flush=True
            ),
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "status",
                        "batch_seconds",
                        "total_seconds",
                        "has_backend_save",
                    ]
                }
            ),
            flush=True,
        )
        print(
            f"Verified {result['audit']['verified_parts']} single-solid parts; no envelope overlaps",
            flush=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
