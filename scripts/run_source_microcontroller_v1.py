#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from motif_transfer.multihorizon_replay import file_hash
from motif_transfer.source_microcontroller import (
    analyze_microcontroller_rows,
    run_microcontroller_snapshot,
    validate_source_microcontroller_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen source event micro-controller h1/h2/h4/h8 forks."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-runtime", required=True, type=Path)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_source_microcontroller_plan(plan)
    source_runtime = args.source_runtime.resolve()
    if not (source_runtime / "env_wrappers/subprocess_env.py").is_file():
        raise SystemExit(f"invalid source runtime: {source_runtime}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    rows_path = args.output / "microcontroller_rows.jsonl"
    manifest_path = args.output / "manifest.json"

    sys.path.insert(0, str(source_runtime))
    # Reuse the already receipt-audited source adapter.  This import does not
    # contact the model endpoint; the v1 micro-controller is deterministic.
    from run_source_multihorizon_replay import SourceForkEnvironment

    maximum_steps = {
        str(row["game"]): int(row["maximum_steps"])
        for row in plan["inputs"]
    }
    manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "plan": str(args.plan.resolve()),
        "plan_file_sha256": file_hash(args.plan),
        "plan_content_sha256": plan["plan_sha256"],
        "source_runtime": str(source_runtime),
        "completed_snapshots": 0,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    all_rows = []
    try:
        with rows_path.open("a", encoding="utf-8") as stream:
            for snapshot in plan["snapshots"]:
                game = str(snapshot["game"])
                rows = run_microcontroller_snapshot(
                    lambda game=game: SourceForkEnvironment(
                        game=game, maximum_steps=maximum_steps[game]
                    ),
                    snapshot=snapshot,
                    branch_map=plan["branch_map"],
                    stall_window=int(plan["stall_window"]),
                )
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                all_rows.extend(rows)
                manifest["completed_snapshots"] += 1
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps({
                    "completed": manifest["completed_snapshots"],
                    "game": game,
                    "snapshots": len(plan["snapshots"]),
                    "status_counts": {
                        status: sum(row["status"] == status for row in rows)
                        for status in sorted({row["status"] for row in rows})
                    },
                }, sort_keys=True), flush=True)
    except Exception as error:
        manifest.update({
            "status": "FAILED_BEFORE_COMPLETE",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    report = analyze_microcontroller_rows(all_rows)
    report.update({
        "plan_content_sha256": plan["plan_sha256"],
        "rows_sha256": file_hash(rows_path),
    })
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update({
        "status": "COMPLETE",
        "rows_file": rows_path.name,
        "rows_sha256": file_hash(rows_path),
        "report_file": report_path.name,
        "report_sha256": file_hash(report_path),
        "gates": report["gates"],
    })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "gates": report["gates"],
        "output": str(args.output.resolve()),
        "selected_snapshots": report["selected_snapshots"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
