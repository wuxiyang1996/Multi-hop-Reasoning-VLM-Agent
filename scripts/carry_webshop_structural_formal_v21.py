#!/usr/bin/env python3
"""Carry an unopened V20 formal reserve into the qualified V21 runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_semantic_reserve import require_semantic_reserve  # noqa: E402


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path,
        default=REPO / "configs/webshop_structural_v20_frozen.json",
    )
    parser.add_argument(
        "--qualification-manifest", type=Path,
        default=REPO / "configs/webshop_structural_v21_qualification_frozen.json",
    )
    parser.add_argument(
        "--qualification-report", type=Path,
        default=REPO / "runs/webshop_structural_transfer_v21_qualification/report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/webshop_structural_v21_formal_frozen.json",
    )
    args = parser.parse_args()

    source = _read(args.source_manifest)
    qualification_manifest = _read(args.qualification_manifest)
    qualification = _read(args.qualification_report)
    if source.get("status") != "FROZEN_BEFORE_ANY_V20_PROVIDER_CALL_OR_OUTCOME":
        raise SystemExit("V20 source reserve is not frozen")
    if source.get("formal_outcomes_read_or_run") is not False:
        raise SystemExit("V20 formal reserve was already opened")
    if qualification_manifest.get("status") != (
        "FROZEN_BEFORE_ANY_V21_PROVIDER_CALL_OR_OUTCOME"
    ):
        raise SystemExit("V21 qualification manifest is not frozen")
    if qualification.get("status") != "V21_TRANSPORT_QUALIFICATION_PASSED":
        raise SystemExit("V21 transport qualification did not pass")
    if not all((qualification.get("gates") or {}).values()):
        raise SystemExit("V21 qualification gates are incomplete")
    expected_qualification_tasks = [
        row["task_id"]
        for row in qualification_manifest["roles"]["transport_qualification"]
    ]
    if qualification.get("tasks") != expected_qualification_tasks:
        raise SystemExit("V21 qualification report/manifest task mismatch")

    formal = list(source["roles"]["formal_reserve"])
    if len(formal) != 32:
        raise SystemExit("carry-forward formal reserve must contain 32 tasks")
    qualification_rows = list(
        qualification_manifest["roles"]["transport_qualification"]
    )
    audit = require_semantic_reserve(
        formal,
        consumed_rows=qualification_rows,
        required_unique_goals=32,
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )
    body = {
        "schema_version": "webshop-structural-v21-reserve-v1",
        "artifact_role": "V21_QUALIFIED_CARRY_FORWARD_V20_UNOPENED_FORMAL",
        "status": "FROZEN_BEFORE_ANY_V21_PROVIDER_CALL_OR_OUTCOME",
        "claim_boundary": (
            "The 32 formal goal snapshots were frozen in V20 and never run. "
            "V21 code changes used only V20 qualification outcomes; a disjoint "
            "fresh V21 qualification passed before this carry-forward artifact."
        ),
        "goal_seed": source["goal_seed"],
        "server_goal_count": source["server_goal_count"],
        "number_of_registered_tasks_required": (
            source["number_of_registered_tasks_required"]
        ),
        "roles": {"transport_qualification": [], "formal_reserve": formal},
        "preflight": {
            "formal_vs_v21_qualification": audit,
            "source_v20_manifest_formal_unopened": True,
            "v21_qualification_all_gates_passed": True,
        },
        "carry_forward_lineage": {
            "source_manifest_artifact_sha256": source["artifact_sha256"],
            "source_manifest_file_sha256": file_sha256(args.source_manifest),
            "qualification_manifest_artifact_sha256": (
                qualification_manifest["artifact_sha256"]
            ),
            "qualification_manifest_file_sha256": file_sha256(
                args.qualification_manifest
            ),
            "qualification_report_sha256": qualification["report_sha256"],
            "qualification_report_file_sha256": file_sha256(
                args.qualification_report
            ),
            "formal_task_goal_hashes": [
                row["goal_sha256"] for row in formal
            ],
        },
        "transport_contract": source["transport_contract"],
        "runtime_hashes": {
            "freezer": file_sha256(Path(__file__)),
            "server_launcher": file_sha256(REPO / (
                "scripts/run_webshop_structural_server_v18.py"
            )),
            "transport_adapter": file_sha256(REPO / (
                "src/motif_transfer/webshop_deterministic_transport_v18.py"
            )),
            "frozen_goal_adapter": file_sha256(REPO / (
                "src/motif_transfer/webshop_frozen_goal_transport.py"
            )),
        },
        "formal_outcomes_read_or_run": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": artifact["status"],
        "formal_tasks": len(formal),
        "goal_seed": artifact["goal_seed"],
        "semantic_audit_passed": audit["passed"],
        "formal_outcomes_read_or_run": False,
        "artifact_sha256": artifact["artifact_sha256"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
