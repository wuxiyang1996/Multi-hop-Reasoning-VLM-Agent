#!/usr/bin/env python3
"""Freeze the corrected RELATE-only V11 claim before fresh adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.contracts import stable_hash


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen artifact hash: {field}")


def _receipt(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-candidate", type=Path, required=True)
    parser.add_argument("--v10-gate-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v10-harness-code", type=Path, required=True)
    parser.add_argument("--v9-graph-code", type=Path, required=True)
    parser.add_argument("--slot-ledger-code", type=Path, required=True)
    parser.add_argument("--v10-runner-code", type=Path, required=True)
    parser.add_argument("--shared-v9-runner-code", type=Path, required=True)
    parser.add_argument("--shared-v8-runner-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-seed", type=int, default=99701)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V11 candidate: {args.output}")
    v10 = _read(args.v10_candidate)
    _validate_hash(v10, "candidate_sha256")
    if v10.get("schema_version") != (
        "budgeted-executable-source-graph-alfworld-candidate-v10"
    ):
        raise SystemExit("wrong V10 parent candidate")
    if int(v10["experiment_parameters"]["max_steps"]) != 60:
        raise SystemExit("V10 parent did not freeze the 60-step endpoint")
    gate = _read(args.v10_gate_report)
    _validate_hash(gate, "report_sha256")
    if gate.get("status") != "ADAPTATION_GATE_FAILED_STOP":
        raise SystemExit("V11 requires the stopped V10 gate")
    failed = [name for name, value in gate["gates"].items() if not value]
    if failed != ["changed_effects_each_claimed_effect"]:
        raise SystemExit("V10 failed for more than the claim-coverage mismatch")
    authentic = gate["summaries"]["authentic_slot_ir"]
    if (
        authentic["changed_by_effect"]["BIND"] != 0
        or authentic["changed_by_effect"]["RELATE"] < 1
    ):
        raise SystemExit("V10 does not support the RELATE-only correction")
    manifest = _read(args.manifest)
    _validate_hash(manifest, "manifest_sha256")
    manifest_schema = "budgeted-relation-edge-alfworld-manifest-v11"
    manifest_status = "FROZEN_BEFORE_ANY_V11_ADAPTATION_RESET"
    if manifest.get("schema_version") != manifest_schema:
        raise SystemExit("wrong V11 manifest schema")
    if manifest.get("status") != manifest_status:
        raise SystemExit("V11 manifest was not frozen")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V11 selection used target outcomes")
    body = dict(v10)
    parent_hash = str(body.pop("candidate_sha256"))
    body["schema_version"] = (
        "budgeted-relation-edge-alfworld-candidate-v11"
    )
    body["experiment_version"] = "v11"
    body["claim_boundary"] = (
        "RELATE_ONLY_CLAIM_FIXED_FROM_CONSUMED_V10_GATE; SIXTY_STEP_"
        "ENDPOINT_AND_CONTROLS_UNCHANGED; FROZEN_BEFORE_FRESH_V11_"
        "ADAPTATION; CONFIRMATION_FORBIDDEN_UNTIL_GATE_PASSES; "
        "EXISTING_VALID_UNSEEN_FORBIDDEN"
    )
    body["v10_parent_candidate"] = _receipt(args.v10_candidate) | {
        "candidate_sha256": parent_hash
    }
    body["consumed_v10_gate_report"] = _receipt(
        args.v10_gate_report
    ) | {
        "report_sha256": gate["report_sha256"],
        "use_authority": "CLAIM_COVERAGE_FIX_ONLY",
        "only_failed_gate": failed[0],
        "changed_by_effect": authentic["changed_by_effect"],
    }
    body["manifest"] = _receipt(args.manifest) | {
        "manifest_sha256": manifest["manifest_sha256"]
    }
    body["manifest_schema"] = manifest_schema
    body["manifest_status"] = manifest_status
    body["implementation"] = {
        "v10_harness": _receipt(args.v10_harness_code),
        "v9_graph_executor": _receipt(args.v9_graph_code),
        "slot_ledger": _receipt(args.slot_ledger_code),
        "v10_runner": _receipt(args.v10_runner_code),
        "shared_v9_runner": _receipt(args.shared_v9_runner_code),
        "shared_v8_runner_helpers": _receipt(
            args.shared_v8_runner_code
        ),
    }
    body["experiment_parameters"] = dict(body["experiment_parameters"])
    body["experiment_parameters"]["runner_seed"] = args.runner_seed
    body["experiment_parameters"]["claim_fix_authority"] = (
        "ONLY_CHANGED_CLAIMED_EFFECTS_FROM_BIND_AND_RELATE_TO_RELATE"
    )
    body["transfer_scope"] = dict(body["transfer_scope"])
    body["transfer_scope"]["name"] = (
        "BUDGETED_EXECUTED_BIND_TO_RELATE_SOURCE_EDGE"
    )
    body["transfer_scope"]["claimed_changed_effects"] = ["RELATE"]
    body["permissions"] = dict(body["permissions"])
    body["permissions"]["forbidden"] = list(
        body["permissions"]["forbidden"]
    ) + [
        "CHANGE_V10_THRESHOLDS_OR_CONTROLS",
        "CLAIM_CHANGED_BIND_ENTRY_NODE",
    ]
    result = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": result["candidate_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "max_steps": result["experiment_parameters"]["max_steps"],
        "runner_seed": args.runner_seed,
        "claimed_changed_effects": ["RELATE"],
        "confirmation_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
