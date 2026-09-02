#!/usr/bin/env python3
"""Replay frozen CLEVRER V2 decisions through the anonymous source controller.

This is an outcome-blind controller-substitution audit.  It never opens the
formal labels: it checks whether the source-induced arm is exactly reproduced
when its old named routing layer is replaced by the anonymous state-delta
program induced from source-game ledgers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.anonymous_video_harness import route_grounded_candidate
from motif_transfer.contracts import stable_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def audit(controller: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    rows = predictions.get("rows") or []
    mismatches: list[str] = []
    commit_count = fallback_count = 0
    route_counts: dict[str, int] = {}
    for row in rows:
        route = route_grounded_candidate(
            controller, candidate_qualified=bool(row["source_commit"]),
        )
        for operator_id in route[:-1]:
            route_counts[operator_id] = route_counts.get(operator_id, 0) + 1
        disposition = route[-1]
        commit_count += disposition == "COMMIT"
        fallback_count += disposition == "FALLBACK"
        neural = row["predictions"]["neural_only"]
        expected = row["explicit_prediction"] if disposition == "COMMIT" else neural
        reasons = []
        if expected != row["predictions"]["source_induced"]:
            reasons.append("source_prediction")
        if expected != row["predictions"]["target_written_isomorphic"]:
            reasons.append("isomorphic_prediction")
        if neural != row["predictions"]["source_permuted"]:
            reasons.append("permuted_fail_closed")
        if reasons:
            mismatches.append(f"{row['task_id']}:{','.join(reasons)}")

    gates = {
        "controller_source_only_qualified": controller.get("status") == "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED",
        "all_frozen_tasks_replayed": len(rows) == 1600,
        "anonymous_route_total": commit_count + fallback_count == len(rows),
        "source_action_equivalence": not mismatches,
        "isomorphic_action_equivalence": not mismatches,
        "source_permuted_fail_closed_to_shared_fallback": not mismatches,
        "formal_answers_not_read": True,
    }
    body = {
        "schema_version": "clevrer-anonymous-harness-substitution-v1",
        "status": "CLEVRER_ANONYMOUS_HARNESS_SUBSTITUTION_VERIFIED" if all(gates.values()) else "CLEVRER_ANONYMOUS_HARNESS_SUBSTITUTION_FAILED",
        "authority": "FROZEN_PREDICTIONS_ONLY;NO_FORMAL_ANSWERS",
        "answers_read": False,
        "controller_artifact_sha256": controller["artifact_sha256"],
        "prediction_artifact_sha256": predictions["predictions_sha256"],
        "tasks": len(rows),
        "anonymous_commits": commit_count,
        "anonymous_fallbacks": fallback_count,
        "anonymous_operator_route_counts": dict(sorted(route_counts.items())),
        "mismatches": mismatches,
        "gates": gates,
        "claim_boundary": "Replaces controller routing only; target-native VM, grounder, parser, executor, and fallback remain frozen and designer-specified.",
    }
    body["report_sha256"] = stable_hash(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--controller", type=Path, default=Path("runs/anonymous_video_harness_v1/controller.json"))
    parser.add_argument("--predictions", type=Path, default=Path("runs/clevrer_full_raw_video_v2/five_arm_predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    controller_path, predictions_path = resolve(args.controller), resolve(args.predictions)
    result = audit(load(controller_path), load(predictions_path))
    result["controller_file_sha256"] = file_sha256(controller_path)
    result["predictions_file_sha256"] = file_sha256(predictions_path)
    # Recompute after adding file-level receipts.
    result["report_sha256"] = stable_hash({key: value for key, value in result.items() if key != "report_sha256"})
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "tasks", "anonymous_commits", "anonymous_fallbacks", "report_sha256"
    )}, indent=2))
    return 0 if result["status"] == "CLEVRER_ANONYMOUS_HARNESS_SUBSTITUTION_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
