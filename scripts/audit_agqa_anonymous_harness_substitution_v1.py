#!/usr/bin/env python3
"""Verify anonymous-controller substitution on two consumed AGQA Layer-B results.

The inputs contain evaluator labels because they are completed result files.
This audit deliberately never indexes the label/correctness fields, but it is
not a fresh evaluation and the report states that boundary explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from motif_transfer.anonymous_video_harness import route_grounded_candidate
from motif_transfer.contracts import stable_hash


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _commit(row: Mapping[str, Any], kind: str) -> bool:
    if kind == "broad":
        return bool(row["source_open_world_commit"])
    if kind == "temporal":
        return bool(row["source_multiview_commit"])
    raise ValueError(f"unknown AGQA cohort kind: {kind}")


def audit_one(
    controller: Mapping[str, Any], result: Mapping[str, Any], *, kind: str,
) -> dict[str, Any]:
    mismatches: list[str] = []
    commits = fallbacks = 0
    for row in result.get("rows") or ():
        route = route_grounded_candidate(controller, candidate_qualified=_commit(row, kind))
        disposition = route[-1]
        commits += disposition == "COMMIT"
        fallbacks += disposition == "FALLBACK"
        predictions = row["predictions"]
        reasons = []
        if predictions["source_induced"] != predictions["target_written_isomorphic"]:
            reasons.append("isomorphic")
        if disposition == "FALLBACK" and predictions["source_induced"] != predictions["neural_only"]:
            reasons.append("fallback")
        if predictions["source_permuted"] != predictions["neural_only"]:
            reasons.append("permuted")
        if reasons:
            mismatches.append(f"{row['task_id']}:{','.join(reasons)}")
    expected = int((result.get("summaries") or {}).get("source_induced", {}).get("symbolic_commits", -1))
    gates = {
        "anonymous_controller_qualified": controller.get("status") == "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED",
        "commit_count_matches_frozen_result": commits == expected,
        "all_tasks_routed": commits + fallbacks == len(result.get("rows") or ()),
        "source_isomorphic_equivalence": not mismatches,
        "abstention_uses_shared_neural_fallback": not mismatches,
        "matched_permuted_fails_closed": not mismatches,
    }
    return {
        "kind": kind,
        "tasks": len(result.get("rows") or ()),
        "anonymous_commits": commits,
        "anonymous_fallbacks": fallbacks,
        "input_result_report_sha256": result.get("report_sha256"),
        "gates": gates,
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--controller", type=Path, default=Path("runs/anonymous_video_harness_v1/controller.json"))
    parser.add_argument("--broad", type=Path, default=Path("runs/agqa2_layer_b_raw_video_v1/qualification_v4/five_arm_epistemic_qualification_full512_v4.json"))
    parser.add_argument("--temporal", type=Path, default=Path("runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/five_arm_typed_temporal_full256_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/agqa2_layer_b_raw_video_v1/anonymous_harness_substitution_v1.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    controller_path, broad_path, temporal_path = map(resolve, (args.controller, args.broad, args.temporal))
    controller = _load(controller_path)
    cohorts = [
        audit_one(controller, _load(broad_path), kind="broad"),
        audit_one(controller, _load(temporal_path), kind="temporal"),
    ]
    gates = {
        "both_existing_replications_equivalent": all(all(row["gates"].values()) for row in cohorts),
        "fresh_official_test_claim_forbidden": True,
    }
    body = {
        "schema_version": "agqa-anonymous-harness-substitution-v1",
        "status": "AGQA_EXISTING_LAYER_B_ANONYMOUS_SUBSTITUTION_VERIFIED" if all(gates.values()) else "AGQA_EXISTING_LAYER_B_ANONYMOUS_SUBSTITUTION_FAILED",
        "controller_artifact_sha256": controller["artifact_sha256"],
        "controller_file_sha256": _sha(controller_path),
        "input_file_sha256s": {"broad": _sha(broad_path), "temporal": _sha(temporal_path)},
        "cohorts": cohorts,
        "gates": gates,
        "fresh_evaluation": False,
        "claim_boundary": "Controller-substitution audit over two already-consumed raw-video results; not a new untouched AGQA official-test formal.",
    }
    body["report_sha256"] = stable_hash(body)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": body["status"], "cohorts": cohorts,
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if body["status"] == "AGQA_EXISTING_LAYER_B_ANONYMOUS_SUBSTITUTION_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
