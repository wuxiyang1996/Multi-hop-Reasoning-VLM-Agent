#!/usr/bin/env python3
"""Adapt frozen V2 typed-role receipts to the unchanged Layer-B VM contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_contracts import LayerBTaskStateReceipt
from motif_transfer.agqa_query_grounder_v2 import (
    adapt_query_grounding_v2,
    query_grounding_v2_from_dict,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--query-grounding", type=Path, required=True)
    parser.add_argument("--minimum-candidate-confidence", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Layer-B adapter output is immutable")

    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    query = json.loads(args.query_grounding.read_text())
    if query.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("V2 query grounding is not frozen")
    if len({cohort["cohort_sha256"], runtime["cohort_sha256"], query["cohort_sha256"]}) != 1:
        raise ValueError("cohort, semantics, and V2 grounding differ")
    if any(query.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("V2 query grounding crossed the authority boundary")
    semantic_by_task = {
        str(row["task_id"]): _semantic(row["receipt"])
        for row in runtime["rows"] if row.get("receipt") is not None
    }
    public_by_task = {str(row["task_id"]): row for row in cohort["rows"]}
    outputs = []
    for row in query["rows"]:
        task_id = str(row["task_id"])
        if task_id not in semantic_by_task or task_id not in public_by_task:
            raise ValueError(f"V2 task {task_id} lacks public semantics")
        semantic = semantic_by_task[task_id]
        v2_receipt = query_grounding_v2_from_dict(row["receipt"])
        grounding = adapt_query_grounding_v2(
            v2_receipt,
            semantic,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
        )
        state = LayerBTaskStateReceipt.create(semantic, grounding)
        outputs.append({
            "cohort_position": int(row["cohort_position"]),
            "task_id": task_id,
            "video_id": str(public_by_task[task_id]["video_id"]),
            "semantic_receipt": asdict(semantic),
            "grounding_receipt": asdict(grounding),
            "task_state_receipt": asdict(state),
            "query_grounding_v2_receipt_sha256": v2_receipt.receipt_sha256,
        })
    outputs.sort(key=lambda row: row["cohort_position"])
    expected = [str(row["task_id"]) for row in cohort["rows"]]
    if [row["task_id"] for row in outputs] != expected:
        raise ValueError("V2 adapter does not exactly cover the frozen cohort in order")
    body = {
        "schema_version": "agqa-query-grounder-v2-layer-b-adapter-v1",
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES",
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "query_grounding_report_sha256": query["report_sha256"],
        "minimum_candidate_confidence": args.minimum_candidate_confidence,
        "rows": outputs,
        "all_harness_arms_share_exact_receipts": True,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "provider_calls": 0,
        "inputs": {
            "cohort_sha256": _sha256(args.cohort),
            "semantic_runtime_sha256": _sha256(args.semantic_runtime),
            "query_grounding_sha256": _sha256(args.query_grounding),
        },
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"],
        "rows": len(outputs),
        "symbolically_grounded_rows": sum(bool(row["grounding_receipt"]["events"]) for row in outputs),
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
