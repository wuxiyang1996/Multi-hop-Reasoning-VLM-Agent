#!/usr/bin/env python3
"""Outcome-blind compatibility audit for full source algebra on AGQA Layer B."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_contracts import AGQASemanticSlotReceipt, SemanticSlotNode
from motif_transfer.contracts import stable_hash
from motif_transfer.video_source_applicability import (
    authorize_video_applicability, classify_agqa_family,
)
from motif_transfer.video_target_signature_binding import permuted_algebra


def _receipt(value: dict) -> AGQASemanticSlotReceipt:
    slots = tuple(SemanticSlotNode(
        slot_id=str(row["slot_id"]), kind=str(row["kind"]), surface=str(row["surface"]),
        children=tuple(row.get("children") or ()),
        attributes=tuple(tuple(pair) for pair in row.get("attributes") or ()),
    ) for row in value["slots"])
    receipt = AGQASemanticSlotReceipt(
        task_id=str(value["task_id"]), question_sha256=str(value["question_sha256"]),
        answer_kind=str(value["answer_kind"]), root_slot_id=str(value["root_slot_id"]),
        slots=slots, parser_sha256=str(value["parser_sha256"]),
        parser_training_authority=str(value["parser_training_authority"]),
        functional_program_read_at_runtime=bool(value["functional_program_read_at_runtime"]),
        operator_sequence_emitted=bool(value["operator_sequence_emitted"]),
        answer_read=bool(value["answer_read"]), target_outcome_read=bool(value["target_outcome_read"]),
        receipt_sha256=str(value["receipt_sha256"]),
    )
    receipt.validate(); return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--source-algebra", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("audit is immutable")
    runtime = json.loads(args.semantic_runtime.read_text(encoding="utf-8"))
    algebra = json.loads(args.source_algebra.read_text(encoding="utf-8"))
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    if runtime.get("answer_read") or runtime.get("functional_program_read") or runtime.get("scene_graph_read"):
        raise ValueError("semantic runtime crossed authority boundary")
    permuted = permuted_algebra(algebra); rows = []; counts = Counter()
    for row in runtime["rows"]:
        semantic = _receipt(row["receipt"])
        family = classify_agqa_family(semantic)
        authentic = authorize_video_applicability(
            algebra=algebra, binding_spec=bindings, task_id=semantic.task_id,
            target_domain="agqa", parser_receipt_sha256=semantic.receipt_sha256,
            question_family=family,
        )
        control = authorize_video_applicability(
            algebra=permuted, binding_spec=bindings, task_id=semantic.task_id,
            target_domain="agqa", parser_receipt_sha256=semantic.receipt_sha256,
            question_family=family,
        )
        counts[f"family:{family}"] += 1; counts[f"authentic:{authentic.status}"] += 1
        counts[f"permuted:{control.status}"] += 1
        rows.append({"task_id": semantic.task_id, "family": family,
                     "semantic_receipt_sha256": semantic.receipt_sha256,
                     "authentic": asdict(authentic), "permuted": asdict(control),
                     "target_outcome_read": False})
    gates = {
        "all_semantic_roots_classified": all(row["family"] is not None for row in rows),
        "authentic_full_signature_coverage": all(row["authentic"]["status"] == "AUTHORIZED" for row in rows),
        "matched_permuted_always_abstains": all(row["permuted"]["status"] == "ABSTAINED" for row in rows),
        "no_target_outcome_read": all(not row["target_outcome_read"] for row in rows),
    }
    body = {"schema_version": "agqa-raw-video-full-source-algebra-compatibility-v1",
            "status": "FULL_SOURCE_ALGEBRA_COMPATIBLE" if all(gates.values()) else "COMPATIBILITY_FAILED",
            "semantic_runtime_sha256": runtime["runtime_sha256"],
            "source_algebra_sha256": algebra["artifact_sha256"],
            "bindings_sha256": stable_hash(bindings), "rows": rows,
            "counts": dict(sorted(counts.items())), "gates": gates,
            "answers_read": False, "functional_programs_read": False,
            "claim_boundary": "Applicability/provenance bridge only; does not reuse consumed QA outcomes as new efficacy evidence."}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "rows": len(rows), "counts": body["counts"],
                      "gates": gates, "report_sha256": body["report_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__": raise SystemExit(main())
