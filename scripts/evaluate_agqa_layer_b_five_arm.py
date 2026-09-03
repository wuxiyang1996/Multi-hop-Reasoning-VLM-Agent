#!/usr/bin/env python3
"""Evaluate five matched Harness arms only after all Layer-B runtime artifacts freeze."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import re
import zipfile

from motif_transfer.agqa_layer_b_contracts import (
    AGQASemanticSlotReceipt, GroundedEvent, RawVideoEventGraphReceipt,
    SemanticSlotNode,
)
from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import ARMS, plan_harness_arm
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


def _semantic(raw: dict) -> AGQASemanticSlotReceipt:
    slots = tuple(SemanticSlotNode(
        slot_id=str(slot["slot_id"]), kind=str(slot["kind"]), surface=str(slot["surface"]),
        children=tuple(slot.get("children", ())),
        attributes=tuple(tuple(pair) for pair in slot.get("attributes", ())),
    ) for slot in raw["slots"])
    value = AGQASemanticSlotReceipt(**{**raw, "slots": slots}); value.validate(); return value


def _grounding(raw: dict) -> RawVideoEventGraphReceipt:
    events = tuple(GroundedEvent(
        event_id=str(event["event_id"]), subject=str(event["subject"]),
        predicate=str(event["predicate"]), object=str(event["object"]),
        start_frame=int(event["start_frame"]), end_frame=int(event["end_frame"]),
        evidence_frames=tuple(event["evidence_frames"]), confidence=float(event["confidence"]),
        semantic_slot_ids=tuple(event.get("semantic_slot_ids", ())),
    ) for event in raw["events"])
    value = RawVideoEventGraphReceipt(**{
        **raw, "events": events,
        "selected_frame_indices": tuple(raw["selected_frame_indices"]),
        "selected_frame_sha256s": tuple(raw["selected_frame_sha256s"]),
    }); value.validate(); return value


def _normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    for prefix in ("the answer is ", "it is ", "they were ", "they are "):
        if text.startswith(prefix): text = text[len(prefix):].strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return {"true": "yes", "false": "no"}.get(text, text)


def _matches(prediction: object, gold: object) -> bool:
    predicted, expected = _normalize(prediction), _normalize(gold)
    if expected in {"yes", "no", "before", "after"}:
        return bool(predicted) and predicted.split(maxsplit=1)[0] == expected
    return predicted == expected


def _mcnemar(left: list[bool], right: list[bool]) -> dict[str, object]:
    wins = sum(a and not b for a, b in zip(left, right))
    losses = sum(b and not a for a, b in zip(left, right)); n = wins + losses
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n) if n else 1.0
    return {"wins": wins, "losses": losses, "discordant": n,
            "exact_two_sided_p": min(1.0, 2 * tail)}


def _gold_rows(archive: Path, entry: str, wanted: set[str]) -> dict[str, dict]:
    output = {}
    with zipfile.ZipFile(archive) as bundle, bundle.open(entry) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                if task_id in wanted:
                    output[task_id] = row
                    if len(output) == len(wanted): break
    if set(output) != wanted:
        raise ValueError(f"official evaluator rows missing: {sorted(wanted-set(output))[:5]}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "qualification", "formal"), required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Layer-B evaluation is immutable")
    cohort = json.loads(args.cohort.read_text())
    grounding_report = json.loads(args.grounding.read_text())
    fallback_report = json.loads(args.fallback.read_text())
    source = json.loads(args.source_capabilities.read_text())
    if grounding_report["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("grounding was not frozen before evaluation")
    if fallback_report["status"] != "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("fallback was not frozen before evaluation")
    if len({cohort["cohort_sha256"], grounding_report["cohort_sha256"],
            fallback_report["cohort_sha256"]}) != 1:
        raise ValueError("five-arm evaluator inputs refer to different cohorts")
    if fallback_report["grounding_report_sha256"] != grounding_report["report_sha256"]:
        raise ValueError("fallback did not consume the frozen shared grounding")
    if not grounding_report["all_harness_arms_share_exact_receipts"] or not fallback_report["shared_by_all_five_arms"]:
        raise ValueError("matched-arm invariants were not frozen")
    forbidden = ("answer_read", "official_scene_graph_read", "functional_program_read", "source_controller_read")
    if any(grounding_report.get(key) for key in forbidden) or any(fallback_report.get(key) for key in forbidden):
        raise ValueError("runtime artifact crossed an authority boundary")
    fallback = {str(row["task_id"]): str(row["prediction"]) for row in fallback_report["rows"]}
    semantic_runtime = json.loads((args.cohort.parent / "semantic_runtime.json").read_text())
    compact_by_task = {
        str(row["task_id"]): str(row["predicted_semantics"])
        for row in semantic_runtime["rows"]
    }
    wanted = {str(row["task_id"]) for row in cohort["rows"]}
    # This is the first outcome-bearing operation. It happens after parser,
    # frames, event graphs, fallback predictions, and all receipts froze.
    evaluator = _gold_rows(args.archive, args.entry, wanted)
    all_ops = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(edge) for edge in source["authorized_compositions"])
    rows = []
    for raw in grounding_report["rows"]:
        task_id = str(raw["task_id"]); semantic = _semantic(raw["semantic_receipt"])
        grounding = _grounding(raw["grounding_receipt"])
        compact = compact_by_task[task_id]
        plans = {arm: plan_harness_arm(
            semantic, arm=arm, source_capabilities=source, all_vm_operators=all_ops,
        ) for arm in ARMS}
        strict_execution = execute_layer_b_semantics(
            compact_semantics=compact, grounding=grounding, semantic=semantic,
            authorized_operators=all_ops, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        eager_execution = execute_layer_b_semantics(
            compact_semantics=compact, grounding=grounding, semantic=semantic,
            authorized_operators=all_ops, authorized_compositions=None,
            ambiguity_policy="EAGER",
        )
        predictions = {}
        for arm in ARMS:
            execution = eager_execution if arm == "generic_scaffold" else strict_execution
            symbolic = arm in {"generic_scaffold", "source_induced", "target_written_isomorphic"}
            predictions[arm] = (str(execution.receipt.prediction)
                if symbolic and plans[arm].status == "PLANNED" and execution.receipt.status == "COMMITTED"
                else fallback[task_id])
        gold = str(evaluator[task_id]["answer"])
        rows.append({
            "task_id": task_id, "video_id": raw["video_id"], "gold_answer_evaluator_only": gold,
            "fallback_prediction": fallback[task_id],
            "strict_symbolic_execution": asdict(strict_execution.receipt),
            "generic_eager_execution": asdict(eager_execution.receipt),
            "plans": {arm: asdict(plan) for arm, plan in plans.items()},
            "predictions": predictions,
            "correct": {arm: _matches(value, gold) for arm, value in predictions.items()},
            "official_answer_first_read_after_all_runtime_artifacts_froze": True,
        })
    correct = {arm: [row["correct"][arm] for row in rows] for arm in ARMS}
    n = len(rows)
    summaries = {arm: {
        "correct": sum(correct[arm]), "total": n, "accuracy": sum(correct[arm]) / n,
        "symbolic_commits": sum(
            arm in {"generic_scaffold", "source_induced", "target_written_isomorphic"}
            and (row["generic_eager_execution"]["status"] if arm == "generic_scaffold"
                 else row["strict_symbolic_execution"]["status"]) == "COMMITTED" for row in rows
        ),
    } for arm in ARMS}
    comparisons = {baseline: _mcnemar(correct["source_induced"], correct[baseline])
                   for baseline in ("neural_only", "generic_scaffold", "source_permuted")}
    source_target_equivalent = all(
        row["predictions"]["source_induced"] == row["predictions"]["target_written_isomorphic"]
        for row in rows
    )
    gates = {
        "source_beats_neural": summaries["source_induced"]["correct"] > summaries["neural_only"]["correct"],
        "source_beats_generic": summaries["source_induced"]["correct"] > summaries["generic_scaffold"]["correct"],
        "source_vs_neural_significant": comparisons["neural_only"]["exact_two_sided_p"] < .05,
        "negative_transfer_losses_at_most_five_percent": comparisons["neural_only"]["losses"] <= math.floor(.05*n),
        "source_permuted_not_better_than_source": summaries["source_permuted"]["correct"] <= summaries["source_induced"]["correct"],
        "target_written_isomorphic_action_equivalence": source_target_equivalent,
    }
    body = {
        "schema_version": "agqa-layer-b-five-arm-evaluation-v1", "stage": args.stage,
        "status": "LAYER_B_GATES_PASSED" if all(gates.values()) else "LAYER_B_GATES_FAILED",
        "cohort_sha256": cohort["cohort_sha256"], "grounding_report_sha256": grounding_report["report_sha256"],
        "fallback_report_sha256": fallback_report["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "rows": rows, "summaries": summaries, "comparisons": comparisons, "gates": gates,
        "frames_shared": True, "frame_budget_shared": True, "grounder_shared": True,
        "parser_shared": True, "executor_shared": True, "fallback_shared": True,
        "raw_video_end_to_end_only": True, "official_scene_graph_used_at_runtime": False,
        "official_answers_read_only_by_post_freeze_evaluator": True,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": body["status"], "summaries": summaries,
                      "comparisons": comparisons, "gates": gates,
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
