#!/usr/bin/env python3
"""Replay both validated video routes through the shared-grounding contract.

This is a compatibility audit over consumed reports, not a fresh success
experiment.  It proves that CLEVRER and AGQA2 predictions can be represented
under one matched controller/evidence protocol without reopening videos or
calling a provider.  AGQA V62 did not execute source-permuted/generic arms, so
those two audit-only controls are strict fail-closed copies of its frozen
direct prediction and are explicitly not promoted to formal evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.video_transfer_measurement import (  # noqa: E402
    GroundingMode,
    GroundingToolBudget,
    SharedVideoGroundingReceipt,
    VideoTransferDecision,
    evaluate_matched_transfer,
)


ZERO = GroundingToolBudget(0, 0, 0)


def _agqa_answer_equivalent(prediction, gold) -> bool:
    def normalize(value):
        text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
        for prefix in ("the answer is ", "it is ", "they were ", "they are "):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        text = re.sub(r"^(?:a|an|the)\s+", "", text)
        return {"true": "yes", "false": "no"}.get(text, text)

    predicted, expected = normalize(prediction), normalize(gold)
    if expected in {"yes", "no", "before", "after"}:
        return bool(predicted) and predicted.split(maxsplit=1)[0] == expected
    return predicted == expected


def _decision(
    grounding: SharedVideoGroundingReceipt, *, arm: str, prediction: str,
    controller: str, source_program: str | None = None,
) -> VideoTransferDecision:
    return VideoTransferDecision.create(
        grounding=grounding, arm=arm, prediction=prediction,
        controller_sha256=stable_hash(controller),
        source_program_sha256=source_program,
    )


def _clevrer(path: Path):
    report = json.loads(path.read_text())
    source_program = str(report["source_program"]["artifact_sha256"])
    groundings = []
    decisions = []
    gold = {}
    for row in report["rows"]:
        task_id = str(row["sample_id"])
        state = {
            "task_id": task_id,
            "legacy_target_state_sha256": row["unified_authority"]["utility"][
                "target_state_sha256"
            ],
            "proof_receipts_sha256": row["proof_receipts_sha256"],
            "target_receipt_sha256": stable_hash(row["target_receipt"]),
            "outcome_read": False,
        }
        grounding = SharedVideoGroundingReceipt.create(
            benchmark="clevrer", task_id=task_id, split="consumed_formal_replay",
            mode=GroundingMode.MODEL_TOOL_EVENT_GRAPH, state=state,
            evidence_source_sha256=row["proof_receipts_sha256"],
            tool_backend_sha256=stable_hash(report["target_interface"]),
            allowed_tools=("frozen_target_receipt_replay",), tool_budget=ZERO,
        )
        conditions = row["conditions"]
        groundings.append(grounding)
        decisions.extend((
            _decision(
                grounding, arm="neural_only",
                prediction=conditions["neural_only_explicit_relation"]["answer"],
                controller="clevrer-neural-explicit-v15",
            ),
            _decision(
                grounding, arm="source_induced",
                prediction=conditions["authentic_source_induced_goal_relation"]["answer"],
                controller="clevrer-source-induced-v15",
                source_program=source_program,
            ),
            _decision(
                grounding, arm="source_permuted",
                prediction=conditions["source_permuted_uplift"]["answer"],
                controller="clevrer-source-permuted-v15",
                source_program=stable_hash([source_program, "permuted"]),
            ),
            _decision(
                grounding, arm="generic_scaffold",
                prediction=conditions["generic_error_scaffold"]["answer"],
                controller="clevrer-generic-scaffold-v15",
            ),
            _decision(
                grounding, arm="target_native_ceiling",
                prediction=conditions["target_native_representation_ceiling"]["answer"],
                controller="clevrer-target-native-ceiling-v15",
            ),
        ))
        gold[task_id] = str(row["gold_answer_evaluator_only"])
    return groundings, decisions, gold, report


def _agqa(path: Path):
    report = json.loads(path.read_text())
    source_program = stable_hash(report["source_induced_primitives"])
    groundings = []
    decisions = []
    gold = {}
    for row in report["rows"]:
        task_id = str(row["task_id"])
        state = {
            "task_id": task_id,
            "runtime_receipt_sha256": row["runtime_receipt_sha256"],
            "question_sha256": row["question_sha256"],
            "public_plan_sha256": stable_hash(row.get("public_plan")),
            "typed_binding_sha256": stable_hash(row.get("candidate_typed_prediction")),
            "outcome_read": False,
        }
        grounding = SharedVideoGroundingReceipt.create(
            benchmark="agqa2", task_id=task_id, split="consumed_formal_replay",
            mode=GroundingMode.MODEL_TOOL_EVENT_GRAPH, state=state,
            evidence_source_sha256=str(row["video_sha256"]),
            tool_backend_sha256=str(row["grounder_sha256"]),
            allowed_tools=("frozen_target_receipt_replay",), tool_budget=ZERO,
        )
        direct = str(row["direct_response"])
        groundings.append(grounding)
        decisions.extend((
            _decision(
                grounding, arm="neural_only", prediction=direct,
                controller="agqa2-direct-v62",
            ),
            _decision(
                grounding, arm="source_induced",
                prediction=str(row["source_prediction"]),
                controller="agqa2-source-induced-v62",
                source_program=source_program,
            ),
            # These controls were absent in V62.  A malformed/unavailable
            # source route must preserve the already-frozen direct response.
            _decision(
                grounding, arm="source_permuted", prediction=direct,
                controller="agqa2-strict-permuted-fallback-audit-v1",
                source_program=stable_hash([source_program, "permuted"]),
            ),
            _decision(
                grounding, arm="generic_scaffold",
                prediction=str(row.get("candidate_typed_prediction") or direct),
                controller="agqa2-target-written-always-use-typed-candidate-audit-v1",
            ),
        ))
        gold[task_id] = str(row["gold_answer_evaluator_only"])
    return groundings, decisions, gold, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clevrer", type=Path,
        default=REPO / "runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json",
    )
    parser.add_argument(
        "--agqa", type=Path,
        default=REPO / "runs/agqa2_full_distribution_v62/base_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/video_shared_grounding_v1_compatibility.json",
    )
    args = parser.parse_args()
    c_ground, c_decisions, c_gold, c_report = _clevrer(args.clevrer)
    a_ground, a_decisions, a_gold, a_report = _agqa(args.agqa)
    measured = evaluate_matched_transfer(
        groundings=tuple(c_ground + a_ground),
        decisions=tuple(c_decisions + a_decisions),
        gold_answers=c_gold | a_gold,
        answer_equivalence={"agqa2": _agqa_answer_equivalent},
    )
    body = {
        "schema_version": "two-video-shared-grounding-compatibility-v1",
        "status": "TWO_VIDEO_SHARED_GROUNDING_COMPATIBILITY_PASSED",
        "fresh_evidence": False,
        "claim_boundary": (
            "Consumed-report compatibility audit only. It opens no video and "
            "adds no oracle-grounded or end-to-end formal success claim."
        ),
        "benchmarks": ["clevrer", "agqa2"],
        "source_reports": {
            "clevrer_status": c_report["status"],
            "clevrer_report_sha256": c_report["report_sha256"],
            "agqa_status": a_report["status"],
            "agqa_report_sha256": a_report["report_sha256"],
        },
        "matched_measurement": measured,
        "controls": {
            "clevrer": "all controller arms are original V15 formal outputs",
            "agqa2": (
                "source/neural are original V62 outputs; source-permuted is an "
                "audit-only strict direct fallback and generic is the target-written "
                "always-use-typed-candidate policy; neither is new formal evidence"
            ),
        },
        "gates": {
            "both_benchmarks_present": {row["benchmark"] for row in measured["summaries"]}
            == {"clevrer", "agqa2"},
            "all_arms_share_one_grounding_per_task": all(
                row["all_arms_shared_exact_grounding"]
                for row in measured["summaries"]
            ),
            "grounding_modes_not_pooled": measured["grounding_modes_combined"] is False,
            "no_provider_calls": True,
        },
        "next_authoritative_step": (
            "Run a newly frozen answer/program-blind oracle-grounded matched "
            "cohort; keep model/tool grounding as a separately reported track."
        ),
    }
    if not all(body["gates"].values()):
        body["status"] = "TWO_VIDEO_SHARED_GROUNDING_COMPATIBILITY_FAILED"
    result = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "summaries": measured["summaries"],
        "report_sha256": result["report_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
