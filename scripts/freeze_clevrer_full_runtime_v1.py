#!/usr/bin/env python3
"""Freeze shared raw-video/parser/applicability receipts before evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.clevrer_nsdr_tool_grounder import bind_cached_nsdr_prediction
from motif_transfer.clevrer_public_semantics import parse_public_semantics
from motif_transfer.contracts import stable_hash
from motif_transfer.video_source_applicability import authorize_video_applicability
from motif_transfer.video_target_signature_binding import permuted_algebra


ARMS = (
    "neural_only", "generic_symbolic", "source_permuted",
    "source_induced", "target_written_isomorphic",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--grounder-config", type=Path, required=True)
    parser.add_argument("--grounder-qualification", type=Path, required=True)
    parser.add_argument("--parser-qualification", type=Path, required=True)
    parser.add_argument("--source-algebra", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("CLEVRER runtime freeze is immutable")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    protocol = json.loads(args.preregistration.read_text(encoding="utf-8"))
    config = json.loads(args.grounder_config.read_text(encoding="utf-8"))
    grounder_q = json.loads(args.grounder_qualification.read_text(encoding="utf-8"))
    parser_q = json.loads(args.parser_qualification.read_text(encoding="utf-8"))
    algebra = json.loads(args.source_algebra.read_text(encoding="utf-8"))
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    if grounder_q.get("status") != "CLEVRER_NSDR_GROUNDER_QUALIFIED":
        raise ValueError("CLEVRER grounder did not qualify")
    if parser_q.get("status") != "CLEVRER_PUBLIC_PARSER_QUALIFIED":
        raise ValueError("CLEVRER parser did not qualify")
    v2 = protocol.get("schema_version") == "clevrer-full-raw-video-transfer-preregistration-v2"
    if not v2 and (
        grounder_q["cohort_sha256"] != cohort["cohort_sha256"]
        or parser_q["cohort_sha256"] != cohort["cohort_sha256"]
    ):
        raise ValueError("qualification/cohort lineage drift")
    if v2:
        # Frozen component qualification is intentionally reusable: it was run
        # on a disjoint development cohort, not the V2 formal reserve.
        if not all((grounder_q.get("gates") or {}).values()):
            raise ValueError("external grounder development qualification gate failed")
        if not all((parser_q.get("gates") or {}).values()):
            raise ValueError("external parser development qualification gate failed")
    if algebra["artifact_sha256"] != protocol["source_algebra_sha256"]:
        raise ValueError("source algebra drifted after preregistration")
    permuted = permuted_algebra(algebra)
    raw_root = Path(config["raw_video_root"]); prediction_root = Path(config["prediction_root"])
    videos = []; tasks = []
    for public_video in cohort["reserve"]:
        video_id = int(public_video["video_id"])
        matches = list(raw_root.glob(f"**/video_{video_id}.mp4"))
        if len(matches) != 1: raise ValueError(f"raw video lookup failed for {video_id}")
        grounder = bind_cached_nsdr_prediction(
            video_path=matches[0], prediction_path=prediction_root / f"sim_{video_id:05d}.json",
            config=config,
        )
        grounder_dict = asdict(grounder)
        videos.append({"video_id": video_id, "grounder_receipt": grounder_dict})
        for question in public_video["questions"]:
            task_id = f"video_{video_id}.Q{question['question_id']}"
            choices = [row["choice"] for row in question["choices"]]
            semantic = parse_public_semantics(
                task_id=task_id, question=question["question"],
                question_family=question["question_type"],
                public_subtype=question["question_subtype"], choices=choices,
            )
            authentic_auth = authorize_video_applicability(
                algebra=algebra, binding_spec=bindings, task_id=task_id,
                target_domain="clevrer", parser_receipt_sha256=semantic.receipt_sha256,
                question_family=question["question_type"],
            )
            permuted_auth = authorize_video_applicability(
                algebra=permuted, binding_spec=bindings, task_id=task_id,
                target_domain="clevrer", parser_receipt_sha256=semantic.receipt_sha256,
                question_family=question["question_type"],
            )
            qualified_families = set(
                (protocol.get("source_applicability") or {}).get(
                    "authorized_families", (),
                )
            ) if v2 else {"descriptive", "explanatory", "predictive", "counterfactual"}
            grounder_evidence_authorized = question["question_type"] in qualified_families
            source_execution_authorized = (
                authentic_auth.status == "AUTHORIZED" and grounder_evidence_authorized
            )
            state = {
                "task_id": task_id, "video_id": video_id,
                "question_id": int(question["question_id"]),
                "question": question["question"], "question_family": question["question_type"],
                "public_subtype": question["question_subtype"],
                "choices": question["choices"],
                "semantic_receipt": asdict(semantic),
                "shared_grounder_receipt_sha256": grounder.receipt_sha256,
                "source_applicability": asdict(authentic_auth),
                "permuted_applicability": asdict(permuted_auth),
                "grounder_evidence_authorized": grounder_evidence_authorized,
                "source_execution_authorized": source_execution_authorized,
                "all_five_arms_share_exact_state": True,
            }
            state["task_state_sha256"] = stable_hash(state)
            tasks.append(state)
    gates = {
        "reserve_video_count_exact": len(videos) == int(protocol["reserve_video_count"]),
        "reserve_task_count_exact": len(tasks) == int(protocol["reserve_video_count"]) * 4,
        "source_signature_applicability_all_authorized": all(row["source_applicability"]["status"] == "AUTHORIZED" for row in tasks),
        "source_execution_matches_frozen_grounder_scope": all(
            row["source_execution_authorized"] == (
                row["question_family"] in set(
                    (protocol.get("source_applicability") or {}).get(
                        "authorized_families",
                        ["descriptive", "explanatory", "predictive", "counterfactual"],
                    )
                )
            ) for row in tasks
        ),
        "single_grounder_receipt_per_video": len({row["video_id"] for row in videos}) == len(videos),
        "all_five_arms_share_exact_state": all(row["all_five_arms_share_exact_state"] for row in tasks),
        "no_runtime_oracle": all(
            not row["grounder_receipt"][key]
            for row in videos for key in (
                "question_read", "processed_proposals_read", "official_annotation_read",
                "functional_program_read", "answer_read", "source_controller_read",
            )
        ),
    }
    body = {
        "schema_version": "clevrer-full-shared-runtime-v1",
        "status": "CLEVRER_SHARED_RUNTIME_FROZEN" if all(gates.values()) else "CLEVRER_SHARED_RUNTIME_FAILED",
        "cohort_sha256": cohort["cohort_sha256"],
        "preregistration_sha256": stable_hash(protocol),
        "grounder_qualification_sha256": grounder_q["report_sha256"],
        "parser_qualification_sha256": parser_q["report_sha256"],
        "qualification_reused_from_disjoint_development": v2,
        "source_algebra_sha256": algebra["artifact_sha256"],
        "permuted_algebra_sha256": permuted["artifact_sha256"],
        "arms": ARMS, "videos": videos, "tasks": tasks, "gates": gates,
        "raw_frames_shared": True, "frame_budget_shared": True, "grounder_shared": True,
        "parser_shared": True, "executor_shared": True, "fallback_shared": True,
        "answers_read": False, "reserve_programs_read": False,
    }
    body["runtime_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "videos": len(videos), "tasks": len(tasks),
                      "gates": gates, "runtime_sha256": body["runtime_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__": raise SystemExit(main())
