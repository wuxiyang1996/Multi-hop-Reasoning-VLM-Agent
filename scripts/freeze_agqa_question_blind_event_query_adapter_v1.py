#!/usr/bin/env python3
"""Freeze the deterministic task-query adapter before opening dev outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-protocol", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--event-inventory", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--event-grounder-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("event-query adapter protocol is immutable")
    acquisition = json.loads(args.acquisition_protocol.read_text())
    cohort = json.loads(args.cohort.read_text())
    inventory = json.loads(args.event_inventory.read_text())
    if inventory.get("status") != "QUESTION_BLIND_EVENT_INVENTORY_FROZEN_BEFORE_TASK_QUERY_OR_OUTCOME":
        raise ValueError("event inventory is not frozen before task query")
    if inventory.get("cohort_sha256") != cohort.get("cohort_sha256"):
        raise ValueError("event inventory and cohort differ")
    if any(inventory.get(key) for key in (
        "question_read", "answer_read", "official_scene_graph_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
        "per_video_action_genome_annotation_read",
    )):
        raise ValueError("event inventory crossed its authority boundary")
    body = {
        "schema_version": "agqa-question-blind-event-query-adapter-development-v1",
        "status": "FROZEN_AFTER_EVENT_ACQUISITION_BEFORE_TASK_QUERY_OR_DEVELOPMENT_OUTCOME",
        "claim_boundary": "Consumed AGQA train development only; never transfer evidence.",
        "parent_acquisition_protocol_file_sha256": _sha256(args.acquisition_protocol),
        "candidate_generation_threshold": 0.0,
        "question_blind_event_acquisition": acquisition["question_blind_event_acquisition"],
        "query_adapter": {
            "task_visible_inputs": [
                "public question-derived semantic slots",
                "frozen query temporal scope",
                "frozen question-blind event inventory"
            ],
            "ranking": "maximum provider confidence among exact public-predicate and typed-role events inside the frozen parser scope",
            "generic_interact_query": "fail closed when the public parser supplies no explicit answer-bearing predicate",
            "candidate_count": "one top stable track or abstain",
            "candidate_status_at_generation": "SUPPORTED_AT_ZERO_THRESHOLD",
            "tie_break": ["higher support count", "lexicographic stable track ID"],
            "learned_or_outcome_fitted_parameters": False
        },
        "threshold_selection": {
            "candidate_grid": [
                0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675,
                0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875,
                0.9, 0.925, 0.95, 0.975
            ],
            "objective": "maximum unique-supported coverage",
            "tie_break": "higher threshold",
            "constraints": {
                "unique_supported_count_minimum": acquisition["grounding_gates"]["unique_supported_count_minimum"],
                "unique_supported_precision_wilson_95_lower_bound_minimum": acquisition["grounding_gates"]["unique_supported_precision_wilson_95_lower_bound_minimum"],
                "unique_supported_coverage_minimum": acquisition["grounding_gates"]["unique_supported_coverage_minimum"]
            }
        },
        "grounding_gates": {
            key: value for key, value in acquisition["grounding_gates"].items()
            if key not in {
                "unique_supported_count_minimum",
                "unique_supported_precision_wilson_95_lower_bound_minimum",
                "unique_supported_coverage_minimum",
                "one_global_threshold_required"
            }
        },
        "immutable_inputs": {
            "cohort_sha256": cohort["cohort_sha256"],
            "semantic_runtime_file_sha256": _sha256(args.semantic_runtime),
            "sgdet_file_sha256": _sha256(args.sgdet),
            "query_plans_file_sha256": _sha256(args.query_plans),
            "event_inventory_file_sha256": _sha256(args.event_inventory),
            "event_inventory_report_sha256": inventory["report_sha256"],
            "compiler_file_sha256": _sha256(args.compiler),
            "event_grounder_module_sha256": _sha256(args.event_grounder_module)
        },
        "authority": acquisition["authority"],
        "decision": {
            "if_pass": "Freeze this acquisition and adapter for one new video-disjoint qualification cohort.",
            "if_fail": "Do not allocate qualification; revise only on this consumed development cohort."
        }
    }
    body["protocol_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"],
        "event_inventory_report_sha256": inventory["report_sha256"],
        "protocol_sha256": body["protocol_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
