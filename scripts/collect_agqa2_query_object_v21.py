#!/usr/bin/env python3
"""Run cross-model 2-of-3 QUERY_OBJECT consensus without changing V20."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_query_object_consensus import (  # noqa: E402
    calibrate_query_object_consensus,
)
from motif_transfer.agqa_query_object_grounder import atomic_query_object_plan  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_query_object_v20 as v20  # noqa: E402


def _collect_query_object_runtime_v21(
    sample: Mapping[str, Any], *, config: Mapping[str, Any], api_key: str,
    sources: Sequence[Any], grounder_sha256: str, cache_root: Path,
) -> dict[str, Any]:
    question = str(sample["question"])
    plan = parse_public_question_plan(question)
    if plan is None or not atomic_query_object_plan(plan):
        raise ValueError("sample is not an atomic QUERY_OBJECT public question")
    spec = config["query_object_grounder"]
    receipts, attempts, video_metadata = [], [], []
    for view_name, model in (
        ("ontology_primary", spec["model"]),
        ("ontology_secondary", spec["secondary_model"]),
    ):
        receipt, view_attempts, metadata = v20._ontology_call(
            plan=plan, video_path=Path(sample["video_path"]), model=model,
            media=config["media"], api_key=api_key,
            cache_dir=cache_root / str(sample["task_id"]),
        )
        receipts.append(receipt)
        attempts.extend(dict(row) | {"view": view_name} for row in view_attempts)
        video_metadata.append(metadata)

    # The matched direct call remains inside the base runtime and starts only
    # after both ontology receipts and the isolated-relation receipt are frozen.
    base = v20._collect_runtime(
        sample, config=config, api_key=api_key, sources=sources,
        grounder_sha256=grounder_sha256, cache_root=cache_root,
    )
    if base["query_plan"]["comparison"] != "QUERY_OBJECT":
        raise ValueError("base runtime changed the QUERY_OBJECT route")
    calibration = config["query_object_calibration"]
    calibrated = calibrate_query_object_consensus(
        base_decision=base["target_native_execution"]["decision"],
        direct_response=base["direct_response"], ontology_receipts=receipts,
        minimum_confidences=calibration["minimum_ontology_confidences"],
        minimum_neural_votes=int(calibration["minimum_neural_votes"]),
    )
    body = deepcopy(base)
    body.pop("runtime_receipt_sha256", None)
    body.update({
        "object_ontology_receipt": receipts[0].as_dict(),
        "object_ontology_receipts": [row.as_dict() for row in receipts],
        "object_ontology_attempts": attempts,
        "object_ontology_video_metadata": video_metadata,
        "object_ontology_call_started_before_direct": True,
        "object_ontology_original_question_read": False,
        "object_ontology_answer_candidates_read": False,
        "calibrated_target_native_execution": calibrated,
        "calibration_started_after_typed_and_direct_receipts_froze": True,
        "grounder_sha256": grounder_sha256,
    })
    return body | {"runtime_receipt_sha256": stable_hash(body)}


def collect(**kwargs) -> dict[str, Any]:
    config_path = Path(kwargs["config_path"])
    config = json.loads(config_path.read_text())
    for label in ("consensus_module", "parent_query_object_collector"):
        path = REPO_ROOT / config["query_object_grounder"][label]
        expected = config["query_object_grounder"][f"{label}_sha256"]
        if v20._sha256(path) != expected:
            raise ValueError(f"QUERY_OBJECT V21 {label} hash mismatch")
    original = v20._collect_query_object_runtime
    v20._collect_query_object_runtime = _collect_query_object_runtime_v21
    try:
        result = v20.collect(**kwargs)
    finally:
        v20._collect_query_object_runtime = original
    body = deepcopy(result)
    body.pop("report_sha256", None)
    body.update({
        "schema_version": "agqa2-query-object-consensus-report-v21",
        "status": result["status"].replace("V20", "V21"),
        "object_ontology_models": [
            config["query_object_grounder"]["model"]["id"],
            config["query_object_grounder"]["secondary_model"]["id"],
        ],
    })
    body.pop("object_ontology_model", None)
    final = body | {"report_sha256": stable_hash(body)}
    output_path = Path(kwargs["output_path"])
    output_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", type=Path,
                        default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "metrics", "controls", "qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
