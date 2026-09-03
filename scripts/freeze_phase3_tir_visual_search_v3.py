#!/usr/bin/env python3
"""Freeze a new TIR holdout and retain untouched qualification/formal pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_tir_nonmaze import validate_grounder_artifact  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gates(expected: int, ceiling: int, contrast: int) -> dict:
    return {
        "expected_tasks": expected,
        "minimum_ceiling_successes": ceiling,
        "minimum_source_action_contrasts": contrast,
        "minimum_permuted_action_contrasts": contrast,
        "minimum_selected_effect_types": 2,
        "maximum_negative_transfer_rate": 0.0,
        "required_gate_names": [
            "expected_task_count", "target_native_ceiling_capable",
            "source_changes_target_policy",
            "authentic_differs_from_permuted",
            "multiple_source_effect_types_selected",
            "maximum_negative_transfer", "source_not_below_neural",
            "source_strictly_beats_neural",
            "source_strictly_beats_permuted",
            "source_strictly_beats_generic",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path,
        default=REPO / "configs/phase3_tir_visual_search_v2_splits.json",
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=REPO / "runs/phase3_tir_visual_search_v7_frozen/artifact.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/phase3_tir_visual_search_v3_frozen.json",
    )
    parser.add_argument(
        "--diagnostic-reuse-holdout", action="store_true",
        help="Emit an explicitly consumed, development-only diagnostic manifest.",
    )
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text())
    parent_body = dict(parent)
    parent_claimed = str(parent_body.pop("config_sha256", ""))
    if not parent_claimed or stable_hash(parent_body) != parent_claimed:
        raise SystemExit("parent TIR split manifest hash mismatch")
    artifact = json.loads(args.artifact.read_text())
    validate_grounder_artifact(artifact)
    if artifact.get("status") != "DEVELOPMENT_GROUNDER_FROZEN_BEFORE_NEW_HOLDOUT":
        raise SystemExit("TIR grounder is not frozen for a new holdout")

    reserve = list(map(str, parent["splits"]["unopened_reserve"]))
    if len(reserve) < 20:
        raise SystemExit("parent TIR reserve is unexpectedly short")
    if args.diagnostic_reuse_holdout:
        current = REPO / "configs/phase3_tir_visual_search_v3_frozen.json"
        if not current.is_file():
            raise SystemExit("no frozen V3 holdout exists for diagnostic reuse")
        current_payload = json.loads(current.read_text())
        new_holdout = list(map(
            str, current_payload["splits"]["development_holdout"],
        ))
    else:
        new_holdout = reserve[:8]
    # These IDs were content-blind reservations in V2.  No receipt containing
    # them may exist before this manifest is frozen.
    receipt_pattern = '"sample_id": "'
    exposed = []
    for path in (REPO / "runs").rglob("*.json"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if receipt_pattern not in text:
            continue
        if any(f'{receipt_pattern}{sample_id}"' in text for sample_id in new_holdout):
            exposed.append(str(path.relative_to(REPO)))
    if exposed and not args.diagnostic_reuse_holdout:
        raise SystemExit(f"new holdout already has receipts: {exposed[:5]}")

    splits = {
        "development_train": list(map(str, parent["splits"]["development_train"])),
        "development_validation": list(map(
            str, parent["splits"]["development_validation"],
        )),
        "development_holdout": new_holdout,
        "qualification": list(map(str, parent["splits"]["qualification"])),
        "formal": list(map(str, parent["splits"]["formal"])),
        "unopened_reserve": (
            [sample_id for sample_id in reserve if sample_id not in set(new_holdout)]
            if args.diagnostic_reuse_holdout else reserve[8:]
        ),
    }
    flattened = [sample_id for values in splits.values() for sample_id in values]
    if len(flattened) != len(set(flattened)):
        raise SystemExit("TIR V3 split overlap")
    dependencies = (
        "scripts/collect_phase3_tir_visual_search_v3.py",
        "scripts/collect_phase3_tir_visual_search_v2.py",
        "scripts/collect_phase3_tir_nonmaze.py",
        "src/motif_transfer/phase3_tir_nonmaze.py",
        "src/motif_transfer/phase3_attempt_runtime.py",
        "src/motif_transfer/phase3_source_portfolio.py",
        "src/motif_transfer/phase3_typed_effect_induction.py",
        "src/motif_transfer/visual_wrapper_bridge.py",
    )
    body = {
        "schema_version": "phase3-tir-visual-search-manifest-v3",
        "status": (
            "FROZEN_CONSUMED_DEVELOPMENT_DIAGNOSTIC_ONLY"
            if args.diagnostic_reuse_holdout else
            "FROZEN_BEFORE_NEW_DEVELOPMENT_HOLDOUT_QUALIFICATION_FORMAL"
        ),
        "claim_boundary": (
            "PROSPECTIVE_NON_MAZE_TIR_REPLICATION;SAME_SOURCE_INDUCED_IR;"
            "ONLY_TARGET_NATIVE_ACTIVE_VISION_GROUNDER_REPLACED"
        ),
        "parent_config": {
            "path": str(args.parent.relative_to(REPO)),
            "config_sha256": parent["config_sha256"],
            "qualification_or_formal_consumed": False,
        },
        "development_history": {
            "qwen_v4_receipts": "runs/phase3_tir_visual_search_v4",
            "split_grounder_v6_receipts": (
                "runs/phase3_tir_visual_search_v6_split_grounder"
            ),
            "consumed_design_report": (
                "runs/phase3_tir_visual_search_v7_frozen/"
                "consumed_design_report.json"
            ),
            "consumed_design_report_file_sha256": file_sha256(
                REPO / "runs/phase3_tir_visual_search_v7_frozen/"
                "consumed_design_report.json"
            ),
            "new_holdout_used_to_select_protocol": bool(
                args.diagnostic_reuse_holdout
            ),
        },
        "dataset": dict(parent["dataset"]),
        "selection": {
            "rule": (
                "Take the first eight IDs from the already content-blind V2 "
                "unopened-reserve hash order as a new development holdout; "
                "retain the V2 qualification and formal pools byte-for-byte."
            ),
            "prompt_image_or_answer_read_for_new_holdout_selection": False,
            "new_holdout_receipt_files_before_freeze": len(exposed),
            "new_holdout_ids_sha256": stable_hash(new_holdout),
        },
        "splits": splits,
        "conditions": list(parent["conditions"]),
        "source_programs": list(parent["source_programs"]),
        "source_ir": dict(parent["source_ir"]),
        "grounder": {
            "path": str(args.artifact.relative_to(REPO)),
            "file_sha256": file_sha256(args.artifact),
            "artifact_sha256": artifact["artifact_sha256"],
            "thresholds": dict(artifact["thresholds"]),
            "thresholds_frozen_before_new_holdout": True,
            "qualification_or_formal_outcomes_read": False,
        },
        "target_mdp": {
            "state": "LOW_RES_CONTEXT_PLUS_ACCUMULATED_REGION_EVIDENCE",
            "action": "visual_reasoning_wrapper.zoom_region",
            "candidate_operands": "FOUR_TARGET_NEURAL_HYPOTHESIS_REGIONS",
            "schedule": "CONTEXT_H1_TO_LOCAL_H4_TO_NEIGHBORS_H8",
            "transition_horizons": [1, 4, 8],
            "budget": 8,
            "observed_effect": (
                "TARGET_NEURAL_EXECUTED_ENDPOINT_QUALITY_TO_HIGH_OR_LOW"
            ),
            "gold_visible_to_grounder_source_or_runtime": False,
        },
        "wrapper": dict(parent["wrapper"]),
        "models": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "proposal_grounder": "google/gemini-3.7-flash",
            "answer_readout": "qwen/qwen3-vl-32b-instruct",
            "temperature": 0,
        },
        # Execution compatibility: the collector uses this as its default
        # transport model before applying the two content-bound overrides.
        "model": dict(parent["model"]),
        "media": {
            **parent["media"],
            "overview_max_side": 384,
        },
        "development_holdout_gates": _gates(8, 6, 3),
        "qualification_gates": _gates(10, 8, 3),
        "formal_gates": _gates(24, 18, 6),
        "authorization_chain": {
            "qualification_requires": "TIR_PHASE3_DEVELOPMENT_HOLDOUT_PASSED",
            "formal_requires": "TIR_PHASE3_QUALIFICATION_PASSED",
            "fail_closed_in_collector": True,
        },
        "integrity": {
            "code_sha256": {
                path: file_sha256(REPO / path) for path in dependencies
            },
        },
    }
    output = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"],
        "config_sha256": output["config_sha256"],
        "splits": {key: len(value) for key, value in splits.items()},
        "new_holdout_ids_sha256": output["selection"]["new_holdout_ids_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
