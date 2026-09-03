#!/usr/bin/env python3
"""Freeze fresh TIR visual-search splits after the color dev gate failed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _load_v1_freezer():
    path = REPO / "scripts/freeze_phase3_tir_nonmaze_v1.py"
    spec = importlib.util.spec_from_file_location("phase3_tir_freeze_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load TIR exposure-audit base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_v1_freezer()
AUDITED_IDS = ("22", "42")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/TIR-Bench"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/phase3_tir_visual_search_v2_splits.json",
    )
    args = parser.parse_args()
    dataset_file = args.dataset_root / "TIR-Bench.json"
    rows = json.loads(dataset_file.read_text())
    known_ids = {str(row["id"]) for row in rows}
    roots = [
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-experiment-clean"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"),
    ]
    historical = BASE._historical_tir_reservations(
        roots, known_ids,
        ignored_path_substrings=("phase3_tir_visual_search_v2",),
    )
    # Answer value is never used for ranking or allocation.  Schema eligibility
    # is necessary because the common evaluator is exact A--F classification.
    eligible = [
        str(row["id"]) for row in rows
        if row.get("task") == "visual_search" and not row.get("image_2")
        and str(row.get("answer")) in tuple("ABCDEF")
    ]
    if len(eligible) != 78:
        raise SystemExit(f"unexpected visual_search A--F population: {len(eligible)}")
    eligible_set = set(eligible)
    historical_development = sorted(
        (set(historical) | set(AUDITED_IDS)) & eligible_set,
        key=lambda sample_id: hashlib.sha256(
            f"phase3-tir-visual-search-v2-dev\0{sample_id}".encode()
        ).hexdigest(),
    )
    remaining = sorted(
        eligible_set - set(historical_development),
        key=lambda sample_id: hashlib.sha256(
            f"phase3-tir-visual-search-v2-fresh\0{sample_id}".encode()
        ).hexdigest(),
    )
    needed_development = max(0, 24 - len(historical_development))
    development = [*historical_development, *remaining[:needed_development]]
    fresh = remaining[needed_development:]
    if len(development) < 24 or len(fresh) < 34:
        raise SystemExit("insufficient disjoint visual_search population")
    development = development[:24]
    splits = {
        "development_train": development[:16],
        "development_validation": development[16:24],
        "qualification": fresh[:10],
        "formal": fresh[10:34],
        "unopened_reserve": fresh[34:],
    }
    flattened = [sample_id for values in splits.values() for sample_id in values]
    if len(flattened) != len(set(flattened)):
        raise SystemExit("TIR V2 splits overlap")
    source_programs = []
    source_dir = REPO / "configs/phase3_source_induction_v3/frozen_reserve/programs"
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        source_programs.append({
            "path": str(path.relative_to(REPO)),
            "file_sha256": BASE.file_sha256(path),
            "artifact_sha256": payload["artifact_sha256"],
        })
    body = {
        "schema_version": "phase3-tir-visual-search-split-manifest-v2",
        "status": "FROZEN_BEFORE_ANY_TIR_V2_TARGET_CALL",
        "claim_boundary": (
            "PROSPECTIVE_SAME_SOURCE_IR_REPLICATION_ON_NON_MAZE_TIR_VISUAL_SEARCH;"
            "NEURAL_ANCHOR_TARGET_GROUNDER;QUALIFICATION_AND_FORMAL_LOCKED"
        ),
        "predecessor_development_gate": {
            "config": "configs/phase3_tir_nonmaze_v1_splits.json",
            "receipts": "runs/phase3_tir_nonmaze_v2/development_validation_receipts.json",
            "status": "TARGET_EVIDENCE_HEADROOM_FAILED_2_OF_8_EQUALS_BASELINE",
            "qualification_or_formal_consumed": False,
            "reason_for_new_family": (
                "COLOR_MIXED_COUNT_ILLUSION_AND_SEMANTIC_COLOR_TASKS_DO_NOT_MATCH_"
                "AVAILABLE_REGION_TOOLS;PRIOR_CONSUMED_VISUAL_SEARCH_DEV_SHOWED_4_OF_4_"
                "ONE_CROP_ORACLE_HEADROOM"
            ),
        },
        "dataset": {
            "path": str(dataset_file.resolve()),
            "sha256": BASE.file_sha256(dataset_file),
            "benchmark": "TIR-Bench",
            "family": "visual_search",
            "eligible_a_to_f_population": len(eligible),
            "formal_prompt_image_answer_read_before_freeze": False,
            "answer_schema_used_for_eligibility_only": True,
            "answer_value_used_for_selection_or_order": False,
        },
        "selection": {
            "rule": (
                "Use historically TIR-reserved/executed visual_search IDs plus IDs "
                "22/42 as development first; fill development to 24 by a frozen hash. "
                "Order every remaining A--F single-image visual_search ID by sha256("
                "'phase3-tir-visual-search-v2-fresh\\0'+id), allocate 10 qualification "
                "and the next 24 formal. No prompt, image, or answer value is read for "
                "fresh ordering."
            ),
            "historical_development_ids": historical_development,
            "historical_reservation_receipt_sha256": stable_hash(historical),
            "prompt_or_image_used_for_fresh_order": False,
            "answer_value_used_for_fresh_order": False,
        },
        "splits": splits,
        "conditions": [
            "neural_only", "source_induced", "source_permuted",
            "generic_scaffold", "target_native_ceiling",
        ],
        "source_programs": source_programs,
        "source_ir": {
            "runtime": "src/motif_transfer/phase3_attempt_runtime.py",
            "portfolio": "src/motif_transfer/phase3_source_portfolio.py",
            "typed_effect_induction": "src/motif_transfer/phase3_typed_effect_induction.py",
            "program_updated_for_tir": False,
            "source_identity_used_as_runtime_feature": False,
        },
        "target_mdp": {
            "state": "ACCUMULATED_MULTISCALE_REGION_EVIDENCE",
            "action": "zoom_region",
            "neural_operands": "FOUR_TARGET_NEURAL_ANCHORS",
            "transition_horizons": [1, 4, 8],
            "budget": 8,
            "formal_gold_visible_to_grounder_or_source": False,
        },
        "wrapper": {
            "root": "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent",
            "execution_authority": "visual_reasoning_wrapper.zoom_region",
        },
        "model": {
            "provider": "openrouter",
            "id": "qwen/qwen3-vl-32b-instruct",
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 240,
            "max_retries": 3,
            "temperature": 0,
        },
        "media": {
            "maximum_source_pixels": 300000000,
            "native_working_max_side": 2048,
            "overview_max_side": 768,
            "evidence_max_side": 1280,
            "jpeg_quality": 88,
        },
        "qualification_gates": {
            "expected_tasks": 10,
            "minimum_ceiling_successes": 8,
            "minimum_source_action_contrasts": 3,
            "minimum_permuted_action_contrasts": 3,
            "minimum_selected_effect_types": 2,
            "maximum_negative_transfer_rate": 0.0,
            "required_gate_names": [
                "expected_task_count", "target_native_ceiling_capable",
                "source_changes_target_policy", "authentic_differs_from_permuted",
                "multiple_source_effect_types_selected", "maximum_negative_transfer",
                "source_not_below_neural", "source_strictly_beats_neural",
                "source_strictly_beats_permuted", "source_strictly_beats_generic",
            ],
        },
        "formal_gates": {
            "expected_tasks": 24,
            "minimum_ceiling_successes": 18,
            "minimum_source_action_contrasts": 6,
            "minimum_permuted_action_contrasts": 6,
            "minimum_selected_effect_types": 2,
            "maximum_negative_transfer_rate": 0.0,
            "required_gate_names": [
                "expected_task_count", "target_native_ceiling_capable",
                "source_changes_target_policy", "authentic_differs_from_permuted",
                "multiple_source_effect_types_selected", "maximum_negative_transfer",
                "source_not_below_neural", "source_strictly_beats_neural",
                "source_strictly_beats_permuted", "source_strictly_beats_generic",
            ],
        },
        "integrity": {
            "code_sha256": {
                path: BASE.file_sha256(REPO / path) for path in (
                    "scripts/collect_phase3_tir_visual_search_v2.py",
                    "scripts/collect_phase3_tir_nonmaze.py",
                    "src/motif_transfer/phase3_tir_nonmaze.py",
                    "src/motif_transfer/phase3_attempt_runtime.py",
                    "src/motif_transfer/phase3_source_portfolio.py",
                    "src/motif_transfer/phase3_typed_effect_induction.py",
                )
            },
        },
    }
    output = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"],
        "eligible": len(eligible),
        "historical_development": len(historical_development),
        "splits": {key: len(value) for key, value in splits.items()},
        "config_sha256": output["config_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
