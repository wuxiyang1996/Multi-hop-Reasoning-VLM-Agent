#!/usr/bin/env python3
"""Open the frozen six-benchmark replay only after five-schema source gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-qualification", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    protocol = _read(args.protocol)
    qualification = _read(args.source_qualification)
    receipt = _read(args.training_receipt)
    if protocol.get("status") != "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES":
        raise SystemExit("five-schema protocol is not pre-update frozen")
    if not (
        qualification.get("status") == "SOURCE_MIXED_HARNESS_GATE_PASSED"
        and all((qualification.get("gates") or {}).values())
        and (qualification.get("protocol") or {}).get("sha256") == _sha(args.protocol)
        and (qualification.get("training_receipt") or {}).get("sha256")
        == _sha(args.training_receipt)
    ):
        raise SystemExit("five-schema source-only qualification did not pass")

    mixed = protocol["mixed_source_dataset"]
    if receipt.get("train_file_sha256") != mixed["train"]["sha256"]:
        raise SystemExit("adapter used the wrong five-schema train split")
    if receipt.get("initial_adapter_file_sha256") != protocol[
        "initial_adapter"
    ]["adapter_model_sha256"]:
        raise SystemExit("adapter did not continue from the frozen V1 controller")
    forbidden = (
        "target_data_used", "target_outcome_used_for_controller_labels",
        "formal_or_qualification_targets_used", "video_target_data_used",
        "target_grounder_training_used_target_outcomes",
    )
    if any(receipt.get(field) is not False for field in forbidden):
        raise SystemExit("five-schema adapter training was not source-only")

    substitution = protocol["six_benchmark_substitution"]
    prereg_path = Path(substitution["preregistration"]["path"])
    dataset_path = Path(substitution["dataset"]["path"])
    index_path = Path(substitution["native_replay_index"]["path"])
    prereg = _read(prereg_path)
    if not (
        _sha(prereg_path) == substitution["preregistration"]["sha256"]
        and prereg.get("status") == substitution.get(
            "preregistration_status",
            "FROZEN_BEFORE_FIVE_SCHEMA_9B_WEIGHT_UPDATE_OR_SUBSTITUTION_INFERENCE",
        )
        and all((prereg.get("gates") or {}).values())
        and _sha(dataset_path) == substitution["dataset"]["sha256"]
        and _sha(index_path) == substitution["native_replay_index"]["sha256"]
    ):
        raise SystemExit("six-benchmark substitution artifacts drifted")

    adapter_model = args.adapter / "adapter_model.safetensors"
    adapter_config = args.adapter / "adapter_config.json"
    if not adapter_model.is_file() or not adapter_config.is_file():
        raise SystemExit("five-schema adapter is incomplete")
    gates = {
        "five_schema_source_gate_passed": True,
        "training_used_only_source_examples": True,
        "substitution_tasks_frozen_before_five_schema_weight_update": True,
        "substitution_dataset_hash_matches_preregistration": True,
        "task_selection_used_no_target_outcome": prereg["gates"][
            "formal_outcomes_not_parsed_or_used_by_freezer"
        ],
        "all_six_benchmarks_present": prereg["gates"][
            "exact_six_benchmark_groups"
        ],
        "agqa_two_primitive_route_present": prereg["gates"][
            "agqa_has_two_source_primitives_per_task"
        ],
        "target_actor_grounder_composer_utility_verifier_executor_updates_absent": True,
    }
    payload = {
        "schema_version": "harness-9b-six-benchmark-substitution-activation-v1",
        "status": "FROZEN_SIX_BENCHMARK_SUBSTITUTION_EVALUATION_READY",
        "authority": (
            "FIVE_SCHEMA_SOURCE_ONLY_GATE_PASSED;SUBSTITUTION_SET_PRETRAINING_FROZEN;"
            "NO_TARGET_WEIGHT_UPDATE;NO_TARGET_OUTCOME_SELECTION"
        ),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": _sha(args.protocol)},
        "source_qualification": {
            "path": str(args.source_qualification.resolve()),
            "sha256": _sha(args.source_qualification),
        },
        "target_preregistration": {
            "path": str(prereg_path.resolve()), "sha256": _sha(prereg_path),
        },
        "evaluation_file": {
            "path": str(dataset_path.resolve()), "sha256": _sha(dataset_path),
            "rows": substitution["dataset"]["examples"],
        },
        "native_replay_index": {
            "path": str(index_path.resolve()), "sha256": _sha(index_path),
            "tasks": substitution["native_replay_index"]["tasks"],
        },
        "frozen_model": {
            "base_model": protocol["model"],
            "adapter_path": str(args.adapter.resolve()),
            "adapter_model_sha256": _sha(adapter_model),
            "adapter_config_sha256": _sha(adapter_config),
            "training_receipt_sha256": _sha(args.training_receipt),
        },
        "gates": gates,
        "preregistered_model_evaluation_gates": substitution["gates"],
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"], "gates": gates,
        "evaluation_file": payload["evaluation_file"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
