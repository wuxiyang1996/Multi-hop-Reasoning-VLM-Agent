#!/usr/bin/env python3
"""Bind a source-qualified mixed 9B Harness to the frozen V5 target reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "configs/harness_controller_qwen35_9b_mixed_v1_protocol.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-qualification", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    protocol = _read(args.protocol)
    qualification = _read(args.source_qualification)
    receipt = _read(args.training_receipt)
    if protocol.get("status") != "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES":
        raise SystemExit("mixed Harness protocol is not frozen")
    if (
        qualification.get("status") != "SOURCE_MIXED_HARNESS_GATE_PASSED"
        or not all((qualification.get("gates") or {}).values())
    ):
        raise SystemExit("mixed source gate did not pass; V5 target reserve remains closed")
    qualification_protocol = qualification.get("protocol") or {}
    if qualification_protocol.get("sha256") != _sha(args.protocol):
        raise SystemExit("source qualification used a different frozen protocol")
    qualification_receipt = qualification.get("training_receipt") or {}
    if qualification_receipt.get("sha256") != _sha(args.training_receipt):
        raise SystemExit("source qualification used a different training receipt")

    mixed = protocol["mixed_source_dataset"]
    if receipt.get("train_file_sha256") != mixed["train"]["sha256"]:
        raise SystemExit("adapter was not trained on the frozen mixed source split")
    forbidden_receipt_fields = (
        "target_data_used",
        "target_outcome_used_for_controller_labels",
        "formal_or_qualification_targets_used",
        "video_target_data_used",
        "target_grounder_training_used_target_outcomes",
    )
    if any(receipt.get(key) is not False for key in forbidden_receipt_fields):
        raise SystemExit("adapter training receipt is not source-only")

    reserve_spec = protocol["fresh_target_reserve"]["dataset"]
    reserve_path = _resolve(reserve_spec["path"])
    prereg_spec = protocol["fresh_target_reserve"]["preregistration"]
    prereg_path = _resolve(prereg_spec["path"])
    if _sha(reserve_path) != reserve_spec["sha256"]:
        raise SystemExit("frozen V5 target reserve hash mismatch")
    if _sha(prereg_path) != prereg_spec["sha256"]:
        raise SystemExit("V5 preregistration hash mismatch")
    prereg = _read(prereg_path)
    if (
        prereg.get("status") != "FROZEN_PROSPECTIVE_TO_V5_BEFORE_WEIGHT_UPDATES"
        or not all((prereg.get("gates") or {}).values())
    ):
        raise SystemExit("V5 target reserve preregistration is not gate-clean")
    if prereg["reserve"]["sha256"] != _sha(reserve_path):
        raise SystemExit("V5 reserve does not match its preregistration")
    if prereg["mixed_source_manifest"]["sha256"] != mixed["manifest"]["sha256"]:
        raise SystemExit("V5 reserve was not frozen against this mixed source dataset")

    adapter_model = args.adapter / "adapter_model.safetensors"
    adapter_config = args.adapter / "adapter_config.json"
    if not adapter_model.is_file() or not adapter_config.is_file():
        raise SystemExit("trained adapter is incomplete")

    gates = {
        "mixed_source_gate_passed": True,
        "training_used_only_source_examples": True,
        "reserve_frozen_before_v5_weight_updates": True,
        "reserve_hash_matches_preregistration": True,
        "reserve_disjoint_from_consumed_target_diagnostics": all(
            prereg["gates"][name] for name in (
                "consumed_example_overlap_zero",
                "consumed_pair_overlap_zero",
                "consumed_prompt_overlap_zero",
            )
        ),
        "reserve_selection_prediction_and_outcome_blind": prereg["gates"][
            "selection_is_prediction_and_outcome_blind"
        ],
        "target_actor_and_grounder_weight_updates_absent": True,
    }
    if not all(gates.values()):
        raise SystemExit(f"V5 target activation gates failed: {gates}")

    payload = {
        "schema_version": "harness-controller-target-ir-zero-shot-activation-v5-mixed",
        "status": "FROZEN_TARGET_IR_ZERO_SHOT_EVALUATION_READY",
        "authority": (
            "SOURCE_MIXED_HARNESS_GATE_PASSED;TARGET_RESERVE_FROZEN_BEFORE_V5_WEIGHTS;"
            "NO_TARGET_WEIGHT_UPDATE;NO_SELECTION_BY_MODEL_OUTPUT"
        ),
        "protocol": {
            "path": str(args.protocol.resolve()), "sha256": _sha(args.protocol),
        },
        "source_qualification": {
            "path": str(args.source_qualification.resolve()),
            "sha256": _sha(args.source_qualification),
        },
        "target_preregistration": {
            "path": str(prereg_path.resolve()), "sha256": _sha(prereg_path),
        },
        "evaluation_file": {
            "path": str(reserve_path.resolve()),
            "sha256": _sha(reserve_path),
            "rows": reserve_spec["rows"],
        },
        "frozen_model": {
            "base_model": protocol["model"],
            "adapter_path": str(args.adapter.resolve()),
            "adapter_model_sha256": _sha(adapter_model),
            "adapter_config_sha256": _sha(adapter_config),
            "training_receipt_sha256": _sha(args.training_receipt),
        },
        "gates": gates,
        "preregistered_model_evaluation_gates": protocol[
            "fresh_target_reserve"
        ]["gates"],
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
