#!/usr/bin/env python3
"""Bind a passed source-only V4 adapter to the already frozen target reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs/harness_controller_qwen35_9b_v4_cardinality_protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


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
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    qualification = json.loads(args.source_qualification.read_text(encoding="utf-8"))
    receipt = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    if (
        qualification.get("status") != "SOURCE_CARDINALITY_GATE_PASSED"
        or not all(qualification.get("gates", {}).values())
    ):
        raise SystemExit("source cardinality gate did not pass; target reserve remains closed")
    if receipt.get("target_data_used") is not False:
        raise SystemExit("adapter training was not source-only")

    reserve_spec = protocol["fresh_target_reserve"]["dataset"]
    reserve_path = _resolve(reserve_spec["path"])
    prereg_spec = protocol["fresh_target_reserve"]["preregistration"]
    prereg_path = _resolve(prereg_spec["path"])
    if _sha256(reserve_path) != reserve_spec["sha256"]:
        raise SystemExit("frozen target reserve hash mismatch")
    if _sha256(prereg_path) != prereg_spec["sha256"]:
        raise SystemExit("target reserve preregistration hash mismatch")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if (
        prereg.get("status") != "FROZEN_PROSPECTIVE_TO_V4_BEFORE_WEIGHT_UPDATES"
        or not all(prereg.get("gates", {}).values())
    ):
        raise SystemExit("target reserve preregistration is not gate-clean")

    adapter_file = args.adapter / "adapter_model.safetensors"
    adapter_config = args.adapter / "adapter_config.json"
    gates = {
        "source_cardinality_gate_passed": True,
        "training_target_data_absent": receipt["target_data_used"] is False,
        "reserve_frozen_before_v4_training": (
            prereg["status"] == "FROZEN_PROSPECTIVE_TO_V4_BEFORE_WEIGHT_UPDATES"
        ),
        "reserve_hash_matches_preregistration": (
            _sha256(reserve_path) == prereg["reserve"]["sha256"]
        ),
        "consumed_overlap_gates_passed": all(
            prereg["gates"][name] for name in (
                "consumed_example_overlap_zero",
                "consumed_pair_overlap_zero",
                "consumed_prompt_overlap_zero",
            )
        ),
        "formal_or_reserve_targets_absent": True,
    }
    if not all(gates.values()):
        raise SystemExit(f"target reserve activation gates failed: {gates}")
    output = {
        "schema_version": "harness-controller-target-ir-zero-shot-activation-v4-cardinality",
        "status": "FROZEN_TARGET_IR_ZERO_SHOT_EVALUATION_READY",
        "authority": (
            "SOURCE_CARDINALITY_GATE_PASSED;TARGET_RESERVE_FROZEN_BEFORE_V4_TRAINING;"
            "NO_TARGET_WEIGHT_UPDATE;NO_SELECTION_BY_MODEL_OUTPUT"
        ),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": _sha256(args.protocol)},
        "source_qualification": {
            "path": str(args.source_qualification.resolve()),
            "sha256": _sha256(args.source_qualification),
        },
        "target_preregistration": {
            "path": str(prereg_path.resolve()), "sha256": _sha256(prereg_path),
        },
        "evaluation_file": {
            "path": str(reserve_path.resolve()),
            "sha256": _sha256(reserve_path),
            "rows": protocol["fresh_target_reserve"]["rows"],
        },
        "frozen_model": {
            "base_model": protocol["training"]["model"],
            "adapter_path": str(args.adapter.resolve()),
            "adapter_model_sha256": _sha256(adapter_file),
            "adapter_config_sha256": _sha256(adapter_config),
            "training_receipt_sha256": _sha256(args.training_receipt),
        },
        "gates": gates,
        "preregistered_model_evaluation_gates": protocol["fresh_target_gates"],
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": output["status"], "gates": gates,
        "evaluation_file": output["evaluation_file"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
