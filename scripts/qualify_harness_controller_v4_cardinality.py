#!/usr/bin/env python3
"""Apply the preregistered source-held-out arity gate to a V4 report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V4_CARDINALITY_TRAINING":
        raise SystemExit("V4 cardinality protocol is not frozen")
    source = protocol["source_only_dataset"]
    for name in ("manifest", "train", "validation", "source_held_out"):
        spec = source[name]
        path = _resolve(spec["path"])
        if _sha256(path) != spec["sha256"]:
            raise SystemExit(f"source {name} hash mismatch")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    receipt = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    heldout_path = _resolve(source["source_held_out"]["path"])
    if report.get("dataset_sha256") != _sha256(heldout_path):
        raise SystemExit("source qualification report used the wrong dataset")
    if receipt.get("target_data_used") is not False:
        raise SystemExit("V4 training receipt does not attest source-only weights")
    if receipt.get("train_file_sha256") != source["train"]["sha256"]:
        raise SystemExit("V4 training receipt used the wrong train file")

    thresholds = protocol["source_cardinality_qualification"]
    controller = report["regimes"]["CONTROLLER_LORA"]
    overall = controller["overall"]
    by_arity = controller["by_balance_group"]
    required_arities = [str(value) for value in thresholds["required_arities"]]
    gates: dict[str, bool] = {
        "all_heldout_rows_selected": report["selection"]["all_input_rows_selected"] is True,
        "strict_json_accuracy": overall["valid_json_accuracy"] >= thresholds[
            "strict_json_accuracy_minimum"
        ],
        "overall_decision_accuracy": overall["decision_correct_accuracy"] >= thresholds[
            "overall_decision_accuracy_minimum"
        ],
        "overall_exact_accuracy": overall["exact_json_accuracy"] >= thresholds[
            "overall_exact_accuracy_minimum"
        ],
        "exact_minus_base": report["controller_lora_minus_base_exact_accuracy"] >= thresholds[
            "exact_minus_base_minimum"
        ],
        "every_required_arity_present": set(required_arities) <= set(by_arity),
        "every_arity_decision_accuracy": all(
            by_arity[arity]["decision_correct_accuracy"]
            >= thresholds["every_arity_decision_accuracy_minimum"]
            for arity in required_arities if arity in by_arity
        ),
        "every_arity_exact_accuracy": all(
            by_arity[arity]["exact_json_accuracy"]
            >= thresholds["every_arity_exact_accuracy_minimum"]
            for arity in required_arities if arity in by_arity
        ),
    }
    passed = all(gates.values())
    output: dict[str, Any] = {
        "schema_version": "harness-controller-v4-source-cardinality-qualification-v1",
        "status": (
            "SOURCE_CARDINALITY_GATE_PASSED"
            if passed else "SOURCE_CARDINALITY_GATE_FAILED"
        ),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": _sha256(args.protocol)},
        "training_receipt": {
            "path": str(args.training_receipt.resolve()),
            "sha256": _sha256(args.training_receipt),
            "target_data_used": receipt["target_data_used"],
        },
        "evaluation_report": {
            "path": str(args.report.resolve()), "sha256": _sha256(args.report),
        },
        "overall": overall,
        "by_arity": {arity: by_arity[arity] for arity in required_arities if arity in by_arity},
        "gates": gates,
        "next_legal_step": (
            "ACTIVATE_FROZEN_FRESH_TARGET_IR_RESERVE"
            if passed else "DO_NOT_OPEN_FRESH_TARGET_IR_RESERVE"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": output["status"], "gates": gates}, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
