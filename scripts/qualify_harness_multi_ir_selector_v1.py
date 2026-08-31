#!/usr/bin/env python3
"""Apply the frozen source-held-out gate to the 9B multi-IR selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "configs/harness_controller_qwen35_9b_multi_ir_v1_protocol.json"


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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _read(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_MULTI_IR_SELECTOR_WEIGHT_UPDATES":
        raise SystemExit("multi-IR protocol is not frozen")
    source = protocol["source_only_dataset"]
    for key in ("manifest", "train", "validation", "source_held_out"):
        if _sha(_resolve(source[key]["path"])) != source[key]["sha256"]:
            raise SystemExit(f"multi-IR source {key} hash mismatch")
    initial = protocol["initial_adapter"]
    if _sha(_resolve(initial["path"]) / "adapter_model.safetensors") != initial[
        "adapter_model_sha256"
    ]:
        raise SystemExit("initial V4 adapter hash mismatch")
    source_qualification = _read(_resolve(initial["source_qualification"]))
    if source_qualification.get("status") != initial[
        "source_qualification_status"
    ]:
        raise SystemExit("initial V4 source qualification changed")

    report = _read(args.report)
    receipt = _read(args.training_receipt)
    if report.get("status") not in {
        "SOURCE_MULTI_IR_HELD_OUT_CONTROLLER_GATE_PASSED",
        "SOURCE_MULTI_IR_HELD_OUT_CONTROLLER_GATE_FAILED",
    }:
        raise SystemExit("report is not a source multi-IR held-out evaluation")
    if report.get("dataset_sha256") != source["source_held_out"]["sha256"]:
        raise SystemExit("multi-IR report used the wrong held-out dataset")
    if receipt.get("train_file_sha256") != source["train"]["sha256"]:
        raise SystemExit("multi-IR weights used the wrong train dataset")
    if receipt.get("initial_adapter_file_sha256") != initial[
        "adapter_model_sha256"
    ]:
        raise SystemExit("multi-IR weights used the wrong initial adapter")
    if any(receipt.get(key) is not False for key in (
        "target_data_used",
        "target_outcome_used_for_controller_labels",
        "formal_or_qualification_targets_used",
        "video_target_data_used",
        "target_grounder_training_used_target_outcomes",
    )):
        raise SystemExit("multi-IR training receipt is not source-only")

    threshold = protocol["source_held_out_qualification"]
    controller = report["regimes"]["CONTROLLER_LORA"]
    overall = controller["overall"]
    decisions = controller["by_target_decision"]
    by_ir = controller["by_balance_group"]
    by_control = controller["by_audit_field"][
        "control_variant_audit_only"
    ]
    required_ir = set(threshold["required_ir_kinds"])
    gates = {
        "all_heldout_rows_selected": report["selection"][
            "all_input_rows_selected"
        ] is True,
        "strict_json_accuracy": overall["valid_json_accuracy"] >= threshold[
            "strict_json_accuracy_minimum"
        ],
        "overall_decision_accuracy": overall[
            "decision_correct_accuracy"
        ] >= threshold["overall_decision_accuracy_minimum"],
        "overall_exact_accuracy": overall["exact_json_accuracy"] >= threshold[
            "overall_exact_accuracy_minimum"
        ],
        "select_skill_recall": decisions["SELECT_SKILL"][
            "decision_correct_accuracy"
        ] >= threshold["select_skill_decision_recall_minimum"],
        "abstain_recall": decisions["ABSTAIN"][
            "decision_correct_accuracy"
        ] >= threshold["abstain_decision_recall_minimum"],
        "every_required_ir_present": required_ir <= set(by_ir),
        "every_ir_decision_accuracy": all(
            by_ir[name]["decision_correct_accuracy"] >= threshold[
                "every_ir_decision_accuracy_minimum"
            ] for name in required_ir if name in by_ir
        ),
        "every_ir_exact_accuracy": all(
            by_ir[name]["exact_json_accuracy"] >= threshold[
                "every_ir_exact_accuracy_minimum"
            ] for name in required_ir if name in by_ir
        ),
        "every_control_decision_accuracy": bool(by_control) and all(
            row["decision_correct_accuracy"] >= threshold[
                "every_control_decision_accuracy_minimum"
            ] for row in by_control.values()
        ),
        "every_control_exact_accuracy": bool(by_control) and all(
            row["exact_json_accuracy"] >= threshold[
                "every_control_exact_accuracy_minimum"
            ] for row in by_control.values()
        ),
        "exact_minus_base": report[
            "controller_lora_minus_base_exact_accuracy"
        ] >= threshold["exact_minus_base_minimum"],
    }
    passed = all(gates.values())
    output = {
        "schema_version": "harness-multi-ir-selector-source-qualification-v1",
        "status": (
            "SOURCE_MULTI_IR_SELECTOR_GATE_PASSED"
            if passed else "SOURCE_MULTI_IR_SELECTOR_GATE_FAILED"
        ),
        "protocol": {
            "path": str(args.protocol.resolve()), "sha256": _sha(args.protocol),
        },
        "training_receipt": {
            "path": str(args.training_receipt.resolve()),
            "sha256": _sha(args.training_receipt),
        },
        "evaluation_report": {
            "path": str(args.report.resolve()), "sha256": _sha(args.report),
        },
        "overall": overall,
        "by_ir_kind": by_ir,
        "by_control_variant": by_control,
        "gates": gates,
        "next_legal_step": (
            "FREEZE_TARGET_NATIVE_PHASE8_SELECTOR_REPLAY"
            if passed else "DO_NOT_ACTIVATE_TARGET_SELECTOR"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": output["status"], "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
