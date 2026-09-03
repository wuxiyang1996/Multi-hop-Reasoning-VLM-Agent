#!/usr/bin/env python3
"""Apply frozen scalar-retention and multi-IR gates to mixed Harness SFT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.portable_paths import resolve_repo_artifact  # noqa: E402
DEFAULT_PROTOCOL = REPO / "configs/harness_controller_qwen35_9b_mixed_v1_protocol.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    return resolve_repo_artifact(value, REPO)


def _verified(spec: dict[str, Any]) -> Path:
    path = _resolve(spec["path"])
    if _sha(path) != spec["sha256"]:
        raise SystemExit(f"protocol hash mismatch: {path}")
    return path


def _source_only_receipt(
    receipt: dict[str, Any], protocol: dict[str, Any], train_hash: str,
) -> None:
    if receipt.get("train_file_sha256") != train_hash:
        raise SystemExit("mixed adapter used the wrong train split")
    if receipt.get("initial_adapter_file_sha256") != protocol[
        "initial_adapter"
    ]["adapter_model_sha256"]:
        raise SystemExit("mixed adapter used the wrong initial adapter")
    forbidden = (
        "target_data_used",
        "target_outcome_used_for_controller_labels",
        "formal_or_qualification_targets_used",
        "video_target_data_used",
        "target_grounder_training_used_target_outcomes",
    )
    if any(receipt.get(key) is not False for key in forbidden):
        raise SystemExit("mixed adapter training was not source-only")


def _missing_schema_by_arity(
    dataset: Path, predictions_path: Path,
) -> dict[str, dict[str, Any]]:
    predictions = {
        str(row["example_id"]): row
        for row in _rows(predictions_path)
        if row.get("regime") == "CONTROLLER_LORA"
    }
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in _rows(dataset):
        if "CARDINALITY_EQUIVARIANT_MISSING_SCHEMA_CONTROL" not in set(
            row.get("control_variants") or ()
        ):
            continue
        arity = str(row["candidate_count_audit_only"])
        prediction = predictions.get(str(row["example_id"]))
        target = json.loads(row["completion"])
        parsed = prediction.get("parsed") if prediction else None
        counts[arity]["rows"] += 1
        counts[arity]["decision_correct"] += int(
            isinstance(parsed, dict)
            and parsed.get("decision") == target["decision"]
        )
        counts[arity]["exact"] += int(parsed == target)
    return {
        arity: {
            "rows": count["rows"],
            "decision_accuracy": count["decision_correct"] / count["rows"],
            "exact_accuracy": count["exact"] / count["rows"],
        }
        for arity, count in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--scalar-report", type=Path, required=True)
    parser.add_argument("--scalar-predictions", type=Path, required=True)
    parser.add_argument("--multi-ir-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _read(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES":
        raise SystemExit("mixed protocol is not frozen")
    mixed = protocol["mixed_source_dataset"]
    manifest_path = _verified(mixed["manifest"])
    train_path = _verified(mixed["train"])
    _verified(mixed["validation"])
    manifest = _read(manifest_path)
    if not (
        manifest.get("status")
        == "FROZEN_SOURCE_ONLY_MIXED_HARNESS_SUPERVISION"
        and all((manifest.get("gates") or {}).values())
    ):
        raise SystemExit("mixed source manifest is not gate-clean")
    initial = protocol["initial_adapter"]
    if _sha(_resolve(initial["path"]) / "adapter_model.safetensors") != initial[
        "adapter_model_sha256"
    ]:
        raise SystemExit("initial V4 adapter drifted")
    if _read(_resolve(initial["source_qualification"])).get(
        "status"
    ) != initial["source_qualification_status"]:
        raise SystemExit("initial V4 source qualification drifted")
    receipt = _read(args.training_receipt)
    _source_only_receipt(receipt, protocol, _sha(train_path))

    source = protocol["source_held_out_gates"]
    scalar_spec = source["scalar_executor"]
    multi_spec = source["multi_ir_selector"]
    scalar_dataset = _verified(scalar_spec)
    multi_dataset = _verified(multi_spec)
    scalar = _read(args.scalar_report)
    multi = _read(args.multi_ir_report)
    if scalar.get("dataset_sha256") != scalar_spec["sha256"]:
        raise SystemExit("scalar report used the wrong held-out split")
    if multi.get("dataset_sha256") != multi_spec["sha256"]:
        raise SystemExit("multi-IR report used the wrong held-out split")

    s = scalar["regimes"]["CONTROLLER_LORA"]
    so = s["overall"]
    by_arity = s["by_balance_group"]
    required_arities = {str(value) for value in range(2, 13)}
    missing = _missing_schema_by_arity(
        scalar_dataset, args.scalar_predictions,
    )
    scalar_gates = {
        "all_rows_selected": scalar["selection"]["all_input_rows_selected"] is True,
        "strict_json_accuracy": so["valid_json_accuracy"] >= scalar_spec[
            "strict_json_accuracy_minimum"
        ],
        "overall_decision_accuracy": so[
            "decision_correct_accuracy"
        ] >= scalar_spec["overall_decision_accuracy_minimum"],
        "overall_exact_accuracy": so["exact_json_accuracy"] >= scalar_spec[
            "overall_exact_accuracy_minimum"
        ],
        "all_arities_present": required_arities <= set(by_arity),
        "every_arity_decision_accuracy": all(
            by_arity[arity]["decision_correct_accuracy"] >= scalar_spec[
                "every_arity_decision_accuracy_minimum"
            ] for arity in required_arities if arity in by_arity
        ),
        "every_arity_exact_accuracy": all(
            by_arity[arity]["exact_json_accuracy"] >= scalar_spec[
                "every_arity_exact_accuracy_minimum"
            ] for arity in required_arities if arity in by_arity
        ),
        "missing_schema_control_present_at_every_arity": (
            required_arities <= set(missing)
        ),
        "missing_schema_decision_accuracy_at_every_arity": all(
            missing[arity]["decision_accuracy"] >= scalar_spec[
                "missing_schema_every_arity_decision_accuracy_minimum"
            ] for arity in required_arities if arity in missing
        ),
        "exact_minus_base": scalar[
            "controller_lora_minus_base_exact_accuracy"
        ] >= scalar_spec["exact_minus_base_minimum"],
    }

    m = multi["regimes"]["CONTROLLER_LORA"]
    mo = m["overall"]
    by_decision = m["by_target_decision"]
    by_ir = m["by_balance_group"]
    by_control = m["by_audit_field"]["control_variant_audit_only"]
    required_ir = set(multi_spec["required_ir_kinds"])
    multi_gates = {
        "all_rows_selected": multi["selection"]["all_input_rows_selected"] is True,
        "strict_json_accuracy": mo["valid_json_accuracy"] >= multi_spec[
            "strict_json_accuracy_minimum"
        ],
        "overall_decision_accuracy": mo[
            "decision_correct_accuracy"
        ] >= multi_spec["overall_decision_accuracy_minimum"],
        "overall_exact_accuracy": mo["exact_json_accuracy"] >= multi_spec[
            "overall_exact_accuracy_minimum"
        ],
        "select_skill_recall": by_decision["SELECT_SKILL"][
            "decision_correct_accuracy"
        ] >= multi_spec["select_skill_decision_recall_minimum"],
        "abstain_recall": by_decision["ABSTAIN"][
            "decision_correct_accuracy"
        ] >= multi_spec["abstain_decision_recall_minimum"],
        "all_ir_kinds_present": required_ir <= set(by_ir),
        "every_ir_decision_accuracy": all(
            by_ir[name]["decision_correct_accuracy"] >= multi_spec[
                "every_ir_decision_accuracy_minimum"
            ] for name in required_ir if name in by_ir
        ),
        "every_ir_exact_accuracy": all(
            by_ir[name]["exact_json_accuracy"] >= multi_spec[
                "every_ir_exact_accuracy_minimum"
            ] for name in required_ir if name in by_ir
        ),
        "every_control_decision_accuracy": bool(by_control) and all(
            row["decision_correct_accuracy"] >= multi_spec[
                "every_control_decision_accuracy_minimum"
            ] for row in by_control.values()
        ),
        "every_control_exact_accuracy": bool(by_control) and all(
            row["exact_json_accuracy"] >= multi_spec[
                "every_control_exact_accuracy_minimum"
            ] for row in by_control.values()
        ),
        "exact_minus_base": multi[
            "controller_lora_minus_base_exact_accuracy"
        ] >= multi_spec["exact_minus_base_minimum"],
    }
    passed = all(scalar_gates.values()) and all(multi_gates.values())
    substitution_protocol = "six_benchmark_substitution" in protocol
    output = {
        "schema_version": (
            "harness-mixed-source-qualification-v2-five-schema"
            if substitution_protocol
            else "harness-mixed-source-qualification-v1"
        ),
        "status": (
            "SOURCE_MIXED_HARNESS_GATE_PASSED"
            if passed else "SOURCE_MIXED_HARNESS_GATE_FAILED"
        ),
        "protocol": {
            "path": str(args.protocol.resolve()), "sha256": _sha(args.protocol),
        },
        "training_receipt": {
            "path": str(args.training_receipt.resolve()),
            "sha256": _sha(args.training_receipt),
        },
        "scalar_executor": {
            "report": {"path": str(args.scalar_report.resolve()), "sha256": _sha(args.scalar_report)},
            "overall": so, "by_arity": by_arity,
            "missing_schema_by_arity": missing, "gates": scalar_gates,
        },
        "multi_ir_selector": {
            "report": {"path": str(args.multi_ir_report.resolve()), "sha256": _sha(args.multi_ir_report)},
            "overall": mo, "by_ir_kind": by_ir,
            "by_control_variant": by_control, "gates": multi_gates,
        },
        "gates": {
            "scalar_executor_retained_and_schema_closed": all(
                scalar_gates.values()
            ),
            "multi_ir_selector_qualified": all(multi_gates.values()),
            "training_source_only": True,
        },
        "next_legal_step": (
            (
                "ACTIVATE_FROZEN_SIX_BENCHMARK_SUBSTITUTION_REPLAY"
                if substitution_protocol
                else "ACTIVATE_FROZEN_V5_TARGET_IR_RESERVE"
            )
            if passed else (
                "DO_NOT_OPEN_SIX_BENCHMARK_SUBSTITUTION_REPLAY"
                if substitution_protocol
                else "DO_NOT_OPEN_V5_TARGET_IR_RESERVE"
            )
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": output["status"], "gates": output["gates"],
        "scalar_gates": scalar_gates, "multi_ir_gates": multi_gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
