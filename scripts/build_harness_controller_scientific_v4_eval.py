#!/usr/bin/env python3
"""Freeze target-IR zero-shot evaluation for the source-only V3 controller.

This builder never reads a target training split and never updates model weights.
It verifies the complete V2 -> V3 source-only adapter lineage, proves prompt/example
disjointness, and adds an audit-only target group that is absent from the prompt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/harness_controller_scientific_v4.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs/harness_controller_scientific_v4_zero_shot"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _verify(spec: dict[str, Any]) -> Path:
    path = _resolve(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    expected = str(spec["sha256"])
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _group(row: dict[str, Any]) -> str:
    domain = str(row.get("target_domain_audit_only", ""))
    if domain == "video":
        benchmark = str(row.get("target_benchmark_audit_only", ""))
        if not benchmark:
            raise ValueError(f"video row lacks benchmark: {row.get('example_id')}")
        return f"video/{benchmark}"
    if not domain:
        raise ValueError(f"row lacks target domain: {row.get('example_id')}")
    return domain


def _instruction_prefix(prompt: str) -> str:
    marker = "\n\nOBJECTIVE="
    if marker not in prompt:
        raise ValueError("controller prompt lacks OBJECTIVE marker")
    return prompt.split(marker, 1)[0]


def _completion_decision(row: dict[str, Any]) -> str:
    completion = json.loads(str(row["completion"]))
    required = {
        "binding", "decision", "next_symbolic_state", "operator_id", "reason",
    }
    if set(completion) != required:
        raise ValueError(
            f"unexpected completion schema for {row.get('example_id')}: "
            f"{sorted(completion)}"
        )
    return str(completion["decision"])


def build(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_TARGET_ZERO_SHOT_EVALUATION":
        raise ValueError("scientific V4 protocol is not frozen")

    verified_inputs: dict[str, dict[str, str]] = {}

    def verify_named(name: str, spec: dict[str, Any]) -> Path:
        path = _verify(spec)
        verified_inputs[name] = {
            "path": str(path.resolve()), "sha256": _sha256(path),
        }
        return path

    source_specs = config["source_only_supervision"]
    verify_named("source_v3_build_config", source_specs["build_config"])
    source_manifest_path = verify_named("source_v3_manifest", source_specs["manifest"])
    source_paths = [
        verify_named("source_v3_train", source_specs["train"]),
        verify_named("source_v3_validation", source_specs["validation"]),
        verify_named("source_v3_model_unseen_eval", source_specs["model_unseen_eval"]),
    ]

    lineage = config["initial_source_only_adapter_lineage"]
    initial_receipt_path = verify_named(
        "source_v2_training_receipt", lineage["training_receipt"],
    )
    verify_named("source_v2_adapter", lineage["adapter_model"])
    verify_named("source_v2_manifest", lineage["dataset_manifest"])
    source_paths.extend([
        verify_named("source_v2_train", lineage["train"]),
        verify_named("source_v2_validation", lineage["validation"]),
        verify_named("source_v2_source_held_out", lineage["source_held_out"]),
    ])

    frozen = config["frozen_model"]
    adapter_model = Path(frozen["adapter_path"])
    adapter_model = _resolve(str(adapter_model)) / "adapter_model.safetensors"
    if _sha256(adapter_model) != frozen["adapter_model_sha256"]:
        raise ValueError("frozen V3 adapter hash mismatch")
    adapter_config = adapter_model.parent / "adapter_config.json"
    if _sha256(adapter_config) != frozen["adapter_config_sha256"]:
        raise ValueError("frozen V3 adapter config hash mismatch")
    v3_receipt = adapter_model.parent.parent / "training_receipt.json"
    if _sha256(v3_receipt) != frozen["training_receipt_sha256"]:
        raise ValueError("V3 training receipt hash mismatch")
    heldout_report_path = adapter_model.parent.parent / "model_unseen_source_family_report.json"
    if _sha256(heldout_report_path) != frozen["source_family_heldout_report_sha256"]:
        raise ValueError("V3 source-family held-out report hash mismatch")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    initial_receipt = json.loads(initial_receipt_path.read_text(encoding="utf-8"))
    v3_training_receipt = json.loads(v3_receipt.read_text(encoding="utf-8"))
    heldout_report = json.loads(heldout_report_path.read_text(encoding="utf-8"))

    target_spec = config["target_zero_shot_evaluation"]["input_validation"]
    target_path = verify_named("target_validation", target_spec)
    target_rows = _read_jsonl(target_path)
    expected_rows = int(config["target_zero_shot_evaluation"]["expected_rows"])
    if len(target_rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} target rows, found {len(target_rows)}")

    source_rows: list[dict[str, Any]] = []
    for path in source_paths:
        source_rows.extend(_read_jsonl(path))

    source_prompts = {str(row["prompt"]) for row in source_rows}
    source_ids = {str(row["example_id"]) for row in source_rows}
    target_prompts = {str(row["prompt"]) for row in target_rows}
    target_ids = {str(row["example_id"]) for row in target_rows}
    prompt_intersection = source_prompts & target_prompts
    id_intersection = source_ids & target_ids

    source_prefixes = {_instruction_prefix(str(row["prompt"])) for row in source_rows}
    target_prefixes = {_instruction_prefix(str(row["prompt"])) for row in target_rows}
    forbidden_prompt_tokens = (
        "webshop", "alfworld", "discoveryworld", "tirbench", "clevrer",
        "agqa", "candy_crush", "tetris", "thunder_force", "strider",
        "streets_of_rage", "gymv_columns",
    )
    identity_leaks = [
        str(row["example_id"])
        for row in target_rows
        if any(token in str(row["prompt"]).lower() for token in forbidden_prompt_tokens)
    ]

    frozen_rows = []
    for row in target_rows:
        frozen_row = dict(row)
        frozen_row["target_eval_group_audit_only"] = _group(row)
        _completion_decision(frozen_row)
        frozen_rows.append(frozen_row)
    frozen_rows.sort(key=lambda row: str(row["example_id"]))

    group_counts = Counter(_group(row) for row in frozen_rows)
    expected_groups = {
        str(key): int(value)
        for key, value in config["target_zero_shot_evaluation"]["expected_groups"].items()
    }
    decision_counts = Counter(_completion_decision(row) for row in frozen_rows)
    group_decision_counts = Counter(
        (_group(row), _completion_decision(row)) for row in frozen_rows
    )
    control_counts = Counter(
        str(row.get("control_variant_audit_only", "MISSING")) for row in frozen_rows
    )

    prereg = config["preregistered_gates"]
    gates = {
        "source_v2_target_data_absent": (
            lineage.get("target_data_used") is False
            and initial_receipt.get("target_data_used") is False
        ),
        "source_v3_target_data_absent": (
            source_manifest.get("gates", {}).get("target_data_absent") is True
            and v3_training_receipt.get("target_data_used") is False
            and int(frozen.get("target_examples_used_for_weight_updates", -1)) == 0
        ),
        "source_family_heldout_gate_passed": (
            heldout_report.get("status")
            == frozen["source_family_gate_status"]
        ),
        "target_validation_rows_exact": len(frozen_rows) == expected_rows,
        "target_group_counts_exact": dict(group_counts) == expected_groups,
        "both_controller_decisions_present": (
            set(decision_counts) == {"ABSTAIN", "EXECUTE_OPERATOR"}
        ),
        "both_decisions_present_in_every_target_group": all(
            group_decision_counts[(group, decision)] > 0
            for group in expected_groups
            for decision in ("ABSTAIN", "EXECUTE_OPERATOR")
        ),
        "authentic_and_matched_controls_present": (
            control_counts["AUTHENTIC_TARGET_NEURAL_GROUNDING"] > 0
            and len(control_counts) >= 4
        ),
        "source_target_prompt_intersection_exact": (
            len(prompt_intersection)
            == int(prereg["source_target_prompt_intersection"])
        ),
        "source_target_example_id_intersection_exact": (
            len(id_intersection)
            == int(prereg["source_target_example_id_intersection"])
        ),
        "controller_instruction_schema_identical": (
            len(source_prefixes) == 1
            and target_prefixes == source_prefixes
        ),
        "target_identity_absent_from_prompts": not identity_leaks,
        "all_validation_rows_selected": (
            config["target_zero_shot_evaluation"]["selection"]
            == "ALL_FROZEN_VALIDATION_ROWS"
        ),
        "formal_or_reserve_targets_absent": (
            config["target_zero_shot_evaluation"]["formal_or_reserve_targets_used"]
            is False
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"scientific V4 freeze gates failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "zero_shot_eval.jsonl"
    _write_jsonl(eval_path, frozen_rows)
    manifest = {
        "schema_version": "harness-controller-target-zero-shot-eval-v4",
        "status": "FROZEN_TARGET_IR_ZERO_SHOT_EVALUATION_READY",
        "authority": (
            "FROZEN_SOURCE_ONLY_V3_WEIGHTS;ALL_FIVE_DOMAIN_VALIDATION_ROWS;"
            "NO_TARGET_WEIGHT_UPDATE;NO_FORMAL_OR_RESERVE_TARGETS"
        ),
        "protocol": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "verified_inputs": verified_inputs,
        "frozen_model": {
            "base_model": frozen["base_model"],
            "adapter_path": str(adapter_model.parent.resolve()),
            "adapter_model_sha256": _sha256(adapter_model),
            "adapter_config_sha256": _sha256(adapter_config),
            "training_receipt_sha256": _sha256(v3_receipt),
            "source_family_heldout_report_sha256": _sha256(heldout_report_path),
        },
        "evaluation_file": {
            "path": str(eval_path.resolve()),
            "sha256": _sha256(eval_path),
            "rows": len(frozen_rows),
        },
        "summary": {
            "group_counts": dict(sorted(group_counts.items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "group_decision_counts": {
                f"{group}/{decision}": count
                for (group, decision), count in sorted(group_decision_counts.items())
            },
            "control_variant_counts": dict(sorted(control_counts.items())),
            "source_rows_audited": len(source_rows),
            "source_target_prompt_intersection": len(prompt_intersection),
            "source_target_example_id_intersection": len(id_intersection),
        },
        "gates": gates,
        "preregistered_model_evaluation_gates": prereg,
        "claim_boundary": config["claim_boundary"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "rows": manifest["evaluation_file"]["rows"],
        "groups": manifest["summary"]["group_counts"],
        "gates": manifest["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
