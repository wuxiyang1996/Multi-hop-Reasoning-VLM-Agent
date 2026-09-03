#!/usr/bin/env python3
"""Bind a source-qualified V1 adapter to the five-schema source-only update."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST = REPO / "configs/harness_controller_qwen35_9b_mixed_v2_request.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_spec(path: Path, *, examples: int | None = None) -> dict[str, Any]:
    value = {"path": str(path.resolve()), "sha256": _sha(path)}
    if examples is not None:
        value["examples"] = examples
    return value


def _rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def freeze(request_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    request = _read(request_path)
    if request.get("status") not in {
        "REQUEST_FROZEN_BEFORE_FIVE_SCHEMA_WEIGHT_UPDATE",
        "REQUEST_FROZEN_BEFORE_SOURCE_ONLY_PERMUTATION_UPDATE",
    }:
        raise ValueError("V2 protocol request is not frozen")

    dataset_dir = _resolve(str(request["mixed_source_dataset_dir"]))
    manifest_path = dataset_dir / "manifest.json"
    manifest = _read(manifest_path)
    if not (
        manifest.get("status") == "FROZEN_SOURCE_ONLY_MIXED_HARNESS_SUPERVISION"
        and manifest.get("target_data_used") is False
        and manifest.get("target_outcome_used_for_controller_labels") is False
        and all((manifest.get("gates") or {}).values())
    ):
        raise ValueError("V2 mixed source dataset is not source-only and gate-clean")
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    scalar_path = _resolve(str(request["scalar_heldout"]))
    multi_path = _resolve(str(request["multi_ir_heldout"]))

    initial_adapter = _resolve(str(request["initial_adapter"]))
    initial_model = initial_adapter / "adapter_model.safetensors"
    initial_config = initial_adapter / "adapter_config.json"
    initial_receipt_path = _resolve(str(request["initial_training_receipt"]))
    initial_qualification_path = _resolve(str(request["initial_source_qualification"]))
    initial_receipt = _read(initial_receipt_path)
    initial_qualification = _read(initial_qualification_path)
    initial_qualification_status = str(request.get(
        "initial_source_qualification_status", "SOURCE_MIXED_HARNESS_GATE_PASSED",
    ))
    if not (
        initial_qualification.get("status") == initial_qualification_status
        and all((initial_qualification.get("gates") or {}).values())
    ):
        raise ValueError("V1 mixed adapter did not pass both source-only gates")
    forbidden = (
        "target_data_used", "target_outcome_used_for_controller_labels",
        "formal_or_qualification_targets_used", "video_target_data_used",
        "target_grounder_training_used_target_outcomes",
    )
    if any(initial_receipt.get(field) is not False for field in forbidden):
        raise ValueError("V1 initial adapter was not trained source-only")

    substitution_path = _resolve(str(request["substitution_preregistration"]))
    substitution = _read(substitution_path)
    substitution_status = str(request.get(
        "expected_substitution_status",
        "FROZEN_BEFORE_FIVE_SCHEMA_9B_WEIGHT_UPDATE_OR_SUBSTITUTION_INFERENCE",
    ))
    if not (
        substitution.get("status") == substitution_status
        and all((substitution.get("gates") or {}).values())
    ):
        raise ValueError("six-benchmark substitution set is not frozen")
    substitution_dataset = Path(substitution["route_selector_replay"]["path"])
    if _sha(substitution_dataset) != substitution["route_selector_replay"]["sha256"]:
        raise ValueError("substitution dataset hash drifted")

    scalar_gates = dict(request["source_held_out_gates"]["scalar_executor"])
    scalar_gates.update(_file_spec(scalar_path, examples=_rows(scalar_path)))
    multi_gates = dict(request["source_held_out_gates"]["multi_ir_selector"])
    multi_gates.update(_file_spec(multi_path, examples=_rows(multi_path)))
    body = {
        "schema_version": str(request.get(
            "output_protocol_schema_version",
            "harness-controller-qwen35-9b-mixed-protocol-v2",
        )),
        "status": "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES",
        "model": str(request["model"]),
        "request": {"path": str(request_path.resolve()), "sha256": _sha(request_path)},
        "mixed_source_dataset": {
            "manifest": _file_spec(manifest_path),
            "train": _file_spec(train_path, examples=_rows(train_path)),
            "validation": _file_spec(validation_path, examples=_rows(validation_path)),
            "target_examples": 0,
        },
        "source_held_out_gates": {
            "scalar_executor": scalar_gates,
            "multi_ir_selector": multi_gates,
        },
        "initial_adapter": {
            "role": str(request.get(
                "initial_adapter_role", "SOURCE_ONLY_FOUR_SCHEMA_MIXED_CONTROLLER",
            )),
            "path": str(initial_adapter.resolve()),
            "adapter_model_sha256": _sha(initial_model),
            "adapter_config_sha256": _sha(initial_config),
            "source_qualification": str(initial_qualification_path.resolve()),
            "source_qualification_status": initial_qualification_status,
            "source_qualification_sha256": _sha(initial_qualification_path),
            "training_receipt_sha256": _sha(initial_receipt_path),
        },
        "six_benchmark_substitution": {
            "preregistration": _file_spec(substitution_path),
            "preregistration_status": substitution_status,
            "dataset": _file_spec(
                substitution_dataset,
                examples=int(substitution["route_selector_replay"]["rows"]),
            ),
            "native_replay_index": {
                "path": substitution["native_replay_index"]["path"],
                "sha256": substitution["native_replay_index"]["sha256"],
                "tasks": substitution["native_replay_index"]["tasks"],
            },
            "gates": substitution["preregistered_gates"],
        },
        "training": dict(request["training"]),
        "claim_boundary": str(request["claim_boundary"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = freeze(args.request.resolve(), args.output.resolve())
    print(json.dumps({
        "status": protocol["status"],
        "initial_adapter": protocol["initial_adapter"],
        "mixed_source_dataset": protocol["mixed_source_dataset"],
        "six_benchmark_substitution": protocol["six_benchmark_substitution"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
