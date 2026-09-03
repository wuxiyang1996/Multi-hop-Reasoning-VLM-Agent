#!/usr/bin/env python3
"""Build source-only SFT for the heterogeneous Phase-8 program selector."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.multi_ir_selector_training import (  # noqa: E402
    SELECT_SKILL,
    build_multi_ir_selector_examples,
    format_multi_ir_selector_prompt,
)
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    SourceIRContract,
    goal_acquisition_artifact_contract,
    goal_relation_artifact_contract,
    relational_artifact_contract,
    structural_program_contract,
    temporal_function_artifact_contract,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _load_contracts(
    config: Mapping[str, Any],
) -> tuple[list[SourceIRContract], dict[str, str], dict[str, Any]]:
    source = config["source_artifacts"]
    structural_manifest_path = _resolve(source["structural_manifest"])
    structural_report_path = _resolve(source["structural_confirmation"])
    structural_manifest = _read(structural_manifest_path)
    structural_report = _read(structural_report_path)
    _self_hash(structural_manifest, "manifest_sha256")
    _self_hash(structural_report, "report_sha256")
    if not (
        structural_report.get("status") == "SOURCE_STRUCTURAL_FRESH_VALIDATED"
        and all((structural_report.get("gates") or {}).values())
        and structural_report.get("manifest_sha256")
        == structural_manifest.get("manifest_sha256")
    ):
        raise ValueError("finite structural source qualification failed")

    contracts: list[SourceIRContract] = []
    confirmations: dict[str, str] = {}
    input_files: dict[str, Any] = {
        "structural_manifest": {
            "path": str(structural_manifest_path.resolve()),
            "sha256": _sha(structural_manifest_path),
        },
        "structural_confirmation": {
            "path": str(structural_report_path.resolve()),
            "sha256": _sha(structural_report_path),
        },
    }
    for name, record in sorted(structural_manifest["source_programs"].items()):
        program_path = _resolve(record["path"])
        if _sha(program_path) != record["file_sha256"]:
            raise ValueError(f"frozen structural file hash drifted: {name}")
        program = _read(program_path)
        if program.get("program_sha256") != record["program_sha256"]:
            raise ValueError(f"frozen structural program hash drifted: {name}")
        contract = structural_program_contract(
            program,
            source_confirmation_sha256=structural_report["report_sha256"],
            source_intervention_qualified=True,
        )
        contracts.append(contract)
        confirmations[contract.program_sha256] = structural_report[
            "report_sha256"
        ]
        input_files[f"structural_program_{name}"] = {
            "path": str(program_path.resolve()), "sha256": _sha(program_path),
        }

    relational_artifact_path = _resolve(source["relational_artifact"])
    relational_confirmation_path = _resolve(source["relational_confirmation"])
    relational_artifact = _read(relational_artifact_path)
    relational_confirmation = _read(relational_confirmation_path)
    _self_hash(relational_artifact, "artifact_sha256")
    _self_hash(relational_confirmation, "report_sha256")
    if not (
        relational_confirmation.get("status")
        == "SOURCE_RELATIONAL_STRUCTURAL_FRESH_VALIDATED"
        and relational_confirmation.get("source_gate_passed") is True
        and all((relational_confirmation.get("gates") or {}).values())
        and relational_confirmation.get("artifact_sha256")
        == relational_artifact.get("artifact_sha256")
    ):
        raise ValueError("relational source qualification failed")
    relational = relational_artifact_contract(
        relational_artifact,
        source_confirmation_sha256=relational_confirmation["report_sha256"],
        source_intervention_qualified=True,
    )
    contracts.append(relational)
    confirmations[relational.program_sha256] = relational_confirmation[
        "report_sha256"
    ]
    input_files["relational_artifact"] = {
        "path": str(relational_artifact_path.resolve()),
        "sha256": _sha(relational_artifact_path),
    }
    input_files["relational_confirmation"] = {
        "path": str(relational_confirmation_path.resolve()),
        "sha256": _sha(relational_confirmation_path),
    }

    acquisition_artifact_path = _resolve(source["acquisition_artifact"])
    acquisition_confirmation_path = _resolve(source["acquisition_confirmation"])
    acquisition_artifact = _read(acquisition_artifact_path)
    acquisition_confirmation = _read(acquisition_confirmation_path)
    acquisition = goal_acquisition_artifact_contract(
        acquisition_artifact, confirmation=acquisition_confirmation,
    )
    if not acquisition.source_intervention_qualified:
        raise ValueError("goal-acquisition source qualification failed")
    contracts.append(acquisition)
    confirmations[acquisition.program_sha256] = acquisition_confirmation[
        "report_sha256"
    ]
    input_files["acquisition_artifact"] = {
        "path": str(acquisition_artifact_path.resolve()),
        "sha256": _sha(acquisition_artifact_path),
    }
    input_files["acquisition_confirmation"] = {
        "path": str(acquisition_confirmation_path.resolve()),
        "sha256": _sha(acquisition_confirmation_path),
    }

    goal_relation_artifact_path = _resolve(source["goal_relation_artifact"])
    goal_relation_confirmation_path = _resolve(
        source["goal_relation_confirmation"]
    )
    goal_relation_artifact = _read(goal_relation_artifact_path)
    goal_relation_confirmation = _read(goal_relation_confirmation_path)
    goal_relation = goal_relation_artifact_contract(
        goal_relation_artifact, confirmation=goal_relation_confirmation,
    )
    if not goal_relation.source_intervention_qualified:
        raise ValueError("goal-relation source qualification failed")
    contracts.append(goal_relation)
    confirmations[goal_relation.program_sha256] = goal_relation_confirmation[
        "report_sha256"
    ]
    input_files["goal_relation_artifact"] = {
        "path": str(goal_relation_artifact_path.resolve()),
        "sha256": _sha(goal_relation_artifact_path),
    }
    input_files["goal_relation_confirmation"] = {
        "path": str(goal_relation_confirmation_path.resolve()),
        "sha256": _sha(goal_relation_confirmation_path),
    }

    # Optional source-only temporal functions extend the catalog without
    # changing the V1 build.  Each function must have been independently
    # confirmed on held-out source interventions.  The source-game label is
    # used only to bind the correct confirmation row and is never serialized
    # into a model prompt or completion.
    temporal_specs = source.get("temporal_function_artifacts") or ()
    for index, spec in enumerate(temporal_specs):
        artifact_path = _resolve(spec["artifact"])
        confirmation_path = _resolve(spec["confirmation"])
        artifact = _read(artifact_path)
        confirmation = _read(confirmation_path)
        _self_hash(confirmation, "report_sha256")
        source_game = str(spec["source_game"])
        matching_lineages = [
            row for row in confirmation.get("lineages") or ()
            if str(row.get("source_game")) == source_game
        ]
        if len(matching_lineages) != 1:
            raise ValueError(
                f"temporal source confirmation is ambiguous: {source_game}"
            )
        lineage = matching_lineages[0]
        program = artifact.get("source_function_program") or {}
        confirmed = bool(
            lineage.get("status") == "V4_SOURCE_DOMAIN_FUNCTION_CONFIRMED"
            and lineage.get("source_function_program_sha256")
            == program.get("program_sha256")
            and all((lineage.get("gates") or {}).values())
        )
        temporal = temporal_function_artifact_contract(
            artifact,
            source_confirmation_sha256=confirmation["report_sha256"],
            source_intervention_qualified=confirmed,
        )
        if not temporal.source_intervention_qualified:
            raise ValueError(
                f"temporal source qualification failed: {source_game}"
            )
        contracts.append(temporal)
        confirmations[temporal.program_sha256] = confirmation["report_sha256"]
        input_files[f"temporal_artifact_{index}"] = {
            "path": str(artifact_path.resolve()), "sha256": _sha(artifact_path),
        }
        input_files[f"temporal_confirmation_{index}"] = {
            "path": str(confirmation_path.resolve()),
            "sha256": _sha(confirmation_path),
        }

    if len({row.program_sha256 for row in contracts}) != len(contracts):
        raise ValueError("source catalog contains duplicate program hashes")
    return contracts, confirmations, input_files


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/harness_multi_ir_selector_sft_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config)
    contracts, confirmations, input_files = _load_contracts(config)
    ir_kind_by_program = {
        row.program_sha256: row.ir_kind for row in contracts
    }
    split_repetitions = {
        split: range(int(bounds[0]), int(bounds[1]))
        for split, bounds in config["split_repetition_ranges"].items()
    }
    if set(split_repetitions) != {"train", "validation", "source_held_out"}:
        raise ValueError("split repetition ranges are incomplete")
    integer_sets = [set(value) for value in split_repetitions.values()]
    if any(
        left & right
        for index, left in enumerate(integer_sets)
        for right in integer_sets[index + 1:]
    ):
        raise ValueError("split repetition ranges overlap")

    examples = []
    for split, repetitions in split_repetitions.items():
        examples.extend(build_multi_ir_selector_examples(
            contracts=contracts, split=split, repetitions=repetitions,
            confirmation_by_program=confirmations,
        ))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    structured_path = args.output_dir / "structured.jsonl"
    split_paths = {
        split: args.output_dir / f"{split}.jsonl" for split in split_repetitions
    }
    streams = {
        split: path.open("w", encoding="utf-8")
        for split, path in split_paths.items()
    }
    prompts: dict[str, set[str]] = defaultdict(set)
    decisions: dict[str, Counter[str]] = defaultdict(Counter)
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        with structured_path.open("w", encoding="utf-8") as structured:
            for example in sorted(examples, key=lambda row: row.example_id):
                structured.write(json.dumps(
                    asdict(example), sort_keys=True, ensure_ascii=False,
                ) + "\n")
                prompt = format_multi_ir_selector_prompt(example.input_payload)
                completion = json.dumps(
                    example.target_payload, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":"),
                )
                if prompt in prompts[example.split]:
                    raise ValueError("duplicate prompt within split")
                prompts[example.split].add(prompt)
                decisions[example.split][example.target_payload["decision"]] += 1
                variants[example.split][example.control_variant] += 1
                model_row = {
                    "example_id": example.example_id,
                    "objective": "SELECT_TRANSFER_PROGRAM",
                    "prompt": prompt,
                    "completion": completion,
                    "control_variant_audit_only": example.control_variant,
                    "source_ir_kind_audit_only": ir_kind_by_program[
                        example.source_program_sha256
                    ],
                    "evidence_receipt_ids": list(example.evidence_receipt_ids),
                    "derivation": example.derivation,
                }
                streams[example.split].write(json.dumps(
                    model_row, sort_keys=True, ensure_ascii=False,
                ) + "\n")
    finally:
        for stream in streams.values():
            stream.close()

    prompt_sets = list(prompts.values())
    expected_contract_count = int(config.get("expected_source_program_count", 6))
    required_ir_kinds = set(config.get("required_ir_kinds") or {
            "FINITE_STRUCTURAL_DELTA_SEQUENCE",
            "RECURRENT_GOAL_RELATION_PROGRAM",
            "RECURRENT_RELATIONAL_TRANSITION_PROGRAM",
            "RECURRENT_GOAL_ACQUISITION_RELATION_PROGRAM",
    })
    count_gate_name = (
        "six_distinct_source_programs_present"
        if expected_contract_count == 6
        else "expected_distinct_source_programs_present"
    )
    kind_gate_name = (
        "four_required_ir_kinds_present"
        if len(required_ir_kinds) == 4
        else "expected_required_ir_kinds_present"
    )
    gates = {
        count_gate_name: len(contracts) == expected_contract_count,
        kind_gate_name: {
            row.ir_kind for row in contracts
        } == required_ir_kinds,
        "all_source_contracts_qualified": all(
            row.source_intervention_qualified for row in contracts
        ),
        "no_target_data_used": all(not row.target_data_used for row in examples),
        "all_examples_validate": all(row.validate() for row in examples),
        "select_and_abstain_in_every_split": all(
            set(decisions[split]) == {SELECT_SKILL, "ABSTAIN"}
            for split in split_repetitions
        ),
        "every_control_in_every_split": all(
            len(variants[split]) == len(contracts) + 7
            for split in split_repetitions
        ),
        "no_prompt_overlap_across_splits": all(
            left.isdisjoint(right)
            for index, left in enumerate(prompt_sets)
            for right in prompt_sets[index + 1:]
        ),
        "no_source_identity_in_model_prompts": all(
            all(
                token not in prompt
                for token in (
                    "sokoban", "minigrid", "doorkey", "put_near",
                    "unlock_pickup",
                )
            )
            for split_prompts in prompts.values() for prompt in split_prompts
        ),
        "no_target_domain_or_native_action_in_model_prompts": all(
            "target_domain" not in prompt and "native_action" not in prompt
            for split_prompts in prompts.values() for prompt in split_prompts
        ),
    }
    if not all(gates.values()):
        raise SystemExit(f"multi-IR SFT gates failed: {gates}")

    manifest_body = {
        "schema_version": str(config.get(
            "output_schema_version", "harness-multi-ir-selector-sft-v1",
        )),
        "status": "FROZEN_SOURCE_ONLY_MULTI_IR_SELECTOR_SUPERVISION",
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "authority": (
            "LABELS_FROM_FROZEN_SOURCE_INDUCED_STRUCTURAL_CONTRACTS_ONLY;"
            "NO_TARGET_EXAMPLE;NO_TARGET_OUTCOME;NO_NATIVE_ACTION;"
            "NO_NAMED_POLICY_TEMPLATE"
        ),
        "input_files": input_files,
        "source_contracts_audit_only": [
            {
                "program_sha256": row.program_sha256,
                "contract_sha256": row.contract_sha256,
                "ir_kind": row.ir_kind,
                "confirmation_sha256": confirmations[row.program_sha256],
            }
            for row in contracts
        ],
        "split_repetition_ranges": config["split_repetition_ranges"],
        "files": {
            "structured": {
                "path": str(structured_path.resolve()),
                "sha256": _sha(structured_path),
                "examples": len(examples),
            },
            **{
                split: {
                    "path": str(path.resolve()), "sha256": _sha(path),
                    "examples": len(prompts[split]),
                    "decisions": dict(sorted(decisions[split].items())),
                    "control_variants": dict(sorted(variants[split].items())),
                }
                for split, path in split_paths.items()
            },
        },
        "gates": gates,
        "claim_boundary": str(config.get("claim_boundary")),
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    _write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "examples": len(examples),
        "files": manifest["files"],
        "gates": gates,
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
