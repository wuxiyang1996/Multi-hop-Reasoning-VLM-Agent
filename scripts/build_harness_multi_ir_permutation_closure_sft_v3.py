#!/usr/bin/env python3
"""Build balanced source-only catalog-permutation closure supervision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.multi_ir_selector_training import (  # noqa: E402
    ABSTAIN,
    CONTROL_VARIANTS,
    SELECT_SKILL,
    MultiIRSelectorExample,
    _make_example,
    _mutated_requirement,
    _renamed_catalog,
    format_multi_ir_selector_prompt,
    requirement_from_contract,
)
from motif_transfer.structural_ir_applicability import SourceIRContract  # noqa: E402
from scripts.build_harness_multi_ir_selector_sft_v1 import _load_contracts  # noqa: E402


DEFAULT_CONFIG = REPO / "configs/harness_multi_ir_permutation_closure_sft_v3.json"
DEFAULT_OUTPUT = REPO / "runs/harness_multi_ir_permutation_closure_sft_v3"
ABSTENTION_VARIANTS = (
    *CONTROL_VARIANTS,
    "AMBIGUOUS_DUPLICATE_CONTRACT",
    "SOURCE_PROGRAM_UNQUALIFIED",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _abstention_example(
    *, contract: SourceIRContract, split: str, variant: str,
    catalog: Sequence[Mapping[str, Any]], requirement: Mapping[str, Any],
    matched_alias: str, evidence_receipt_ids: Sequence[str], seed: str,
) -> MultiIRSelectorExample:
    if variant in CONTROL_VARIANTS:
        control_catalog = [dict(row) for row in catalog]
        control_requirement = _mutated_requirement(requirement, variant)
    elif variant == "AMBIGUOUS_DUPLICATE_CONTRACT":
        control_catalog = [dict(row) for row in catalog]
        duplicate = dict(next(
            row for row in catalog if str(row["catalog_id"]) == matched_alias
        ))
        duplicate["catalog_id"] = f"P{len(catalog)}_{seed[:10]}"
        control_catalog.append(duplicate)
        control_requirement = dict(requirement)
    elif variant == "SOURCE_PROGRAM_UNQUALIFIED":
        control_catalog = []
        for row in catalog:
            candidate = dict(row)
            if str(candidate["catalog_id"]) == matched_alias:
                candidate["source_intervention_qualified"] = False
            control_catalog.append(candidate)
        control_requirement = dict(requirement)
    else:
        raise ValueError(f"unknown abstention variant: {variant}")
    example = _make_example(
        contract=contract, split=split, control_variant=variant,
        catalog=control_catalog, requirement=control_requirement,
        evidence_receipt_ids=evidence_receipt_ids,
    )
    if example.target_payload["decision"] != ABSTAIN:
        raise ValueError("permutation-closure negative did not abstain")
    return example


def _closure_examples(
    *, contracts: Sequence[SourceIRContract], confirmations: Mapping[str, str],
    split: str, repetitions: range,
) -> list[MultiIRSelectorExample]:
    output: list[MultiIRSelectorExample] = []
    for repetition in repetitions:
        for contract_index, contract in enumerate(contracts):
            requirement = requirement_from_contract(contract)
            receipts = (
                contract.contract_sha256,
                confirmations[contract.program_sha256],
            )
            for presentation in range(2):
                seed = stable_hash({
                    "authority": "SOURCE_ONLY_CATALOG_PERMUTATION_CLOSURE_V3",
                    "split": split,
                    "repetition": repetition,
                    "program": contract.program_sha256,
                    "presentation": presentation,
                })
                catalog, aliases = _renamed_catalog(contracts, seed=seed)
                matched_alias = aliases[contract.program_sha256]
                positive = _make_example(
                    contract=contract, split=split,
                    control_variant=f"PERMUTATION_CLOSURE_POSITIVE_{presentation}",
                    catalog=catalog, requirement=requirement,
                    evidence_receipt_ids=receipts,
                )
                if positive.target_payload["decision"] != SELECT_SKILL:
                    raise ValueError("source closure positive did not select")
                output.append(positive)
                variant = ABSTENTION_VARIANTS[
                    (repetition + contract_index + presentation) % len(ABSTENTION_VARIANTS)
                ]
                output.append(_abstention_example(
                    contract=contract, split=split, variant=variant,
                    catalog=catalog, requirement=requirement,
                    matched_alias=matched_alias, evidence_receipt_ids=receipts,
                    seed=seed,
                ))
    return output


def build(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    config = _read(config_path)
    catalog_config_path = _resolve(str(config["source_catalog_config"]))
    catalog_config = _read(catalog_config_path)
    contracts, confirmations, source_inputs = _load_contracts(catalog_config)
    expected_kinds = set(config["required_ir_kinds"])
    if not (
        len(contracts) == int(config["expected_source_program_count"])
        and {row.ir_kind for row in contracts} == expected_kinds
        and all(row.source_intervention_qualified for row in contracts)
    ):
        raise ValueError("source catalog is not the frozen seven-program authority")

    ranges = {
        split: range(int(bounds[0]), int(bounds[1]))
        for split, bounds in config["split_repetition_ranges"].items()
    }
    if set(ranges) != {"train", "validation", "source_held_out"}:
        raise ValueError("closure split ranges are incomplete")
    range_sets = [set(values) for values in ranges.values()]
    if any(
        left & right
        for index, left in enumerate(range_sets)
        for right in range_sets[index + 1:]
    ):
        raise ValueError("closure split repetition ranges overlap")

    examples = {
        split: _closure_examples(
            contracts=contracts, confirmations=confirmations,
            split=split, repetitions=repetitions,
        )
        for split, repetitions in ranges.items()
    }
    retention_root = _resolve(str(config["retention_heldout_dataset_dir"]))
    retention_manifest_path = retention_root / "manifest.json"
    retention_manifest = _read(retention_manifest_path)
    retention_path = retention_root / "source_held_out.jsonl"
    if not (
        retention_manifest.get("status")
        == "FROZEN_SOURCE_ONLY_MULTI_IR_SELECTOR_SUPERVISION"
        and all((retention_manifest.get("gates") or {}).values())
        and _sha(retention_path)
        == retention_manifest["files"]["source_held_out"]["sha256"]
    ):
        raise ValueError("retention source held-out split is not gate-clean")
    retention_rows = _rows(retention_path)

    output_dir.mkdir(parents=True)
    file_specs: dict[str, Any] = {}
    prompt_sets: dict[str, set[str]] = {}
    closure_counts: dict[str, Any] = {}
    ir_by_program = {row.program_sha256: row.ir_kind for row in contracts}
    for split in ("train", "validation", "source_held_out"):
        model_rows = []
        decisions = Counter()
        variants = Counter()
        by_ir = Counter()
        for example in examples[split]:
            if not example.validate():
                raise ValueError("source permutation-closure example failed validation")
            prompt = format_multi_ir_selector_prompt(example.input_payload)
            completion = json.dumps(
                example.target_payload, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"),
            )
            decisions[str(example.target_payload["decision"])] += 1
            variants[example.control_variant] += 1
            by_ir[ir_by_program[example.source_program_sha256]] += 1
            model_rows.append({
                "example_id": example.example_id,
                "objective": "SELECT_TRANSFER_PROGRAM",
                "prompt": prompt,
                "completion": completion,
                "control_variant_audit_only": example.control_variant,
                "source_ir_kind_audit_only": ir_by_program[
                    example.source_program_sha256
                ],
                "evidence_receipt_ids": list(example.evidence_receipt_ids),
                "derivation": "FROZEN_SOURCE_CONTRACT_PERMUTATION_CLOSURE",
            })
        closure_counts[split] = {
            "rows": len(model_rows),
            "decisions": dict(sorted(decisions.items())),
            "control_variants": dict(sorted(variants.items())),
            "by_ir_kind": dict(sorted(by_ir.items())),
        }
        if split == "source_held_out":
            model_rows.extend(dict(row) for row in retention_rows)
        ids = [str(row["example_id"]) for row in model_rows]
        prompts = [str(row["prompt"]) for row in model_rows]
        if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
            raise ValueError(f"duplicate closure row or prompt in {split}")
        prompt_sets[split] = set(prompts)
        model_rows.sort(key=lambda row: hashlib.sha256(
            str(row["example_id"]).encode()
        ).hexdigest())
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in model_rows:
                stream.write(json.dumps(
                    row, sort_keys=True, ensure_ascii=False,
                ) + "\n")
        file_specs[split] = {
            "path": str(path.resolve()), "sha256": _sha(path),
            "examples": len(model_rows),
        }

    prompt_values = list(prompt_sets.values())
    gates = {
        "seven_source_programs_and_five_ir_kinds": (
            len(contracts) == 7 and {row.ir_kind for row in contracts} == expected_kinds
        ),
        "all_source_contracts_intervention_qualified": all(
            row.source_intervention_qualified for row in contracts
        ),
        "balanced_positive_and_abstention_closure_in_every_split": all(
            counts["decisions"].get(SELECT_SKILL)
            == counts["decisions"].get(ABSTAIN)
            for counts in closure_counts.values()
        ),
        "every_source_program_symmetric_in_every_split": all(
            len(set(counts["by_ir_kind"].values())) <= 2
            for counts in closure_counts.values()
        ),
        "all_abstention_controls_covered_in_every_split": all(
            set(ABSTENTION_VARIANTS) <= set(counts["control_variants"])
            for counts in closure_counts.values()
        ),
        "retention_heldout_merged_without_training_leakage": (
            bool(retention_rows)
            and file_specs["source_held_out"]["examples"]
            == closure_counts["source_held_out"]["rows"] + len(retention_rows)
        ),
        "no_prompt_overlap_across_splits": all(
            left.isdisjoint(right)
            for index, left in enumerate(prompt_values)
            for right in prompt_values[index + 1:]
        ),
        "no_target_identity_action_or_outcome_in_closure_prompts": all(
            all(token not in prompt.lower() for token in (
                "webshop", "alfworld", "discoveryworld", "tirbench", "clevrer",
                "agqa2", "native_action", "formal_task_id", "target_outcome",
            ))
            for split, prompts in prompt_sets.items()
            for prompt in prompts
            if split != "source_held_out" or prompt not in {
                str(row["prompt"]) for row in retention_rows
            }
        ),
        "source_heldout_retention_manifest_gate_clean": all(
            retention_manifest["gates"].values()
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"source permutation-closure gates failed: {gates}")
    body = {
        "schema_version": config["output_schema_version"],
        "status": "FROZEN_SOURCE_ONLY_MULTI_IR_SELECTOR_SUPERVISION",
        "authority": (
            "SOURCE_INDUCED_SEVEN_PROGRAM_CONTRACTS_ONLY;SYMMETRIC_CATALOG_"
            "PERMUTATION_AND_ALIAS_CLOSURE;NO_TARGET_EXAMPLE_ERROR_OR_OUTCOME"
        ),
        "config": {"path": str(config_path.resolve()), "sha256": _sha(config_path)},
        "source_catalog_config": {
            "path": str(catalog_config_path.resolve()),
            "sha256": _sha(catalog_config_path),
        },
        "source_inputs": source_inputs,
        "source_contracts_audit_only": [
            {
                "program_sha256": row.program_sha256,
                "contract_sha256": row.contract_sha256,
                "ir_kind": row.ir_kind,
                "confirmation_sha256": confirmations[row.program_sha256],
            }
            for row in contracts
        ],
        "retention_heldout": {
            "manifest": {
                "path": str(retention_manifest_path.resolve()),
                "sha256": _sha(retention_manifest_path),
            },
            "file": {"path": str(retention_path.resolve()), "sha256": _sha(retention_path)},
            "rows": len(retention_rows),
        },
        "split_repetition_ranges": config["split_repetition_ranges"],
        "closure_counts": closure_counts,
        "files": file_specs,
        "target_data_used": False,
        "target_outcome_used_for_controller_labels": False,
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "files": manifest["files"],
        "closure_counts": manifest["closure_counts"],
        "gates": manifest["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
