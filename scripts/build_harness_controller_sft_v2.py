#!/usr/bin/env python3
"""Build family-held-out SFT for the neural symbolic Harness controller."""

from __future__ import annotations

from dataclasses import asdict
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.harness_controller_training import (  # noqa: E402
    build_controller_sft_examples,
    format_controller_prompt,
    summarize_controller_sft_examples,
)
from motif_transfer.phase3_source_induction import read_jsonl  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    typed_intervention_sets_from_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/harness_controller_sft_v2.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/harness_controller_sft_v2",
    )
    parser.add_argument(
        "--heldout-rows-root", type=Path,
        help="Optional alternate model-unseen source-held-out receipt root.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    config = _read_object(args.config)
    families = config.get("families")
    if not isinstance(families, Mapping) or len(families) != 6:
        raise SystemExit("V2 requires exactly six declared source families")
    program_root = _resolve(str(config["program_root"]))
    development_root = _resolve(str(config["development_rows_root"]))
    reserve_root = _resolve(str(config["fresh_reserve_rows_root"]))
    augment_cardinality = bool(
        config.get("augment_cardinality_equivariance", False)
    )
    missing_schema_all_cardinalities = bool(
        config.get("augment_missing_schema_all_cardinalities", False)
    )
    if missing_schema_all_cardinalities and not augment_cardinality:
        raise SystemExit(
            "all-cardinality missing-schema controls require cardinality augmentation"
        )
    cardinality_grid = tuple(map(int, config.get("cardinality_grid") or ()))
    if augment_cardinality and (
        not cardinality_grid
        or cardinality_grid != tuple(range(2, max(cardinality_grid) + 1))
    ):
        raise SystemExit(
            "cardinality grid must be a consecutive, duplicate-free range starting at two"
        )

    examples = []
    input_receipts = []
    family_splits: dict[str, list[str]] = {
        "train": [], "validation": [], "source_held_out": [],
    }
    for family, spec_value in sorted(families.items()):
        if not isinstance(spec_value, Mapping):
            raise SystemExit(f"invalid family spec: {family}")
        split = str(spec_value["split"])
        if split not in family_splits:
            raise SystemExit(f"invalid split for {family}: {split}")
        family_splits[split].append(str(family))
        program_path = program_root / f"{family}.json"
        if split == "source_held_out" and args.heldout_rows_root is not None:
            rows_root = args.heldout_rows_root
            if not rows_root.is_absolute():
                rows_root = REPO / rows_root
        else:
            rows_root = (
                _resolve(str(spec_value["rows_root"]))
                if "rows_root" in spec_value else
                (development_root if split == "train" else reserve_root)
            )
        rows_path = rows_root / str(family) / "rows.jsonl"
        artifact = _read_object(program_path)
        program = artifact.get("source_function_program")
        if not isinstance(program, Mapping):
            raise SystemExit(f"program artifact malformed: {program_path}")
        intervention_sets, conversion_audit = typed_intervention_sets_from_rows(
            read_jsonl(rows_path),
            primary_horizon=int(spec_value["primary_horizon"]),
        )
        if not intervention_sets:
            raise SystemExit(f"no eligible intervention sets: {family}")
        family_examples = build_controller_sft_examples(
            source_family=str(family), split=split, program=program,
            intervention_sets=intervention_sets,
            augment_retry_equivariance=(
                split == "train"
                and bool(config.get("augment_retry_equivariance", False))
            ),
            augment_cardinality_equivariance=augment_cardinality,
            augment_missing_schema_all_cardinalities=(
                missing_schema_all_cardinalities
            ),
            cardinality_grid=cardinality_grid,
        )
        examples.extend(family_examples)
        input_receipts.append({
            "source_family": str(family),
            "split": split,
            "program_file": str(program_path.relative_to(REPO)),
            "program_file_sha256": _sha256(program_path),
            "source_function_program_sha256": str(program["program_sha256"]),
            "program_status": str(program["status"]),
            "rows_file": str(rows_path.relative_to(REPO)),
            "rows_file_sha256": _sha256(rows_path),
            "eligible_intervention_sets": len(intervention_sets),
            "examples": len(family_examples),
            "conversion_audit": conversion_audit,
        })

    summary = summarize_controller_sft_examples(examples)
    formatted: dict[str, dict[str, list[tuple[Any, str]]]] = {
        split: {} for split in family_splits
    }
    for example in sorted(examples, key=lambda row: row.example_id):
        prompt = format_controller_prompt(
            objective=example.objective,
            input_payload=example.input_payload,
        )
        completion = json.dumps(
            example.target_payload, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
        formatted[example.split].setdefault(prompt, []).append((example, completion))
    ambiguous_prompts = sum(
        len({completion for _, completion in support}) != 1
        for by_prompt in formatted.values() for support in by_prompt.values()
    )
    prompt_sets = [set(formatted[split]) for split in family_splits]
    leaked_control_tokens = tuple(config.get("controls") or ()) + ("control_variant",)
    model_rows = [
        support[0][0]
        for by_prompt in formatted.values() for support in by_prompt.values()
    ]
    sft_summary = {
        "examples": len(model_rows),
        "raw_structured_examples": len(examples),
        "duplicate_prompt_support_removed": len(examples) - len(model_rows),
        "split_counts": {
            split: len(by_prompt) for split, by_prompt in formatted.items()
        },
        "objective_counts": dict(sorted(Counter(
            row.objective for row in model_rows
        ).items())),
        "decision_counts": dict(sorted(Counter(
            str(row.target_payload["decision"]) for row in model_rows
        ).items())),
        "ambiguous_prompt_count": ambiguous_prompts,
    }
    if augment_cardinality:
        sft_summary["candidate_count_by_split"] = {
            split: dict(sorted(Counter(
                len(row.input_payload["candidate_effects"])
                for support in by_prompt.values()
                for row, _ in support[:1]
            ).items()))
            for split, by_prompt in formatted.items()
        }
    family_sets = [set(family_splits[name]) for name in family_splits]
    gates = {
        "exact_six_source_families": len(families) == 6,
        "whole_family_splits_disjoint": all(
            left.isdisjoint(right)
            for index, left in enumerate(family_sets)
            for right in family_sets[index + 1:]
        ),
        "all_three_splits_nonempty": all(family_splits.values()),
        "all_examples_valid": bool(summary["all_valid"]),
        "source_identity_hidden_from_prompts": not summary["source_identity_in_prompt"],
        "native_actions_absent": not summary["native_action_tokens_exported"],
        "target_data_absent": not summary["target_data_used"],
        "named_policy_templates_absent": not summary["named_policy_templates_used"],
        "selection_and_transition_supervision_present": set(
            summary["objective_counts"]
        ) == {"SELECT_OPERATOR", "APPLY_TRANSITION"},
        "execution_and_abstention_targets_present": {
            "EXECUTE_OPERATOR", "ABSTAIN"
        } <= set(sft_summary["decision_counts"]),
        "control_metadata_hidden_from_prompts": not any(
            token in prompt
            for by_prompt in formatted.values() for prompt in by_prompt
            for token in leaked_control_tokens
        ),
        "one_completion_per_prompt": ambiguous_prompts == 0,
        "model_prompt_completion_pairs_deduplicated": (
            sum(len(by_prompt) for by_prompt in formatted.values())
            == len(model_rows)
        ),
        "model_prompts_disjoint_across_splits": all(
            left.isdisjoint(right)
            for index, left in enumerate(prompt_sets)
            for right in prompt_sets[index + 1:]
        ),
    }
    if augment_cardinality:
        gates.update({
            "cardinality_grid_is_consecutive_from_two": (
                cardinality_grid
                == tuple(range(2, max(cardinality_grid, default=1) + 1))
            ),
            "every_cardinality_present_in_train": set(cardinality_grid) <= {
                count for count in sft_summary["candidate_count_by_split"]["train"]
            },
            "every_cardinality_present_in_validation": set(cardinality_grid) <= {
                count for count in sft_summary["candidate_count_by_split"]["validation"]
            },
            "every_cardinality_present_in_source_held_out": set(cardinality_grid) <= {
                count for count in sft_summary["candidate_count_by_split"]["source_held_out"]
            },
        })
    if not all(gates.values()):
        raise SystemExit(f"Harness V2 dataset gates failed: {gates}")

    args.output_dir.mkdir(parents=True)
    structured_path = args.output_dir / "structured.jsonl"
    split_paths = {
        split: args.output_dir / f"{split}.jsonl" for split in family_splits
    }
    streams = {
        split: path.open("w", encoding="utf-8")
        for split, path in split_paths.items()
    }
    try:
        with structured_path.open("w", encoding="utf-8") as structured:
            for example in sorted(examples, key=lambda row: row.example_id):
                structured.write(json.dumps(
                    asdict(example), sort_keys=True, ensure_ascii=False,
                ) + "\n")
            for split, by_prompt in formatted.items():
                for prompt, support in sorted(by_prompt.items()):
                    representative, completion = support[0]
                    model_row = {
                        "example_id": representative.example_id,
                        "objective": representative.objective,
                        "control_variants": sorted({
                            row.control_variant for row, _ in support
                        }),
                        "prompt": prompt,
                        "completion": completion,
                        "derivations": sorted({row.derivation for row, _ in support}),
                        "duplicate_support_count": len(support),
                        "supporting_example_ids": sorted(
                            row.example_id for row, _ in support
                        ),
                        "evidence_receipt_ids": sorted({
                            receipt for row, _ in support
                            for receipt in row.evidence_receipt_ids
                        }),
                    }
                    if augment_cardinality:
                        model_row.update({
                            "candidate_count_audit_only": len(
                                representative.input_payload["candidate_effects"]
                            ),
                            "source_family_audit_only": representative.source_family,
                        })
                    streams[split].write(json.dumps(
                        model_row, sort_keys=True, ensure_ascii=False,
                    ) + "\n")
    finally:
        for stream in streams.values():
            stream.close()

    manifest = {
        "schema_version": "harness-controller-sft-dataset-v2",
        "status": "FROZEN_SOURCE_ONLY_CONTROLLER_SUPERVISION",
        "build_config": str(args.config.resolve()),
        "build_config_sha256": _sha256(args.config),
        "authority": (
            "LABELS_FROM_FROZEN_SOURCE_INDUCED_PROGRAM_EXECUTION_ONLY;"
            "NO_TARGET_OUTCOME;NO_HUMAN_POLICY_TEMPLATE"
        ),
        "family_splits": family_splits,
        "input_receipts": input_receipts,
        "structured_file": {
            "path": str(structured_path.resolve()),
            "sha256": _sha256(structured_path),
        },
        "sft_files": {
            split: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "examples": len(formatted[split]),
            }
            for split, path in split_paths.items()
        },
        "summary": summary,
        "sft_summary": sft_summary,
        "gates": gates,
        "prompt_policy": {
            "game_identity_exposed": False,
            "native_actions_exposed": False,
            "reward_or_success_exposed": False,
            "target_action_authority": False,
            "reasoning_trace_requested": False,
            "json_only": True,
        },
        "training_sequence": [
            "SOURCE_ONLY_CONTROLLER_SFT",
            "HELD_OUT_SOURCE_FAMILY_EXECUTION_GATE",
            "TARGET_DOMAIN_OPD_IF_AND_ONLY_IF_SFT_GATE_PASSES",
            "GRPO_ONLY_AS_LATER_ABLATION_WITH_SYMBOLIC_VERIFIER_REWARD",
        ],
        "claim_boundary": str(config["claim_boundary"]),
    }
    if augment_cardinality:
        manifest.update({
            "target_data_used": False,
            "cardinality_equivariance": {
                "enabled": True,
                "grid": list(cardinality_grid),
                "missing_schema_all_cardinalities": (
                    missing_schema_all_cardinalities
                ),
                "authority": (
                    "FROZEN_SOURCE_INTERVENTION_SETS_AND_SOURCE_PROGRAM_EXECUTOR_ONLY;"
                    "NO_TARGET_EXAMPLES_OR_TARGET_CARDINALITY_COUNTS_READ_BY_BUILDER"
                ),
            },
        })
    _write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "summary": summary,
        "sft_summary": sft_summary,
        "family_splits": family_splits, "gates": gates,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
