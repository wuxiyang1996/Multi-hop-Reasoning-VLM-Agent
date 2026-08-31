#!/usr/bin/env python3
"""Evaluate exact symbolic execution on source or frozen target-IR rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_by_decision(
    rows: list[dict[str, Any]], per_decision: int,
    balance_field: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = json.loads(row["completion"])
        balance = str(row.get(balance_field, "ALL")) if balance_field else "ALL"
        grouped[(balance, str(target["decision"]))].append(row)
    selected = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                str(row["example_id"]).encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[:per_decision])
    return selected


def _strict_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--per-decision", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=1792)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--evaluation-kind",
        choices=(
            "source_family_heldout",
            "source_multi_ir_heldout",
            "target_development_validation",
            "target_ir_zero_shot",
            "target_model_substitution",
        ),
        default="source_family_heldout",
    )
    parser.add_argument(
        "--balance-field",
        help="Optional audit-only JSONL field used to balance each decision stratum",
    )
    parser.add_argument(
        "--audit-field",
        action="append",
        default=[],
        help="Additional audit-only field to report and gate; may be repeated",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        help="Frozen V4 manifest binding the dataset and source-only adapter hashes",
    )
    args = parser.parse_args()

    manifest_required_kinds = {
        "target_ir_zero_shot", "target_model_substitution",
    }
    if args.evaluation_kind in manifest_required_kinds and args.evaluation_manifest is None:
        parser.error(f"{args.evaluation_kind} requires --evaluation-manifest")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    input_rows = _read_jsonl(args.dataset)
    rows = _select_by_decision(input_rows, args.per_decision, args.balance_field)
    if not rows:
        raise SystemExit("held-out controller dataset is empty")
    evaluation_manifest = None
    if args.evaluation_manifest is not None:
        evaluation_manifest = json.loads(
            args.evaluation_manifest.read_text(encoding="utf-8")
        )
        expected_manifest_status = (
            "FROZEN_SIX_BENCHMARK_SUBSTITUTION_EVALUATION_READY"
            if args.evaluation_kind == "target_model_substitution"
            else "FROZEN_TARGET_IR_ZERO_SHOT_EVALUATION_READY"
        )
        if (
            evaluation_manifest.get("status") != expected_manifest_status
            or not all(evaluation_manifest.get("gates", {}).values())
        ):
            raise SystemExit("evaluation manifest is not frozen and gate-clean")
        expected_dataset_hash = evaluation_manifest["evaluation_file"]["sha256"]
        if _sha256(args.dataset) != expected_dataset_hash:
            raise SystemExit("evaluation dataset does not match frozen manifest")
        adapter_file = args.adapter / "adapter_model.safetensors"
        expected_adapter_hash = evaluation_manifest["frozen_model"][
            "adapter_model_sha256"
        ]
        if _sha256(adapter_file) != expected_adapter_hash:
            raise SystemExit("controller adapter does not match frozen manifest")
    if args.evaluation_kind in manifest_required_kinds and len(rows) != len(input_rows):
        raise SystemExit(
            f"{args.evaluation_kind} requires all frozen rows; increase --per-decision"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def generate(adapter_enabled: bool) -> list[str]:
        outputs = []
        context = nullcontext() if adapter_enabled else model.disable_adapter()
        with context, torch.inference_mode():
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                encoded = tokenizer(
                    [row["prompt"] for row in batch], return_tensors="pt",
                    padding=True, truncation=True,
                    max_length=args.max_input_length,
                ).to(model.device)
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                suffix = generated[:, encoded["input_ids"].shape[1]:]
                outputs.extend(tokenizer.batch_decode(
                    suffix, skip_special_tokens=True,
                ))
        return outputs

    regimes = {}
    prediction_rows = []
    for regime, enabled in (("BASE", False), ("CONTROLLER_LORA", True)):
        generated = generate(enabled)
        counters = Counter()
        by_decision: dict[str, Counter] = defaultdict(Counter)
        by_balance_group: dict[str, Counter] = defaultdict(Counter)
        by_audit_field: dict[str, dict[str, Counter]] = {
            field: defaultdict(Counter) for field in args.audit_field
        }
        for row, text in zip(rows, generated):
            target = json.loads(row["completion"])
            parsed = _strict_json(text)
            target_decision = str(target["decision"])
            valid = parsed is not None
            exact = bool(valid and parsed == target)
            decision_correct = bool(
                valid and parsed.get("decision") == target["decision"]
            )
            state_correct = bool(
                valid and parsed.get("next_symbolic_state")
                == target.get("next_symbolic_state")
            )
            binding_correct = bool(
                valid and parsed.get("binding") == target.get("binding")
            )
            reason_correct = bool(
                valid and parsed.get("reason") == target["reason"]
            )
            values = {
                "rows": 1, "valid_json": int(valid), "exact_json": int(exact),
                "decision_correct": int(decision_correct),
                "state_correct": int(state_correct),
                "binding_correct": int(binding_correct),
                "reason_correct": int(reason_correct),
            }
            counters.update(values)
            by_decision[target_decision].update(values)
            if args.balance_field:
                by_balance_group[str(row.get(args.balance_field, "MISSING"))].update(values)
            for field in args.audit_field:
                by_audit_field[field][str(row.get(field, "MISSING"))].update(values)
            prediction_rows.append({
                "regime": regime,
                "example_id": row["example_id"],
                "objective": row["objective"],
                "target_decision": target_decision,
                "generated_text": text,
                "parsed": parsed,
                "target": target,
                "exact_json": exact,
            })

        def rates(bucket: Counter) -> dict[str, Any]:
            total = int(bucket["rows"])
            return {
                "rows": total,
                **{
                    f"{name}_accuracy": bucket[name] / total
                    for name in (
                        "valid_json", "exact_json", "decision_correct",
                        "state_correct", "binding_correct", "reason_correct",
                    )
                },
            }

        regimes[regime] = {
            "overall": rates(counters),
            "by_target_decision": {
                decision: rates(bucket)
                for decision, bucket in sorted(by_decision.items())
            },
            "by_balance_group": {
                group: rates(bucket)
                for group, bucket in sorted(by_balance_group.items())
            },
            "by_audit_field": {
                field: {
                    group: rates(bucket)
                    for group, bucket in sorted(groups.items())
                }
                for field, groups in sorted(by_audit_field.items())
            },
        }

    base_exact = regimes["BASE"]["overall"]["exact_json_accuracy"]
    lora = regimes["CONTROLLER_LORA"]
    lora_exact = lora["overall"]["exact_json_accuracy"]
    non_abstain = [
        values["decision_correct_accuracy"]
        for decision, values in lora["by_target_decision"].items()
        if decision != "ABSTAIN"
    ]
    gates = {
        "strict_json_accuracy_at_least_0p98": (
            lora["overall"]["valid_json_accuracy"] >= 0.98
        ),
        "decision_accuracy_at_least_0p95": (
            lora["overall"]["decision_correct_accuracy"] >= 0.95
        ),
        "exact_symbolic_output_accuracy_at_least_0p90": lora_exact >= 0.90,
        "exact_output_beats_base_by_0p20": lora_exact - base_exact >= 0.20,
        "non_abstain_decision_recall_at_least_0p90": (
            bool(non_abstain) and min(non_abstain) >= 0.90
        ),
    }
    if args.balance_field:
        balance_groups = lora["by_balance_group"]
        gates.update({
            "every_balance_group_decision_accuracy_at_least_0p90": (
                bool(balance_groups) and min(
                    values["decision_correct_accuracy"]
                    for values in balance_groups.values()
                ) >= 0.90
            ),
            "every_balance_group_exact_accuracy_at_least_0p80": (
                bool(balance_groups) and min(
                    values["exact_json_accuracy"]
                    for values in balance_groups.values()
                ) >= 0.80
            ),
        })
    for field in args.audit_field:
        groups = lora["by_audit_field"][field]
        gate_prefix = "".join(
            character if character.isalnum() else "_" for character in field
        ).strip("_")
        gates.update({
            f"every_{gate_prefix}_decision_accuracy_at_least_0p90": (
                bool(groups) and min(
                    values["decision_correct_accuracy"] for values in groups.values()
                ) >= 0.90
            ),
            f"every_{gate_prefix}_exact_accuracy_at_least_0p80": (
                bool(groups) and min(
                    values["exact_json_accuracy"] for values in groups.values()
                ) >= 0.80
            ),
        })
    if args.evaluation_kind == "target_model_substitution":
        balance_groups = lora["by_balance_group"]
        audit_groups = {
            field: lora["by_audit_field"][field]
            for field in args.audit_field
        }
        gates = {
            "all_frozen_rows_selected": len(rows) == len(input_rows),
            "strict_json_accuracy_is_one": (
                lora["overall"]["valid_json_accuracy"] == 1.0
            ),
            "decision_accuracy_is_one": (
                lora["overall"]["decision_correct_accuracy"] == 1.0
            ),
            "exact_route_selection_accuracy_is_one": (
                lora["overall"]["exact_json_accuracy"] == 1.0
            ),
            "every_benchmark_exact_accuracy_is_one": bool(balance_groups) and all(
                values["exact_json_accuracy"] == 1.0
                for values in balance_groups.values()
            ),
            "every_audit_stratum_exact_accuracy_is_one": all(
                groups and all(
                    values["exact_json_accuracy"] == 1.0
                    for values in groups.values()
                )
                for groups in audit_groups.values()
            ),
            "zero_wrong_or_false_positive_program_authorizations": (
                lora["overall"]["exact_json_accuracy"] == 1.0
            ),
        }
    if args.evaluation_kind == "source_family_heldout":
        schema_version = "harness-controller-heldout-evaluation-v2"
        passed_status = "SOURCE_FAMILY_HELD_OUT_CONTROLLER_GATE_PASSED"
        failed_status = "SOURCE_FAMILY_HELD_OUT_CONTROLLER_GATE_FAILED"
        authority = "FROZEN_UNSEEN_SOURCE_FAMILY;GREEDY_EXACT_JSON_EXECUTION"
        claim_boundary = (
            "This proves execution generalization to one unseen source-program "
            "family. Target-domain skill transfer still requires matched target "
            "evaluation with the symbolic verifier and target-native grounder."
        )
    elif args.evaluation_kind == "source_multi_ir_heldout":
        schema_version = "harness-controller-source-multi-ir-evaluation-v1"
        passed_status = "SOURCE_MULTI_IR_HELD_OUT_CONTROLLER_GATE_PASSED"
        failed_status = "SOURCE_MULTI_IR_HELD_OUT_CONTROLLER_GATE_FAILED"
        authority = (
            "FROZEN_SOURCE_ONLY_STRUCTURAL_CONTRACTS;HELD_OUT_ALPHA_RENAMING;"
            "GREEDY_EXACT_JSON_SELECTION"
        )
        claim_boundary = (
            "This evaluates held-out anonymous catalog permutations and controls "
            "for the trained source-induced IR schemas. It is not a held-out "
            "source family and does not by itself prove target-domain success."
        )
    elif args.evaluation_kind == "target_development_validation":
        schema_version = "harness-controller-target-development-evaluation-v1"
        passed_status = "TARGET_DEVELOPMENT_CONTROLLER_GATE_PASSED"
        failed_status = "TARGET_DEVELOPMENT_CONTROLLER_GATE_FAILED"
        authority = (
            "TARGET_DEVELOPMENT_VALIDATION;GREEDY_EXACT_JSON_EXECUTION;"
            "NO_FORMAL_OR_QUALIFICATION_TARGETS"
        )
        claim_boundary = (
            "This is a target-development controller diagnostic. It cannot prove "
            "target transfer and cannot replace untouched formal evaluation."
        )
    elif args.evaluation_kind == "target_ir_zero_shot":
        schema_version = "harness-controller-target-ir-zero-shot-evaluation-v4"
        passed_status = "TARGET_IR_ZERO_SHOT_CONTROLLER_GATE_PASSED"
        failed_status = "TARGET_IR_ZERO_SHOT_CONTROLLER_GATE_FAILED"
        authority = (
            "FROZEN_SOURCE_ONLY_V3_CONTROLLER;ALL_FROZEN_TARGET_IR_ROWS;"
            "GREEDY_EXACT_JSON_EXECUTION;NO_TARGET_WEIGHT_UPDATE;"
            "NO_FORMAL_OR_RESERVE_TARGETS"
        )
        claim_boundary = (
            "A pass proves that the source-only 9B controller executes anonymous "
            "symbolic programs on target-grounded IR without target weight updates. "
            "It does not prove target-grounder causality, formal downstream success, "
            "or non-heuristic video grounding."
        )
    else:
        schema_version = "harness-controller-six-benchmark-substitution-evaluation-v1"
        passed_status = "SIX_BENCHMARK_MODEL_SUBSTITUTION_ROUTE_GATE_PASSED"
        failed_status = "SIX_BENCHMARK_MODEL_SUBSTITUTION_ROUTE_GATE_FAILED"
        authority = (
            "ALL_PREOUTCOME_MANIFEST_TASK_IDENTITIES;GREEDY_EXACT_JSON_SELECTION;"
            "SEVEN_SOURCE_PROGRAMS;FIVE_ANONYMOUS_IR_SCHEMAS;"
            "NO_TARGET_WEIGHT_UPDATE;NO_TARGET_OUTCOME_FILTER"
        )
        claim_boundary = (
            "A pass proves exact 9B substitution for every source-program route "
            "decision in six previously validated target benchmarks. Existing "
            "success can be bridged only after the separate native receipt and "
            "success-critical action-equivalence audit passes. This is locked "
            "retrospective model substitution, not a fresh target success run."
        )
    payload = {
        "schema_version": schema_version,
        "status": (
            passed_status if all(gates.values()) else failed_status
        ),
        "authority": authority,
        "evaluation_kind": args.evaluation_kind,
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "adapter_model_sha256": _sha256(args.adapter / "adapter_model.safetensors"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "evaluation_manifest": (
            {
                "path": str(args.evaluation_manifest.resolve()),
                "sha256": _sha256(args.evaluation_manifest),
                "status": evaluation_manifest.get("status"),
            }
            if args.evaluation_manifest is not None else None
        ),
        "selection": {
            "method": "SHA256_BALANCED_BY_TARGET_DECISION",
            "per_decision": args.per_decision,
            "balance_field": args.balance_field,
            "rows": len(rows),
            "input_rows": len(input_rows),
            "all_input_rows_selected": len(rows) == len(input_rows),
            "balance_group_counts": dict(sorted(Counter(
                str(row.get(args.balance_field, "ALL"))
                if args.balance_field else "ALL"
                for row in rows
            ).items())),
            "target_decision_counts": dict(sorted(Counter(
                json.loads(row["completion"])["decision"] for row in rows
            ).items())),
            "example_ids": [row["example_id"] for row in rows],
        },
        "regimes": regimes,
        "controller_lora_minus_base_exact_accuracy": lora_exact - base_exact,
        "gates": gates,
        "claim_boundary": claim_boundary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row in prediction_rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": payload["status"], "rows": len(rows),
        "base_exact_accuracy": base_exact,
        "controller_lora_exact_accuracy": lora_exact,
        "gates": gates, "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
