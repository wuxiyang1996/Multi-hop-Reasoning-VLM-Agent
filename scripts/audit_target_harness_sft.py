#!/usr/bin/env python3
"""Independently audit a frozen target-Harness SFT dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


def _percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[int(fraction * (len(values) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    structured = _read_jsonl(args.dataset_dir / "structured.jsonl")
    model_rows = {
        split: _read_jsonl(args.dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    video_provenance_path = args.dataset_dir / "video_provenance.jsonl"
    video_provenance_rows = (
        _read_jsonl(video_provenance_path)
        if video_provenance_path.is_file() else []
    )
    video_provenance = {
        str(row["state_receipt_sha256"]): row for row in video_provenance_rows
    }
    metadata = {str(row["example_id"]): row for row in structured}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )

    lengths: list[int] = []
    grouped_lengths: dict[tuple[str, int], list[int]] = defaultdict(list)
    prompt_sets: dict[str, set[str]] = {}
    example_ids: list[str] = []
    for split, rows in model_rows.items():
        prompt_sets[split] = {str(row["prompt"]) for row in rows}
        for row in rows:
            example_id = str(row["example_id"])
            raw = metadata[example_id]
            prompt_ids = tokenizer(
                str(row["prompt"]), add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            completion_ids = tokenizer(
                str(row["completion"]) + tokenizer.eos_token,
                add_special_tokens=False, truncation=False,
            )["input_ids"]
            length = len(prompt_ids) + len(completion_ids)
            lengths.append(length)
            example_ids.append(example_id)
            grouped_lengths[(
                str(raw["target_domain"]),
                len(raw["input_payload"]["candidate_effects"]),
            )].append(length)

    lower_prompts = "\n".join(
        str(row["prompt"]).lower()
        for rows in model_rows.values() for row in rows
    )
    lower_completions = "\n".join(
        str(row["completion"]).lower()
        for rows in model_rows.values() for row in rows
    )
    source_families = {str(row["source_family"]) for row in structured}
    control_variants = {str(row["control_variant"]) for row in structured}
    domains = {str(row["target_domain"]) for row in structured}
    forbidden = (
        "webshop", "alfworld", "discoveryworld", "tirbench", "video",
        "clevrer", "agqa", "target_benchmark", "raw_frame", "question",
        "answer", "functional_program", "scene_graph", "anchor", "relation",
        "trajectory", "explicit_relation",
        "selected_action", "expert_action",
        "official_success", "official_reward", "native_actions",
        "target_domain", "go to ", "click('", "fill('", "move ", "take ",
        "teleport_to_object", "pickup", '"use"', "zoom_region",
        "extract_colors",
    )
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in structured:
        pairs[str(row["pair_id"])].append(row)
    domain_split_kind_counts = Counter(
        (
            str(row["target_domain"]), str(row["split"]),
            "authentic" if row["control_variant"]
            == "AUTHENTIC_TARGET_NEURAL_GROUNDING" else "control",
        )
        for row in structured
    )
    task_sets = {
        (domain, split): {
            str(row["target_task_sha256"])
            for row in structured
            if row["target_domain"] == domain and row["split"] == split
        }
        for domain in domains
        for split in ("train", "validation")
    }
    state_sets = {
        split: {
            str(row["evidence_receipt_ids"][0])
            for row in structured if row["split"] == split
        }
        for split in ("train", "validation")
    }
    video_rows = [
        row for row in structured if str(row["target_domain"]) == "video"
    ]
    video_benchmark_split_kind = Counter()
    for row in video_rows:
        receipt = str(row["evidence_receipt_ids"][0])
        meta = video_provenance.get(receipt) or {}
        video_benchmark_split_kind[(
            str(meta.get("target_benchmark")), str(row["split"]),
            "authentic" if row["control_variant"]
            == "AUTHENTIC_TARGET_NEURAL_GROUNDING" else "control",
        )] += 1
    video_groups = {
        split: {
            str(row["video_group_sha256"])
            for row in video_provenance_rows if row.get("split") == split
        }
        for split in ("train", "validation")
    }
    file_hash_gates = {
        name: _sha256(args.dataset_dir / f"{name}.jsonl")
        == str(manifest["files"][name]["sha256"])
        for name in ("structured", "train", "validation")
    }
    if video_provenance_rows:
        file_hash_gates["video_provenance"] = (
            _sha256(video_provenance_path)
            == str(manifest["files"]["video_provenance"]["sha256"])
        )
    video_gates = {
        "video_provenance_unique": (
            not video_provenance_rows
            or len(video_provenance) == len(video_provenance_rows)
        ),
        "all_video_examples_join_provenance": (
            not video_rows
            or all(
                str(row["evidence_receipt_ids"][0]) in video_provenance
                for row in video_rows
            )
        ),
        "exact_two_video_benchmarks": (
            not video_rows
            or {
                str(row["target_benchmark"])
                for row in video_provenance_rows
            } == {"clevrer", "agqa2"}
        ),
        "exact_video_benchmark_split_kind_quotas": (
            not video_rows
            or all(
                video_benchmark_split_kind[(benchmark, split, kind)] == expected
                for benchmark in ("clevrer", "agqa2")
                for split, expected in (("train", 225), ("validation", 75))
                for kind in ("authentic", "control")
            )
        ),
        "video_groups_disjoint_across_splits": (
            not video_rows
            or video_groups["train"].isdisjoint(video_groups["validation"])
        ),
        "video_authority_contract_preserved": (
            not video_rows
            or all(
                row.get("raw_frames_exposed_to_harness") is False
                and row.get("question_or_answer_exposed_to_harness") is False
                and row.get("native_action_or_relation_exposed_to_harness") is False
                and row.get("formal_or_reserve_target_used") is False
                and row.get("target_outcome_used_for_controller_label") is False
                for row in video_provenance_rows
            )
        ),
        "model_video_benchmark_metadata_matches_provenance": (
            not video_rows
            or all(
                row.get("target_benchmark_audit_only")
                == video_provenance[str(row["evidence_receipt_ids"][0])][
                    "target_benchmark"
                ]
                for rows in model_rows.values() for row in rows
                if row.get("target_domain_audit_only") == "video"
            )
        ),
    }
    gates = {
        "manifest_status_frozen": manifest.get("status") in {
            "FROZEN_TARGET_DEVELOPMENT_HARNESS_SUPERVISION",
            "FROZEN_FOUR_TARGET_DEVELOPMENT_HARNESS_SUPERVISION",
            "FROZEN_FIVE_TARGET_DEVELOPMENT_HARNESS_SUPERVISION",
        },
        "manifest_builder_gates_all_pass": all(
            manifest.get("gates", {}).values()
        ),
        "manifest_file_hashes_match": all(file_hash_gates.values()),
        "structured_ids_unique": len(metadata) == len(structured),
        "model_ids_unique": len(example_ids) == len(set(example_ids)),
        "model_ids_match_structured": set(example_ids) == set(metadata),
        "target_domains_match_manifest": domains
        == set(map(str, manifest.get("summary", {}).get("target_domains") or domains)),
        "train_validation_prompts_disjoint": prompt_sets["train"].isdisjoint(
            prompt_sets["validation"]
        ),
        "forbidden_prompt_tokens_absent": not any(
            token in lower_prompts for token in forbidden
        ),
        "source_identity_absent_from_prompts": not any(
            family.lower() in lower_prompts for family in source_families
        ),
        "control_identity_absent_from_prompts": not any(
            variant.lower() in lower_prompts for variant in control_variants
        ),
        "outcome_authority_absent_from_completions": not any(
            token in lower_completions
            for token in ("reward", "success", "native_action")
        ),
        "exact_domain_split_kind_quotas": all(
            domain_split_kind_counts[(domain, split, kind)] == expected
            for domain in domains
            for split, expected in (("train", 450), ("validation", 150))
            for kind in ("authentic", "control")
        ),
        "matched_pairs_complete_and_changed": all(
            len(rows) == 2
            and sum(
                row["control_variant"]
                == "AUTHENTIC_TARGET_NEURAL_GROUNDING"
                for row in rows
            ) == 1
            and len({
                row["target_domain"] for row in rows
            }) == 1
            and len({row["split"] for row in rows}) == 1
            and len({row["source_family"] for row in rows}) == 1
            and rows[0]["target_payload"] != rows[1]["target_payload"]
            for rows in pairs.values()
        ),
        "all_authentic_rows_execute": all(
            row["target_payload"]["decision"] == "EXECUTE_OPERATOR"
            for row in structured
            if row["control_variant"]
            == "AUTHENTIC_TARGET_NEURAL_GROUNDING"
        ),
        "target_tasks_disjoint_across_splits": all(
            task_sets[(domain, "train")].isdisjoint(
                task_sets[(domain, "validation")]
            )
            for domain in domains
        ),
        "target_states_disjoint_across_splits": state_sets["train"].isdisjoint(
            state_sets["validation"]
        ),
        "all_rows_preserve_target_authority_contract": all(
            row.get("target_data_used") is True
            and row.get("target_outcome_used") is False
            and row.get("native_action_exposed") is False
            for row in structured
        ),
        "controller_training_contract_matches_audit": (
            str(manifest.get("controller_training_contract", {}).get("model"))
            == args.model
            and int(manifest.get("controller_training_contract", {}).get(
                "maximum_sequence_length", -1,
            )) == args.max_length
        ),
        "all_sequences_fit": max(lengths) <= args.max_length,
        **video_gates,
    }
    report = {
        "schema_version": "target-harness-sft-independent-audit-v2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "model": args.model,
        "maximum_sequence_length": args.max_length,
        "counts": {
            "structured": len(structured),
            "train": len(model_rows["train"]),
            "validation": len(model_rows["validation"]),
            "decisions": dict(sorted(Counter(
                str(row["target_payload"]["decision"])
                for row in structured
            ).items())),
            "pairs": len(pairs),
            "domain_split_kind": {
                "/".join(key): value
                for key, value in sorted(domain_split_kind_counts.items())
            },
            "video_benchmark_split_kind": {
                "/".join(key): value
                for key, value in sorted(video_benchmark_split_kind.items())
            },
            "video_provenance_rows": len(video_provenance_rows),
        },
        "token_lengths": {
            "minimum": min(lengths),
            "median": _percentile(lengths, 0.5),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "maximum": max(lengths),
            "over_maximum": sum(value > args.max_length for value in lengths),
            "by_domain_and_candidate_count": {
                f"{domain}/{count}": {
                    "examples": len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "over_maximum": sum(
                        value > args.max_length for value in values
                    ),
                }
                for (domain, count), values in sorted(grouped_lengths.items())
            },
        },
        "file_hash_gates": file_hash_gates,
        "gates": gates,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
