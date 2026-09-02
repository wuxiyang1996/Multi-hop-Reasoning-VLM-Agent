#!/usr/bin/env python3
"""Build deterministic AGQA question -> operator-free semantic-slot data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_semantic_slots import serialize_compact_semantic_target
from motif_transfer.agqa_typed_program import parse_program
from motif_transfer.contracts import stable_hash


def selected(task_id: str, split: str, threshold: int) -> bool:
    value = int(hashlib.sha256(f"agqa-layer-b-semantic-v1:{split}:{task_id}".encode()).hexdigest()[:16], 16)
    return value % 1_000_000 < threshold


def root(program: str) -> str:
    expression = parse_program(program)
    return getattr(expression, "function", "ATOM")


def transform(source: Path, output: Path, *, split: str, threshold: int) -> dict:
    rows = kept = 0; roots: Counter[str] = Counter()
    with source.open(encoding="utf-8") as reader, output.open("x", encoding="utf-8") as writer:
        for line in reader:
            rows += 1; row = json.loads(line)
            if not selected(str(row["task_id"]), split, threshold):
                continue
            target = serialize_compact_semantic_target(str(row["program"]))
            if any(token in target for token in ("PROJECT", "TEMPORAL_SELECT", "FILTER_EQ", "UNIQUE")):
                raise ValueError("semantic target leaked VM operator vocabulary")
            writer.write(json.dumps({
                "task_id": row["task_id"], "video_id": row["video_id"],
                "input": "parse AGQA semantics: " + row["question"],
                "target": target, "answer_read": False, "scene_graph_read": False,
                "functional_program_in_model_input": False,
            }, sort_keys=True) + "\n")
            kept += 1; roots[root(str(row["program"]))] += 1
    return {"source_rows": rows, "selected_rows": kept, "root_functions": dict(sorted(roots.items()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-threshold", type=int, default=200_000)
    parser.add_argument("--validation-threshold", type=int, default=200_000)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError("semantic parser data freeze is immutable")
    args.output_dir.mkdir(parents=True)
    train = transform(args.train_source, args.output_dir / "train.jsonl", split="train", threshold=args.train_threshold)
    validation = transform(args.validation_source, args.output_dir / "validation.jsonl", split="validation", threshold=args.validation_threshold)
    gates = {
        "train_has_all_root_families": len(train["root_functions"]) == 8,
        "validation_has_all_root_families": len(validation["root_functions"]) == 8,
        "train_size_at_least_200k": train["selected_rows"] >= 200_000,
        "validation_size_at_least_25k": validation["selected_rows"] >= 25_000,
    }
    body = {
        "schema_version": "agqa-layer-b-semantic-parser-data-v1",
        "status": "SEMANTIC_PARSER_DATA_FROZEN" if all(gates.values()) else "SEMANTIC_PARSER_DATA_GATE_FAILED",
        "selection": "SHA256_THRESHOLD_BY_TASK_ID", "train_threshold": args.train_threshold,
        "validation_threshold": args.validation_threshold, "train": train, "validation": validation,
        "authority": "AGQA_TRAIN_DEVELOPMENT_QUESTION_AND_PROGRAM_TO_OPERATOR_FREE_SEMANTICS_ONLY",
        "answers_read": False, "scene_graphs_read": False, "formal_test_read": False,
        "runtime_functional_program_allowed": False, "gates": gates,
    }
    body["manifest_sha256"] = stable_hash(body)
    (args.output_dir / "manifest.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__": raise SystemExit(main())
