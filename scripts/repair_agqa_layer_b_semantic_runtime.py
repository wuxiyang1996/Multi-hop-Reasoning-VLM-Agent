#!/usr/bin/env python3
"""Outcome-blind deterministic length repair for invalid Layer-B semantics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target
from motif_transfer.contracts import stable_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--input-runtime", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("semantic repair output is immutable")
    cohort = json.loads(args.cohort.read_text()); runtime = json.loads(args.input_runtime.read_text())
    if runtime["status"] != "SEMANTIC_RUNTIME_FROZEN_BEFORE_VIDEO_OR_OUTCOME":
        raise ValueError("input semantic runtime is not frozen")
    if runtime["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("semantic repair cohort mismatch")
    invalid = [row for row in runtime["rows"] if row["status"] != "SEMANTIC_SLOTS_FROZEN"]
    if not invalid:
        raise ValueError("semantic runtime has nothing to repair")
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16,
    ).to("cuda").eval()
    repaired = {}
    with torch.inference_mode():
        for old in invalid:
            task_id = str(old["task_id"]); row = public[task_id]
            encoded = tokenizer(
                "parse AGQA semantics: " + str(row["question"]), truncation=True,
                max_length=192, return_tensors="pt",
            ).to("cuda")
            generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, num_beams=1)
            prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
            repair_kind = "DETERMINISTIC_SAME_MODEL_LENGTH_ONLY_512_TO_1024"
            try:
                receipt = parse_compact_semantic_target(
                    prediction, task_id=task_id, question_sha256=row["question_sha256"],
                    parser_sha256=runtime["parser_sha256"],
                    parser_training_authority="AGQA_TRAIN_DEV_TO_OPERATOR_FREE_COMPACT_SEMANTICS",
                )
            except ValueError:
                # A seq2seq decoder can stop after a complete final atom while
                # omitting only trailing call delimiters.  Permit exactly the
                # unique minimal suffix that balances parentheses; never add,
                # remove, or reorder semantic content.
                missing = prediction.count("(") - prediction.count(")")
                if missing <= 0:
                    raise
                balanced = prediction + ")" * missing
                receipt = parse_compact_semantic_target(
                    balanced, task_id=task_id, question_sha256=row["question_sha256"],
                    parser_sha256=runtime["parser_sha256"],
                    parser_training_authority="AGQA_TRAIN_DEV_TO_OPERATOR_FREE_COMPACT_SEMANTICS",
                )
                prediction = balanced
                repair_kind = "UNIQUE_MINIMAL_TRAILING_PARENTHESIS_BALANCE"
            repaired[task_id] = {
                **old, "status": "SEMANTIC_SLOTS_FROZEN", "predicted_semantics": prediction,
                "receipt": asdict(receipt), "reason": None,
                "repair": repair_kind,
            }
    rows = [repaired.get(str(row["task_id"]), row) for row in runtime["rows"]]
    body = {
        **{key: value for key, value in runtime.items()
           if key not in {"rows", "valid", "invalid", "runtime_sha256"}},
        "rows": rows, "valid": len(rows), "invalid": 0,
        "base_runtime_sha256": runtime["runtime_sha256"],
        "repaired_task_ids": sorted(repaired),
        "repair_authority": "QUESTION_ONLY_SAME_FROZEN_MODEL_LENGTH_EXTENSION_NO_OUTCOME_OR_PROGRAM",
        "max_new_tokens_for_invalid_only": args.max_new_tokens,
    }
    body["runtime_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "repaired": sorted(repaired),
                      "runtime_sha256": body["runtime_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
