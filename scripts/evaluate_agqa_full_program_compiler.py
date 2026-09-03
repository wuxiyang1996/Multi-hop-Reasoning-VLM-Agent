#!/usr/bin/env python3
"""Evaluate a frozen question-only compiler on held-out target programs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from motif_transfer.agqa_typed_program import (
    ProgramSyntaxError, compile_receipt, parse_program, serialize_program,
)
from motif_transfer.contracts import stable_hash


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("compiler evaluation is immutable")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    capabilities = json.loads(args.capabilities.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (
        args.data_dir / "validation.jsonl"
    ).open(encoding="utf-8")]
    validation_total = len(rows)
    if args.start_row:
        rows = rows[args.start_row:]
    if args.max_rows:
        # Fixed prefix is deterministic and outcome-blind; it is diagnostic,
        # never a replacement for the full held-out qualification.
        rows = rows[:args.max_rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    counts: Counter[str] = Counter()
    by_root: dict[str, Counter[str]] = {}
    mismatches = []
    with torch.inference_mode():
        for batch in batches(rows, args.batch_size):
            encoded = tokenizer(
                ["compile AGQA: " + row["question"] for row in batch],
                padding=True, truncation=True, max_length=192,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **encoded, max_new_tokens=512, num_beams=1,
            )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, prediction in zip(batch, predictions):
                gold = row["program"]
                root = gold.split("(", 1)[0]
                bucket = by_root.setdefault(root, Counter())
                try:
                    canonical = serialize_program(parse_program(prediction))
                    counts["syntax_valid"] += 1; bucket["syntax_valid"] += 1
                    admission = compile_receipt(
                        canonical, capabilities["authorized_operators"],
                        capabilities.get("authorized_compositions"),
                    )
                    if admission["status"] == "COMPILED":
                        counts["source_admitted"] += 1; bucket["source_admitted"] += 1
                    exact = canonical == gold
                except ProgramSyntaxError:
                    canonical = None
                    exact = False
                counts["rows"] += 1; bucket["rows"] += 1
                if exact:
                    counts["exact"] += 1; bucket["exact"] += 1
                elif len(mismatches) < 200:
                    mismatches.append({
                        "task_id": row["task_id"], "question": row["question"],
                        "gold": gold, "prediction": prediction,
                        "canonical_prediction": canonical,
                    })
    total = counts["rows"]
    metrics = {
        "rows": total,
        "syntax_valid_rate": counts["syntax_valid"] / total,
        "source_admission_rate": counts["source_admitted"] / total,
        "program_exact_rate": counts["exact"] / total,
    }
    end_row = args.start_row + len(rows)
    full = args.start_row == 0 and end_row >= validation_total
    passed = (
        full and metrics["syntax_valid_rate"] >= .995
        and metrics["source_admission_rate"] >= .995
        and metrics["program_exact_rate"] >= .98
    )
    body = {
        "schema_version": "agqa-full-program-compiler-heldout-eval-v1",
        "status": "COMPILER_QUALIFIED" if passed else (
            "DIAGNOSTIC_COMPLETE" if not full else "COMPILER_NOT_QUALIFIED"
        ),
        "authority": "HELDOUT_TARGET_DEVELOPMENT_PROGRAMS;NO_FORMAL_PROGRAM_OR_ANSWER",
        "formal_programs_read": False, "formal_answers_read": False,
        "full_validation": full, "metrics": metrics,
        "row_range": {"start": args.start_row, "end": end_row,
                      "validation_total": validation_total},
        "by_root": {name: dict(value) for name, value in sorted(by_root.items())},
        "mismatch_sample": mismatches,
        "model_dir": str(args.model_dir),
        "source_capability_artifact_sha256": capabilities["artifact_sha256"],
        "supervision_manifest_sha256": manifest["manifest_sha256"],
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "metrics": metrics,
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0 if passed or not full else 1


if __name__ == "__main__":
    raise SystemExit(main())
