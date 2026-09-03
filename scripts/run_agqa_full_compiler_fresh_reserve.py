#!/usr/bin/env python3
"""Freeze compiler outputs on fresh questions before opening formal fields."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from motif_transfer.agqa_typed_program import compile_receipt
from motif_transfer.contracts import stable_hash


def batches(values, size):
    for start in range(0, len(values), size): yield values[start:start + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--compiler-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("fresh compiler outputs are immutable")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    capabilities = json.loads(args.capabilities.read_text(encoding="utf-8"))
    qualification = json.loads(args.compiler_qualification.read_text(encoding="utf-8"))
    if qualification["status"] != "COMPILER_QUALIFIED":
        raise ValueError("full held-out compiler qualification did not pass")
    if cohort["answer_read"] or cohort["functional_program_read"]:
        raise ValueError("fresh cohort boundary already crossed")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    rows = cohort["rows"]; outputs = []; counts: Counter[str] = Counter()
    with torch.inference_mode():
        for batch in batches(rows, args.batch_size):
            encoded = tokenizer(["compile AGQA: " + row["question"] for row in batch],
                                padding=True, truncation=True, max_length=192,
                                return_tensors="pt").to("cuda")
            generated = model.generate(**encoded, max_new_tokens=512, num_beams=1)
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, program in zip(batch, predictions):
                admission = compile_receipt(
                    program, capabilities["authorized_operators"],
                    capabilities["authorized_compositions"],
                )
                counts[admission["status"]] += 1
                outputs.append({
                    "task_id": row["task_id"], "video_id": row["video_id"],
                    "question_sha256": stable_hash(row["question"]),
                    "predicted_program": program,
                    "program_admission": admission,
                    "answer_read": False, "oracle_program_read": False,
                })
    body = {
        "schema_version": "agqa-full-fresh-compiler-runtime-v1",
        "status": "FRESH_PROGRAMS_FROZEN_BEFORE_OUTCOME_ACCESS",
        "cohort_sha256": cohort["cohort_sha256"],
        "compiler_qualification_report_sha256": qualification["report_sha256"],
        "source_capability_artifact_sha256": capabilities["artifact_sha256"],
        "counts": dict(counts), "rows": outputs,
        "answer_read": False, "oracle_program_read": False,
    }
    body["runtime_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "counts": body["counts"],
                      "runtime_sha256": body["runtime_sha256"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
