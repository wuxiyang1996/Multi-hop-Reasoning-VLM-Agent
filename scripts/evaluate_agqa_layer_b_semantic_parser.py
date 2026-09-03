#!/usr/bin/env python3
"""Held-out evaluation for the operator-free AGQA semantic parser."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target
from motif_transfer.agqa_typed_program import parse_program, serialize_program
from motif_transfer.contracts import stable_hash


def batches(rows, size):
    for start in range(0, len(rows), size): yield rows[start:start+size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("semantic parser evaluation is immutable")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (args.data_dir/"validation.jsonl").open()]
    validation_total=len(rows); rows=rows[args.start_row:]
    full = args.start_row==0 and (not args.max_rows or args.max_rows >= validation_total)
    if args.max_rows: rows = rows[:args.max_rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    counts=Counter(); roots: dict[str,Counter[str]]={}; mismatches=[]
    parser_sha=stable_hash({"model_dir":str(args.model_dir),"manifest":manifest["manifest_sha256"]})
    with torch.inference_mode():
        for batch in batches(rows,args.batch_size):
            encoded=tokenizer([r["input"] for r in batch],padding=True,truncation=True,max_length=192,return_tensors="pt").to("cuda")
            generated=model.generate(**encoded,max_new_tokens=512,num_beams=1)
            predictions=tokenizer.batch_decode(generated,skip_special_tokens=True)
            for row,prediction in zip(batch,predictions):
                gold=serialize_program(parse_program(row["target"])); root=gold.split("(",1)[0]
                bucket=roots.setdefault(root,Counter()); counts["rows"]+=1; bucket["rows"]+=1
                try:
                    canonical=serialize_program(parse_program(prediction))
                    parse_compact_semantic_target(canonical,task_id=row["task_id"],
                        question_sha256=stable_hash(row["input"]),parser_sha256=parser_sha,
                        parser_training_authority="AGQA_TRAIN_DEV_TO_OPERATOR_FREE_COMPACT_SEMANTICS")
                    counts["valid"]+=1; bucket["valid"]+=1
                except Exception:
                    canonical=None
                if canonical==gold: counts["exact"]+=1; bucket["exact"]+=1
                elif len(mismatches)<200: mismatches.append({"task_id":row["task_id"],"gold":gold,"prediction":prediction})
    n=counts["rows"]; metrics={"rows":n,"semantic_valid_rate":counts["valid"]/n,"semantic_exact_rate":counts["exact"]/n}
    passed=full and metrics["semantic_valid_rate"]>=.995 and metrics["semantic_exact_rate"]>=.98 and len(roots)==8
    body={
        "schema_version":"agqa-layer-b-semantic-parser-heldout-v1",
        "status":"SEMANTIC_PARSER_QUALIFIED" if passed else ("DIAGNOSTIC_COMPLETE" if not full else "SEMANTIC_PARSER_NOT_QUALIFIED"),
        "full_validation":full,"metrics":metrics,"by_semantic_root":{k:dict(v) for k,v in sorted(roots.items())},
        "row_range":{"start":args.start_row,"end":args.start_row+len(rows),"validation_total":validation_total},
        "mismatch_sample":mismatches,"supervision_manifest_sha256":manifest["manifest_sha256"],
        "answers_read":False,"scene_graphs_read":False,"formal_test_read":False,
    }
    body["report_sha256"]=stable_hash(body); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":body["status"],"metrics":metrics,"report_sha256":body["report_sha256"]},indent=2))
    return 0 if passed or not full else 1


if __name__ == "__main__": raise SystemExit(main())
