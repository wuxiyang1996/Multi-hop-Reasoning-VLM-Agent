#!/usr/bin/env python3
"""Freeze question-only Layer-B semantic slots before raw-video grounding."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from motif_transfer.agqa_layer_b_authority import cohort_crossed_authority
from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target
from motif_transfer.contracts import stable_hash


def batches(rows,size):
    for start in range(0,len(rows),size): yield rows[start:start+size]


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--qualification",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=32)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("semantic runtime is immutable")
    cohort=json.loads(args.cohort.read_text()); qualification=json.loads(args.qualification.read_text())
    if qualification["status"]!="SEMANTIC_PARSER_QUALIFIED": raise ValueError("semantic parser is not fully qualified")
    # Accept both the legacy runtime schema and the newer projection schema.
    # Missing authority fields are not silently accepted: the V1 projection
    # must explicitly say that evaluator-only values were not projected.
    if cohort_crossed_authority(cohort):
        raise ValueError("semantic runtime cohort crossed target authority boundary")
    parser_sha=stable_hash({"model_dir":str(args.model_dir),"qualification":qualification["report_sha256"]})
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir)
    model=AutoModelForSeq2SeqLM.from_pretrained(args.model_dir,dtype=torch.bfloat16).to("cuda").eval()
    outputs=[]; invalid=0
    with torch.inference_mode():
        for batch in batches(cohort["rows"],args.batch_size):
            encoded=tokenizer(["parse AGQA semantics: "+row["question"] for row in batch],padding=True,
                              truncation=True,max_length=192,return_tensors="pt").to("cuda")
            generated=model.generate(**encoded,max_new_tokens=512,num_beams=1)
            predictions=tokenizer.batch_decode(generated,skip_special_tokens=True)
            for row,prediction in zip(batch,predictions):
                try:
                    receipt=parse_compact_semantic_target(
                        prediction,task_id=row["task_id"],question_sha256=row["question_sha256"],
                        parser_sha256=parser_sha,
                        parser_training_authority="AGQA_TRAIN_DEV_TO_OPERATOR_FREE_COMPACT_SEMANTICS",
                    ); value={"status":"SEMANTIC_SLOTS_FROZEN","receipt":asdict(receipt)}
                except Exception as exc:
                    # Outcome-blind syntax-only repair: allow only the unique
                    # minimal suffix that closes trailing call delimiters.
                    missing=prediction.count("(")-prediction.count(")")
                    try:
                        if missing<=0: raise exc
                        prediction=prediction+")"*missing
                        receipt=parse_compact_semantic_target(
                            prediction,task_id=row["task_id"],question_sha256=row["question_sha256"],
                            parser_sha256=parser_sha,
                            parser_training_authority="AGQA_TRAIN_DEV_TO_OPERATOR_FREE_COMPACT_SEMANTICS",
                        )
                        value={"status":"SEMANTIC_SLOTS_FROZEN","receipt":asdict(receipt),
                               "syntax_repair":"UNIQUE_MINIMAL_TRAILING_PARENTHESIS_BALANCE"}
                    except Exception:
                        invalid+=1; value={"status":"SEMANTIC_PARSER_ABSTAINED","reason":str(exc),"receipt":None}
                outputs.append({"task_id":row["task_id"],"video_id":row["video_id"],
                                "question_sha256":row["question_sha256"],"predicted_semantics":prediction,
                                **value,"answer_read":False,"functional_program_read":False,"scene_graph_read":False})
    body={"schema_version":"agqa-layer-b-semantic-runtime-v1",
          "status":"SEMANTIC_RUNTIME_FROZEN_BEFORE_VIDEO_OR_OUTCOME","cohort_sha256":cohort["cohort_sha256"],
          "qualification_sha256":qualification["report_sha256"],"parser_sha256":parser_sha,
          "rows":outputs,"valid":len(outputs)-invalid,"invalid":invalid,
          "answer_read":False,"functional_program_read":False,"scene_graph_read":False}
    body["runtime_sha256"]=stable_hash(body); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":body["status"],"rows":len(outputs),"valid":body["valid"],"invalid":invalid,
                      "runtime_sha256":body["runtime_sha256"]},indent=2)); return 0 if not invalid else 1


if __name__=="__main__": raise SystemExit(main())
