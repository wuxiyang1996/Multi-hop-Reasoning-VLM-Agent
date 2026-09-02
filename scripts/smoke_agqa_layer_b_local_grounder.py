#!/usr/bin/env python3
"""Outcome-blind one-row smoke for a frozen local AGQA raw-video grounder."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from vllm import LLM, SamplingParams

from motif_transfer.agqa_layer_b_contracts import GroundedEvent, RawVideoEventGraphReceipt
from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video


SYSTEM = """You are an answer-blind video event grounder. You receive chronological frame panels and an operator-free semantic goal. Return JSON only:
{"events":[{"event_id":"E0","subject":"person","predicate":"...","object":"...","start_frame":0,"end_frame":3,"evidence_frames":[0,3],"confidence":0.8,"semantic_slot_ids":["S1"]}],"uncertainties":["..."]}
Frame indices refer to the labeled proxy frames F0..F23. Include only events and relations visibly supported by pixels and relevant to the semantic goal. Every event must cite 1-3 evidence frames inside its interval. Use an empty events list when evidence is insufficient. Never answer the question. Never emit an answer, choice, official scene graph, functional program, VM operator, source game, or correctness judgment."""


def sha_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def data_url(payload: bytes) -> str:
    return "data:image/jpeg;base64,"+base64.b64encode(payload).decode()


def parse_json(text: str) -> dict:
    text=re.sub(r"^```(?:json)?|```$","",text.strip(),flags=re.I).strip()
    start=text.find("{"); end=text.rfind("}")
    if start<0 or end<start: raise ValueError("grounder did not return a JSON object")
    value=json.loads(text[start:end+1])
    forbidden={"answer","choice","correct","functional_program","program","source_game","sg_grounding"}
    if forbidden & {str(key).casefold() for key in value}: raise ValueError("grounder crossed answer/program boundary")
    return value


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--video-root",type=Path,required=True)
    parser.add_argument("--model",default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frame-count",type=int,default=24)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("grounder smoke output is immutable")
    row=None
    with args.data.open() as handle:
        for line in handle:
            candidate=json.loads(line); path=args.video_root/f"{candidate['video_id']}.mp4"
            if path.exists(): row=candidate; video_path=path; break
    if row is None: raise RuntimeError("no semantic development row has a local video")
    question=str(row["input"]).removeprefix("parse AGQA semantics: ")
    semantic_receipt=parse_compact_semantic_target(
        str(row["target"]),task_id=str(row["task_id"]),question_sha256=stable_hash(question),
        parser_sha256=stable_hash("DEVELOPMENT_GOLD_LOWERING_SMOKE_ONLY"),
        parser_training_authority="CONSUMED_DEVELOPMENT_GOLD_LOWERING_FOR_SCHEMA_SMOKE_ONLY",
    )
    frames,seconds,metadata=_sample_video(video_path,frame_count=args.frame_count,max_side=448)
    panels=_panels(frames,seconds,{"frames_per_panel":6,"panel_frame_width":224,"jpeg_quality":82})
    content=[{"type":"text","text":f"Question (do not answer): {question}\nOperator-free semantic goal: {row['target']}"}]
    for index,panel in enumerate(panels):
        content.extend([{"type":"text","text":f"Chronological panel {index+1}:"},
                        {"type":"image_url","image_url":{"url":data_url(panel)}}])
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":content}]
    llm=LLM(model=args.model,dtype="bfloat16",max_model_len=16384,gpu_memory_utilization=.90,
            trust_remote_code=True,limit_mm_per_prompt={"image":len(panels)})
    outputs=llm.chat(messages,sampling_params=SamplingParams(temperature=0,max_tokens=1024),
                     chat_template_kwargs={"enable_thinking":False})
    raw=outputs[0].outputs[0].text; payload=parse_json(raw)
    events=[]
    for index,event in enumerate(payload.get("events",())):
        events.append(GroundedEvent(
            event_id=f"E{index}",subject=str(event.get("subject", "person")),
            predicate=str(event.get("predicate", "")),object=str(event.get("object", "")),
            start_frame=int(event["start_frame"]),end_frame=int(event["end_frame"]),
            evidence_frames=tuple(sorted(set(int(x) for x in event["evidence_frames"]))),
            confidence=float(event.get("confidence",0.0)),
            semantic_slot_ids=tuple(str(x) for x in event.get("semantic_slot_ids",())),
        ))
    frame_hashes=[]
    for frame in frames:
        frame_hashes.append(stable_hash({"mode":frame.mode,"size":frame.size,"pixels_sha256":hashlib.sha256(frame.tobytes()).hexdigest()}))
    backend=stable_hash({"model":args.model,"prompt":SYSTEM,"thinking":False,"frame_count":args.frame_count,
                         "panels":len(panels),"temperature":0})
    receipt=RawVideoEventGraphReceipt.create(
        task_id=str(row["task_id"]),video_sha256=sha_file(video_path),
        semantic_slots_sha256=semantic_receipt.receipt_sha256,selected_frame_indices=tuple(range(len(frames))),
        selected_frame_sha256s=tuple(frame_hashes),events=events,grounder_backend_sha256=backend,
        frame_budget=args.frame_count,provider_calls=0,
    )
    body={"schema_version":"agqa-layer-b-local-grounder-smoke-v1","status":"SCHEMA_SMOKE_PASSED",
          "task_id":row["task_id"],"video_id":row["video_id"],"question_sha256":stable_hash(question),
          "semantic_target_sha256":stable_hash(row["target"]),"semantic_receipt_sha256":semantic_receipt.receipt_sha256,
          "grounding_receipt":asdict(receipt),
          "raw_response":raw,"video_metadata":metadata,"panel_sha256s":[hashlib.sha256(x).hexdigest() for x in panels],
          "answer_read":False,"official_scene_graph_read":False,"provider_calls":0}
    body["report_sha256"]=stable_hash(body); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":body["status"],"task_id":row["task_id"],"events":len(events),"report_sha256":body["report_sha256"]},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
