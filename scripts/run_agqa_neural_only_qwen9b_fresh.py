#!/usr/bin/env python3
"""Frozen Qwen3.5-9B target actor over shared outcome-blind STSG facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from motif_transfer.agqa_oracle_query_mdp import load_agqa_id_to_text
from motif_transfer.contracts import stable_hash
from motif_transfer.official_video_event_graph import load_builtin_only_pickle


def clean(value) -> str:
    value = re.sub(r"```(?:json)?|```", "", str(value), flags=re.I).strip()
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', value, flags=re.I)
    if match: value = match.group(1)
    value = re.sub(r"^(?:answer\s*:\s*)", "", value, flags=re.I)
    return " ".join(value.strip(" \t\n\r\"'.").replace("_", " ").casefold().split())


def refs(value):
    if isinstance(value, dict): value = value.get("vertices", ())
    if not isinstance(value, (list, tuple)): return ()
    return tuple(str(row.get("id")) if isinstance(row, dict) else str(row) for row in value)


def graph_summary(graph, ontology) -> str:
    actions = set(); objects = set(); relations = {}
    for key, row in graph.items():
        if not isinstance(row, dict): continue
        if row.get("type") == "action":
            label = clean(row.get("phrase") or ontology.get(str(row.get("charades", "")), ""))
            frames = [int(x) for x in row.get("all_f", ()) if str(x).isdigit()]
            if label and frames: actions.add((label, min(frames), max(frames)))
        elif str(key).startswith("o"):
            label = clean(ontology.get(str(row.get("class", "")), ""))
            if label and label not in {"person", "none"}: objects.add(label)
        elif str(key)[:1] in {"r", "v"}:
            relation = clean(ontology.get(str(row.get("class", "")), ""))
            frame_text = str(key).split("/")[-1]
            frame = int(frame_text) if frame_text.isdigit() else -1
            for ref in refs(row.get("objects")):
                obj = clean(ontology.get(ref.split("/", 1)[0], ""))
                if relation and obj and obj not in {"person", "none"}:
                    relations.setdefault((relation, obj), []).append(frame)
    lines = ["ACTIONS (label,start_frame,end_frame):"]
    lines.extend(f"- {label} | {start} | {end}" for label, start, end in sorted(actions, key=lambda x: (x[1], x[2], x[0])))
    lines.append("RELATIONS (relation,object,first_frame,last_frame):")
    lines.extend(f"- {rel} | {obj} | {min(frames)} | {max(frames)}"
                 for (rel, obj), frames in sorted(relations.items()))
    lines.append("VISIBLE OBJECT CLASSES: " + ", ".join(sorted(objects)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--stsg", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("neural runtime is immutable")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    if cohort["answer_read"] or cohort["functional_program_read"]:
        raise ValueError("cohort authority boundary crossed")
    corpus = load_builtin_only_pickle(args.stsg)
    ontology = load_agqa_id_to_text(args.ontology)
    summaries = {video: graph_summary(corpus[video], ontology)
                 for video in {row["video_id"] for row in cohort["rows"]}}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for row in cohort["rows"]:
        messages = [{"role": "system", "content": (
            "Answer the video question using only the supplied scene facts. "
            "Return only the shortest answer label, with no explanation."
        )}, {"role": "user", "content": (
            summaries[row["video_id"]] + "\n\nQUESTION: " + row["question"] + "\nANSWER:"
        )}]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        ))
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=.90, trust_remote_code=True)
    generated = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=32, skip_special_tokens=True,
    ))
    rows = []
    for public, output in zip(cohort["rows"], generated):
        raw = output.outputs[0].text
        rows.append({"task_id": public["task_id"], "video_id": public["video_id"],
                     "prediction": clean(raw), "raw_response": raw,
                     "answer_read": False, "oracle_program_read": False})
    body = {"schema_version": "agqa-neural-only-qwen35-9b-stsg-v1",
            "status": "NEURAL_PREDICTIONS_FROZEN_BEFORE_OUTCOME_ACCESS",
            "model": args.model, "thinking_enabled": False,
            "grounding": "SHARED_OFFICIAL_STSG_COMPACT_FACTS",
            "cohort_sha256": cohort["cohort_sha256"], "rows": rows,
            "answer_read": False, "oracle_program_read": False}
    body["runtime_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "rows": len(rows),
                      "runtime_sha256": body["runtime_sha256"]}, indent=2))


if __name__ == "__main__": main()
