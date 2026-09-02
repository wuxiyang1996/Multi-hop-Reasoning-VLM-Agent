#!/usr/bin/env python3
"""Outcome-blind intrinsic audit of Layer-B planning and shared VM execution."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import ARMS, plan_harness_arm
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--grounding",type=Path,required=True)
    parser.add_argument("--semantic-runtime",type=Path,required=True)
    parser.add_argument("--source-capabilities",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("Layer-B intrinsic audit is immutable")
    report=json.loads(args.grounding.read_text()); runtime=json.loads(args.semantic_runtime.read_text())
    source=json.loads(args.source_capabilities.read_text())
    if report["status"]!="RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES": raise ValueError("grounding not frozen")
    if report["semantic_runtime_sha256"]!=runtime["runtime_sha256"]: raise ValueError("semantic/grounding mismatch")
    compact={str(row["task_id"]):str(row["predicted_semantics"]) for row in runtime["rows"]}
    ops=tuple(source["authorized_operators"]); edges=tuple(tuple(x) for x in source["authorized_compositions"])
    rows=[]; reasons=Counter(); plan_status=Counter(); event_counts=Counter()
    for raw in report["rows"]:
        task_id=str(raw["task_id"]); semantic=_semantic(raw["semantic_receipt"]); grounding=_grounding(raw["grounding_receipt"])
        plans={arm:plan_harness_arm(semantic,arm=arm,source_capabilities=source,all_vm_operators=ops) for arm in ARMS}
        execution=execute_layer_b_semantics(compact_semantics=compact[task_id],grounding=grounding,
                                            semantic=semantic,authorized_operators=ops,
                                            authorized_compositions=edges,ambiguity_policy="STRICT")
        eager=execute_layer_b_semantics(compact_semantics=compact[task_id],grounding=grounding,
                                        semantic=semantic,authorized_operators=ops,
                                        authorized_compositions=None,ambiguity_policy="EAGER")
        reasons[execution.receipt.reason]+=1; event_counts[len(grounding.events)]+=1
        for arm,plan in plans.items(): plan_status[f"{arm}:{plan.status}"]+=1
        rows.append({"task_id":task_id,"events":len(grounding.events),"rejected_events":len(raw.get("rejected_events",())),
                     "execution":asdict(execution.receipt),"generic_eager_execution":asdict(eager.receipt),
                     "plans":{arm:asdict(plan) for arm,plan in plans.items()}})
    commits=sum(row["execution"]["status"]=="COMMITTED" for row in rows)
    body={"schema_version":"agqa-layer-b-intrinsic-execution-audit-v1",
          "status":"INTRINSIC_AUDIT_COMPLETE_BEFORE_OUTCOMES","rows":rows,
          "summary":{"tasks":len(rows),"commits":commits,"commit_rate":commits/len(rows),
                     "execution_reasons":dict(reasons),"plan_status":dict(plan_status),
                     "event_count_histogram":{str(k):v for k,v in sorted(event_counts.items())},
                     "rejected_events":sum(row["rejected_events"] for row in rows)},
          "grounding_report_sha256":report["report_sha256"],"source_capability_sha256":source["artifact_sha256"],
          "answers_read":False,"official_scene_graph_read":False,"official_program_read":False}
    body["report_sha256"]=stable_hash(body); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":body["status"],"summary":body["summary"],"report_sha256":body["report_sha256"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
