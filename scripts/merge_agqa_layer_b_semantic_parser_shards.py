#!/usr/bin/env python3
"""Merge disjoint held-out semantic-parser shards into one qualification."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--shards",type=Path,nargs="+",required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("merged qualification is immutable")
    shards=[json.loads(p.read_text()) for p in args.shards]; ranges=sorted((d["row_range"]["start"],d["row_range"]["end"],d) for d in shards)
    total=ranges[0][2]["row_range"]["validation_total"]
    if ranges[0][0]!=0 or ranges[-1][1]!=total or any(a[1]!=b[0] for a,b in zip(ranges,ranges[1:])):
        raise ValueError("semantic qualification shards are not an exact partition")
    counts=Counter(); roots: dict[str,Counter[str]]={}; mismatches=[]
    for _,_,d in ranges:
        n=d["metrics"]["rows"]; counts["rows"]+=n
        counts["valid"]+=round(d["metrics"]["semantic_valid_rate"]*n)
        counts["exact"]+=round(d["metrics"]["semantic_exact_rate"]*n)
        for root,row in d["by_semantic_root"].items(): roots.setdefault(root,Counter()).update(row)
        mismatches.extend(d["mismatch_sample"][:max(0,200-len(mismatches))])
    metrics={"rows":counts["rows"],"semantic_valid_rate":counts["valid"]/counts["rows"],"semantic_exact_rate":counts["exact"]/counts["rows"]}
    passed=counts["rows"]==total and metrics["semantic_valid_rate"]>=.995 and metrics["semantic_exact_rate"]>=.98 and len(roots)==8
    body={"schema_version":"agqa-layer-b-semantic-parser-heldout-v1","status":"SEMANTIC_PARSER_QUALIFIED" if passed else "SEMANTIC_PARSER_NOT_QUALIFIED",
          "full_validation":True,"metrics":metrics,"by_semantic_root":{k:dict(v) for k,v in sorted(roots.items())},"mismatch_sample":mismatches,
          "shard_report_sha256s":[d["report_sha256"] for _,_,d in ranges],"answers_read":False,"scene_graphs_read":False,"formal_test_read":False}
    body["report_sha256"]=stable_hash(body); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":body["status"],"metrics":metrics,"report_sha256":body["report_sha256"]},indent=2)); return 0 if passed else 1


if __name__=="__main__": raise SystemExit(main())
