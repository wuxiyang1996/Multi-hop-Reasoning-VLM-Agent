#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import asdict
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from motif_transfer.agqa_aggregate_temporal_transfer import bind_aggregate_temporal_pair_program
from motif_transfer.agqa_asymmetric_support_calibrator import AsymmetricExample,induce_asymmetric_rule
from motif_transfer.agqa_interval_reliability_calibrator import binding_geometry
from motif_transfer.agqa_temporal_support_calibrator import maximum_interval_span
from motif_transfer.agqa_view_reliability_calibrator import singleton_view_kind
from motif_transfer.contracts import stable_hash
INPUTS=(("v38","runs/agqa2_aggregate_temporal_v38_development/report.json","runs/agqa2_robust_temporal_v36_development/base_report.json"),("v40","runs/agqa2_aggregate_temporal_v41_completion/report.json","runs/agqa2_aggregate_temporal_v41_completion/base_report.json"),("v44","runs/agqa2_view_reliability_v44_qualification/report.json","runs/agqa2_view_reliability_v43_qualification/base_report.json"),("v47","runs/agqa2_interval_reliability_v47_qualification/report.json","runs/agqa2_interval_reliability_v46_qualification/base_report.json"),("v49","runs/agqa2_temporal_support_v49_qualification/report.json","runs/agqa2_temporal_support_v49_qualification/base_report.json"),("v52","runs/agqa2_directional_support_v52_qualification/report.json","runs/agqa2_directional_support_v52_qualification/base_report.json"))
def load(p):
 x=json.loads((ROOT/p).read_text());b=dict(x);h=b.pop("report_sha256");assert stable_hash(b)==h;return x
def main():
 rows=[];line=[]
 for s,rp,bp in INPUTS:
  r=load(rp);b=load(bp);bm={str(x["task_id"]):x for x in b["rows"]}
  for o in r["rows_detail"]:
   x=bm[str(o["task_id"])];z=bind_aggregate_temporal_pair_program(task_id=str(x["task_id"]),target_state_sha256=str(x["runtime_receipt_sha256"]),target_grounder_sha256=str(r["target_grounder_sha256"]),source_program_sha256=str(r["source_program_sha256"]),obligation_kind=str(x["query_plan"]["obligation_kind"]),operand_runs=x["operand_runs"],grounder_qualified=True,formal_outcome_read=False);g,sp=binding_geometry(z);rows.append(AsymmetricExample(s,str(x["task_id"]),z.authorized_relation is not None,z.resolved_relation,singleton_view_kind(z),g,sp,maximum_interval_span(z),bool(o["source_correct"]),bool(o["target_native_correct"])))
  line.append({"split":s,"report_sha256":r["report_sha256"],"base_report_sha256":b["report_sha256"],"rows":len(r["rows_detail"])})
 rule,c=induce_asymmetric_rule(rows);core={"schema_version":"agqa2-asymmetric-support-training-v54","status":"V54_TRAINED_ON_950_CONSUMED_ROWS_BEFORE_NEW_QUALIFICATION","runtime_authority":"ABSTENTION_ONLY","source_program_or_ir_changed":False,"feature_space":["VIEW","GAP","SPREAD","RELATION_CONDITIONAL_INTERVAL_SPAN"],"finite_candidate_count":len(c),"rule":asdict(rule),"training_lineage":line,"candidate_rule_table":list(c),"confirmatory_claim":False};a=core|{"artifact_sha256":stable_hash(core)};p=ROOT/"configs/agqa2_asymmetric_support_v54/training_artifact.json";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":a["status"],"rule":a["rule"],"artifact_sha256":a["artifact_sha256"]},indent=2))
if __name__=="__main__":main()
