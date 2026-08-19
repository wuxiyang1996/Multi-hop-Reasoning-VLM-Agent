#!/usr/bin/env python3
"""Induce V51 directional support from 700 consumed AGQA rows."""

from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_aggregate_temporal_transfer import bind_aggregate_temporal_pair_program
from motif_transfer.agqa_directional_support_calibrator import DirectionalSupportExample, induce_directional_support_rule
from motif_transfer.agqa_interval_reliability_calibrator import binding_geometry
from motif_transfer.agqa_temporal_support_calibrator import maximum_interval_span
from motif_transfer.agqa_view_reliability_calibrator import singleton_view_kind
from motif_transfer.contracts import stable_hash

INPUTS = (
    ("v38", "runs/agqa2_aggregate_temporal_v38_development/report.json", "runs/agqa2_robust_temporal_v36_development/base_report.json"),
    ("v40", "runs/agqa2_aggregate_temporal_v41_completion/report.json", "runs/agqa2_aggregate_temporal_v41_completion/base_report.json"),
    ("v44", "runs/agqa2_view_reliability_v44_qualification/report.json", "runs/agqa2_view_reliability_v43_qualification/base_report.json"),
    ("v47", "runs/agqa2_interval_reliability_v47_qualification/report.json", "runs/agqa2_interval_reliability_v46_qualification/base_report.json"),
    ("v49", "runs/agqa2_temporal_support_v49_qualification/report.json", "runs/agqa2_temporal_support_v49_qualification/base_report.json"),
)
OUTPUT = "configs/agqa2_directional_support_v51/training_artifact.json"

def _verified(path):
    value=json.loads(path.read_text()); body=dict(value); claimed=body.pop("report_sha256")
    if stable_hash(body)!=claimed: raise ValueError(f"report hash mismatch: {path}")
    return value

def _metrics(rule, rows):
    selected=[x for x in rows if x.aggregate_authorized and x.resolved_relation in set(rule.allowed_relations) and (x.singleton_view is None or x.singleton_view in set(rule.allowed_singleton_views)) and x.minimum_cross_pair_gap>=rule.minimum_cross_pair_gap and x.maximum_within_operand_endpoint_spread<=rule.maximum_within_operand_endpoint_spread and x.maximum_interval_span>=rule.minimum_max_interval_span]
    w=sum(x.source_correct and not x.target_native_correct for x in selected); l=sum(x.target_native_correct and not x.source_correct for x in selected)
    return {"authorizations":len(selected),"wins":w,"losses":l,"net_gain":w-l}

def main():
    examples=[]; lineage=[]
    for split,rp,bp in INPUTS:
        report=_verified(REPO_ROOT/rp); base=_verified(REPO_ROOT/bp); bm={str(x["task_id"]):x for x in base["rows"]}
        for outcome in report["rows_detail"]:
            row=bm[str(outcome["task_id"])]
            binding=bind_aggregate_temporal_pair_program(task_id=str(row["task_id"]),target_state_sha256=str(row["runtime_receipt_sha256"]),target_grounder_sha256=str(report["target_grounder_sha256"]),source_program_sha256=str(report["source_program_sha256"]),obligation_kind=str(row["query_plan"]["obligation_kind"]),operand_runs=row["operand_runs"],grounder_qualified=True,formal_outcome_read=False)
            gap,spread=binding_geometry(binding)
            examples.append(DirectionalSupportExample(split=split,task_id=str(row["task_id"]),aggregate_authorized=binding.authorized_relation is not None,resolved_relation=binding.resolved_relation,singleton_view=singleton_view_kind(binding),minimum_cross_pair_gap=gap,maximum_within_operand_endpoint_spread=spread,maximum_interval_span=maximum_interval_span(binding),source_correct=bool(outcome["source_correct"]),target_native_correct=bool(outcome["target_native_correct"])))
        lineage.append({"split":split,"report":rp,"report_sha256":report["report_sha256"],"base_report":bp,"base_report_sha256":base["report_sha256"],"rows":len(report["rows_detail"])})
    rule,candidates=induce_directional_support_rule(examples)
    per_split={split:_metrics(rule,[x for x in examples if x.split==split]) for split,_,_ in INPUTS}
    core={"schema_version":"agqa2-directional-support-training-artifact-v51","status":"V51_TRAINED_ON_700_CONSUMED_ROWS_BEFORE_NEW_QUALIFICATION","training_authority":"CONSUMED_V38_V40_V44_V47_AND_FAILED_V49_ROWS_ONLY;NO_FUTURE_QUALIFICATION_OR_FORMAL_DATA","source_program_or_ir_changed":False,"target_interval_or_relation_changed":False,"runtime_authority":"ABSTENTION_ONLY;CANNOT_INVENT_OR_EDIT_A_BINDING","feature_space":["RESOLVED_RELATION","SINGLETON_VIEW_KIND","MINIMUM_CROSS_PAIR_GAP","MAXIMUM_WITHIN_OPERAND_ENDPOINT_SPREAD","MAXIMUM_INTERVAL_SPAN"],"finite_candidate_count":len(candidates),"selection_objective":"MINIMIZE_OBSERVED_NEGATIVE_TRANSFER_THEN_MAXIMIZE_NET_GAIN_WINS_AND_COVERAGE_WITH_FIXED_MDL_TIE_BREAK","rule":asdict(rule),"selected_rule_per_consumed_split":per_split,"candidate_rule_table":list(candidates),"training_lineage":lineage,"training_example_count":len(examples),"future_policy":"REQUIRE_ONE_NEW_VIDEO_DISJOINT_TRAIN_QUALIFICATION_BEFORE_ANY_NEW_TEST_FORMAL","confirmatory_claim":False}
    artifact=core|{"artifact_sha256":stable_hash(core)}; out=REPO_ROOT/OUTPUT; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(artifact,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":artifact["status"],"finite_candidate_count":len(candidates),"selected_rule":artifact["rule"],"per_split":per_split,"artifact_sha256":artifact["artifact_sha256"]},indent=2,sort_keys=True))

if __name__=="__main__": main()
