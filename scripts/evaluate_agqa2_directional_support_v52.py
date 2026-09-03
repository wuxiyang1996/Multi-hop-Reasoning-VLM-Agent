#!/usr/bin/env python3
"""Evaluate the frozen V51 directional-support rule."""
from __future__ import annotations
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO_ROOT/"src"),str(REPO_ROOT)]
from motif_transfer.agqa_aggregate_temporal_transfer import bind_aggregate_temporal_pair_program,build_aggregate_temporal_harness,build_aggregate_temporal_route,decide_aggregate_temporal_relation,unified_aggregate_temporal_grounding
from motif_transfer.agqa_directional_support_calibrator import DirectionalSupportRule,apply_directional_support_rule
from motif_transfer.contracts import stable_hash
import scripts.collect_agqa2_robust_temporal_v34_formal as core

def _load(config):
    spec=config["directional_support_calibration"];a=json.loads((REPO_ROOT/spec["artifact"]).read_text());b=dict(a);h=b.pop("artifact_sha256")
    if stable_hash(b)!=h or h!=spec["artifact_sha256"]:raise ValueError("directional-support artifact hash mismatch")
    rule=DirectionalSupportRule.from_mapping(a["rule"])
    if rule.rule_sha256!=spec["rule_sha256"]:raise ValueError("directional-support rule hash mismatch")
    return rule,a

def evaluate_calibrated(*,config_path,base_report_path,output_path,formal):
    config=json.loads(config_path.read_text());rule,artifact=_load(config)
    def binding(**kwargs):return apply_directional_support_rule(bind_aggregate_temporal_pair_program(**kwargs),rule)
    replacements={"bind_robust_temporal_pair_program":binding,"build_temporal_harness":build_aggregate_temporal_harness,"build_temporal_route":build_aggregate_temporal_route,"decide_temporal_relation":decide_aggregate_temporal_relation,"unified_temporal_grounding":unified_aggregate_temporal_grounding};originals={k:getattr(core,k) for k in replacements}
    try:
        for k,v in replacements.items():setattr(core,k,v)
        result=core.evaluate(config_path=config_path,base_report_path=base_report_path,output_path=output_path)
    finally:
        for k,v in originals.items():setattr(core,k,v)
    body=deepcopy(result);body.pop("report_sha256",None);qualified=all(body["qualification_gates"].values())
    body.update({"schema_version":"agqa2-directional-support-v53-formal-report-v1" if formal else "agqa2-directional-support-v52-qualification-report-v1","status":("AGQA2_DIRECTIONAL_SUPPORT_V53_FORMAL_QUALIFIED" if formal and qualified else "AGQA2_DIRECTIONAL_SUPPORT_V53_FORMAL_NOT_QUALIFIED" if formal else "AGQA2_DIRECTIONAL_SUPPORT_V52_QUALIFICATION_QUALIFIED" if qualified else "AGQA2_DIRECTIONAL_SUPPORT_V52_QUALIFICATION_NOT_QUALIFIED"),"split":"fresh_formal" if formal else "fresh_train_qualification","confirmatory_claim":bool(formal and qualified),"calibration_artifact_sha256":artifact["artifact_sha256"],"calibration_rule_sha256":rule.rule_sha256,"allowed_relations":list(rule.allowed_relations),"allowed_singleton_views":list(rule.allowed_singleton_views),"minimum_cross_pair_gap":rule.minimum_cross_pair_gap,"maximum_within_operand_endpoint_spread":rule.maximum_within_operand_endpoint_spread,"minimum_max_interval_span":rule.minimum_max_interval_span,"runtime_calibrator_authority":"ABSTENTION_ONLY;NO_INTERVAL_RELATION_OR_BINDING_CREATION_OR_EDIT","current_outcome_used_for_calibration":False,"prior_failed_splits_reclassified_as_success":False})
    final=body|{"report_sha256":stable_hash(body)};output_path.parent.mkdir(parents=True,exist_ok=True);output_path.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n");return final

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=REPO_ROOT/"configs/agqa2_directional_support_v52_qualification.json");p.add_argument("--base-report",type=Path,default=REPO_ROOT/"runs/agqa2_directional_support_v52_qualification/base_report.json");p.add_argument("--output",type=Path,default=REPO_ROOT/"runs/agqa2_directional_support_v52_qualification/report.json");a=p.parse_args();r=evaluate_calibrated(config_path=a.config.resolve(),base_report_path=a.base_report.resolve(),output_path=a.output.resolve(),formal=False);print(json.dumps({k:r[k] for k in ("status","rows","source_executor_authorizations","source_vs_target_native","qualification_gates","reported_provider_cost_usd","report_sha256")},indent=2,sort_keys=True))
if __name__=="__main__":main()
