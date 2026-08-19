#!/usr/bin/env python3
"""Freeze a 250-row fresh qualification for V51 directional support."""
from __future__ import annotations
from copy import deepcopy
import json,sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO_ROOT/"src"),str(REPO_ROOT)]
from motif_transfer.agqa_directional_support_calibrator import directional_support_target_grounder_sha256
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _load_sources
from scripts.collect_agqa2_active_grounding_v3 import _evaluation_protocol_core,_grounder_semantic_core
import scripts.freeze_agqa2_temporal_support_v49_qualification as v49

NONCE="agqa2-v52-directional-support-train-qualification-250";N=250
ART="configs/agqa2_directional_support_v51/training_artifact.json";PARENT="configs/agqa2_temporal_support_v49_qualification.json";PM="configs/agqa2_temporal_selective_v19_development_manifest.json"
SEL="configs/agqa2_directional_support_v52_qualification_selection.json";MAN="configs/agqa2_directional_support_v52_qualification_manifest.json";PRE="configs/agqa2_directional_support_v52_qualification_preregistration.json";CFG="configs/agqa2_directional_support_v52_qualification.json";DL="runs/agqa2_directional_support_v52_download/receipt.json"
ADAPTER="src/motif_transfer/agqa_aggregate_temporal_transfer.py";CAL="src/motif_transfer/agqa_directional_support_calibrator.py";EVAL="scripts/evaluate_agqa2_directional_support_v52.py"
sha=v49._sha256;verified=v49._verified

def selection(parent):
    old=(v49.NONCE,v49.SAMPLE_COUNT);v49.NONCE,v49.SAMPLE_COUNT=NONCE,N
    try:x=v49._new_selection(parent)
    finally:v49.NONCE,v49.SAMPLE_COUNT=old
    x=dict(x);x.pop("manifest_sha256");x.update({"schema_version":"agqa2-directional-support-selection-v52","status":"FROZEN_V52_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V52_CALLS","split":"development","claim_boundary":"TWO_HUNDRED_FIFTY_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_ATOMIC_BEFORE_AFTER_ROWS_FROM_OFFICIAL_TRAIN_METADATA;V51_DIRECTIONAL_SUPPORT_RULE_QUALIFICATION_ONLY","selection_nonce":NONCE,"selection_metadata_split":"official_train_balanced","prior_v52_neural_grounder_exposure":False,"answer_read_during_freeze":False});x.pop("prior_v49_neural_grounder_exposure",None);return x|{"manifest_sha256":stable_hash(x)}
def seal(sel):
    x=dict(v49._seal(sel));x.pop("manifest_sha256");x.update({"schema_version":"agqa2-directional-support-manifest-v52","status":"FROZEN_V52_TRAIN_QUALIFICATION_VIDEOS_UNSEEN"});return x|{"manifest_sha256":stable_hash(x)}

def main():
    run=REPO_ROOT/"runs/agqa2_directional_support_v52_qualification"
    if run.exists() and any(run.rglob("*.json")):raise RuntimeError("V52 already has runtime artifacts")
    ap=REPO_ROOT/ART;artifact=verified(ap,"artifact_sha256")
    if artifact["status"]!="V51_TRAINED_ON_700_CONSUMED_ROWS_BEFORE_NEW_QUALIFICATION":raise ValueError("V51 artifact not frozen")
    sp=REPO_ROOT/SEL;sel=verified(sp,"manifest_sha256") if sp.exists() else selection(verified(REPO_ROOT/PM,"manifest_sha256"));sp.write_text(json.dumps(sel,indent=2,sort_keys=True)+"\n")
    missing=[r["video_id"] for r in sel["samples"] if not Path(r["video_path"]).is_file()]
    if missing:print(json.dumps({"status":sel["status"],"selection_manifest_sha256":sel["manifest_sha256"],"sample_count":N,"missing_video_count":len(missing),"missing_video_ids":missing,"next":"download exact frozen videos then rerun"},indent=2));return
    receipt=json.loads((REPO_ROOT/DL).read_text())
    if receipt.get("status")!="COMPLETE" or receipt.get("selection_manifest_sha256")!=sel["manifest_sha256"] or len(receipt.get("videos") or [])!=N:raise ValueError("V52 download receipt mismatch")
    manifest=seal(sel);mp=REPO_ROOT/MAN;mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    parent=json.loads((REPO_ROOT/PARENT).read_text());config=deepcopy(parent);config.pop("temporal_support_calibration",None);config.update({"schema_version":"agqa2-directional-support-v52-qualification-config-v1","status":"FROZEN_V52_DIRECTIONAL_SUPPORT_QUALIFICATION","split":"development","claim_boundary":manifest["claim_boundary"],"manifest":MAN,"manifest_file_sha256":sha(mp),"expected_manifest_status":manifest["status"],"expected_preregistration_status":"FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL","report_version":"V52_QUALIFICATION_BASE"})
    config["qualification_gates"]={"required_valid_runtime_rows":N,"minimum_route_correct":N,"minimum_decisive_executions":N+1,"minimum_decisive_accuracy":0.0,"minimum_typed_vs_direct_wins":0,"maximum_typed_vs_direct_losses":N,"required_source_permuted_abstentions":N,"required_target_written_equivalent_matches":N,"maximum_reported_provider_cost_usd":2.25}
    sources,_=_load_sources(config);parent_grounder=stable_hash(_grounder_semantic_core(config,sources))
    if parent_grounder!=parent["expected_grounder_sha256"]:raise AssertionError("V52 acquisition drift")
    base_eval=stable_hash(_evaluation_protocol_core(config));ad=REPO_ROOT/ADAPTER;cal=REPO_ROOT/CAL;ev=REPO_ROOT/EVAL
    target=directional_support_target_grounder_sha256(parent_grounder_sha256=parent_grounder,aggregate_adapter_sha256=sha(ad),normalization_module_sha256=config["syntax_transport_normalization"]["normalization_module_sha256"],acquisition_collector_sha256=config["grounder"]["collector_sha256"],calibrator_module_sha256=sha(cal),calibration_artifact_sha256=artifact["artifact_sha256"])
    r=artifact["rule"];route={"wins":r["training_wins"],"losses":r["training_losses"],"ties":r["training_ties"],"decision":"SELECT_SKILL","reason":"V51_RISK_FIRST_DIRECTIONAL_SUPPORT_INDUCTION"}
    gates={"required_valid_rows":N,"required_unique_videos":N,"minimum_source_authorizations":25,"minimum_source_wins":8,"maximum_source_losses":1,"minimum_source_minus_target_correct":7,"maximum_exact_one_sided_pvalue":0.05,"required_effect_shuffled_abstentions":N,"required_wrong_source_abstentions":N,"required_generic_scaffold_matches":N,"required_target_written_equivalent_matches":N,"maximum_reported_provider_cost_usd":2.25}
    protocol={"schema_version":"agqa2-directional-support-v52-qualification-protocol-v1","sample_count":N,"source_program_sha256":config["postground"]["source_program_sha256"],"target_grounder_sha256":target,"target_executor_sha256":config["postground"]["target_executor_sha256"],"aggregate_adapter_sha256":sha(ad),"calibrator_module_sha256":sha(cal),"calibration_artifact_sha256":artifact["artifact_sha256"],"calibration_rule_sha256":r["rule_sha256"],"evaluator_module_sha256":sha(ev),"runtime_calibrator_authority":"ABSTENTION_ONLY;NO_INTERVAL_RELATION_OR_BINDING_CREATION_OR_EDIT","runtime_features":artifact["feature_space"],"fallback":"PRESERVE_MATCHED_TARGET_NATIVE_DIRECT_ON_ABSTENTION","gates":gates,"confirmatory_claim":False};ph=stable_hash(protocol)
    prereg={"schema_version":"agqa2-directional-support-v52-preregistration-v1","status":"FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL","v52_status":"FROZEN_BEFORE_ANY_V52_QUALIFICATION_PROVIDER_OR_OUTCOME_CALL","claim_boundary":manifest["claim_boundary"],"selection_manifest_sha256":sel["manifest_sha256"],"qualification_manifest_sha256":manifest["manifest_sha256"],"download_receipt_file_sha256":sha(REPO_ROOT/DL),"qualified_v33_development_report_sha256":artifact["artifact_sha256"],"qualified_development_artifact_sha256":artifact["artifact_sha256"],"v51_training_artifact":ART,"v51_training_artifact_file_sha256":sha(ap),"v51_training_artifact_sha256":artifact["artifact_sha256"],"source_program_sha256":config["postground"]["source_program_sha256"],"development_calibration":route,"base_evaluation_protocol_sha256":base_eval,"postground_evaluation_protocol":protocol,"postground_evaluation_protocol_sha256":ph,"qualification_gates":gates,"cost_projection":{"projected_250_row_cost_usd":1.74,"frozen_cap_usd":2.25},"failure_policy":{"qualification":"RUN_ONCE;NO_POST_OUTCOME_THRESHOLD_CHANGE","failed_gate":"STOP_BEFORE_NEW_TEST_FORMAL","passed":"FREEZE_ONE_NEW_VIDEO_DISJOINT_TEST_FORMAL"},"confirmatory_claim_allowed":False}
    pp=REPO_ROOT/PRE;pp.write_text(json.dumps(prereg,indent=2,sort_keys=True)+"\n");config.update({"preregistration":PRE,"preregistration_file_sha256":sha(pp),"expected_grounder_sha256":parent_grounder,"expected_evaluation_protocol_sha256":base_eval,"directional_support_calibration":{"module":CAL,"module_sha256":sha(cal),"artifact":ART,"artifact_file_sha256":sha(ap),"artifact_sha256":artifact["artifact_sha256"],"rule_sha256":r["rule_sha256"]}});config["postground"].update({"adapter_module":ADAPTER,"adapter_module_sha256":sha(ad),"evaluator_module":EVAL,"evaluator_module_sha256":sha(ev),"target_grounder_sha256":target,"development_calibration":route,"evaluation_protocol_sha256":ph,"formal_gates":gates});cp=REPO_ROOT/CFG;cp.write_text(json.dumps(config,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":prereg["v52_status"],"selection_manifest_sha256":sel["manifest_sha256"],"manifest_sha256":manifest["manifest_sha256"],"sample_count":N,"parent_grounder_sha256":parent_grounder,"target_grounder_sha256":target,"evaluation_protocol_sha256":ph,"provider_cost_cap_usd":2.25,"config_file_sha256":sha(cp)},indent=2))
if __name__=="__main__":main()
