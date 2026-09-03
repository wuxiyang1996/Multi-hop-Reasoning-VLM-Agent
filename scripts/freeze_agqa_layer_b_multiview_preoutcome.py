#!/usr/bin/env python3
"""Freeze typed Layer-B multi-view coverage before opening outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import (
    plan_harness_arm, source_permuted_compositions,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--preregistration',type=Path,required=True)
    p.add_argument('--cohort',type=Path,required=True); p.add_argument('--semantic-runtime',type=Path,required=True)
    p.add_argument('--routed-grounding',type=Path,required=True); p.add_argument('--grounding-view',type=Path,action='append',required=True)
    p.add_argument('--fallback',type=Path,required=True); p.add_argument('--source-capabilities',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('multi-view pre-outcome receipt is immutable')
    if len(a.grounding_view)!=2: raise ValueError('exactly two frozen grounding views are required')
    prereg=json.loads(a.preregistration.read_text()); cohort=json.loads(a.cohort.read_text())
    runtime=json.loads(a.semantic_runtime.read_text()); routed=json.loads(a.routed_grounding.read_text())
    views=[json.loads(path.read_text()) for path in a.grounding_view]
    fallback=json.loads(a.fallback.read_text()); source=json.loads(a.source_capabilities.read_text())
    if prereg['status']!='FROZEN_BEFORE_ANY_TYPED_REPLICATION_RUNTIME_OR_OUTCOME': raise ValueError('invalid preregistration')
    if prereg['cohort']['public_cohort_sha256']!=cohort['cohort_sha256']: raise ValueError('cohort mismatch')
    reports=[runtime,routed,*views,fallback]
    if len({x['cohort_sha256'] for x in reports}|{cohort['cohort_sha256']})!=1: raise ValueError('artifact cohort mismatch')
    if runtime['valid']!=len(cohort['rows']) or runtime['invalid']: raise ValueError('semantic runtime incomplete')
    if any(x['status']!='RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES' for x in [routed,*views]): raise ValueError('grounding not frozen')
    if any(not x['all_harness_arms_share_exact_receipts'] for x in [routed,*views]): raise ValueError('grounding not arm-shared')
    if fallback['status']!='SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES' or not fallback['shared_by_all_five_arms']: raise ValueError('fallback not shared')
    if fallback['grounding_report_sha256']!=routed['report_sha256']: raise ValueError('fallback/routed binding mismatch')
    compact={str(x['task_id']):str(x['predicted_semantics']) for x in runtime['rows']}
    view_rows=[{str(x['task_id']):x for x in view['rows']} for view in views]
    wanted={str(x['task_id']) for x in cohort['rows']}
    if any(set(x)!=wanted for x in [compact,*view_rows]): raise ValueError('view coverage mismatch')
    ops=tuple(str(x) for x in source['authorized_operators']); edges=tuple(tuple(x) for x in source['authorized_compositions'])
    perm_edges=source_permuted_compositions(ops,edges); all_vm=ops+('SEMANTIC_EQUALS',)
    rows=[]; source_commits=permuted_commits=0
    for public in cohort['rows']:
        tid=str(public['task_id']); executions=[]; permuted=[]; semantic=None
        for by_task in view_rows:
            raw=by_task[tid]; current=_semantic(raw['semantic_receipt'])
            if semantic is not None and current.receipt_sha256!=semantic.receipt_sha256: raise ValueError(f'{tid}: semantic view mismatch')
            semantic=current; graph=_grounding(raw['grounding_receipt'])
            executions.append(execute_layer_b_semantics(compact_semantics=compact[tid],grounding=graph,semantic=semantic,
                authorized_operators=ops,authorized_compositions=edges,ambiguity_policy='STRICT'))
            permuted.append(execute_layer_b_semantics(compact_semantics=compact[tid],grounding=graph,semantic=semantic,
                authorized_operators=ops,authorized_compositions=perm_edges,ambiguity_policy='STRICT'))
        source_plan=plan_harness_arm(semantic,arm='source_induced',source_capabilities=source,all_vm_operators=all_vm)
        perm_plan=plan_harness_arm(semantic,arm='source_permuted',source_capabilities=source,all_vm_operators=all_vm)
        source_commit=(source_plan.status=='PLANNED' and all(x.receipt.status=='COMMITTED' for x in executions)
                       and len({str(x.receipt.prediction) for x in executions})==1)
        perm_commit=(perm_plan.status=='PLANNED' and all(x.receipt.status=='COMMITTED' for x in permuted)
                     and len({str(x.receipt.prediction) for x in permuted})==1)
        source_commits+=int(source_commit); permuted_commits+=int(perm_commit)
        rows.append({'task_id':tid,'source_plan':asdict(source_plan),'permuted_plan':asdict(perm_plan),
                     'source_view_executions':[asdict(x.receipt) for x in executions],
                     'permuted_view_executions':[asdict(x.receipt) for x in permuted],
                     'source_multiview_commit':source_commit,'permuted_multiview_commit':perm_commit})
    coverage=source_commits/len(rows); threshold=float(prereg['gates']['outcome_blind_source_execution_coverage_at_least'])
    matched=len(set(perm_edges))==len(set(edges)) and set(perm_edges)!=set(edges)
    passed=coverage>=threshold and matched
    body={'schema_version':'agqa-layer-b-multiview-preoutcome-v1',
          'status':'ALL_RUNTIME_ARTIFACTS_FROZEN_BEFORE_OUTCOMES' if passed else 'PRE_OUTCOME_GATE_FAILED',
          'preregistration_file_sha256':_sha(a.preregistration),'cohort_sha256':cohort['cohort_sha256'],
          'semantic_runtime_sha256':runtime['runtime_sha256'],'routed_grounding_report_sha256':routed['report_sha256'],
          'grounding_view_report_sha256s':[x['report_sha256'] for x in views],
          'fallback_report_sha256':fallback['report_sha256'],'source_capability_sha256':source['artifact_sha256'],
          'source_commits':source_commits,'source_permuted_commits':permuted_commits,'tasks':len(rows),
          'source_execution_coverage':coverage,'coverage_threshold':threshold,'coverage_gate_passed':coverage>=threshold,
          'matched_permutation_gate_passed':matched,'rows':rows,'answers_read':False,'official_scene_graph_read':False,
          'functional_program_read':False,'next_and_only_outcome_operation':'TYPED_MULTIVIEW_FIVE_ARM_EVALUATOR_ONCE'}
    body['receipt_sha256']=stable_hash(body); a.output.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:body[k] for k in ['status','source_commits','source_permuted_commits','tasks','source_execution_coverage','coverage_gate_passed','receipt_sha256']},indent=2))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
