#!/usr/bin/env python3
"""One-shot typed temporal Layer-B multi-view five-arm evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import ARMS, plan_harness_arm
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _gold_rows,_grounding,_matches,_mcnemar,_semantic


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--preregistration',type=Path,required=True)
    p.add_argument('--pre-outcome-receipt',type=Path,required=True); p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--semantic-runtime',type=Path,required=True); p.add_argument('--routed-grounding',type=Path,required=True)
    p.add_argument('--fallback',type=Path,required=True); p.add_argument('--source-capabilities',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True); p.add_argument('--entry',default='AGQA_balanced/train_balanced.txt')
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('typed multi-view evaluation is immutable')
    prereg=json.loads(a.preregistration.read_text()); pre=json.loads(a.pre_outcome_receipt.read_text())
    cohort=json.loads(a.cohort.read_text()); runtime=json.loads(a.semantic_runtime.read_text())
    routed=json.loads(a.routed_grounding.read_text()); fallback_report=json.loads(a.fallback.read_text())
    source=json.loads(a.source_capabilities.read_text())
    if prereg['status']!='FROZEN_BEFORE_ANY_TYPED_REPLICATION_RUNTIME_OR_OUTCOME': raise ValueError('invalid preregistration')
    if pre['status']!='ALL_RUNTIME_ARTIFACTS_FROZEN_BEFORE_OUTCOMES': raise ValueError('pre-outcome gate failed')
    if prereg['cohort']['public_cohort_sha256']!=cohort['cohort_sha256'] or pre['cohort_sha256']!=cohort['cohort_sha256']: raise ValueError('cohort mismatch')
    if len({cohort['cohort_sha256'],runtime['cohort_sha256'],routed['cohort_sha256'],fallback_report['cohort_sha256']})!=1: raise ValueError('artifact cohort mismatch')
    if fallback_report['grounding_report_sha256']!=routed['report_sha256'] or fallback_report['report_sha256']!=pre['fallback_report_sha256']: raise ValueError('fallback binding mismatch')
    wanted={str(x['task_id']) for x in cohort['rows']}; compact={str(x['task_id']):str(x['predicted_semantics']) for x in runtime['rows']}
    routed_rows={str(x['task_id']):x for x in routed['rows']}; fallback={str(x['task_id']):str(x['prediction']) for x in fallback_report['rows']}
    pre_rows={str(x['task_id']):x for x in pre['rows']}
    if any(set(x)!=wanted for x in [compact,routed_rows,fallback,pre_rows]): raise ValueError('artifact coverage mismatch')
    evaluator=_gold_rows(a.archive,a.entry,wanted); ops=tuple(str(x) for x in source['authorized_operators']); all_vm=ops+('SEMANTIC_EQUALS',)
    rows=[]
    for public in cohort['rows']:
        tid=str(public['task_id']); raw=routed_rows[tid]; semantic=_semantic(raw['semantic_receipt']); graph=_grounding(raw['grounding_receipt'])
        plans={arm:plan_harness_arm(semantic,arm=arm,source_capabilities=source,all_vm_operators=all_vm) for arm in ARMS}
        eager=execute_layer_b_semantics(compact_semantics=compact[tid],grounding=graph,semantic=semantic,
            authorized_operators=ops,authorized_compositions=None,ambiguity_policy='EAGER')
        pre_row=pre_rows[tid]; source_commit=bool(pre_row['source_multiview_commit']); perm_commit=bool(pre_row['permuted_multiview_commit'])
        source_prediction=str(pre_row['source_view_executions'][0]['prediction']) if source_commit else fallback[tid]
        perm_prediction=str(pre_row['permuted_view_executions'][0]['prediction']) if perm_commit else fallback[tid]
        generic_commit=plans['generic_scaffold'].status=='PLANNED' and eager.receipt.status=='COMMITTED'
        predictions={'neural_only':fallback[tid],
                     'generic_scaffold':str(eager.receipt.prediction) if generic_commit else fallback[tid],
                     'source_permuted':perm_prediction,'source_induced':source_prediction,
                     'target_written_isomorphic':source_prediction}
        gold=str(evaluator[tid]['answer'])
        rows.append({'task_id':tid,'video_id':str(public['video_id']),'gold_answer_evaluator_only':gold,
                     'plans':{arm:asdict(plan) for arm,plan in plans.items()},'generic_execution':asdict(eager.receipt),
                     'generic_commit':generic_commit,'source_multiview_commit':source_commit,
                     'source_permuted_multiview_commit':perm_commit,'predictions':predictions,
                     'correct':{arm:_matches(value,gold) for arm,value in predictions.items()}})
    correct={arm:[row['correct'][arm] for row in rows] for arm in ARMS}; n=len(rows)
    summaries={arm:{'correct':sum(correct[arm]),'total':n,'accuracy':sum(correct[arm])/n,
        'symbolic_commits':sum(row['source_multiview_commit'] if arm in {'source_induced','target_written_isomorphic'}
          else row['source_permuted_multiview_commit'] if arm=='source_permuted'
          else row['generic_commit'] if arm=='generic_scaffold' else 0 for row in rows)} for arm in ARMS}
    comparisons={baseline:_mcnemar(correct['source_induced'],correct[baseline]) for baseline in ('neural_only','generic_scaffold','source_permuted')}
    versus_neural={arm:_mcnemar(correct[arm],correct['neural_only']) for arm in ('generic_scaffold','source_permuted','source_induced')}
    max_losses=math.floor(float(prereg['gates']['negative_transfer_fraction_at_most'])*n)
    feasible=[arm for arm in ('generic_scaffold','source_permuted','source_induced') if versus_neural[arm]['losses']<=max_losses]
    gates={'source_beats_neural':summaries['source_induced']['correct']>summaries['neural_only']['correct'],
           'source_vs_neural_significant':comparisons['neural_only']['exact_two_sided_p']<.05,
           'source_negative_transfer_bounded':comparisons['neural_only']['losses']<=max_losses,
           'source_beats_matched_permuted':summaries['source_induced']['correct']>summaries['source_permuted']['correct'],
           'source_vs_matched_permuted_significant':comparisons['source_permuted']['exact_two_sided_p']<.05,
           'source_is_best_feasible_symbolic_arm':all(summaries['source_induced']['correct']>=summaries[x]['correct'] for x in feasible),
           'target_written_isomorphic_action_equivalence':all(r['predictions']['source_induced']==r['predictions']['target_written_isomorphic'] for r in rows),
           'pre_outcome_coverage_gate_passed':bool(pre['coverage_gate_passed'])}
    body={'schema_version':'agqa-layer-b-typed-temporal-multiview-five-arm-v1',
          'status':'TYPED_TEMPORAL_LAYER_B_GATES_PASSED' if all(gates.values()) else 'TYPED_TEMPORAL_LAYER_B_GATES_FAILED',
          'claim_scope':'SELECTIVE_SOURCE_COMPATIBLE_ORDERED_EFFECT_TRANSFER_TO_RAW_VIDEO_DURATION_REASONING',
          'cohort_sha256':cohort['cohort_sha256'],'pre_outcome_receipt_sha256':pre['receipt_sha256'],
          'routed_grounding_report_sha256':routed['report_sha256'],'grounding_view_report_sha256s':pre['grounding_view_report_sha256s'],
          'fallback_report_sha256':fallback_report['report_sha256'],'source_capability_sha256':source['artifact_sha256'],
          'negative_transfer_max_losses':max_losses,'feasible_symbolic_arms':feasible,'rows':rows,
          'summaries':summaries,'comparisons':comparisons,'versus_neural':versus_neural,'gates':gates,
          'frames_grounding_views_parser_executor_fallback_shared':True,'only_symbolic_harness_differs':True,
          'raw_video_end_to_end_only':True,'official_scene_graph_used_at_runtime':False}
    body['report_sha256']=stable_hash(body); a.output.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:body[k] for k in ['status','summaries','comparisons','versus_neural','feasible_symbolic_arms','gates','report_sha256']},indent=2))
    return 0 if all(gates.values()) else 1


if __name__=='__main__': raise SystemExit(main())
