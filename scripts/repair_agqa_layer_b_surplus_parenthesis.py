#!/usr/bin/env python3
"""Outcome-blind removal of a unique surplus trailing DSL parenthesis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target
from motif_transfer.contracts import stable_hash


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--input-runtime',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('syntax repair output is immutable')
    cohort=json.loads(a.cohort.read_text()); runtime=json.loads(a.input_runtime.read_text())
    if runtime['cohort_sha256']!=cohort['cohort_sha256']: raise ValueError('cohort mismatch')
    public={str(x['task_id']):x for x in cohort['rows']}; repaired={}
    for old in runtime['rows']:
        if old['status']=='SEMANTIC_SLOTS_FROZEN': continue
        prediction=str(old['predicted_semantics']); excess=prediction.count(')')-prediction.count('(')
        if old.get('reason')!='trailing token: )' or excess<=0 or not prediction.endswith(')'*excess):
            raise ValueError(f"{old['task_id']}: invalid output has no unique surplus-parenthesis repair")
        candidate=prediction[:-excess]
        row=public[str(old['task_id'])]
        receipt=parse_compact_semantic_target(candidate,task_id=str(old['task_id']),
            question_sha256=row['question_sha256'],parser_sha256=runtime['parser_sha256'],
            parser_training_authority='AGQA_TRAIN_DEV_TO_OPERATOR_FREE_COMPACT_SEMANTICS')
        repaired[str(old['task_id'])]={**old,'status':'SEMANTIC_SLOTS_FROZEN','predicted_semantics':candidate,
            'receipt':asdict(receipt),'reason':None,'syntax_repair':'UNIQUE_MINIMAL_SURPLUS_TRAILING_PARENTHESIS_REMOVAL'}
    if not repaired: raise ValueError('runtime has nothing repairable')
    rows=[repaired.get(str(x['task_id']),x) for x in runtime['rows']]
    body={**{k:v for k,v in runtime.items() if k not in {'rows','valid','invalid','runtime_sha256'}},
          'rows':rows,'valid':len(rows),'invalid':0,'base_runtime_sha256':runtime['runtime_sha256'],
          'repaired_task_ids':sorted(repaired),'repair_authority':'SYNTAX_ONLY_NO_ANSWER_PROGRAM_VIDEO_OR_PROVIDER'}
    body['runtime_sha256']=stable_hash(body); a.output.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'rows':len(rows),'repaired':sorted(repaired),'runtime_sha256':body['runtime_sha256']},indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
