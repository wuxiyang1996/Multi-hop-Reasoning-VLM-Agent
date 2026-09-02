#!/usr/bin/env python3
"""Rebuild frozen grounding receipts with deterministic perceptual slot binding."""
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
from motif_transfer.agqa_layer_b_contracts import GroundedEvent, LayerBTaskStateReceipt, RawVideoEventGraphReceipt
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa_layer_b_local_grounding import _canonical_slot_bindings, _parse_json
from scripts.evaluate_agqa_layer_b_five_arm import _semantic

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('revalidated grounding is immutable')
    source=json.loads(a.input.read_text()); rows=[]
    backend=stable_hash({'upstream_grounder_backend_sha256':source['grounder_backend_sha256'],
                         'binding':'DETERMINISTIC_PERCEPTUAL_LEXICAL_PROJECTION_V1','max_bindings':6})
    for raw in source['rows']:
        semantic=_semantic(raw['semantic_receipt']); old=raw['grounding_receipt']; rejected=[]; events=[]
        try: payload=_parse_json(raw['raw_response'])
        except Exception as exc:
            payload={'events':[]}; rejected.append({'raw_event_index':None,'reason':f'{type(exc).__name__}:{exc}','raw_event_sha256':stable_hash(raw['raw_response'])})
        for index,event in enumerate(payload.get('events',())):
            try:
                candidate=GroundedEvent(event_id=f'E{len(events)}',subject=str(event.get('subject','person')),
                    predicate=str(event.get('predicate','')),object=str(event.get('object','')),
                    start_frame=int(event['start_frame']),end_frame=int(event['end_frame']),
                    evidence_frames=tuple(sorted(set(int(x) for x in event['evidence_frames']))),
                    confidence=float(event.get('confidence',0)),semantic_slot_ids=_canonical_slot_bindings(event,semantic))
                candidate.validate(len(old['selected_frame_indices'])); events.append(candidate)
            except Exception as exc:
                rejected.append({'raw_event_index':index,'reason':f'{type(exc).__name__}:{exc}','raw_event_sha256':stable_hash(event)})
        receipt=RawVideoEventGraphReceipt.create(task_id=raw['task_id'],video_sha256=old['video_sha256'],
            semantic_slots_sha256=semantic.receipt_sha256,selected_frame_indices=old['selected_frame_indices'],
            selected_frame_sha256s=old['selected_frame_sha256s'],events=events,grounder_backend_sha256=backend,
            frame_budget=old['frame_budget'],provider_calls=old['provider_calls'])
        state=LayerBTaskStateReceipt.create(semantic,receipt); row=dict(raw); row.update(
            grounding_receipt=asdict(receipt),task_state_receipt=asdict(state),rejected_events=rejected,
            binding_revalidated_without_model_call=True); rows.append(row)
    body={k:v for k,v in source.items() if k not in {'rows','report_sha256','grounder_backend_sha256','rejected_event_count'}}
    body.update(rows=rows,grounder_backend_sha256=backend,rejected_event_count=sum(len(r['rejected_events']) for r in rows),
                binding_revalidation='DETERMINISTIC_PERCEPTUAL_LEXICAL_PROJECTION_V1',additional_model_invocations=0)
    body['report_sha256']=stable_hash(body); a.output.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':body['status'],'rows':len(rows),'rejected_event_count':body['rejected_event_count'],'report_sha256':body['report_sha256']},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
