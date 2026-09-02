#!/usr/bin/env python3
"""Outcome-blind Qwen3-VL-235B grounding pilot for AGQA Layer B."""
from __future__ import annotations
import argparse, hashlib, json, runpy
from dataclasses import asdict
from pathlib import Path
from openai import OpenAI
from motif_transfer.agqa_layer_b_contracts import GroundedEvent, LayerBTaskStateReceipt, RawVideoEventGraphReceipt
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _cached_provider_call, _panel_content, _provider_json_call
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video
from scripts.collect_agqa_layer_b_local_grounding import SYSTEM, _canonical_slot_bindings, _contains_forbidden_key, _frame_hash, _sha_file, _slot_prompt
from scripts.evaluate_agqa_layer_b_five_arm import _semantic

def provider_json_with_retries(client, *, model, system, content, max_tokens, response_format, attempts=3):
    """Retry malformed/truncated provider JSON without changing the frozen request."""
    last = None
    for _ in range(attempts):
        try:
            return _provider_json_call(client, model=model, system=system, content=content,
                max_tokens=max_tokens, response_format=response_format)
        except (json.JSONDecodeError, ValueError) as exc:
            last = exc
    raise RuntimeError(f'provider failed structured JSON after {attempts} identical attempts') from last

def response_format(frame_count:int, slot_ids:list[str])->dict:
    event={"type":"object","additionalProperties":False,"properties":{
        "event_id":{"type":"string"},"subject":{"type":"string"},"predicate":{"type":"string"},"object":{"type":"string"},
        "start_frame":{"type":"integer","minimum":0,"maximum":frame_count-1},
        "end_frame":{"type":"integer","minimum":0,"maximum":frame_count-1},
        "evidence_frames":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"integer","minimum":0,"maximum":frame_count-1}},
        "confidence":{"type":"number","minimum":0,"maximum":1},
        "semantic_slot_ids":{"type":"array","minItems":1,"maxItems":6,"items":{"type":"string","enum":slot_ids}},
    },"required":["event_id","subject","predicate","object","start_frame","end_frame","evidence_frames","confidence","semantic_slot_ids"]}
    schema={"type":"object","additionalProperties":False,"properties":{
        "events":{"type":"array","maxItems":24,"items":event},"uncertainties":{"type":"array","items":{"type":"string"}}
    },"required":["events","uncertainties"]}
    return {"type":"json_schema","json_schema":{"name":"agqa_layer_b_event_graph_v1","strict":True,"schema":schema}}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--cohort',type=Path,required=True); p.add_argument('--semantic-runtime',type=Path,required=True)
    p.add_argument('--keys',type=Path,default=Path('/fs/gamma-projects/vlm-robot/keys.py')); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True); p.add_argument('--model',default='qwen/qwen3-vl-235b-a22b-instruct')
    p.add_argument('--positions',default='0,4,8,12,16,20,24,28'); p.add_argument('--frame-count',type=int,default=24)
    p.add_argument('--max-tokens',type=int,default=1800)
    p.add_argument('--response-mode',choices=('json_schema','json_object'),default='json_schema')
    p.add_argument('--omit-temperature',action='store_true')
    p.add_argument('--reasoning-effort'); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Qwen235 grounding output is immutable')
    cohort=json.loads(a.cohort.read_text()); runtime=json.loads(a.semantic_runtime.read_text())
    if runtime['cohort_sha256']!=cohort['cohort_sha256'] or runtime['status']!='SEMANTIC_RUNTIME_FROZEN_BEFORE_VIDEO_OR_OUTCOME': raise ValueError('semantic/cohort mismatch')
    semantic_rows={str(r['task_id']):r for r in runtime['rows']}
    positions=(list(range(len(cohort['rows']))) if a.positions.strip().casefold() == 'all'
               else [int(x) for x in a.positions.replace(':',',').split(',') if x.strip()])
    selected=[cohort['rows'][i] for i in positions]; key=runpy.run_path(str(a.keys)).get('OPENROUTER_API_KEY')
    if not key: raise ValueError('OpenRouter API key unavailable')
    client=OpenAI(api_key=key,base_url='https://openrouter.ai/api/v1',timeout=300,max_retries=2)
    model={"id":a.model,"omit_temperature":a.omit_temperature}
    if a.reasoning_effort:
        model['reasoning']={'effort':a.reasoning_effort}
    rows=[]; total_cost=0.0; incremental_cost=0.0; calls=0
    system=SYSTEM.replace('F0..F23',f'F0..F{a.frame_count-1}')
    backend=stable_hash({
        'model':model,'system':system,'frame_count':a.frame_count,
        'sampling':'uniform_full_video','local_contract':'AGQA_LAYER_B_EVENT_GRAPH_V1',
        'response_mode':a.response_mode,
        'temperature':None if a.omit_temperature else 0,
    })
    for position,row in zip(positions,selected):
        task_id=str(row['task_id']); semantic=_semantic(semantic_rows[task_id]['receipt']); video=Path(row['video_path'])
        frames,seconds,metadata=_sample_video(video,frame_count=a.frame_count,max_side=448)
        panels=_panels(frames,seconds,{"frames_per_panel":6,"panel_frame_width":224,"jpeg_quality":82})
        content=[{"type":"text","text":f"Question for perceptual relevance only (never answer it): {row['question']}\nFrozen semantic slots:\n{_slot_prompt(semantic)}"}]+_panel_content(panels)
        perceptual=[s.slot_id for s in semantic.slots if s.kind in {'LITERAL','ENTITY','ACTION','RELATION'}]
        core={'prompt_version':'AGQA_LAYER_B_QWEN235_GROUNDER_V1','model':model,'task_id':task_id,'question_sha256':row['question_sha256'],
              'semantic_receipt_sha256':semantic.receipt_sha256,'panel_sha256s':[hashlib.sha256(x).hexdigest() for x in panels],
              'frame_count':a.frame_count,'max_tokens':a.max_tokens}
        provider_error = None
        try:
            requested_format=(response_format(a.frame_count,perceptual)
                              if a.response_mode=='json_schema' else {'type':'json_object'})
            payload,usage,reused=_cached_provider_call(cache_dir=a.cache_dir,call_name=f'ground_{task_id}',input_core={**core,'response_mode':a.response_mode},
                invoke=lambda:provider_json_with_retries(client,model=model,system=system,content=content,max_tokens=a.max_tokens,response_format=requested_format))
        except RuntimeError as exc:
            # The candidate fails closed after identical structured-output
            # retries.  A second frozen candidate may still supply evidence;
            # malformed text is never interpreted or repaired into events.
            payload = {'events': [], 'uncertainties': ['PROVIDER_JSON_REJECTED']}
            usage = {'reported_cost_usd': 0.0}
            reused = False
            provider_error = f'{type(exc).__name__}:{exc}'
        if (not isinstance(payload,dict) or not isinstance(payload.get('events'),list)
                or len(payload['events'])>24 or _contains_forbidden_key(payload)):
            provider_error = provider_error or 'ValueError:provider payload failed local Layer-B contract validation'
            payload = {'events': [], 'uncertainties': ['PROVIDER_CONTRACT_REJECTED']}
        calls+=int(not reused); total_cost+=float(usage.get('reported_cost_usd',0)); incremental_cost+=float(usage.get('reported_cost_usd',0)) if not reused else 0.0; events=[]; rejected=[]
        for index,e in enumerate(payload['events']):
            try:
                candidate=GroundedEvent(f'E{len(events)}',str(e['subject']),str(e['predicate']),str(e['object']),int(e['start_frame']),int(e['end_frame']),
                    tuple(sorted(set(int(x) for x in e['evidence_frames']))),float(e['confidence']),_canonical_slot_bindings(e,semantic))
                candidate.validate(len(frames)); events.append(candidate)
            except Exception as exc: rejected.append({'raw_event_index':index,'reason':f'{type(exc).__name__}:{exc}','raw_event_sha256':stable_hash(e)})
        receipt=RawVideoEventGraphReceipt.create(task_id=task_id,video_sha256=_sha_file(video),semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=tuple(range(len(frames))),selected_frame_sha256s=tuple(_frame_hash(f) for f in frames),events=events,
            grounder_backend_sha256=backend,frame_budget=a.frame_count,provider_calls=1)
        state=LayerBTaskStateReceipt.create(semantic,receipt)
        rows.append({'task_id':task_id,'video_id':row['video_id'],'cohort_position':position,'semantic_receipt':asdict(semantic),
                     'grounding_receipt':asdict(receipt),'task_state_receipt':asdict(state),'raw_payload':payload,'usage':usage,'cache_reused':reused,
                     'rejected_events':rejected,'provider_error':provider_error,
                     'video_metadata':metadata,'panel_sha256s':[hashlib.sha256(x).hexdigest() for x in panels]})
        print(json.dumps({'task_id':task_id,'events':len(events),'cost_usd':usage.get('reported_cost_usd',0),'cache_reused':reused}),flush=True)
    body={'schema_version':'agqa-layer-b-qwen235-grounding-pilot-v1','status':'RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES','pilot':True,
          'cohort_sha256':cohort['cohort_sha256'],'semantic_runtime_sha256':runtime['runtime_sha256'],'grounder_backend_sha256':backend,
          'model':a.model,'frame_budget':a.frame_count,'selected_positions':positions,'cohort_rows_total':len(cohort['rows']),'provider_calls':calls,
          'reported_receipt_provider_cost_usd':total_cost,'incremental_provider_cost_usd':incremental_cost,'rows':rows,'all_harness_arms_share_exact_receipts':True,'answer_read':False,
          'official_scene_graph_read':False,'functional_program_read':False,'source_controller_read':False}
    body['report_sha256']=stable_hash(body); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':body['status'],'rows':len(rows),'provider_calls':calls,'receipt_cost_usd':total_cost,'incremental_cost_usd':incremental_cost,'report_sha256':body['report_sha256']},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
