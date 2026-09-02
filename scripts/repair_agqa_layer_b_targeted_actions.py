#!/usr/bin/env python3
"""Add outcome-blind, semantic-targeted action verification to a frozen grounding."""
from __future__ import annotations
import argparse, hashlib, json, runpy
from dataclasses import asdict
from pathlib import Path
from openai import OpenAI
from motif_transfer.agqa_layer_b_contracts import GroundedEvent, LayerBTaskStateReceipt, RawVideoEventGraphReceipt
from motif_transfer.contracts import stable_hash
from motif_transfer.agqa_semantic_slots import action_anchor_obligations
from scripts.collect_agqa2_active_grounding_v3 import _cached_provider_call, _panel_content, _provider_json_call
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _provider_json_with_retries(client, *, model, system, content, max_tokens, response_format, attempts=3):
    """Retry malformed provider JSON without changing the frozen request."""
    last = None
    for _ in range(attempts):
        try:
            return _provider_json_call(
                client, model=model, system=system, content=content,
                max_tokens=max_tokens, response_format=response_format,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last = exc
    raise RuntimeError(
        f"provider failed structured JSON after {attempts} identical attempts"
    ) from last


def _frame_hash(frame):
    return stable_hash({
        'mode': frame.mode,
        'size': frame.size,
        'pixels_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
    })

def response_format(frame_count):
    interval={"type":"object","additionalProperties":False,"properties":{
        "start_frame":{"type":"integer","minimum":0,"maximum":frame_count-1},"end_frame":{"type":"integer","minimum":0,"maximum":frame_count-1},
        "evidence_frames":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"integer","minimum":0,"maximum":frame_count-1}},
        "confidence":{"type":"number","minimum":0,"maximum":1}},"required":["start_frame","end_frame","evidence_frames","confidence"]}
    schema={"type":"object","additionalProperties":False,"properties":{"observed":{"type":"boolean"},
        "intervals":{"type":"array","maxItems":4,"items":interval},"uncertainties":{"type":"array","items":{"type":"string"}}},
        "required":["observed","intervals","uncertainties"]}
    return {"type":"json_schema","json_schema":{"name":"agqa_exact_action_intervals_v1","strict":True,"schema":schema}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cohort',type=Path,required=True); p.add_argument('--input',type=Path,required=True); p.add_argument('--keys',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--model',default='qwen/qwen3-vl-32b-instruct'); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('targeted grounding output immutable')
    cohort=json.loads(a.cohort.read_text()); base=json.loads(a.input.read_text()); public={str(r['task_id']):r for r in cohort['rows']}
    key=runpy.run_path(str(a.keys)).get('OPENROUTER_API_KEY'); client=OpenAI(api_key=key,base_url='https://openrouter.ai/api/v1',timeout=300,max_retries=2)
    model={'id':a.model}; rows=[]; calls=0; incremental_cost=0.; total_cost=float(base.get('reported_receipt_provider_cost_usd',0))
    system=('You are an answer-blind exact-action video grounding tool. Given one target action and chronological frames, mark observed=true only when that exact action is directly visible. '
            'Reject preparation, aftermath, related actions, static proximity, and uncertain object identity. Return every distinct supported interval in F-index coordinates. Never answer a question or execute temporal/logical reasoning.')
    backend=stable_hash({'upstream':base['grounder_backend_sha256'],'model':a.model,'tool':'SEMANTIC_TARGETED_EXACT_ACTION_INTERVALS_V1','frame_budget':base['frame_budget'],'system':system})
    for raw in base['rows']:
        task=str(raw['task_id']); semantic=_semantic(raw['semantic_receipt']); old=_grounding(raw['grounding_receipt']); sample=public[task]
        frames,seconds,metadata=_sample_video(Path(sample['video_path']),frame_count=base['frame_budget'],max_side=448)
        if tuple(_frame_hash(frame) for frame in frames) != old.selected_frame_sha256s:
            raise ValueError('targeted tool did not receive the exact frozen frames')
        panels=_panels(frames,seconds,{"frames_per_panel":6,"panel_frame_width":224,"jpeg_quality":82}); events=list(old.events); receipts=[]
        for phrase,slot_id in action_anchor_obligations(semantic):
            content=[{"type":"text","text":f"Exact target action to verify: {phrase}\nDo not answer any question."}]+_panel_content(panels)
            core={'prompt_version':'AGQA_TARGETED_ACTION_V1','model':model,'task_id':task,'phrase':phrase,'slot_id':slot_id,
                  'panel_sha256s':[hashlib.sha256(x).hexdigest() for x in panels],'frame_budget':base['frame_budget']}
            provider_error = None
            try:
                payload,usage,reused=_cached_provider_call(cache_dir=a.cache_dir,call_name=f'{task}_{stable_hash(phrase)[:10]}',input_core=core,
                    invoke=lambda:_provider_json_with_retries(client,model=model,system=system,content=content,max_tokens=700,response_format=response_format(base['frame_budget'])))
            except RuntimeError as exc:
                # Fail closed after identical retries.  A provider formatting
                # failure is never converted into positive visual evidence.
                payload = {'observed': False, 'intervals': [], 'uncertainties': ['PROVIDER_JSON_REJECTED']}
                usage = {'reported_cost_usd': 0.0}
                reused = False
                provider_error = f'{type(exc).__name__}:{exc}'
            calls+=int(not reused); cost=float(usage.get('reported_cost_usd',0)); total_cost+=cost; incremental_cost+=cost if not reused else 0
            accepted=0
            if payload['observed']:
                for interval in payload['intervals']:
                    try:
                        event=GroundedEvent(f'E{len(events)}','person',phrase,'',int(interval['start_frame']),int(interval['end_frame']),
                            tuple(sorted(set(int(x) for x in interval['evidence_frames']))),float(interval['confidence']),(slot_id,)); event.validate(len(frames)); events.append(event); accepted+=1
                    except Exception: pass
            receipts.append({'phrase':phrase,'slot_id':slot_id,'payload':payload,'usage':usage,'cache_reused':reused,'accepted_intervals':accepted,'provider_error':provider_error})
        receipt=RawVideoEventGraphReceipt.create(task_id=task,video_sha256=old.video_sha256,semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=old.selected_frame_indices,selected_frame_sha256s=old.selected_frame_sha256s,events=events,
            grounder_backend_sha256=backend,frame_budget=old.frame_budget,provider_calls=old.provider_calls+len(receipts))
        state=LayerBTaskStateReceipt.create(semantic,receipt); row=dict(raw); row.update(grounding_receipt=asdict(receipt),task_state_receipt=asdict(state),targeted_action_receipts=receipts); rows.append(row)
        print(json.dumps({'task_id':task,'obligations':len(receipts),'added_events':len(events)-len(old.events)}),flush=True)
    body={k:v for k,v in base.items() if k not in {'rows','report_sha256','grounder_backend_sha256','provider_calls','reported_receipt_provider_cost_usd','incremental_provider_cost_usd'}}
    body.update(rows=rows,grounder_backend_sha256=backend,provider_calls=sum(r['grounding_receipt']['provider_calls'] for r in rows),
        reported_receipt_provider_cost_usd=total_cost,incremental_provider_cost_usd=incremental_cost,targeted_action_tool=True,additional_provider_calls=calls)
    body['report_sha256']=stable_hash(body); a.output.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n'); print(json.dumps({'rows':len(rows),'calls':calls,'incremental_cost_usd':incremental_cost,'report_sha256':body['report_sha256']},indent=2))
if __name__=='__main__': main()
