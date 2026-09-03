#!/usr/bin/env python3
"""Bind an AGQA query to a label-unique candidate in a frozen event graph.

The expensive visual stage is question-blind and already frozen.  This stage
reads its public-ontology events plus a public question and an answer-blind
temporal parse.  It emits a recall-only candidate proposal; it cannot authorize
the Harness and never reads an AGQA answer, official STSG/program, source
controller, target outcome, or answer alternatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI

from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _cached_provider_call
from scripts.collect_agqa_question_blind_typed_event_inventory_v1 import (
    _provider_call_with_contract_retries,
    _request_cache_contract,
)


SYSTEM = """You are the recall-only query binder for a frozen target-native video evidence graph, not an answer evaluator, final answer generator, safety authorizer, or source-domain agent. The graph was extracted from raw video without seeing a question and can contain missed, fragmented, or noisy events. Given one public question and its already-parsed typed temporal structure, propose the single graph entity most plausibly filling the queried OUTER role. Distinguish objects in anchor clauses from the queried object. Cite only supplied event IDs, and cite at least one event involving the proposed entity. The proposal can never authorize the Harness; a separate raw-pixel verifier decides that. Never use a gold answer, answer alternatives, official scene graph, functional program, source controller, target outcome, or dataset prior. Return only the required JSON schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_inventory(video: dict) -> tuple[list[dict], dict[str, str]]:
    grouped: dict[str, dict] = {}
    label_by_track = {}
    for track in video["stable_tracks"]:
        track_id = str(track["track_id"])
        label = str(track["canonical_label"])
        label_by_track[track_id] = label
        if label == "person":
            continue
        row = grouped.setdefault(label, {
            "canonical_label": label, "track_ids": [],
            "visible_frame_ids": [], "detector_confidence": 0.0,
        })
        row["track_ids"].append(track_id)
        row["visible_frame_ids"].extend(int(x) for x in track["evidence_frames"])
        row["detector_confidence"] = max(
            float(row["detector_confidence"]), float(track["confidence"]),
        )
    candidates = []
    for label in sorted(grouped):
        row = grouped[label]
        candidates.append({
            "candidate_id": f"O{len(candidates)}",
            "canonical_label": label,
            "track_ids": sorted(set(row["track_ids"])),
            "visible_frame_ids": sorted(set(row["visible_frame_ids"])),
            "detector_confidence": float(row["detector_confidence"]),
        })
    return candidates, label_by_track


def _event_rows(
    video: dict, candidates: list[dict], label_by_track: dict[str, str],
) -> list[dict]:
    candidate_by_label = {
        row["canonical_label"]: row["candidate_id"] for row in candidates
    }
    output = []
    for event in video["events"]:
        label = label_by_track[str(event["object_track_id"])]
        if label == "person" or label not in candidate_by_label:
            continue
        output.append({
            "event_id": str(event["event_id"]),
            "predicate": str(event["predicate"]),
            "candidate_id": candidate_by_label[label],
            "candidate_label": label,
            "object_role": str(event["object_role"]),
            "start_frame": int(event["start_frame"]),
            "end_frame": int(event["end_frame"]),
            "evidence_frames": [int(x) for x in event["evidence_frames"]],
            "confidence": float(event["confidence"]),
        })
    return output


def _response_format(candidate_ids: list[str], event_ids: list[str]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_evidence_graph_query_binding_v1",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["PROPOSED"]},
                "selected_candidate_id": {
                    "type": "string", "enum": candidate_ids,
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "supporting_event_ids": {
                    "type": "array", "minItems": 1, "maxItems": 8,
                    "items": {"type": "string", "enum": event_ids},
                },
            },
            "required": [
                "status", "selected_candidate_id", "confidence",
                "supporting_event_ids",
            ],
        },
    }}


def _validate(payload: dict, *, candidates: set[str], events: list[dict]) -> dict:
    if set(payload) != {
        "status", "selected_candidate_id", "confidence", "supporting_event_ids",
    }:
        raise ValueError("graph-binding payload contains unexpected fields")
    if payload["status"] != "PROPOSED":
        raise ValueError("graph-binding stage has proposal authority only")
    selected = str(payload["selected_candidate_id"])
    confidence = float(payload["confidence"])
    cited = sorted(set(str(x) for x in payload["supporting_event_ids"]))
    if selected not in candidates or not 0 <= confidence <= 1 or not cited:
        raise ValueError("graph-binding proposal is malformed")
    by_id = {row["event_id"]: row for row in events}
    if any(event_id not in by_id for event_id in cited):
        raise ValueError("graph-binding cites an unknown event")
    if not any(by_id[event_id]["candidate_id"] == selected for event_id in cited):
        raise ValueError("graph-binding cites no event involving its candidate")
    return {
        "status": "PROPOSED", "selected_candidate_id": selected,
        "confidence": confidence, "supporting_event_ids": cited,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--event-inventory", type=Path, required=True)
    parser.add_argument("--query-grounding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--provider", default="parasail")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evidence-graph binding output is immutable")

    cohort = json.loads(args.cohort.read_text())
    inventory = json.loads(args.event_inventory.read_text())
    grounding = json.loads(args.query_grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    if inventory.get("status") != (
        "CONSUMED_DEVELOPMENT_EVENT_INVENTORY_NOT_TRANSFER_EVIDENCE"
    ) or grounding.get("status") != (
        "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE"
    ):
        raise ValueError("graph binder requires frozen consumed-development inputs")
    forbidden = (
        "answer_read", "official_scene_graph_read", "official_stsg_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
    )
    if any(inventory.get(key) for key in forbidden) or any(
        grounding.get(key) for key in forbidden
    ):
        raise ValueError("graph-binding input crossed its authority boundary")
    actual = {
        "cohort_file_sha256": _sha256(args.cohort),
        "event_inventory_file_sha256": _sha256(args.event_inventory),
        "query_grounding_file_sha256": _sha256(args.query_grounding),
        "collector_sha256": _sha256(Path(__file__)),
    }
    if actual != {
        key: protocol["immutable_inputs"][key] for key in actual
    }:
        raise ValueError("graph-binding immutable inputs differ")
    runtime = protocol["evidence_graph_query_binding"]
    if args.model != runtime["model"] or args.provider != runtime["provider"]:
        raise ValueError("graph-binding runtime differs from protocol")

    public = {str(row["task_id"]): row for row in cohort["rows"]}
    video_events = {str(row["video_id"]): row for row in inventory["rows"]}
    rows = grounding["rows"][:args.max_tasks] if args.max_tasks else grounding["rows"]
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(
        api_key=key, base_url="https://openrouter.ai/api/v1",
        timeout=300, max_retries=2,
    )
    model = {
        "id": args.model, "omit_temperature": False,
        "seed": int(runtime.get("seed", 0)),
        "provider": {
            "only": [args.provider],
            "allow_fallbacks": bool(runtime.get("provider_allow_fallbacks", False)),
        },
    }

    outputs = []
    calls = 0
    cost = 0.0
    for query in rows:
        task_id = str(query["task_id"])
        video_id = str(query["video_id"])
        video = video_events[video_id]
        candidates, label_by_track = _candidate_inventory(video)
        events = _event_rows(video, candidates, label_by_track)
        involved = {row["candidate_id"] for row in events}
        usable_candidates = [
            row for row in candidates if row["candidate_id"] in involved
        ]
        if not usable_candidates or not events:
            outputs.append({
                "task_id": task_id, "video_id": video_id,
                "status": "ABSTAIN_NO_GRAPH_EVENTS", "selected_candidate_id": None,
                "selected_candidate": None, "confidence": 0.0,
                "supporting_event_ids": [], "candidates": usable_candidates,
                "events": events, "usage": None, "cache_reused": True,
                "provider_error": None,
            })
            continue
        table = "; ".join(
            f"{row['candidate_id']}={row['canonical_label']}"
            for row in usable_candidates
        )
        timeline = "\n".join(
            f"{row['event_id']}: S{row['start_frame']}..S{row['end_frame']} "
            f"person {row['predicate']} {row['candidate_id']}"
            f"({row['candidate_label']}) as {row['object_role']}; "
            f"pixel_frames={row['evidence_frames']}; conf={row['confidence']:.2f}"
            for row in events
        )
        prompt = (
            f"Public question: {public[task_id]['question']}\n"
            f"Queried outer predicate: {query['root_predicate']}\n"
            f"Queried outer role: {query['requested_role']}\n"
            f"Temporal operator: {query['temporal_operator']}\n"
            f"Answer-blind anchor intervals: {query['anchor_intervals']}\n"
            f"Candidate inventory: {table}\n"
            "Chronological question-blind typed evidence graph:\n"
            f"{timeline}\n"
            "Propose the one candidate filling the queried OUTER role. Objects "
            "mentioned in anchor clauses are context, not automatically the query target."
        )
        response_format = _response_format(
            [row["candidate_id"] for row in usable_candidates],
            [row["event_id"] for row in events],
        )
        core = {
            "protocol": "AGQA_EVIDENCE_GRAPH_QUERY_BINDING_V1",
            "protocol_file_sha256": _sha256(args.protocol),
            "task_id": task_id,
            "question_sha256": public[task_id]["question_sha256"],
            "event_inventory_report_sha256": inventory["report_sha256"],
            "query_grounding_report_sha256": grounding["report_sha256"],
            "candidate_inventory_sha256": stable_hash(usable_candidates),
            "event_timeline_sha256": stable_hash(events),
            "model": model, "system_sha256": stable_hash(SYSTEM),
            "prompt_sha256": stable_hash(prompt),
            "request_contract": _request_cache_contract(
                model=model, max_tokens=int(runtime["max_tokens"]),
                response_format=response_format,
                maximum_attempts=int(runtime["maximum_contract_attempts"]),
            ),
        }
        provider_error = None
        try:
            payload, usage, reused = _cached_provider_call(
                cache_dir=args.cache_dir,
                call_name=f"graph_binding_{task_id}", input_core=core,
                invoke=lambda: _provider_call_with_contract_retries(
                    client, model=model, system=SYSTEM,
                    content=[{"type": "text", "text": prompt}],
                    max_tokens=int(runtime["max_tokens"]),
                    response_format=response_format,
                    maximum_attempts=int(runtime["maximum_contract_attempts"]),
                    validator=lambda value: _validate(
                        value,
                        candidates={row["candidate_id"] for row in usable_candidates},
                        events=events,
                    ),
                ),
            )
            decision = _validate(
                payload,
                candidates={row["candidate_id"] for row in usable_candidates},
                events=events,
            )
        except Exception as exc:
            usage = getattr(exc, "usage", {
                "reported_cost_usd": 0.0, "provider_attempts": 0,
                "contract_retry_count": 0, "attempt_receipts": [],
            })
            reused = False
            provider_error = f"{type(exc).__name__}:{exc}"
            decision = {
                "status": "UNKNOWN", "selected_candidate_id": None,
                "confidence": 0.0, "supporting_event_ids": [],
            }
        selected = next((
            row for row in usable_candidates
            if row["candidate_id"] == decision["selected_candidate_id"]
        ), None)
        calls += int(not reused) * int(usage.get("provider_attempts", 1))
        cost += float(usage.get("reported_cost_usd", 0.0))
        outputs.append({
            "task_id": task_id, "video_id": video_id, **decision,
            "selected_candidate": selected, "candidates": usable_candidates,
            "events": events, "usage": usage, "cache_reused": reused,
            "provider_error": provider_error,
        })
        print(json.dumps({
            "task_id": task_id, "status": decision["status"],
            "candidate": selected["canonical_label"] if selected else None,
            "confidence": decision["confidence"], "cost_usd_running": cost,
        }), flush=True)

    report = {
        "schema_version": "agqa-evidence-graph-query-binding-v1",
        "status": "CONSUMED_DEVELOPMENT_GRAPH_BINDING_NOT_TRANSFER_EVIDENCE",
        "protocol_file_sha256": _sha256(args.protocol),
        "cohort_sha256": cohort["cohort_sha256"],
        "event_inventory_report_sha256": inventory["report_sha256"],
        "query_grounding_report_sha256": grounding["report_sha256"],
        "model": model, "rows": outputs, "provider_calls": calls,
        "reported_cost_usd": cost,
        "question_read": True, "answer_read": False,
        "agqa_answer_alternatives_read": False,
        "official_scene_graph_read": False, "official_stsg_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False, "proposal_may_authorize_harness": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
