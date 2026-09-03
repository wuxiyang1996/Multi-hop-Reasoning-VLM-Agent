#!/usr/bin/env python3
"""Collect answer-blind multi-candidate atomic AGQA grounding receipts.

This consumed-development collector fixes the structural defect of a one-best
proposal followed by a binary verifier.  It verifies every fixed detector
candidate independently against one parser-derived atomic predicate.  A
deterministic temporal executor then combines those atomic observations with
separately frozen, answer-blind anchor localizations.

The VLM never receives a question, answer, answer alternatives, temporal
operator, official STSG/program, source controller, or target outcome.  This
stage has proposal authority only and cannot authorize a Harness action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call,
    _panel_content,
    _panels,
)
from scripts.collect_agqa_question_blind_typed_event_inventory_v1 import (
    _provider_call_with_contract_retries,
    _request_cache_contract,
)
from scripts.pilot_agqa_atomic_temporal_grounder_v2 import (
    _annotate,
    _candidate_groups,
    _event_frame_ids,
    _event_response_format,
    _execute_temporal,
    _provider_failure,
    _validate_rows,
)
from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _exact_sgdet_frames,
)


SYSTEM = """You are an answer-blind atomic video proposition verifier, not a question-answering model, candidate ranker, or source-domain agent. P0 is the person track. For every fixed candidate independently, determine whether the chronological raw pixels visibly support exactly the supplied typed proposition between P0 and that candidate. Candidate boxes, IDs, and public-ontology labels are fallible detector proposals, not answer options or facts. SUPPORTED requires direct visible evidence of both candidate identity and the exact predicate/role; mere co-presence is insufficient. Otherwise return UNKNOWN. Do not compare candidates, select a winner, apply a temporal operator, reconstruct a question, or use a gold answer, answer alternatives, official scene graph/STSG, functional program, source controller, target outcome, or dataset prior. Return only the required JSON schema."""


FORBIDDEN_FLAGS = (
    "answer_read",
    "official_scene_graph_read",
    "official_stsg_read",
    "functional_program_read",
    "source_controller_read",
    "target_outcome_read",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _typed_claim(predicate: str, role: str, candidate: dict) -> str:
    label = str(candidate["canonical_label"])
    if role == "patient":
        return f"P0 visibly {predicate} this candidate ({label})"
    if role == "relation_object":
        return (
            f"P0 visibly has the spatial/contact relation '{predicate}' "
            f"relative to this candidate ({label})"
        )
    return f"P0 visibly has predicate '{predicate}' with this candidate ({label})"


def _root_window(
    temporal_operator: str,
    anchor_rows: list[dict],
    *,
    frame_count: int,
    uncertainty: int,
) -> tuple[int, int] | None:
    """Execute the public temporal operator over answer-blind anchor evidence."""

    if frame_count <= 0 or uncertainty < 0:
        raise ValueError("invalid frame count or temporal uncertainty")
    operator = temporal_operator.upper()
    intervals = [
        (
            min(int(value) for value in row["evidence_frame_ids"]),
            max(int(value) for value in row["evidence_frame_ids"]),
        )
        for row in anchor_rows
        if row["status"] == "SUPPORTED" and row["evidence_frame_ids"]
    ]
    if operator == "VIDEO":
        return 0, frame_count - 1
    required = 2 if operator == "BETWEEN" else 1
    if len(intervals) < required:
        return None
    if operator == "BEFORE":
        upper = intervals[0][0] - 1
        return (0, upper) if upper >= 0 else None
    if operator == "AFTER":
        lower = intervals[0][1] + 1
        return (lower, frame_count - 1) if lower < frame_count else None
    if operator == "WHILE":
        return (
            max(0, intervals[0][0] - uncertainty),
            min(frame_count - 1, intervals[0][1] + uncertainty),
        )
    if operator == "BETWEEN":
        first, second = sorted(intervals[:2])
        lower = max(0, first[1] + 1 - uncertainty)
        upper = min(frame_count - 1, second[0] - 1 + uncertainty)
        return (lower, upper) if lower <= upper else None
    raise ValueError(f"unsupported temporal operator: {temporal_operator}")


def _canonicalize_candidate_evidence(
    rows: list[dict], candidates: list[dict],
) -> list[dict]:
    """Fail closed unless cited pixels also contain the fixed candidate track."""

    visible = {
        str(row["candidate_id"]): set(int(value) for value in row["all_frame_ids"])
        for row in candidates
    }
    output = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        evidence = [
            int(value) for value in row["evidence_frame_ids"]
            if int(value) in visible[candidate_id]
        ]
        if row["status"] == "SUPPORTED" and evidence:
            output.append({**row, "evidence_frame_ids": sorted(set(evidence))})
        else:
            output.append({
                "candidate_id": candidate_id,
                "status": "UNKNOWN",
                "confidence": 0.0,
                "evidence_frame_ids": [],
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--anchor-localizations", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--provider", default="parasail")
    parser.add_argument("--top-k-candidates", type=int, default=12)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--maximum-event-frames", type=int, default=24)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("atomic-claim output is immutable")

    cohort = json.loads(args.cohort.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    grounding = json.loads(args.candidate_grounding.read_text())
    anchors = json.loads(args.anchor_localizations.read_text())
    protocol = json.loads(args.protocol.read_text())
    if grounding.get("status") != (
        "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE"
    ):
        raise ValueError("atomic claims require frozen consumed-development grounding")
    if anchors.get("status") != (
        "CONSUMED_DEVELOPMENT_ANCHOR_PILOT_NOT_TRANSFER_EVIDENCE"
    ):
        raise ValueError("atomic claims require frozen consumed-development anchors")
    for artifact in (cohort, sgdet, grounding, anchors):
        if any(bool(artifact.get(key)) for key in FORBIDDEN_FLAGS):
            raise ValueError("atomic-claim input crossed its authority boundary")

    helper = Path(__file__).resolve().parent / "pilot_agqa_atomic_temporal_grounder_v2.py"
    actual_inputs = {
        "cohort_file_sha256": _sha256(args.cohort),
        "sgdet_file_sha256": _sha256(args.sgdet),
        "candidate_grounding_file_sha256": _sha256(args.candidate_grounding),
        "anchor_localizations_file_sha256": _sha256(args.anchor_localizations),
        "collector_sha256": _sha256(Path(__file__)),
        "atomic_helper_sha256": _sha256(helper),
    }
    immutable = protocol["immutable_inputs"]
    if actual_inputs != {key: immutable[key] for key in actual_inputs}:
        raise ValueError("atomic-claim immutable inputs differ")
    runtime = protocol["multicandidate_atomic_claims"]
    for key, value in (
        ("model", args.model),
        ("provider", args.provider),
        ("top_k_candidates", args.top_k_candidates),
        ("candidate_batch_size", args.candidate_batch_size),
        ("maximum_event_frames", args.maximum_event_frames),
    ):
        if runtime[key] != value:
            raise ValueError(f"atomic-claim runtime differs for {key}")

    public_paths = {
        str(row["video_id"]): Path(row["video_path"])
        for row in cohort["rows"]
    }
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    anchor_by_task = {str(row["task_id"]): row for row in anchors["rows"]}
    wanted = set(str(value) for value in (args.task_id or ()))
    source_rows = [
        row for row in grounding["rows"]
        if not wanted or str(row["task_id"]) in wanted
    ]
    if wanted - {str(row["task_id"]) for row in source_rows}:
        raise ValueError("requested task ID is absent")
    if args.max_tasks is not None:
        source_rows = source_rows[:args.max_tasks]

    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        timeout=300,
        max_retries=2,
    )
    model = {
        "id": args.model,
        "omit_temperature": False,
        "seed": int(runtime.get("seed", 0)),
        "provider": {
            "only": [args.provider],
            "allow_fallbacks": bool(runtime.get("provider_allow_fallbacks", False)),
        },
    }

    outputs = []
    provider_calls = 0
    total_cost = 0.0
    for source in source_rows:
        task_id = str(source["task_id"])
        video_id = str(source["video_id"])
        receipt = query_grounding_v2_from_dict(source["receipt"])
        raw_video = raw_by_video[video_id]
        stable = build_stable_tracks(
            raw_video,
            minimum_object_score=float(runtime["minimum_object_score"]),
        )
        predicate = str(source.get("root_predicate") or "").strip()
        role = str(source["requested_role"])
        temporal_operator = str(source["temporal_operator"])
        anchor_rows = list(
            anchor_by_task[task_id].get("anchor_localizations", ())
        )
        root_window = _root_window(
            temporal_operator,
            anchor_rows,
            frame_count=len(raw_video["sampled_original_frame_indices"]),
            uncertainty=int(runtime["temporal_uncertainty_frames"]),
        )
        candidate_groups = (
            _candidate_groups(receipt, root_window, args.top_k_candidates)
            if root_window is not None else []
        )
        anchor_evidence = [
            int(frame) for row in anchor_rows
            for frame in row["evidence_frame_ids"]
        ]
        event_rows = []
        call_receipts = []
        if predicate and root_window is not None:
            for start in range(0, len(candidate_groups), args.candidate_batch_size):
                batch = candidate_groups[start:start + args.candidate_batch_size]
                frame_ids = _event_frame_ids(
                    raw_video,
                    receipt,
                    batch,
                    root_window,
                    anchor_evidence,
                    args.maximum_event_frames,
                )
                images, seconds, scales = _exact_sgdet_frames(
                    public_paths[video_id], raw_video, frame_ids,
                )
                annotated = _annotate(
                    images,
                    frame_ids,
                    scales,
                    raw_video,
                    stable.detection_to_track,
                    stable.retained_detection_indices,
                    batch,
                )
                panels = _panels(
                    annotated,
                    seconds,
                    frames_per_panel=int(runtime["frames_per_panel"]),
                    frame_width=int(runtime["panel_frame_width"]),
                    quality=int(runtime["jpeg_quality"]),
                )
                candidate_ids = [str(row["candidate_id"]) for row in batch]
                claims = "\n".join(
                    f"{row['candidate_id']}: {_typed_claim(predicate, role, row)}"
                    for row in batch
                )
                prompt = (
                    f"Fixed atomic predicate: {predicate}\n"
                    f"Fixed requested role: {role}\n"
                    "Fixed independent propositions:\n"
                    f"{claims}\n"
                    f"Presented chronological sampled-frame IDs: {frame_ids}\n"
                    "Verify every proposition independently. Cite only a frame "
                    "where that candidate's labeled box is visible and the exact "
                    "predicate/role is directly supported. Do not select a best candidate."
                )
                response_format = _event_response_format(candidate_ids, frame_ids)
                core = {
                    "protocol": "AGQA_MULTICANDIDATE_ATOMIC_CLAIMS_V1_CONSUMED_DEV",
                    "protocol_file_sha256": _sha256(args.protocol),
                    "task_id": task_id,
                    "video_sha256": receipt.video_sha256,
                    "predicate": predicate,
                    "requested_role": role,
                    "candidate_groups": [{
                        "candidate_id": row["candidate_id"],
                        "canonical_label": row["canonical_label"],
                        "member_track_ids": row["member_track_ids"],
                    } for row in batch],
                    "presented_frame_ids": frame_ids,
                    "presented_frame_sha256s": [
                        raw_video["selected_frame_sha256s"][index]
                        for index in frame_ids
                    ],
                    "panel_sha256s": [
                        hashlib.sha256(value).hexdigest() for value in panels
                    ],
                    "model": model,
                    "system_sha256": stable_hash(SYSTEM),
                    "prompt_sha256": stable_hash(prompt),
                    "request_contract": _request_cache_contract(
                        model=model,
                        max_tokens=int(runtime["max_tokens"]),
                        response_format=response_format,
                        maximum_attempts=int(runtime["maximum_contract_attempts"]),
                    ),
                }
                error = None
                try:
                    payload, usage, reused = _cached_provider_call(
                        cache_dir=args.cache_dir,
                        call_name=f"atomic_claim_{task_id}_{start // args.candidate_batch_size}",
                        input_core=core,
                        invoke=lambda candidate_ids=candidate_ids, frame_ids=frame_ids, panels=panels, prompt=prompt: _provider_call_with_contract_retries(
                            client,
                            model=model,
                            system=SYSTEM,
                            content=[{"type": "text", "text": prompt}]
                            + _panel_content(panels),
                            max_tokens=int(runtime["max_tokens"]),
                            response_format=response_format,
                            maximum_attempts=int(runtime["maximum_contract_attempts"]),
                            validator=lambda value: _validate_rows(
                                value,
                                "events",
                                "candidate_id",
                                candidate_ids,
                                frame_ids,
                            ),
                        ),
                    )
                    verified = _validate_rows(
                        payload,
                        "events",
                        "candidate_id",
                        candidate_ids,
                        frame_ids,
                    )
                    event_rows.extend(
                        _canonicalize_candidate_evidence(verified, batch)
                    )
                except Exception as exc:
                    usage, error = _provider_failure(exc)
                    reused = False
                    event_rows.extend({
                        "candidate_id": candidate_id,
                        "status": "UNKNOWN",
                        "confidence": 0.0,
                        "evidence_frame_ids": [],
                    } for candidate_id in candidate_ids)
                provider_calls += int(not reused) * int(
                    usage.get("provider_attempts", 0)
                )
                total_cost += float(usage.get("reported_cost_usd", 0.0))
                call_receipts.append({
                    "kind": "MULTICANDIDATE_ATOMIC_CLAIM",
                    "usage": usage,
                    "cache_reused": reused,
                    "provider_error": error,
                    **core,
                })

        decision = (
            _execute_temporal(
                temporal_operator,
                anchor_rows,
                event_rows,
                candidate_groups,
            )
            if predicate and root_window is not None else {
                "status": (
                    "ABSTAIN_NO_ROOT_PREDICATE"
                    if not predicate else "ABSTAIN_ANCHOR_UNGROUNDED"
                ),
                "selected_candidate_id": None,
                "ranking": [],
            }
        )
        outputs.append({
            "task_id": task_id,
            "video_id": video_id,
            "root_predicate": predicate,
            "requested_role": role,
            "temporal_operator": temporal_operator,
            "root_temporal_window": (
                list(root_window) if root_window is not None else None
            ),
            "anchor_localizations": anchor_rows,
            "candidate_groups": candidate_groups,
            "atomic_event_localizations": event_rows,
            "decision": decision,
            "call_receipts": call_receipts,
        })
        print(json.dumps({
            "task_id": task_id,
            "status": decision["status"],
            "selected_label": decision.get("selected_label"),
            "calls_running": provider_calls,
            "cost_usd_running": total_cost,
        }), flush=True)

    report = {
        "schema_version": "agqa-multicandidate-atomic-claims-v1",
        "status": "CONSUMED_DEVELOPMENT_ATOMIC_CLAIMS_NOT_TRANSFER_EVIDENCE",
        "protocol_file_sha256": _sha256(args.protocol),
        "model": model,
        "rows": outputs,
        "provider_calls": provider_calls,
        "reported_cost_usd": total_cost,
        "authority": {
            "question_text_supplied_to_vlm": False,
            "answer_read": False,
            "answer_alternatives_read": False,
            "temporal_operator_supplied_to_vlm": False,
            "official_scene_graph_read": False,
            "official_stsg_read": False,
            "functional_program_read": False,
            "source_controller_read": False,
            "target_outcome_read": False,
            "proposal_may_authorize_harness": False,
        },
        "immutable_input_file_sha256s": actual_inputs,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
