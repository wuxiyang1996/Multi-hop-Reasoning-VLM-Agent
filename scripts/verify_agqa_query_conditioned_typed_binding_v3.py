#!/usr/bin/env python3
"""Independently verify one provisional AGQA typed binding from raw pixels.

The verifier receives the public question and exactly one frozen candidate
label plus a typed predicate, role, and already-resolved temporal scope.  The
question is necessary to distinguish an anchor object from the queried role;
the verifier never sees alternative candidates, an AGQA answer, official
STSG/program, source controller, or target outcome.  Only a SUPPORTED result
from this stage can later become a shared five-arm grounding receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
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
from scripts.collect_agqa_query_conditioned_typed_binding_v3 import (
    _uniform_frame_ids,
)
from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _exact_sgdet_frames,
)


SYSTEM = """You are an independent frozen answer-blind raw-video query-binding verifier, not an answer generator, candidate selector, or source-domain agent. You receive one public question, exactly one provisional public-ontology candidate, and an already-resolved typed temporal scope. Inspect only the chronological raw frames and overlays. Decide whether that candidate visibly fills the role asked by the question. Do not confuse an object mentioned in an anchor clause with the queried object unless pixels establish that it fills both roles. Candidate labels and boxes are fallible perception proposals, not answers or facts. Never choose or name an alternative object, emit a free-form answer, execute a symbolic program, use source knowledge, or consult an official annotation. Absence from sampled pixels is UNKNOWN, not REFUTED. Return only the required JSON schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _spread(values: list[int], maximum: int) -> list[int]:
    ordered = sorted(set(values))
    if maximum <= 0 or not ordered:
        return []
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return [ordered[len(ordered) // 2]]
    return sorted(set(
        ordered[round(index * (len(ordered) - 1) / (maximum - 1))]
        for index in range(maximum)
    ))


def _selected_frame_ids(
    proposal: dict, *, person_visible: set[int], candidate_visible: set[int],
    maximum: int,
) -> list[int]:
    """Prioritize proposal evidence and spread co-visible frames in scope."""

    lower, upper = (int(value) for value in proposal["scope"])
    if maximum <= 0 or lower > upper:
        raise ValueError("invalid verifier frame budget or temporal scope")
    allowed = set(range(lower, upper + 1))
    proposed = [
        int(value) for value in proposal.get("evidence_frame_ids", ())
        if int(value) in allowed
    ]
    co_visible = sorted(person_visible & candidate_visible & allowed)
    candidate_only = sorted(candidate_visible & allowed)
    uniform = _uniform_frame_ids(lower, upper, min(8, maximum))
    priority = (
        _spread(proposed, min(4, maximum))
        + _spread(co_visible, min(8, maximum))
        + _spread(candidate_only, min(6, maximum))
        + uniform
    )
    output = []
    for frame in priority:
        if frame not in output:
            output.append(frame)
        if len(output) == maximum:
            break
    return sorted(output)


def _response_format(frame_ids: list[int]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_typed_binding_independent_verification_v3",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["SUPPORTED", "REFUTED", "UNKNOWN"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_frame_ids": {
                    "type": "array", "maxItems": 4,
                    "items": {"type": "integer", "enum": frame_ids},
                },
            },
            "required": ["status", "confidence", "evidence_frame_ids"],
        },
    }}


def _validate(
    payload: dict, *, frame_ids: set[int], supportable_frame_ids: set[int],
) -> dict:
    if set(payload) != {"status", "confidence", "evidence_frame_ids"}:
        raise ValueError("typed-binding verifier payload has unexpected fields")
    status = str(payload["status"])
    confidence = float(payload["confidence"])
    evidence = sorted(set(int(value) for value in payload["evidence_frame_ids"]))
    if status not in {"SUPPORTED", "REFUTED", "UNKNOWN"}:
        raise ValueError("typed-binding verifier status is invalid")
    if not 0 <= confidence <= 1:
        raise ValueError("typed-binding verifier confidence is invalid")
    if len(evidence) > 4 or any(value not in frame_ids for value in evidence):
        raise ValueError("typed-binding verifier cites an unpresented frame")
    if status in {"SUPPORTED", "REFUTED"} and not evidence:
        raise ValueError("non-UNKNOWN verification requires pixel evidence")
    if status == "SUPPORTED":
        # Keep only citations where both tracks are bound.  A fragmented
        # detector may miss one of several otherwise valid citations; that
        # must not erase the remaining positive pixel evidence.
        evidence = [
            value for value in evidence if value in supportable_frame_ids
        ]
        if not evidence:
            return {
                "status": "UNKNOWN", "confidence": 0.0,
                "evidence_frame_ids": [],
            }
    if status == "UNKNOWN":
        evidence = []
    return {
        "status": status, "confidence": confidence,
        "evidence_frame_ids": evidence,
    }


def _annotate(
    images, *, frame_ids: list[int], scales: list[float], raw: dict,
    stable, person_track_ids: set[str], candidate_track_ids: set[str],
):
    output = [image.copy() for image in images]
    by_frame: dict[int, list[tuple[str, dict]]] = {}
    for detection in raw["objects"]:
        index = int(detection["detection_index"])
        track_id = stable.detection_to_track.get(index)
        if track_id in person_track_ids:
            by_frame.setdefault(int(detection["sampled_frame_index"]), []).append(
                ("P0", detection)
            )
        elif track_id in candidate_track_ids:
            by_frame.setdefault(int(detection["sampled_frame_index"]), []).append(
                ("C0", detection)
            )
    for image, frame_id in zip(output, frame_ids):
        draw = ImageDraw.Draw(image)
        draw.text(
            (8, 8), f"S{frame_id}", fill="white",
            stroke_width=3, stroke_fill="black",
        )
        for marker, detection in by_frame.get(frame_id, ()):
            scale = float(scales[frame_id])
            box = tuple(float(value) / scale for value in detection["bbox_xyxy"])
            color = "lime" if marker == "P0" else "magenta"
            draw.rectangle(box, outline=color, width=5)
            draw.text(
                (box[0] + 2, box[1] + 2), marker, fill=color,
                stroke_width=2, stroke_fill="black",
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--provider", default="parasail")
    parser.add_argument("--maximum-frames", type=int, default=12)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("typed-binding verification output is immutable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    cohort = json.loads(args.cohort.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    proposals = json.loads(args.proposals.read_text())
    protocol = json.loads(args.protocol.read_text())
    if proposals.get("status") != (
        "CONSUMED_DEVELOPMENT_TYPED_BINDING_NOT_TRANSFER_EVIDENCE"
    ):
        raise ValueError("proposals are not frozen consumed-development receipts")
    if not proposals.get(
        "output_is_provisional_until_independent_binary_verification"
    ):
        raise ValueError("proposal artifact does not require independent verification")
    forbidden = (
        "answer_read", "official_scene_graph_read", "official_stsg_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
        "agqa_answer_alternatives_read",
    )
    if any(proposals.get(key) for key in forbidden):
        raise ValueError("proposal artifact crossed its authority boundary")
    actual = {
        "cohort_file_sha256": _sha256(args.cohort),
        "sgdet_file_sha256": _sha256(args.sgdet),
        "proposal_file_sha256": _sha256(args.proposals),
        "verifier_sha256": _sha256(Path(__file__)),
    }
    if actual != {
        key: protocol["immutable_inputs"][key] for key in actual
    }:
        raise ValueError("typed-binding verifier immutable inputs differ")
    runtime = protocol["independent_binary_verifier"]
    if (
        args.model != runtime["model"]
        or args.provider != runtime["provider"]
        or args.maximum_frames != int(runtime["maximum_frames"])
    ):
        raise ValueError("typed-binding verifier runtime differs from protocol")

    paths = {
        str(row["video_id"]): Path(row["video_path"])
        for row in cohort["video_selections"]
    }
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    public_by_task = {str(row["task_id"]): row for row in cohort["rows"]}
    rows = [
        row for index, row in enumerate(proposals["rows"])
        if index % args.shard_count == args.shard_index
    ]
    if args.max_tasks is not None:
        rows = rows[:args.max_tasks]
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
    for proposal in rows:
        task_id = str(proposal["task_id"])
        video_id = str(proposal["video_id"])
        public_row = public_by_task[task_id]
        candidate = proposal.get("selected_candidate")
        if proposal.get("status") != "PROPOSED" or not candidate:
            outputs.append({
                "task_id": task_id, "video_id": video_id,
                "status": "ABSTAIN_NO_PROPOSAL", "confidence": 0.0,
                "evidence_frame_ids": [], "candidate_id": None,
                "candidate_label": None, "presented_frame_ids": [],
                "presented_frame_sha256s": [], "panel_sha256s": [],
                "supportable_evidence_frame_ids": [], "usage": None,
                "cache_reused": True, "provider_error": None,
            })
            continue
        raw = raw_by_video[video_id]
        stable = build_stable_tracks(
            raw, minimum_object_score=float(runtime["minimum_object_score"]),
        )
        person_ids = {
            str(track.track_id) for track in stable.tracks
            if track.canonical_label == "person"
        }
        candidate_ids = set(str(value) for value in candidate["track_ids"])
        person_visible = {
            int(frame) for track in stable.tracks
            if str(track.track_id) in person_ids for frame in track.evidence_frames
        }
        candidate_visible = {
            int(frame) for track in stable.tracks
            if str(track.track_id) in candidate_ids for frame in track.evidence_frames
        }
        frame_ids = _selected_frame_ids(
            proposal, person_visible=person_visible,
            candidate_visible=candidate_visible, maximum=args.maximum_frames,
        )
        supportable = person_visible & candidate_visible & set(frame_ids)
        images, seconds, scales = _exact_sgdet_frames(
            paths[video_id], raw, frame_ids,
        )
        annotated = _annotate(
            images, frame_ids=frame_ids, scales=scales, raw=raw, stable=stable,
            person_track_ids=person_ids, candidate_track_ids=candidate_ids,
        )
        panels = _panels(
            annotated, seconds,
            frames_per_panel=int(runtime["frames_per_panel"]),
            frame_width=int(runtime["panel_frame_width"]),
            quality=int(runtime["jpeg_quality"]),
        )
        frame_hashes = [
            str(raw["selected_frame_sha256s"][index]) for index in frame_ids
        ]
        panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
        prompt = (
            f"Public question: {public_row['question']}\n"
            f"Typed predicate: {proposal['root_predicate']}\n"
            f"Requested typed role: {proposal['requested_role']}\n"
            f"Single provisional candidate C0: {candidate['canonical_label']}\n"
            f"Person overlay: P0\n"
            f"Frozen temporal operator: {proposal['temporal_operator']}\n"
            f"Already-resolved temporal scope: {proposal['scope']}\n"
            f"Presented exact frame IDs: {frame_ids}\n"
            "Verify only whether candidate C0 visibly fills the object role being "
            "asked by the question in this scope. An object named in an anchor clause "
            "is not automatically the queried object. Do not select or name another object."
        )
        response_format = _response_format(frame_ids)
        core = {
            "protocol": "AGQA_QUERY_CONDITIONED_TYPED_BINDING_V3_BINARY_VERIFY",
            "protocol_file_sha256": _sha256(args.protocol),
            "proposal_report_sha256": proposals["report_sha256"],
            "task_id": task_id,
            "question_sha256": public_row["question_sha256"],
            "candidate_id": proposal["selected_candidate_id"],
            "candidate_inventory_sha256": stable_hash(proposal["candidates"]),
            "candidate_label_sha256": stable_hash(candidate["canonical_label"]),
            "candidate_track_ids": sorted(candidate_ids),
            "predicate": proposal["root_predicate"],
            "requested_role": proposal["requested_role"],
            "temporal_operator": proposal["temporal_operator"],
            "scope": proposal["scope"],
            "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes,
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
                call_name=f"typed_binding_verify_{task_id}", input_core=core,
                invoke=lambda: _provider_call_with_contract_retries(
                    client, model=model, system=SYSTEM,
                    content=[{"type": "text", "text": prompt}]
                    + _panel_content(panels),
                    max_tokens=int(runtime["max_tokens"]),
                    response_format=response_format,
                    maximum_attempts=int(runtime["maximum_contract_attempts"]),
                    validator=lambda value: _validate(
                        value, frame_ids=set(frame_ids),
                        supportable_frame_ids=supportable,
                    ),
                ),
            )
            decision = _validate(
                payload, frame_ids=set(frame_ids),
                supportable_frame_ids=supportable,
            )
        except Exception as exc:
            usage = getattr(exc, "usage", {
                "reported_cost_usd": 0.0, "provider_attempts": 0,
                "contract_retry_count": 0, "attempt_receipts": [],
            })
            reused = False
            provider_error = f"{type(exc).__name__}:{exc}"
            decision = {
                "status": "UNKNOWN", "confidence": 0.0,
                "evidence_frame_ids": [],
            }
        cost += float(usage.get("reported_cost_usd", 0.0))
        calls += int(not reused) * int(usage.get("provider_attempts", 1))
        outputs.append({
            "task_id": task_id, "video_id": video_id, **decision,
            "candidate_id": proposal["selected_candidate_id"],
            "candidate_label": candidate["canonical_label"],
            "candidate_track_ids": sorted(candidate_ids),
            "proposal_confidence": proposal["confidence"],
            "proposal_report_sha256": proposals["report_sha256"],
            "supportable_evidence_frame_ids": sorted(supportable),
            "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes, "usage": usage,
            "cache_reused": reused, "provider_error": provider_error,
        })
        print(json.dumps({
            "task_id": task_id, "status": decision["status"],
            "candidate": candidate["canonical_label"],
            "confidence": decision["confidence"],
            "cost_usd_running": cost,
        }), flush=True)

    report = {
        "schema_version": "agqa-query-conditioned-typed-binding-verification-v3",
        "status": "CONSUMED_DEVELOPMENT_TYPED_BINDING_VERIFICATION_NOT_TRANSFER_EVIDENCE",
        "protocol_file_sha256": _sha256(args.protocol),
        "cohort_sha256": cohort["cohort_sha256"],
        "proposal_report_sha256": proposals["report_sha256"],
        "model": model, "maximum_frames": args.maximum_frames,
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "rows": outputs, "provider_calls": calls, "reported_cost_usd": cost,
        "question_text_read": True, "answer_read": False,
        "agqa_answer_alternatives_read": False,
        "official_scene_graph_read": False, "official_stsg_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False,
        "alternative_candidate_selection_allowed": False,
        "candidate_label_emitted_as_answer": False,
        "supported_requires_same_frame_person_and_candidate_track_evidence": True,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
