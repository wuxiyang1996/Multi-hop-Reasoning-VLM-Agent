#!/usr/bin/env python3
"""Collect answer-blind typed object bindings inside frozen AGQA scopes.

This is a target-native perception stage, not an answer evaluator.  A public
question has already been parsed into a predicate, typed role, temporal
operator, and answer-blind anchor intervals.  The collector shows an
off-the-shelf VLM only raw pixels from the resulting scope and a label-unique
inventory derived from the frozen public detector.  It may bind one inventory
ID to the requested role or abstain; it never receives an AGQA answer,
alternative answers, functional program, official STSG, source controller, or
target outcome.

The emitted binding is intentionally provisional.  A separate single-
candidate pixel verifier must confirm it before it can become a shared
grounding receipt for any experimental arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI

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
from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _exact_sgdet_frames,
)


SYSTEM = """You are a frozen target-native video grounding proposal tool, not an answer evaluator or source-domain agent. The input specifies one public typed person-object relation and a symbolic temporal scope. Inspect only the chronological raw frames. When any inventory object is visibly consistent with the requested relation, return the single best-supported detector-inventory ID as a provisional typed binding; an independent later verifier decides whether it is safe to use. Return UNKNOWN only when no inventory object is visibly consistent. Inventory labels are fallible perception proposals, not AGQA answer options. Never use a gold answer, official scene graph, functional program, source controller, target outcome, or dataset prior. Return only the required JSON schema."""


FORCED_PROPOSAL_SYSTEM = """You are the recall stage of a frozen target-native video grounding tool, not an answer evaluator, safety authorizer, or source-domain agent. The input specifies one public typed person-object relation, a symbolic temporal scope, and a nonempty detector inventory. Inspect only the chronological raw frames and always propose the single most plausible inventory ID for the requested typed role, even when evidence is weak. This proposal can never authorize an experimental action: a separate independent raw-pixel verifier is solely responsible for acceptance or rejection. Inventory labels are fallible perception proposals, not AGQA answer options. Never use a gold answer, official scene graph, functional program, source controller, target outcome, or dataset prior. Return only the required JSON schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scope(row: dict, *, frame_count: int, uncertainty: int) -> tuple[int, int] | None:
    """Execute a frozen temporal operator without any visual/outcome tuning."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    operator = str(row.get("temporal_operator") or "").upper()
    intervals = [tuple(int(value) for value in item) for item in row.get("anchor_intervals", ())]
    if operator == "VIDEO":
        return 0, frame_count - 1
    if operator in {"BEFORE", "AFTER", "WHILE"} and len(intervals) < 1:
        return None
    if operator == "BETWEEN" and len(intervals) < 2:
        return None
    if operator == "BEFORE":
        upper = intervals[0][0] - 1
        return (0, upper) if upper >= 0 else None
    if operator == "AFTER":
        lower = intervals[0][1] + 1
        return (lower, frame_count - 1) if lower < frame_count else None
    if operator == "WHILE":
        lower = max(0, intervals[0][0] - uncertainty)
        upper = min(frame_count - 1, intervals[0][1] + uncertainty)
        return (lower, upper) if lower <= upper else None
    if operator == "BETWEEN":
        left, right = sorted(intervals[:2])
        lower = max(0, left[1] + 1 - uncertainty)
        upper = min(frame_count - 1, right[0] - 1 + uncertainty)
        return (lower, upper) if lower <= upper else None
    raise ValueError(f"unsupported temporal operator: {operator}")


def _uniform_frame_ids(lower: int, upper: int, maximum: int) -> list[int]:
    if not 0 <= lower <= upper or maximum <= 0:
        raise ValueError("invalid scope or frame budget")
    size = upper - lower + 1
    if size <= maximum:
        return list(range(lower, upper + 1))
    output = []
    for index in range(maximum):
        value = round(lower + index * (upper - lower) / (maximum - 1))
        if value not in output:
            output.append(value)
    return output


def _label_inventory(stable, *, lower: int, upper: int) -> list[dict]:
    """Deduplicate fragmented detector tracks by canonical public label."""

    grouped: dict[str, dict] = {}
    for track in stable.tracks:
        label = str(track.canonical_label).strip().casefold()
        if label == "person":
            continue
        evidence = sorted(
            int(frame) for frame in track.evidence_frames
            if lower <= int(frame) <= upper
        )
        if not evidence:
            continue
        row = grouped.setdefault(label, {
            "canonical_label": label,
            "track_ids": [],
            "visible_frame_ids": [],
            "detector_confidence": 0.0,
        })
        row["track_ids"].append(str(track.track_id))
        row["visible_frame_ids"].extend(evidence)
        row["detector_confidence"] = max(
            float(row["detector_confidence"]), float(track.confidence),
        )
    output = []
    for label in sorted(grouped):
        row = grouped[label]
        output.append({
            "candidate_id": f"O{len(output)}",
            "canonical_label": label,
            "track_ids": sorted(set(row["track_ids"])),
            "visible_frame_ids": sorted(set(row["visible_frame_ids"])),
            "detector_confidence": float(row["detector_confidence"]),
        })
    return output


def _response_format(
    candidate_ids: list[str], frame_ids: list[int], *, force_proposal: bool = False,
) -> dict:
    statuses = ["PROPOSED"] if force_proposal else ["SUPPORTED", "UNKNOWN"]
    selections = candidate_ids if force_proposal else candidate_ids + ["UNKNOWN"]
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_query_conditioned_typed_binding_v3",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": statuses},
                "selected_candidate_id": {
                    "type": "string", "enum": selections,
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_frame_ids": {
                    "type": "array", "maxItems": 4,
                    "items": {"type": "integer", "enum": frame_ids},
                },
            },
            "required": [
                "status", "selected_candidate_id", "confidence",
                "evidence_frame_ids",
            ],
        },
    }}


def _validate(
    payload: dict, *, candidate_ids: set[str], frame_ids: set[int],
    supportable_by_candidate: dict[str, set[int]], force_proposal: bool = False,
) -> dict:
    if set(payload) != {
        "status", "selected_candidate_id", "confidence", "evidence_frame_ids",
    }:
        raise ValueError("typed-binding payload contains unexpected fields")
    status = str(payload["status"])
    selected = str(payload["selected_candidate_id"])
    confidence = float(payload["confidence"])
    evidence = sorted(set(int(value) for value in payload["evidence_frame_ids"]))
    allowed_statuses = {"PROPOSED"} if force_proposal else {"SUPPORTED", "UNKNOWN"}
    if status not in allowed_statuses or not 0 <= confidence <= 1:
        raise ValueError("typed-binding status/confidence is invalid")
    if len(evidence) > 4 or any(frame not in frame_ids for frame in evidence):
        raise ValueError("typed binding cites an unpresented frame")
    if status in {"SUPPORTED", "PROPOSED"}:
        if selected not in candidate_ids or not evidence:
            raise ValueError("supported typed binding needs one inventory ID and evidence")
        if status == "PROPOSED":
            # This stage has recall authority only.  Requiring the frozen
            # detector to fire on the VLM's exact evidence frame discards
            # visually valid proposals when a fragmented track misses that
            # frame.  Preserve the proposal and expose corroboration
            # separately; only the independent verifier may upgrade it to a
            # shared SUPPORTED receipt.
            return {
                "status": status, "selected_candidate_id": selected,
                "confidence": confidence, "evidence_frame_ids": evidence,
            }
        if any(frame not in supportable_by_candidate[selected] for frame in evidence):
            # The detector cannot corroborate the named object on the cited
            # pixel.  Information-decreasing fail-closed canonicalization is
            # safe; it never upgrades an unsupported binding.
            return {
                "status": "UNKNOWN", "selected_candidate_id": "UNKNOWN",
                "confidence": 0.0, "evidence_frame_ids": [],
            }
        return {
            "status": status, "selected_candidate_id": selected,
            "confidence": confidence, "evidence_frame_ids": evidence,
        }
    return {
        "status": "UNKNOWN", "selected_candidate_id": "UNKNOWN",
        "confidence": 0.0, "evidence_frame_ids": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
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
    parser.add_argument("--maximum-frames", type=int, default=24)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("query-conditioned typed-binding output is immutable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    cohort = json.loads(args.cohort.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    grounding = json.loads(args.query_grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    expected_statuses = {
        "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE",
        "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME",
    }
    if grounding.get("status") not in expected_statuses:
        raise ValueError("query grounding is not a recognized frozen input")
    forbidden = (
        "answer_read", "official_scene_graph_read", "official_stsg_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
    )
    if any(grounding.get(key) for key in forbidden) or any(
        sgdet.get(key) for key in forbidden
    ):
        raise ValueError("typed-binding input crossed its authority boundary")
    immutable = protocol["immutable_inputs"]
    actual = {
        "cohort_file_sha256": _sha256(args.cohort),
        "sgdet_file_sha256": _sha256(args.sgdet),
        "query_grounding_file_sha256": _sha256(args.query_grounding),
        "collector_sha256": _sha256(Path(__file__)),
    }
    if actual != {key: immutable[key] for key in actual}:
        raise ValueError("typed-binding immutable input hashes differ")
    runtime = protocol["query_conditioned_typed_binding"]
    if (
        args.model != runtime["model"]
        or args.provider != runtime["provider"]
        or args.maximum_frames != int(runtime["maximum_frames"])
    ):
        raise ValueError("typed-binding runtime differs from protocol")

    public = {str(row["task_id"]): row for row in cohort["rows"]}
    video_paths = {
        str(row["video_id"]): Path(row["video_path"])
        for row in cohort["video_selections"]
    }
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    sources = [
        row for index, row in enumerate(grounding["rows"])
        if index % args.shard_count == args.shard_index
    ]
    if args.max_tasks is not None:
        sources = sources[:args.max_tasks]
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(
        api_key=key, base_url="https://openrouter.ai/api/v1",
        timeout=300, max_retries=2,
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
    uncertainty = int(runtime["temporal_uncertainty_frames"])
    for row in sources:
        task_id = str(row["task_id"])
        video_id = str(row["video_id"])
        public_row = public[task_id]
        raw = raw_by_video[video_id]
        frame_count = int(raw["model_visible_frame_count"])
        scope = _scope(row, frame_count=frame_count, uncertainty=uncertainty)
        stable = build_stable_tracks(
            raw, minimum_object_score=float(runtime["minimum_object_score"]),
        )
        if scope is None:
            outputs.append({
                "task_id": task_id, "video_id": video_id,
                "status": "ABSTAIN_NO_TEMPORAL_SCOPE",
                "selected_candidate_id": "UNKNOWN", "selected_candidate": None,
                "confidence": 0.0, "evidence_frame_ids": [],
                "scope": None, "candidates": [], "presented_frame_ids": [],
                "presented_frame_sha256s": [], "panel_sha256s": [],
                "usage": None, "cache_reused": True, "provider_error": None,
            })
            continue
        lower, upper = scope
        candidates = _label_inventory(stable, lower=lower, upper=upper)
        if not candidates:
            outputs.append({
                "task_id": task_id, "video_id": video_id,
                "status": "ABSTAIN_NO_VISIBLE_OBJECT_INVENTORY",
                "selected_candidate_id": "UNKNOWN", "selected_candidate": None,
                "confidence": 0.0, "evidence_frame_ids": [],
                "scope": [lower, upper], "candidates": [],
                "presented_frame_ids": [], "presented_frame_sha256s": [],
                "panel_sha256s": [], "usage": None, "cache_reused": True,
                "provider_error": None,
            })
            continue
        frame_ids = _uniform_frame_ids(lower, upper, args.maximum_frames)
        images, seconds, _ = _exact_sgdet_frames(
            video_paths[video_id], raw, frame_ids,
        )
        panels = _panels(
            images, seconds,
            frames_per_panel=int(runtime["frames_per_panel"]),
            frame_width=int(runtime["panel_frame_width"]),
            quality=int(runtime["jpeg_quality"]),
        )
        table = "; ".join(
            f"{candidate['candidate_id']}={candidate['canonical_label']}"
            for candidate in candidates
        )
        force_proposal = bool(runtime.get("force_proposal", False))
        instruction = (
            "Propose the single most plausible inventory ID visibly filling the "
            "requested typed role inside this scope, even if the evidence is weak. "
            "This is a recall-only proposal; a separate verifier will decide whether "
            "it is safe. Do not output an object name."
            if force_proposal else
            "Propose the single best-supported inventory ID visibly filling the "
            "requested typed role inside this scope. A separate verifier will reject "
            "unsafe proposals. Return UNKNOWN only if no listed object is visibly "
            "consistent with the relation. Do not output an object name."
        )
        prompt = (
            f"Public question: {public_row['question']}\n"
            f"Requested typed predicate: {row['root_predicate']}\n"
            f"Requested typed role: {row['requested_role']}\n"
            f"Frozen temporal operator: {row['temporal_operator']}\n"
            f"Already-executed symbolic scope: S{lower}..S{upper}\n"
            f"Presented exact chronological frame IDs: {frame_ids}\n"
            f"Public detector label inventory: {table}\n"
            f"{instruction}"
        )
        panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
        frame_hashes = [
            str(raw["selected_frame_sha256s"][index]) for index in frame_ids
        ]
        candidate_ids = [row["candidate_id"] for row in candidates]
        visible_person = {
            int(frame) for track in stable.tracks
            if track.canonical_label == "person"
            for frame in track.evidence_frames
        }
        supportable = {
            candidate["candidate_id"]: (
                set(candidate["visible_frame_ids"]) & visible_person & set(frame_ids)
            )
            for candidate in candidates
        }
        response_format = _response_format(
            candidate_ids, frame_ids, force_proposal=force_proposal,
        )
        core = {
            "protocol": "AGQA_QUERY_CONDITIONED_TYPED_BINDING_V3",
            "protocol_file_sha256": _sha256(args.protocol),
            "task_id": task_id,
            "question_sha256": str(public_row["question_sha256"]),
            "query_grounding_report_sha256": grounding["report_sha256"],
            "typed_query": {
                "predicate": row["root_predicate"],
                "requested_role": row["requested_role"],
                "temporal_operator": row["temporal_operator"],
                "scope": [lower, upper],
            },
            "candidate_inventory_sha256": stable_hash(candidates),
            "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes,
            "model": model,
            "system_sha256": stable_hash(
                FORCED_PROPOSAL_SYSTEM if force_proposal else SYSTEM
            ),
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
                call_name=f"typed_binding_{task_id}", input_core=core,
                invoke=lambda: _provider_call_with_contract_retries(
                    client, model=model,
                    system=FORCED_PROPOSAL_SYSTEM if force_proposal else SYSTEM,
                    content=[{"type": "text", "text": prompt}]
                    + _panel_content(panels),
                    max_tokens=int(runtime["max_tokens"]),
                    response_format=response_format,
                    maximum_attempts=int(runtime["maximum_contract_attempts"]),
                    validator=lambda payload: _validate(
                        payload, candidate_ids=set(candidate_ids),
                        frame_ids=set(frame_ids),
                        supportable_by_candidate=supportable,
                        force_proposal=force_proposal,
                    ),
                ),
            )
            decision = _validate(
                payload, candidate_ids=set(candidate_ids),
                frame_ids=set(frame_ids), supportable_by_candidate=supportable,
                force_proposal=force_proposal,
            )
        except Exception as exc:
            usage = getattr(exc, "usage", {
                "reported_cost_usd": 0.0, "provider_attempts": 0,
                "contract_retry_count": 0, "attempt_receipts": [],
            })
            reused = False
            provider_error = f"{type(exc).__name__}:{exc}"
            decision = {
                "status": "UNKNOWN", "selected_candidate_id": "UNKNOWN",
                "confidence": 0.0, "evidence_frame_ids": [],
            }
        selected = next((
            candidate for candidate in candidates
            if candidate["candidate_id"] == decision["selected_candidate_id"]
        ), None)
        detector_frame_corroborated = (
            bool(selected)
            and bool(decision["evidence_frame_ids"])
            and all(
                frame in supportable[decision["selected_candidate_id"]]
                for frame in decision["evidence_frame_ids"]
            )
        )
        provider_calls += int(not reused) * int(usage.get("provider_attempts", 1))
        total_cost += float(usage.get("reported_cost_usd", 0.0))
        outputs.append({
            "task_id": task_id, "video_id": video_id,
            **decision, "selected_candidate": selected,
            "scope": [lower, upper], "candidates": candidates,
            "root_predicate": row["root_predicate"],
            "requested_role": row["requested_role"],
            "temporal_operator": row["temporal_operator"],
            "anchor_intervals": row.get("anchor_intervals", []),
            "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes,
            "proposal_detector_frame_corroborated": detector_frame_corroborated,
            "usage": usage, "cache_reused": reused,
            "provider_error": provider_error,
        })
        print(json.dumps({
            "task_id": task_id, "status": decision["status"],
            "candidate": selected["canonical_label"] if selected else None,
            "confidence": decision["confidence"],
            "cost_usd_running": total_cost,
        }), flush=True)

    report = {
        "schema_version": "agqa-query-conditioned-typed-binding-shard-v3",
        "status": "CONSUMED_DEVELOPMENT_TYPED_BINDING_NOT_TRANSFER_EVIDENCE",
        "protocol_file_sha256": _sha256(args.protocol),
        "cohort_sha256": cohort["cohort_sha256"],
        "query_grounding_report_sha256": grounding["report_sha256"],
        "model": model, "maximum_frames": args.maximum_frames,
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "rows": outputs, "provider_calls": provider_calls,
        "reported_cost_usd": total_cost,
        "public_question_read": True,
        "answer_read": False, "official_scene_graph_read": False,
        "official_stsg_read": False, "functional_program_read": False,
        "source_controller_read": False, "target_outcome_read": False,
        "agqa_answer_alternatives_read": False,
        "output_is_provisional_until_independent_binary_verification": True,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
