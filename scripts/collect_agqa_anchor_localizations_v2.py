#!/usr/bin/env python3
"""Collect answer-blind action-anchor intervals for AGQA temporal queries.

The VLM sees only raw content-addressed frames and parser-derived action
phrases.  It never receives the task question, root query predicate, temporal
operator, candidates, answer, official STSG, functional program, game source
controller, or target outcome.  The resulting intervals are consumed by a
deterministic typed temporal executor over a separately acquired
question-blind event graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

from openai import OpenAI

from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call,
    _panel_content,
    _panels,
)
from scripts.collect_agqa_question_blind_typed_event_inventory_v1 import (
    _annotate_frames,
    _detections_by_frame,
    _provider_call_with_contract_retries,
    _request_cache_contract,
)
from scripts.pilot_agqa_atomic_temporal_grounder_v2 import (
    _anchor_response_format,
    _provider_failure,
    _validate_rows,
)
from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _exact_sgdet_frames,
)


FORBIDDEN_INPUT_FLAGS = (
    "answer_read", "official_scene_graph_read", "functional_program_read",
    "source_controller_read", "target_outcome_read",
    "per_video_action_genome_annotation_read",
)


ANCHOR_V2_SYSTEM = """You are an answer-blind video interval localizer. Locate each supplied action or state phrase independently in chronological raw frames. The requested output is the occurrence interval, not a frame showing only the persistent result of an earlier action. For a transition such as opening, closing, taking, putting down, standing up, or turning on/off, cite chronological frames that jointly show the visible change from its start through its completion; a single late frame in the resulting state is insufficient. For a sustained action or relation, cite the earliest and latest directly visible supporting frames. Use UNKNOWN when the shown pixels do not distinguish the occurrence interval. Do not answer a question, compare anchors, apply a temporal operator, infer an object answer, or use information outside the shown pixels. Return only the required JSON schema."""


def _uniform(values: list[int] | range, maximum: int) -> list[int]:
    ordered = sorted(set(int(value) for value in values))
    if maximum <= 0 or not ordered:
        return []
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return [ordered[len(ordered) // 2]]
    indices = [
        round(index * (len(ordered) - 1) / (maximum - 1))
        for index in range(maximum)
    ]
    return [ordered[index] for index in dict.fromkeys(indices)]


def _anchor_specs_v2(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize both scored and parser-only anchor obligations.

    Some valid public parser obligations have no action-class score view.  That
    absence is not an error and must fall back to uniform raw-frame inspection,
    rather than crashing or consulting an outcome-bearing annotation.
    """

    return [{
        "anchor_id": f"A{index}",
        "phrase": str(row["phrase"]),
        "native_frame_index_view": [
            int(value) for value in row.get("native_frame_index_view", ())
        ],
    } for index, row in enumerate(plan.get("action_obligations", ()))]


def _anchor_frame_ids_v2(
    raw_video: dict[str, Any], anchors: list[dict[str, Any]], maximum: int,
) -> list[int]:
    sampled = [int(value) for value in raw_video["sampled_original_frame_indices"]]
    if not sampled:
        return []
    priority = []
    for anchor in anchors:
        mapped = [
            min(
                range(len(sampled)),
                key=lambda index: abs(sampled[index] - int(native)),
            )
            for native in anchor.get("native_frame_index_view", ())
        ]
        priority.extend(_uniform(mapped, 12))
    priority.extend(_uniform(range(len(sampled)), 8))
    output = []
    for value in priority:
        bounded = max(0, min(len(sampled) - 1, int(value)))
        if bounded not in output:
            output.append(bounded)
        if len(output) == maximum:
            break
    return sorted(output)


def _anchor_prompt(anchors: list[dict[str, Any]], frame_ids: list[int]) -> str:
    return (
        "Fixed independent action/state anchor phrases:\n"
        + "\n".join(
            f"{row['anchor_id']}: {row['phrase']}" for row in anchors
        )
        + f"\nPresented chronological sampled-frame IDs: {frame_ids}\n"
        "For a transition, cite boundary frames that jointly demonstrate the "
        "change, not a later frame where only the resulting state persists. For "
        "a sustained action/state, cite its earliest and latest direct evidence. "
        "Localize every phrase independently; do not infer a query predicate, "
        "temporal relation, candidate, or answer."
    )


def _validate_anchor_payload_v2(
    value: Any, anchor_ids: list[str], frame_ids: list[int],
) -> list[dict[str, Any]]:
    """Canonicalize uncertainty to a strict fail-closed representation.

    Qwen occasionally returns ``UNKNOWN`` together with frames it inspected.
    Those frames are not positive evidence.  Dropping them is a deterministic,
    information-decreasing repair; it never upgrades an unknown anchor to
    supported and therefore cannot leak or improve a task answer.
    """

    if isinstance(value, dict) and isinstance(value.get("anchors"), list):
        for row in value["anchors"]:
            if isinstance(row, dict) and row.get("status") == "UNKNOWN":
                row["evidence_frame_ids"] = []
    return _validate_rows(
        value, "anchors", "anchor_id", anchor_ids, frame_ids,
    )


def _anchor_visible_track_ids(stable, anchors: list[dict[str, Any]]) -> frozenset[str]:
    """Select only P0 and public-ontology objects named by an anchor phrase."""

    phrases = " ".join(str(row["phrase"]).casefold() for row in anchors)
    output = {"T0"}
    for track in stable.tracks:
        labels = {str(track.canonical_label).casefold()}
        labels.update(str(value).casefold() for value in track.aliases)
        tokens = {
            token.strip() for label in labels
            for token in label.replace("_", " ").split("/") if token.strip()
        }
        if any(token in phrases for token in tokens):
            output.add(str(track.track_id))
    return frozenset(output)


def _restrict_anchor_frames_to_named_objects(
    frame_ids: list[int], detections: dict[int, list[tuple[str, dict[str, Any]]]],
    visible_track_ids: frozenset[str],
) -> list[int]:
    """Remove frames that cannot visually contain an explicitly named object.

    Object-free action phrases keep the complete action-prior frame set.  For
    an object-bearing phrase, every presented frame must have detector evidence
    for at least one matching public-ontology track, preventing a localizer
    from citing an impossible transition boundary.
    """

    object_ids = visible_track_ids - {"T0"}
    if not object_ids:
        return list(frame_ids)
    restricted = [
        frame_id for frame_id in frame_ids
        if any(track_id in object_ids for track_id, _ in detections.get(frame_id, ()))
    ]
    return restricted if restricted else list(frame_ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_status(consumed_development_pilot: bool) -> str:
    if consumed_development_pilot:
        return "CONSUMED_DEVELOPMENT_ANCHOR_PILOT_NOT_TRANSFER_EVIDENCE"
    return "ANSWER_BLIND_ANCHOR_LOCALIZATIONS_FROZEN_BEFORE_TARGET_OUTCOME"


def _anchor_intervals(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Compile supported evidence into tight deterministic intervals."""

    output = []
    for row in rows:
        evidence = sorted(set(int(value) for value in row["evidence_frame_ids"]))
        if row["status"] == "SUPPORTED" and evidence:
            output.append([min(evidence), max(evidence)])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime-amendment", type=Path)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--provider", default="alibaba")
    parser.add_argument("--maximum-anchor-frames", type=int, default=20)
    parser.add_argument("--minimum-object-score", type=float, default=0.05)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--consumed-development-pilot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("anchor-localization output is immutable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if args.maximum_anchor_frames <= 0:
        raise ValueError("maximum anchor frames must be positive")

    cohort = json.loads(args.cohort.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    plans = json.loads(args.query_plans.read_text())
    protocol = json.loads(args.protocol.read_text())
    amendment = (
        json.loads(args.runtime_amendment.read_text())
        if args.runtime_amendment is not None else None
    )
    if any(bool(sgdet.get(key)) or bool(plans.get(key)) for key in FORBIDDEN_INPUT_FLAGS):
        raise ValueError("anchor-localizer input crossed its authority boundary")
    expected_inputs = protocol["immutable_inputs"]
    actual_inputs = {
        "cohort_sha256": cohort["cohort_sha256"],
        "sgdet_file_sha256": _sha256(args.sgdet),
        "query_plans_file_sha256": _sha256(args.query_plans),
        "anchor_schema_helper_sha256": _sha256(
            Path(__file__).resolve().parent
            / "pilot_agqa_atomic_temporal_grounder_v2.py"
        ),
    }
    if any(expected_inputs.get(key) != value for key, value in actual_inputs.items()):
        raise ValueError(
            f"immutable anchor inputs differ: expected={expected_inputs} "
            f"actual={actual_inputs}"
        )
    current_collector_sha = _sha256(Path(__file__))
    frozen_collector_sha = expected_inputs.get("anchor_collector_sha256")
    if current_collector_sha != frozen_collector_sha:
        expected_amendment = {
            "parent_acquisition_protocol_file_sha256": _sha256(args.protocol),
            "replaced_anchor_collector_sha256": frozen_collector_sha,
            "replacement_anchor_collector_sha256": current_collector_sha,
            "anchor_scope": "CANONICALIZE_UNKNOWN_ANCHOR_TO_EMPTY_EVIDENCE",
            "development_outcomes_opened_before_amendment": False,
            "target_outcomes_read_before_amendment": False,
        }
        if amendment is None or any(
            amendment.get(key) != value
            for key, value in expected_amendment.items()
        ):
            raise ValueError("runtime amendment does not authorize this anchor collector")
    acquisition = protocol["answer_blind_anchor_localization"]
    for key, value in (
        ("model", args.model), ("provider", args.provider),
        ("maximum_anchor_frames", args.maximum_anchor_frames),
        ("minimum_object_score", args.minimum_object_score),
    ):
        if acquisition[key] != value:
            raise ValueError(f"runtime anchor {key} differs from frozen protocol")
    if int(acquisition.get("seed", 0)) != 0:
        raise ValueError("anchor localization currently supports only frozen seed 0")
    if acquisition.get("provider_allow_fallbacks") not in {None, False}:
        raise ValueError("anchor localization must disable provider fallback")
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    wanted = set(str(value) for value in (args.task_id or ()))
    source_rows = [
        row for index, row in enumerate(plans["rows"])
        if index % args.shard_count == args.shard_index
        and (not wanted or str(row["task_id"]) in wanted)
    ]
    if wanted - {str(row["task_id"]) for row in source_rows} and args.shard_count == 1:
        raise ValueError("requested task ID is absent")
    if args.max_tasks is not None:
        source_rows = source_rows[:args.max_tasks]

    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(
        api_key=key, base_url="https://openrouter.ai/api/v1",
        timeout=300, max_retries=2,
    )
    # OpenRouter currently reports no reasoning control for this Qwen instruct
    # model.  Pin one non-reasoning endpoint and deterministic sampling instead
    # of sending the unroutable reasoning.enabled=false parameter.
    model = {
        "id": args.model,
        "omit_temperature": False,
        "seed": 0,
        "provider": {"only": [args.provider], "allow_fallbacks": False},
    }
    outputs = []
    provider_calls = 0
    total_cost = 0.0
    for plan in source_rows:
        task_id = str(plan["task_id"])
        if task_id not in public:
            raise ValueError(f"query plan task is absent from cohort: {task_id}")
        video_id = str(public[task_id]["video_id"])
        if video_id not in raw_by_video:
            raise ValueError(f"video is absent from SGDET receipt: {video_id}")
        raw_video = raw_by_video[video_id]
        anchors = _anchor_specs_v2({
            "action_obligations": plan.get("action_obligations", ())
        })
        call_receipt = None
        localized = []
        error = None
        if anchors:
            stable, detections = _detections_by_frame(
                raw_video, args.minimum_object_score,
            )
            visible_track_ids = _anchor_visible_track_ids(stable, anchors)
            frame_ids = _restrict_anchor_frames_to_named_objects(
                _anchor_frame_ids_v2(
                    raw_video, anchors, args.maximum_anchor_frames,
                ),
                detections,
                visible_track_ids,
            )
            images, seconds, scales = _exact_sgdet_frames(
                Path(public[task_id]["video_path"]), raw_video, frame_ids,
            )
            detections = {
                frame: [row for row in values if row[0] in visible_track_ids]
                for frame, values in detections.items()
            }
            track_labels = {
                str(track.track_id): str(track.canonical_label)
                for track in stable.tracks if track.track_id in visible_track_ids
            }
            annotated = _annotate_frames(
                images, frame_ids, scales, detections, track_labels,
            )
            panels = _panels(
                annotated, seconds, frames_per_panel=2,
                frame_width=448, quality=90,
            )
            anchor_ids = [row["anchor_id"] for row in anchors]
            prompt = _anchor_prompt(anchors, frame_ids)
            response_format = _anchor_response_format(anchor_ids, frame_ids)
            core = {
                "protocol": "AGQA_ANSWER_BLIND_ANCHOR_LOCALIZER_V2",
                "protocol_file_sha256": _sha256(args.protocol),
                "runtime_amendment_file_sha256": (
                    _sha256(args.runtime_amendment)
                    if args.runtime_amendment is not None else None
                ),
                "task_id": task_id, "video_sha256": raw_video["video_sha256"],
                "anchor_phrases": [
                    {"anchor_id": row["anchor_id"], "phrase": row["phrase"]}
                    for row in anchors
                ],
                "visible_public_track_ids": sorted(visible_track_ids),
                "visible_public_track_labels": track_labels,
                "frame_selection_contract": (
                    "ACTION_PRIOR_PLUS_UNIFORM_THEN_NAMED_PUBLIC_OBJECT_VISIBILITY"
                ),
                "presented_frame_ids": frame_ids,
                "presented_frame_sha256s": [
                    raw_video["selected_frame_sha256s"][index] for index in frame_ids
                ],
                "panel_sha256s": [hashlib.sha256(value).hexdigest() for value in panels],
                "model": model,
                "system_sha256": stable_hash(ANCHOR_V2_SYSTEM),
                "prompt_sha256": stable_hash(prompt),
                "request_contract": _request_cache_contract(
                    model=model, max_tokens=640,
                    response_format=response_format, maximum_attempts=2,
                ),
            }
            try:
                payload, usage, reused = _cached_provider_call(
                    cache_dir=args.cache_dir,
                    call_name=f"anchor_v2_{task_id}", input_core=core,
                    invoke=lambda: _provider_call_with_contract_retries(
                        client, model=model, system=ANCHOR_V2_SYSTEM,
                        content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                        max_tokens=640, response_format=response_format,
                        maximum_attempts=2,
                        validator=lambda value: _validate_anchor_payload_v2(
                            value, anchor_ids, frame_ids,
                        ),
                    ),
                )
                localized = _validate_anchor_payload_v2(
                    payload, anchor_ids, frame_ids,
                )
            except Exception as exc:
                usage, error = _provider_failure(exc)
                reused = False
                localized = [{
                    "anchor_id": anchor_id, "status": "UNKNOWN",
                    "confidence": 0.0, "evidence_frame_ids": [],
                } for anchor_id in anchor_ids]
            provider_calls += int(not reused) * int(usage.get("provider_attempts", 0))
            total_cost += float(usage.get("reported_cost_usd", 0.0))
            call_receipt = {
                **core, "usage": usage, "cache_reused": reused,
                "provider_error": error,
            }
        outputs.append({
            "task_id": task_id, "video_id": video_id,
            "anchor_specs": [{
                "anchor_id": row["anchor_id"], "phrase": row["phrase"],
            } for row in anchors],
            "anchor_localizations": localized,
            "anchor_intervals": _anchor_intervals(localized),
            "provider_error": error,
            "call_receipt": call_receipt,
        })
        print(json.dumps({
            "task_id": task_id,
            "anchors": len(anchors), "intervals": _anchor_intervals(localized),
            "provider_calls_running": provider_calls,
            "cost_usd_running": total_cost,
        }), flush=True)

    report = {
        "schema_version": "agqa-answer-blind-anchor-localizations-v2",
        "status": _artifact_status(args.consumed_development_pilot),
        "consumed_development_pilot": bool(args.consumed_development_pilot),
        "model": model,
        "maximum_anchor_frames": args.maximum_anchor_frames,
        "minimum_object_score": args.minimum_object_score,
        "bbox_coordinate_contract": "SGDET_BOXES_ARE_NATIVE_XYXY_RENDER_CLAMP_ONLY",
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "rows": outputs, "provider_calls": provider_calls,
        "reported_cost_usd": total_cost,
        "authority": {
            "question_text_supplied_to_vlm": False,
            "root_query_predicate_supplied_to_vlm": False,
            "temporal_operator_supplied_to_vlm": False,
            "candidate_identity_supplied_to_vlm": False,
            "anchor_phrase_object_tracks_supplied_to_vlm": True,
            "answer_read": False, "official_stsg_read": False,
            "functional_program_read": False, "source_controller_read": False,
            "target_outcome_read": False,
        },
        "input_file_sha256s": {
            "cohort": _sha256(args.cohort), "sgdet": _sha256(args.sgdet),
            "query_plans": _sha256(args.query_plans),
        },
        "protocol_file_sha256": _sha256(args.protocol),
        "runtime_amendment_file_sha256": (
            _sha256(args.runtime_amendment)
            if args.runtime_amendment is not None else None
        ),
        "collector_file_sha256": _sha256(Path(__file__)),
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
