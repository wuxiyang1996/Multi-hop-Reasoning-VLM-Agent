#!/usr/bin/env python3
"""Answer-blind Qwen candidate-ID verification over frozen raw-video tools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import cv2
from openai import OpenAI
from PIL import Image, ImageDraw

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _panels, _provider_json_call,
    _sample_video_range,
)


SYSTEM = """You are a frozen target-native video perception tool, not a question-answering agent. The public question and typed outer predicate specify a perceptual temporal scope. Candidate labels, track boxes, and neural scores are fallible proposals. Inspect the chronological raw-video evidence and select at most one candidate ID that visibly fills the requested OUTER typed role in that scope. Return ABSTAIN when the pixels do not distinguish a unique candidate. Never output an object-name answer, source-domain fact, functional program, controller decision, correctness judgment, or free-text rationale. Return only the required JSON schema."""


def response_format(candidate_ids: list[str], frame_ids: list[int]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_qgv4_candidate_id_verification_v1", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "selected_candidate_id": {
                    "type": "string", "enum": candidate_ids + ["ABSTAIN"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_frame_ids": {
                    # DeepInfra's constrained decoder does not implement the
                    # JSON-Schema uniqueItems keyword.  Runtime canonicalizes
                    # and validates these enum-constrained IDs below.
                    "type": "array", "maxItems": 4,
                    "items": {"type": "integer", "enum": frame_ids},
                },
                "failure_reason": {
                    "type": "string", "enum": [
                        "NONE", "INSUFFICIENT_PIXELS", "NO_CANDIDATE_MATCH",
                        "AMBIGUOUS_TEMPORAL_WINDOW",
                    ],
                },
            },
            "required": [
                "selected_candidate_id", "confidence", "evidence_frame_ids",
                "failure_reason",
            ],
        },
    }}


def fail_closed_call(**kwargs):
    try:
        return _provider_json_call(**kwargs)
    except ValueError as exc:
        payload = {
            "selected_candidate_id": "ABSTAIN", "confidence": 0.0,
            "evidence_frame_ids": [], "failure_reason": "INSUFFICIENT_PIXELS",
        }
        usage = {
            "model": str(kwargs["model"]["id"]), "finish_reason": "schema_failure",
            "prompt_tokens": 0, "completion_tokens": 0,
            "reported_cost_usd": 0.0, "response_sha256": stable_hash(payload),
            "schema_error_type": type(exc).__name__,
        }
        return payload, usage


def candidate_pool(row: dict, maximum: int) -> list[dict]:
    """Create a deterministic label-unique pool from frozen neural rankings."""
    output = []
    seen = set()
    for ranked in row.get("candidate_ranking", ()):
        label = str(ranked["candidate_label"])
        if label in seen or ranked.get("track_id") is None:
            continue
        seen.add(label)
        output.append({
            "candidate_id": f"C{len(output)}", "label": label,
            "track_id": str(ranked["track_id"]),
            "frozen_rank": len(output) + 1,
            "frozen_fusion_score": float(ranked["score"]),
            "sources": list(ranked.get("sources", ())),
            "base_evidence_frames": sorted(set(
                int(value) for value in ranked.get("evidence_frames", ())
            )),
        })
        if len(output) == maximum:
            break
    return output


def selected_frame_ids(
    *, lower: int, upper: int, candidates: list[dict], frame_count: int,
    maximum: int,
) -> list[int]:
    """Budgeted temporal coverage plus rank-ordered track evidence."""
    lower = max(0, min(frame_count - 1, int(lower)))
    upper = max(lower, min(frame_count - 1, int(upper)))
    midpoint = (lower + upper) // 2
    priority = [lower, midpoint, upper]
    priority.extend(
        int(frame)
        for candidate in candidates
        for frame in candidate.get("base_evidence_frames", ())[:1]
        if lower <= int(frame) <= upper
    )
    # Six evenly spaced scope frames prevent candidate proposals from fully
    # controlling what pixels the verifier receives.
    if upper > lower:
        priority.extend(round(lower + index * (upper - lower) / 5) for index in range(6))
    output = []
    for frame in priority:
        value = max(0, min(frame_count - 1, int(frame)))
        if value not in output:
            output.append(value)
        if len(output) == maximum:
            break
    return sorted(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sgdet_frame_sha256(frame) -> str:
    """Reproduce the SGDET receipt's shape+BGR-pixel hash without Torch."""
    if str(frame.dtype) != "uint8" or frame.ndim != 3:
        raise ValueError("decoded frame must be an HxWxC uint8 array")
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in frame.shape)).encode("ascii"))
    digest.update(b"\0BGR_UINT8\0")
    digest.update(frame.tobytes(order="C"))
    return digest.hexdigest()


def _detector_scale(height: int, width: int) -> float:
    """Match the frozen Faster R-CNN ``prep_im_for_blob`` resize."""
    value = 600.0 / min(height, width)
    if round(value * max(height, width)) > 1000:
        value = 1000.0 / max(height, width)
    return value


def _exact_sgdet_frames(
    path: Path, video: dict, frame_ids: list[int],
) -> tuple[list[Image.Image], list[float], dict[int, float]]:
    """Decode the exact content-addressed native frames used by SGDET.

    ``frame_ids`` index the frozen SGDET proxy sequence.  The raw receipt maps
    each proxy position to an original-video coordinate and pixel hash.  This
    prevents a verifier from silently seeing a second, slightly shifted sample.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    native = [int(value) for value in video["sampled_original_frame_indices"]]
    hashes = [str(value) for value in video["selected_frame_sha256s"]]
    if fps <= 0 or total <= 0 or len(native) != len(hashes):
        capture.release()
        raise RuntimeError(f"invalid frozen video receipt: {path}")
    frames: list[Image.Image] = []
    seconds: list[float] = []
    # ``lib.object_detector.detector`` divides ``PRED_BOXES`` by the image
    # preprocessing scale before returning ``entry['boxes']``.  The SGDET
    # receipt therefore stores original decoded-frame coordinates.  Consumers
    # keep the historical ``scales`` interface, but its correct divisor is the
    # identity.  Applying ``_detector_scale`` here would inverse-scale twice
    # and shift every overlay toward the upper-left corner.
    scales: dict[int, float] = {}
    for frame_id in frame_ids:
        if not 0 <= int(frame_id) < len(native):
            capture.release()
            raise ValueError("verifier frame ID exceeds SGDET receipt")
        requested = native[int(frame_id)]
        capture.set(cv2.CAP_PROP_POS_FRAMES, requested)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            capture.release()
            raise RuntimeError(f"failed decoding {path} at native frame {requested}")
        actual = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES)))) - 1
        if actual != requested:
            capture.release()
            raise RuntimeError(
                f"decoder coordinate drift for {path}: requested={requested} actual={actual}"
            )
        if _sgdet_frame_sha256(bgr) != hashes[int(frame_id)]:
            capture.release()
            raise RuntimeError(f"decoded pixel hash differs from SGDET receipt: {path}")
        scales[int(frame_id)] = 1.0
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        seconds.append(requested / fps)
    capture.release()
    return frames, seconds, scales


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet-raw", type=Path, required=True)
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--maximum-candidates", type=int, default=10)
    parser.add_argument("--maximum-frames", type=int, default=12)
    parser.add_argument(
        "--coordinate-mode", choices=("legacy_proxy", "native_exact"),
        default="legacy_proxy",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V4 adjudication shard is immutable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    cohort = json.loads(args.cohort.read_text())
    raw = json.loads(args.sgdet_raw.read_text())
    grounding = json.loads(args.candidate_grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    forbidden = (
        "answer_read", "functional_program_read", "official_scene_graph_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(raw.get(key) for key in forbidden) or any(grounding.get(key) for key in forbidden):
        raise ValueError("input receipt violates authority boundary")
    expected = protocol["immutable_inputs"]
    actual = {
        "cohort_sha256": cohort["cohort_sha256"],
        "query_grounding_file_sha256": _sha256(args.candidate_grounding),
        "sgdet_file_sha256": _sha256(args.sgdet_raw),
    }
    for key, value in actual.items():
        if expected[key] != value:
            raise ValueError(f"immutable input mismatch: {key}")
    if args.maximum_candidates != int(protocol["verifier"]["maximum_candidate_count"]):
        raise ValueError("candidate budget differs from frozen protocol")
    if args.maximum_frames != int(protocol["verifier"]["maximum_presented_unique_frames"]):
        raise ValueError("frame budget differs from frozen protocol")
    if args.model != protocol["verifier"]["model"]:
        raise ValueError("model differs from frozen protocol")
    expected_coordinate_mode = protocol["verifier"].get(
        "coordinate_mode", "legacy_proxy",
    )
    if args.coordinate_mode != expected_coordinate_mode:
        raise ValueError("coordinate mode differs from frozen protocol")

    public = {str(row["task_id"]): row for row in cohort["rows"]}
    raw_by_video = {str(row["video_id"]): row for row in raw["rows"]}
    all_rows = grounding["rows"]
    sources = [
        row for index, row in enumerate(all_rows)
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
    model = {"id": args.model, "omit_temperature": True}
    palette = (
        "red", "lime", "cyan", "yellow", "magenta", "orange",
        "deepskyblue", "violet", "springgreen", "gold",
    )
    outputs = []
    calls = 0
    total_cost = 0.0
    for source in sources:
        task_id = str(source["task_id"])
        public_row = public[task_id]
        video_id = str(source["video_id"])
        video = raw_by_video[video_id]
        candidates = candidate_pool(source, args.maximum_candidates)
        lower, upper = (int(value) for value in source["root_temporal_window"])
        frame_count = int(video["model_visible_frame_count"])
        frame_ids = selected_frame_ids(
            lower=lower, upper=upper, candidates=candidates,
            frame_count=frame_count, maximum=args.maximum_frames,
        )
        if not candidates:
            outputs.append({
                "task_id": task_id, "video_id": video_id,
                "status": "ABSTAIN_NO_NEURAL_CANDIDATES",
                "selected_candidate_id": "ABSTAIN", "selected_candidate": None,
                "confidence": 0.0, "evidence_frame_ids": [], "candidates": [],
                "presented_frame_ids": [], "presented_frame_sha256s": [],
                "panel_sha256s": [], "usage": None, "cache_reused": False,
            })
            continue

        if args.coordinate_mode == "native_exact":
            selected, selected_seconds, detector_scales = _exact_sgdet_frames(
                Path(public_row["video_path"]), video, frame_ids,
            )
            selected = [frame.copy() for frame in selected]
        else:
            frames, seconds, _ = _sample_video_range(
                Path(public_row["video_path"]), frame_count=frame_count,
                max_side=800, start_second=0.0, end_second=None,
            )
            selected = [frames[index].copy() for index in frame_ids]
            selected_seconds = [seconds[index] for index in frame_ids]
            detector_scales = {frame_id: 1.0 for frame_id in frame_ids}
        track_compilation = build_stable_tracks(video, minimum_object_score=0.05)
        candidate_id_by_track = {
            candidate["track_id"]: candidate["candidate_id"] for candidate in candidates
        }
        color_by_id = {
            candidate["candidate_id"]: palette[index % len(palette)]
            for index, candidate in enumerate(candidates)
        }
        objects_by_frame = {}
        for detected in video["objects"]:
            detection_index = int(detected["detection_index"])
            if detection_index not in track_compilation.retained_detection_indices:
                continue
            track_id = track_compilation.detection_to_track.get(detection_index)
            candidate_id = candidate_id_by_track.get(str(track_id))
            if candidate_id is not None:
                objects_by_frame.setdefault(int(detected["sampled_frame_index"]), []).append(
                    (candidate_id, detected)
            )
        for image, frame_id in zip(selected, frame_ids):
            draw = ImageDraw.Draw(image)
            draw.text(
                (8, 8), f"S{frame_id}", fill="white",
                stroke_width=3, stroke_fill="black",
            )
            for candidate_id, detected in objects_by_frame.get(frame_id, ()):
                scale = detector_scales[frame_id]
                box = tuple(
                    float(value) / scale for value in detected["bbox_xyxy"]
                )
                color = color_by_id[candidate_id]
                draw.rectangle(box, outline=color, width=5)
                draw.text(
                    (box[0] + 2, box[1] + 2), candidate_id,
                    fill=color, stroke_width=2, stroke_fill="black",
                )
        panels = _panels(
            selected, selected_seconds,
            frames_per_panel=2, frame_width=448, quality=90,
        )
        table = "; ".join(
            "{}={} [rank={}, tools={}]".format(
                row["candidate_id"], row["label"], row["frozen_rank"],
                "+".join(row["sources"]) or "unknown",
            ) for row in candidates
        )
        prompt = (
            "Public question (scope only; do not answer it): {}\n"
            "Requested outer typed role: {}\n"
            "Requested outer predicate: {}\n"
            "Parsed temporal operator: {}\n"
            "Frozen 64-frame scope: S{}..S{}\n"
            "Displayed chronological sample IDs: {}\n"
            "Fallible candidate IDs: {}\n"
            "Select the unique candidate ID visibly filling the OUTER role in the scope, "
            "or ABSTAIN. Do not return an object name."
        ).format(
            public_row["question"], source["requested_role"],
            source["root_predicate"],
            str(source.get("root_temporal_operator") or "FROM_FROZEN_PARSER"),
            lower, upper, frame_ids, table,
        )
        panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
        frame_hashes = [str(video["selected_frame_sha256s"][index]) for index in frame_ids]
        core = {
            "protocol": "AGQA_QUERY_GROUNDER_V4_QWEN235_CANDIDATE_ID_V1",
            "protocol_file_sha256": _sha256(args.protocol),
            "task_id": task_id, "question_sha256": public_row["question_sha256"],
            "candidate_grounding_report_sha256": grounding["report_sha256"],
            "candidate_table_sha256": stable_hash(candidates),
            "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes, "model": model,
            "coordinate_mode": args.coordinate_mode,
        }
        payload, usage, reused = _cached_provider_call(
            cache_dir=args.cache_dir, call_name="qgv4_" + task_id,
            input_core=core,
            invoke=lambda: fail_closed_call(
                client=client, model=model, system=SYSTEM,
                content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                max_tokens=100,
                response_format=response_format(
                    [row["candidate_id"] for row in candidates], frame_ids,
                ),
            ),
        )
        selected_id = str(payload["selected_candidate_id"])
        selected_candidate = next(
            (row for row in candidates if row["candidate_id"] == selected_id), None
        )
        evidence = sorted(set(int(value) for value in payload["evidence_frame_ids"]))
        confidence = float(payload["confidence"])
        if selected_candidate is None or not evidence:
            selected_id = "ABSTAIN"
            selected_candidate = None
            evidence = []
            confidence = 0.0
        calls += int(not reused)
        total_cost += float(usage.get("reported_cost_usd", 0.0))
        outputs.append({
            "task_id": task_id, "video_id": video_id,
            "status": "BOUND" if selected_candidate else "ABSTAIN",
            "selected_candidate_id": selected_id,
            "selected_candidate": selected_candidate,
            "confidence": confidence, "evidence_frame_ids": evidence,
            "failure_reason": str(payload["failure_reason"]),
            "candidates": candidates, "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes, "usage": usage,
            "cache_reused": reused,
        })
        print(json.dumps({
            "task_id": task_id, "selected_candidate_id": selected_id,
            "confidence": confidence,
            "cost_usd": usage.get("reported_cost_usd", 0.0),
        }), flush=True)

    report = {
        "schema_version": "agqa-query-grounder-v4-qwen235-adjudication-shard-v1",
        "status": "V4_ADJUDICATION_SHARD_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "model": model, "maximum_candidate_count": args.maximum_candidates,
        "maximum_presented_unique_frames": args.maximum_frames,
        "coordinate_mode": args.coordinate_mode,
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "rows": outputs, "provider_calls": calls,
        "reported_cost_usd": total_cost,
        "cohort_sha256": cohort["cohort_sha256"],
        "candidate_grounding_report_sha256": grounding["report_sha256"],
        "protocol_file_sha256": _sha256(args.protocol),
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
