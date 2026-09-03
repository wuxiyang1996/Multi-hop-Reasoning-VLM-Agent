#!/usr/bin/env python3
"""Outcome-blind GroundingDINO relation-phrase to entity-track pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.agqa_open_vocabulary_grounder import (
    Detection, detect_relation_phrase_regions,
)
from motif_transfer.agqa_relation_phrase_binder import (
    bind_phrase_regions_to_tracks, relation_query_phrases,
)
from motif_transfer.agqa_query_grounder_v2 import requested_query_predicates
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _sample_video_range
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--detector-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--box-threshold", type=float, default=.12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("relation phrase pilot output is immutable")
    cohort = json.loads(args.cohort.read_text())
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    runtime = json.loads(args.semantic_runtime.read_text())
    semantics = {str(row["task_id"]): _semantic(row["receipt"]) for row in runtime["rows"]}
    detector = json.loads(args.detector_grounding.read_text())
    sources = detector["rows"][:args.max_tasks] if args.max_tasks else detector["rows"]
    rows = []
    for source in sources:
        task_id = str(source["task_id"]); row = public[task_id]
        predicates = requested_query_predicates(semantics[task_id])
        predicate = predicates[0] if predicates else "related to"
        phrases = relation_query_phrases(predicate)
        frames, _, _ = _sample_video_range(
            Path(row["video_path"]), frame_count=64, max_side=800,
            start_second=0.0, end_second=None)
        regions = detect_relation_phrase_regions(
            frames, frame_indices=source["inspection_indices"], phrases=phrases,
            box_threshold=args.box_threshold, text_threshold=args.box_threshold)
        ontology = tuple(Detection(
            frame_index=int(value["frame_index"]), label=str(value["label"]),
            confidence=float(value["confidence"]),
            bbox_xyxy=tuple(float(x) for x in value["bbox_xyxy"]),
        ) for value in source["detections"])
        binding = bind_phrase_regions_to_tracks(regions, ontology)
        track = None
        if binding is not None:
            track = next((value for value in source["tracks"]
                          if value["canonical_label"] == binding.label), None)
        rows.append({
            "task_id": task_id, "predicate": predicate, "query_phrases": list(phrases),
            "phrase_regions": [asdict(value) for value in regions],
            "binding": ({**asdict(binding), "track_id": track["track_id"]}
                        if binding is not None and track is not None else None),
        })
        print(json.dumps({"task_id": task_id, "predicate": predicate,
                          "regions": len(regions),
                          "binding": binding.label if binding else None}), flush=True)
    body = {
        "schema_version": "agqa-relation-phrase-grounder-pilot-v1",
        "status": "RELATION_PHRASE_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "model": "IDEA-Research/grounding-dino-base", "box_threshold": args.box_threshold,
        "detector_report_sha256": detector["report_sha256"], "rows": rows,
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False, "provider_calls": 0, "reported_cost_usd": 0.0,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
