#!/usr/bin/env python3
"""Outcome-blind SlowFast + Florence-2 + GroundingDINO role pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.agqa_florence_phrase_grounder import ground_relation_phrases
from motif_transfer.agqa_open_vocabulary_grounder import detect_ontology_tracks
from motif_transfer.agqa_query_grounder_v2 import requested_query_predicates
from motif_transfer.agqa_query_object_grounder import AGQA_OBJECT_ONTOLOGY, AGQA_OBJECT_QUERY_TERMS
from motif_transfer.agqa_relation_phrase_binder import (
    bind_phrase_regions_to_tracks, relation_query_phrases,
    slowfast_relation_frame_indices,
)
from motif_transfer.agqa_temporal_localized_query import parse_temporal_localized_object_question
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _sample_video_range
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


ONTOLOGY = ("person",) + AGQA_OBJECT_ONTOLOGY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--action-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--box-threshold", type=float, default=.12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Florence role pilot output is immutable")
    cohort = json.loads(args.cohort.read_text())
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    runtime = json.loads(args.semantic_runtime.read_text())
    semantics = {str(row["task_id"]): _semantic(row["receipt"]) for row in runtime["rows"]}
    action = json.loads(args.action_grounding.read_text())
    rows = []
    for source in action["rows"][:args.max_tasks]:
        task_id = str(source["task_id"]); row = public[task_id]
        predicates = requested_query_predicates(semantics[task_id])
        predicate = predicates[0] if predicates else "related to"
        plan = parse_temporal_localized_object_question(str(row["question"]))
        operator = plan.temporal_operator if plan else "VIDEO"
        indices = slowfast_relation_frame_indices(
            source, temporal_operator=operator, frame_count=48, inspection_frames=8)
        frames, _, _ = _sample_video_range(
            Path(row["video_path"]), frame_count=48, max_side=800,
            start_second=0.0, end_second=None)
        tracks, ontology = detect_ontology_tracks(
            frames, frame_indices=indices, ontology=ONTOLOGY,
            query_terms=("person",) + AGQA_OBJECT_QUERY_TERMS,
            box_threshold=args.box_threshold, text_threshold=args.box_threshold,
            maximum_tracks=12, ontology_chunk_size=6)
        phrases = relation_query_phrases(predicate)
        regions = ground_relation_phrases(frames, frame_indices=indices, phrases=phrases)
        binding = bind_phrase_regions_to_tracks(regions, ontology)
        track = next((value for value in tracks
                      if binding and value.canonical_label == binding.label), None)
        rows.append({
            "task_id": task_id, "predicate": predicate, "temporal_operator": operator,
            "inspection_indices": list(indices), "query_phrases": list(phrases),
            "tracks": [asdict(value) for value in tracks],
            "ontology_detections": [asdict(value) for value in ontology],
            "phrase_regions": [asdict(value) for value in regions],
            "binding": ({**asdict(binding), "track_id": track.track_id}
                        if binding is not None and track is not None else None),
            "action_obligations": source["obligations"],
        })
        print(json.dumps({"task_id": task_id, "regions": len(regions),
                          "binding": binding.label if binding else None}), flush=True)
    body = {
        "schema_version": "agqa-florence-relation-phrase-grounder-pilot-v1",
        "status": "RELATION_PHRASE_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "action_model": action["source"], "action_report_sha256": action["report_sha256"],
        "entity_model": "IDEA-Research/grounding-dino-base",
        "phrase_model": "florence-community/Florence-2-base-ft", "frame_budget": 48,
        "box_threshold": args.box_threshold, "rows": rows, "answer_read": False,
        "official_scene_graph_read": False, "functional_program_read": False,
        "source_controller_read": False, "target_outcome_read": False,
        "provider_calls": 0, "reported_cost_usd": 0.0,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
