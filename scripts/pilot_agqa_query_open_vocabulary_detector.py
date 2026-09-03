#!/usr/bin/env python3
"""Outcome-blind GroundingDINO entity-recall pilot on consumed AGQA development rows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from motif_transfer.agqa_open_vocabulary_grounder import detect_ontology_tracks
from motif_transfer.agqa_query_object_grounder import AGQA_OBJECT_ONTOLOGY, AGQA_OBJECT_QUERY_TERMS
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import _sample_video_range


ONTOLOGY = ("person",) + AGQA_OBJECT_ONTOLOGY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--localized-grounding", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspection-frames", type=int, default=16)
    parser.add_argument("--box-threshold", type=float, default=.18)
    parser.add_argument("--ontology-chunk-size", type=int)
    parser.add_argument("--maximum-tracks", type=int, default=12)
    parser.add_argument("--task-ids", help="Comma-separated answer-blind development task IDs")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("detector pilot output is immutable")
    cohort = json.loads(args.cohort.read_text())
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    grounding = (json.loads(args.localized_grounding.read_text())
                 if args.localized_grounding else None)
    selected = (grounding["rows"] if grounding else [
        {"task_id": row["task_id"]} for row in cohort["rows"]
        if row.get("structural") == "query"
    ])
    if args.task_ids:
        wanted = {value.strip() for value in args.task_ids.split(",") if value.strip()}
        selected = [value for value in selected if str(value["task_id"]) in wanted]
        if {str(value["task_id"]) for value in selected} != wanted:
            raise ValueError("one or more requested task IDs are absent from the public cohort")
    rows = []
    video_cache = {}
    for source in selected:
        task_id = str(source["task_id"]); row = public[task_id]
        if grounding:
            metadata = source["video_metadata"]
            if "localized" in metadata:
                low = float(metadata["localized"]["sample_start_second"])
                high = float(metadata["localized"]["sample_end_second"])
            else:
                low, high = 0.0, float(metadata["duration_seconds"])
        else:
            low, high = 0.0, None
        cache_key = (row["video_id"], low, high)
        if cache_key not in video_cache:
            frames, seconds, _ = _sample_video_range(
                Path(row["video_path"]), frame_count=64, max_side=800,
                start_second=low, end_second=high)
            indices = tuple(dict.fromkeys(round(float(x)) for x in np.linspace(
                0, 63, args.inspection_frames)))
            tracks, detections = detect_ontology_tracks(
                frames, frame_indices=indices, ontology=ONTOLOGY,
                query_terms=("person",) + AGQA_OBJECT_QUERY_TERMS,
                box_threshold=args.box_threshold, text_threshold=args.box_threshold,
                maximum_tracks=args.maximum_tracks,
                ontology_chunk_size=args.ontology_chunk_size)
            video_cache[cache_key] = (tracks, detections, indices, seconds)
        tracks, detections, indices, seconds = video_cache[cache_key]
        rows.append({"task_id": task_id, "tracks": [asdict(x) for x in tracks],
                     "detections": [asdict(x) for x in detections],
                     "inspection_indices": list(indices),
                     "inspection_seconds": [seconds[x] for x in indices]})
        print(json.dumps({"task_id": task_id, "tracks": [x.canonical_label for x in tracks]}), flush=True)
    body = {"schema_version": "agqa-query-open-vocabulary-detector-pilot-v1",
            "status": "DETECTOR_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
            "source_grounding_report_sha256": grounding["report_sha256"] if grounding else None,
            "ontology_sha256": stable_hash(list(ONTOLOGY)),
            "model_id": "IDEA-Research/grounding-dino-base",
            "inspection_frames": args.inspection_frames, "box_threshold": args.box_threshold,
            "maximum_tracks": args.maximum_tracks,
            "ontology_chunk_size": args.ontology_chunk_size,
            "rows": rows, "answer_read": False, "official_scene_graph_read": False,
            "functional_program_read": False, "source_controller_read": False,
            "target_outcome_read": False}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
