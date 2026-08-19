#!/usr/bin/env python3
"""Freeze an outcome-blind, already-consumed V59 engineering split."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_program_transfer import profile_program  # noqa: E402
from motif_transfer.agqa_temporal_localized_query import (  # noqa: E402
    parse_temporal_localized_object_question,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
)
from scripts.collect_agqa2_temporal_localized_query_v59 import (  # noqa: E402
    _sha256,
)


ARCHIVE = Path(
    "/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip"
)
ENTRY = "AGQA_balanced/train_balanced.txt"
VIDEO_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades"
)
BASE_CONFIG = REPO_ROOT / "configs/agqa2_query_object_v27_development.json"
SIGNATURE = "Query>OnlyItem>Iterate>Localize>Filter"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO_ROOT /
        "configs/agqa2_temporal_localized_query_v59_development_manifest.json",
    )
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT /
        "configs/agqa2_temporal_localized_query_v59_development.json",
    )
    args = parser.parse_args()
    if args.sample_count < 1:
        raise ValueError("sample count must be positive")
    candidates = []
    with zipfile.ZipFile(ARCHIVE) as bundle:
        with bundle.open(ENTRY, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for task_id, row in _iter_top_level_object(text):
                program = str(row.get("program", ""))
                profile = profile_program(task_id=task_id, program=program)
                question = str(row.get("question", ""))
                video_id = str(row.get("video_id", ""))
                video_path = VIDEO_ROOT / f"{video_id}.mp4"
                if (
                    ">".join(profile.functions) != SIGNATURE
                    or "[relations," not in program
                    or parse_temporal_localized_object_question(question) is None
                    or not video_path.is_file()
                ):
                    continue
                plan = parse_temporal_localized_object_question(question)
                assert plan is not None
                candidates.append({
                    "priority": stable_hash("v59-development|" + task_id),
                    "temporal_operator": plan.temporal_operator,
                    "task_id": task_id,
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                })
    selected = []
    videos = set()
    by_operator = {
        operator: sorted(
            (row for row in candidates if row["temporal_operator"] == operator),
            key=lambda item: item["priority"],
        )
        for operator in ("BEFORE", "AFTER", "WHILE", "BETWEEN")
    }
    while len(selected) < args.sample_count:
        progressed = False
        for operator in ("BEFORE", "AFTER", "WHILE", "BETWEEN"):
            while by_operator[operator]:
                row = by_operator[operator].pop(0)
                if row["video_id"] not in videos:
                    break
            else:
                continue
            videos.add(row["video_id"])
            selected.append({
                key: value for key, value in row.items() if key != "priority"
            } | {"video_sha256": _sha256(Path(row["video_path"]))})
            progressed = True
            if len(selected) == args.sample_count:
                break
        if not progressed:
            break
    if len(selected) != args.sample_count:
        raise ValueError("not enough local unique-video development candidates")
    manifest_body = {
        "schema_version": "agqa2-temporal-localized-query-manifest-v59",
        "status": "FROZEN_CONSUMED_DEVELOPMENT_ENGINEERING_ONLY",
        "split": "train_consumed_development",
        "selection": (
            "OPERATOR_STRATIFIED_HASH_ORDERED_EXACT_TEMPORAL_LOCALIZED_"
            "RELATION_QUERY;ONE_ROW_PER_LOCAL_VIDEO;NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "selection_salt": "v59-development",
        "sample_count": len(selected),
        "unique_video_count": len(videos),
        "samples": selected,
        "answer_or_scene_graph_read_during_freeze": False,
        "confirmatory_claim": False,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base = json.loads(BASE_CONFIG.read_text())
    config = {
        "schema_version": "agqa2-temporal-localized-query-config-v59",
        "status": "FROZEN_CONSUMED_DEVELOPMENT_ENGINEERING_ONLY",
        "split": "train_consumed_development",
        "claim_boundary": (
            "ALREADY_CONSUMED_OFFICIAL_TRAIN VIDEOS;ENGINEERING_AND_COST_"
            "PREFLIGHT_ONLY;CANNOT_CONFIRM_TRANSFER_OR_FULL_AGQA"
        ),
        "manifest": str(args.manifest.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(args.manifest),
        "dataset": {
            "archive_path": str(ARCHIVE),
            "archive_sha256": _sha256(ARCHIVE),
            "entry": ENTRY,
        },
        "model": base["model"],
        "anchor_secondary_model": base["query_object_grounder"][
            "secondary_model"
        ] | {"max_operand_tokens": 800, "max_direct_tokens": 80},
        "anchor_tiebreak_model": base["query_object_grounder"]["model"] | {
            "max_operand_tokens": 800,
        },
        "ontology_models": [
            base["query_object_grounder"]["model"],
            base["query_object_grounder"]["secondary_model"],
        ],
        "media": base["media"] | {
            "anchor_secondary_frame_count": 32,
            "anchor_secondary_frame_max_side": 512,
            "anchor_secondary_frames_per_panel": 4,
            "anchor_secondary_panel_frame_width": 224,
            "window_frame_count": 32,
            "window_frame_max_side": 640,
            "window_frames_per_panel": 4,
            "window_panel_frame_width": 256,
        },
        "calibration": {
            "anchor_minimum_confidence": 0.5,
            "anchor_maximum_endpoint_spread": 8,
            "minimum_window_frames": 3,
            "ontology_minimum_confidences": [0.8, 0.8],
            "minimum_neural_votes": 2,
        },
        "qualification_gates": {
            "required_valid_rows": args.sample_count,
            "required_unique_videos": args.sample_count,
            "minimum_candidate_predictions": 1,
            "minimum_wins": 0,
            "maximum_losses": args.sample_count,
            "minimum_net_gain": -args.sample_count,
        },
        "sources": base["sources"],
        "expected_grounder_sha256": None,
    }
    args.config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"],
        "sample_count": len(selected),
        "manifest": str(args.manifest),
        "manifest_file_sha256": _sha256(args.manifest),
        "config": str(args.config),
        "config_file_sha256": _sha256(args.config),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
