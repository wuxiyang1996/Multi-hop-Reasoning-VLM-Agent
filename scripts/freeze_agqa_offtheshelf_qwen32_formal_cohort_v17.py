#!/usr/bin/env python3
"""Select the final video/task-disjoint compositional AGQA reserve."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from scripts.freeze_agqa_layer_b_typed_temporal_replication import ROOT_SIGNATURES, _match


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(task_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--exclusion-ledger", type=Path, required=True)
    parser.add_argument("--semantic-pool", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--per-root", type=int, default=128)
    parser.add_argument("--rank-salt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("V17 formal cohort is immutable")
    protocol = json.loads(args.protocol.read_text())
    ledger = json.loads(args.exclusion_ledger.read_text())
    if protocol.get("status") != "QWEN32_COMPOSITIONAL_FORMAL_PROTOCOL_FROZEN_AFTER_DEVELOPMENT":
        raise ValueError("V17 protocol is not eligible")
    if _sha256(Path(__file__)) != protocol["components"]["cohort_freezer_sha256"]:
        raise ValueError("V17 cohort freezer changed after protocol freeze")
    if protocol.get("exclusion_ledger_file_sha256") != _sha256(args.exclusion_ledger):
        raise ValueError("V17 protocol binds a different exclusion ledger")
    if ledger.get("status") != "ALL_EXISTING_AGQA_COHORT_VIDEOS_FROZEN_AS_EXCLUDED":
        raise ValueError("prior-video exclusion ledger is invalid")
    excluded_videos = {str(value) for value in ledger["excluded_video_ids"]}
    excluded_tasks = {str(value) for value in ledger["excluded_task_ids"]}
    candidates: dict[str, dict[str, tuple[str, dict]]] = defaultdict(dict)
    with args.semantic_pool.open() as handle:
        for line in handle:
            row = json.loads(line)
            task_id, video_id = str(row["task_id"]), str(row["video_id"])
            root = str(row["target"]).split("(", 1)[0]
            if root not in ROOT_SIGNATURES or task_id in excluded_tasks or video_id in excluded_videos:
                continue
            digest = _rank(task_id, args.rank_salt)
            prior = candidates[root].get(video_id)
            if prior is None or digest < prior[0]:
                candidates[root][video_id] = (digest, row)
    if set(candidates) != set(ROOT_SIGNATURES):
        raise RuntimeError("V17 compositional roots are incomplete")
    selected = _match(candidates, args.per_root)
    selected_videos = {str(row[2]["video_id"]) for row in selected}
    if selected_videos & excluded_videos or len(selected_videos) != len(selected):
        raise RuntimeError("V17 selection is not video-disjoint")
    public = {
        "schema_version": "agqa-offtheshelf-qwen32-formal-public-v17",
        "status": "FROZEN_QWEN32_COMPOSITIONAL_FORMAL_V17_BEFORE_RUNTIME_OR_OUTCOME",
        "rows": [{
            "task_id": str(row["task_id"]), "video_id": str(row["video_id"]),
            "question": str(row["input"]).removeprefix("parse AGQA semantics: "),
            "question_sha256": stable_hash(str(row["input"]).removeprefix("parse AGQA semantics: ")),
            "video_path": str(args.video_root / f"{row['video_id']}.mp4"),
            "rank_sha256": digest,
        } for _, digest, row in selected],
        "answers_read": False, "scene_graphs_read": False,
        "functional_program_visible_at_runtime": False,
        "source_controller_read": False, "target_outcome_read": False,
    }
    public["cohort_sha256"] = stable_hash(public)
    private = {
        "schema_version": "agqa-offtheshelf-qwen32-formal-private-v17",
        "public_cohort_sha256": public["cohort_sha256"],
        "rows": [{
            "task_id": str(row["task_id"]), "video_id": str(row["video_id"]),
            "semantic_root": root, "gold_semantic_target_sha256": stable_hash(str(row["target"])),
        } for root, _, row in selected],
        "answers_read": False, "scene_graphs_read": False, "target_outcome_read": False,
    }
    private["audit_sha256"] = stable_hash(private)
    selection_core = {
        "schema_version": "agqa-offtheshelf-qwen32-formal-download-selection-v17b",
        "status": "FROZEN_BEFORE_VIDEO_DOWNLOAD_GROUNDING_OR_OUTCOMES",
        "samples": [dict(row) for row in public["rows"]],
        "public_cohort_sha256": public["cohort_sha256"],
        "protocol_file_sha256": _sha256(args.protocol),
        "exclusion_ledger_sha256": ledger["ledger_sha256"],
        "raw_video_archive": {
            "url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
            "archive_prefix": "Charades_v1_480/",
        },
        "answers_read": False, "scene_graphs_read": False,
        "target_outcome_read": False,
    }
    selection = {
        **selection_core, "manifest_sha256": stable_hash(selection_core),
    }
    manifest = {
        "schema_version": "agqa-offtheshelf-qwen32-formal-freeze-v17",
        "status": "AGQA_QWEN32_COMPOSITIONAL_FRESH_FORMAL_V17_FROZEN",
        "claim_scope": "VIDEO_AND_TASK_DISJOINT_BALANCED_TRAIN_COMPOSITIONAL_RESERVE_NOT_OFFICIAL_TEST",
        "rows": len(selected), "videos": len(selected_videos),
        "per_semantic_root": args.per_root, "semantic_roots": sorted(ROOT_SIGNATURES),
        "rank_salt": args.rank_salt,
        "selection_algorithm": "DETERMINISTIC_BIPARTITE_CAPACITY_MATCHING_V1",
        "protocol_file_sha256": _sha256(args.protocol),
        "exclusion_ledger_file_sha256": _sha256(args.exclusion_ledger),
        "exclusion_ledger_sha256": ledger["ledger_sha256"],
        "excluded_prior_videos": len(excluded_videos),
        "public_cohort_sha256": public["cohort_sha256"],
        "private_audit_sha256": private["audit_sha256"],
        "download_selection_sha256": selection["manifest_sha256"],
        "answers_read": False, "scene_graphs_read": False,
        "target_outcome_read": False, "provider_calls": 0,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    args.output_dir.mkdir(parents=True)
    for name, value in (
        ("public_cohort.json", public),
        ("private_semantic_audit.json", private),
        ("download_selection.json", selection),
        ("manifest.json", manifest),
    ):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
