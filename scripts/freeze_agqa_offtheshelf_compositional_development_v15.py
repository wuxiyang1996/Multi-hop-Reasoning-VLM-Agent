#!/usr/bin/env python3
"""Freeze compositional AGQA development tasks on already-consumed V13 videos.

The selector may inspect parser-supervision semantics to stratify question
families, but it never reads an answer, target outcome, official STSG, or any
runtime functional program.  Because the parent videos have already been used
for V13 qualification, this cohort is development evidence only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_authority import cohort_crossed_authority
from motif_transfer.contracts import stable_hash
from scripts.freeze_agqa_layer_b_typed_temporal_replication import (
    ROOT_SIGNATURES,
    _match,
    _source_evidence_classes,
)


def _rank(task_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-pool", type=Path, required=True)
    parser.add_argument("--allowed-cohort", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--per-root", type=int, default=128)
    parser.add_argument("--rank-salt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("compositional development freeze is immutable")
    if args.per_root < 1:
        raise ValueError("per-root must be positive")

    allowed = json.loads(args.allowed_cohort.read_text())
    source = json.loads(args.source_capabilities.read_text())
    if cohort_crossed_authority(allowed):
        raise ValueError("allowed parent cohort crossed the runtime authority boundary")
    if source.get("status") != "SOURCE_CAPABILITIES_INDUCED" or source.get(
        "target_data_read"
    ) is not False:
        raise ValueError("source capability artifact is invalid")
    evidence = _source_evidence_classes(source)
    missing = {
        root: sorted(set(signature) - evidence)
        for root, signature in ROOT_SIGNATURES.items()
        if set(signature) - evidence
    }
    if missing:
        raise ValueError(f"source lacks compositional evidence: {missing}")

    video_paths: dict[str, str] = {}
    parent_tasks = set()
    for row in allowed["rows"]:
        video_id = str(row["video_id"])
        video_paths[video_id] = str(row["video_path"])
        parent_tasks.add(str(row["task_id"]))
    if not video_paths or any(not Path(path).exists() for path in video_paths.values()):
        raise ValueError("allowed V13 videos are not available locally")

    candidates: dict[str, dict[str, tuple[str, dict]]] = defaultdict(dict)
    with args.semantic_pool.open() as handle:
        for line in handle:
            row = json.loads(line)
            task_id, video_id = str(row["task_id"]), str(row["video_id"])
            root = str(row["target"]).split("(", 1)[0]
            if (
                root not in ROOT_SIGNATURES
                or video_id not in video_paths
                or task_id in parent_tasks
            ):
                continue
            digest = _rank(task_id, args.rank_salt)
            previous = candidates[root].get(video_id)
            if previous is None or digest < previous[0]:
                candidates[root][video_id] = (digest, row)
    if set(candidates) != set(ROOT_SIGNATURES):
        raise RuntimeError("development candidate roots are incomplete")
    selected = _match(candidates, args.per_root)

    public = {
        "schema_version": "agqa-offtheshelf-compositional-development-public-v1",
        "status": "CONSUMED_V13_VIDEO_DEVELOPMENT_COHORT_FROZEN_BEFORE_NEW_TASK_OUTCOMES",
        "rows": [
            {
                "task_id": str(row["task_id"]),
                "video_id": str(row["video_id"]),
                "question": str(row["input"]).removeprefix("parse AGQA semantics: "),
                "question_sha256": stable_hash(
                    str(row["input"]).removeprefix("parse AGQA semantics: ")
                ),
                "video_path": video_paths[str(row["video_id"])],
                "rank_sha256": digest,
            }
            for _, digest, row in selected
        ],
        "answers_read": False,
        "scene_graphs_read": False,
        "functional_program_visible_at_runtime": False,
        "answers_projected": False,
        "functional_programs_projected": False,
        "scene_graph_grounding_projected": False,
        "source_controller_read": False,
        "target_outcome_read": False,
    }
    public["cohort_sha256"] = stable_hash(public)
    private = {
        "schema_version": "agqa-offtheshelf-compositional-development-private-v1",
        "public_cohort_sha256": public["cohort_sha256"],
        "rows": [
            {
                "task_id": str(row["task_id"]),
                "video_id": str(row["video_id"]),
                "semantic_root": root,
                "gold_semantic_target_sha256": stable_hash(str(row["target"])),
            }
            for root, _, row in selected
        ],
        "answers_read": False,
        "scene_graphs_read": False,
        "target_outcome_read": False,
    }
    private["audit_sha256"] = stable_hash(private)
    manifest = {
        "schema_version": "agqa-offtheshelf-compositional-development-freeze-v1",
        "status": "DEVELOPMENT_ONLY_FROZEN_ON_CONSUMED_V13_VIDEOS",
        "claim_scope": "DEVELOPMENT_ONLY_NOT_TRANSFER_EVIDENCE",
        "rows": len(selected),
        "videos": len({str(row[2]["video_id"]) for row in selected}),
        "per_semantic_root": args.per_root,
        "semantic_roots": sorted(ROOT_SIGNATURES),
        "rank_salt": args.rank_salt,
        "selection_algorithm": "DETERMINISTIC_BIPARTITE_CAPACITY_MATCHING_V1",
        "parent_v13_cohort_sha256": allowed["cohort_sha256"],
        "parent_videos_already_consumed_for_grounding_qualification": True,
        "new_task_outcomes_read": False,
        "official_stsg_read": False,
        "runtime_functional_program_projected": False,
        "source_capability_sha256": source["artifact_sha256"],
        "public_cohort_sha256": public["cohort_sha256"],
        "private_audit_sha256": private["audit_sha256"],
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    args.output_dir.mkdir(parents=True)
    for name, value in (
        ("public_cohort.json", public),
        ("private_semantic_audit.json", private),
        ("manifest.json", manifest),
    ):
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
