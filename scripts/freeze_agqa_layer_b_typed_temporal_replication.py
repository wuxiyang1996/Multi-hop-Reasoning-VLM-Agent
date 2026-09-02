#!/usr/bin/env python3
"""Freeze a fresh Layer-B cohort by source-induced typed compatibility only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


ROOT_SIGNATURES = {
    "duration_choice": (
        "EFFECT_RANKING", "ORDERED_ENDPOINTS", "GUARDED_BRANCH",
    ),
    "duration_extremum": (
        "EFFECT_RANKING", "ORDERED_ENDPOINTS",
    ),
}


def _rank(task_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def _source_evidence_classes(source: dict) -> set[str]:
    values = set()
    for capability in source["capabilities"].values():
        if capability.get("authorized"):
            values.update(str(value) for value in capability.get("evidence_classes", ()))
    return values


def _match(candidates: dict[str, dict[str, tuple]], per_root: int) -> list[tuple]:
    slots = [
        (root, index)
        for root in sorted(candidates, key=lambda value: (len(candidates[value]), value))
        for index in range(per_root)
    ]
    choices = {
        root: [video for video, _ in sorted(rows.items(), key=lambda item: item[1][0])]
        for root, rows in candidates.items()
    }
    video_to_slot: dict[str, tuple[str, int]] = {}

    def assign(slot: tuple[str, int], seen: set[str]) -> bool:
        for video in choices[slot[0]]:
            if video in seen:
                continue
            seen.add(video)
            previous = video_to_slot.get(video)
            if previous is None or assign(previous, seen):
                video_to_slot[video] = slot
                return True
        return False

    for slot in slots:
        if not assign(slot, set()):
            raise RuntimeError(f"insufficient fresh videos for typed slot {slot}")
    slot_to_video = {slot: video for video, slot in video_to_slot.items()}
    selected = []
    for slot in sorted(slots):
        root, _ = slot; video = slot_to_video[slot]
        digest, row = candidates[root][video]
        selected.append((root, digest, row))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-pool", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--per-root", type=int, default=128)
    parser.add_argument("--rank-salt", required=True)
    parser.add_argument("--exclude-cohort", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("typed temporal replication freeze is immutable")
    source = json.loads(args.source_capabilities.read_text())
    if source["status"] != "SOURCE_CAPABILITIES_INDUCED":
        raise ValueError("source capabilities are not induced")
    if source["induction_authority"] != "SEALED_SOURCE_INTERVENTION_AND_TRANSITION_ARTIFACTS_ONLY":
        raise ValueError("source induction authority mismatch")
    if source["target_data_read"]:
        raise ValueError("source artifact crossed target boundary")
    evidence = _source_evidence_classes(source)
    missing = {root: sorted(set(signature) - evidence)
               for root, signature in ROOT_SIGNATURES.items()
               if set(signature) - evidence}
    if missing:
        raise ValueError(f"source lacks typed evidence: {missing}")

    excluded_tasks: set[str] = set(); excluded_videos: set[str] = set()
    for path in args.exclude_cohort:
        for row in json.loads(path.read_text())["rows"]:
            excluded_tasks.add(str(row["task_id"])); excluded_videos.add(str(row["video_id"]))
    candidates: dict[str, dict[str, tuple]] = defaultdict(dict)
    with args.semantic_pool.open() as handle:
        for line in handle:
            row = json.loads(line); task_id = str(row["task_id"]); video = str(row["video_id"])
            root = str(row["target"]).split("(", 1)[0]
            if root not in ROOT_SIGNATURES or task_id in excluded_tasks or video in excluded_videos:
                continue
            digest = _rank(task_id, args.rank_salt)
            prior = candidates[root].get(video)
            if prior is None or digest < prior[0]:
                candidates[root][video] = (digest, row)
    if set(candidates) != set(ROOT_SIGNATURES):
        raise RuntimeError("typed temporal candidate roots are incomplete")
    selected = _match(candidates, args.per_root)
    args.output_dir.mkdir(parents=True)
    profile = {
        "schema_version": "agqa-layer-b-source-compatible-profile-v1",
        "status": "SOURCE_ONLY_TYPED_COMPATIBILITY_FROZEN",
        "source_capability_sha256": source["artifact_sha256"],
        "source_induction_authority": source["induction_authority"],
        "required_evidence_by_target_signature": {
            root: list(signature) for root, signature in sorted(ROOT_SIGNATURES.items())
        },
        "selection_uses_target_outcome": False,
        "selection_uses_target_answer": False,
        "semantic_equality_is_not_effect_comparison": True,
    }
    profile["profile_sha256"] = stable_hash(profile)
    public = {
        "schema_version": "agqa-layer-b-typed-temporal-public-v1",
        "status": "FROZEN_BEFORE_VIDEO_DOWNLOAD_PARSER_GROUNDER_OR_OUTCOME",
        "rows": [{
            "task_id": row["task_id"], "video_id": row["video_id"],
            "question": str(row["input"]).removeprefix("parse AGQA semantics: "),
            "question_sha256": stable_hash(str(row["input"]).removeprefix("parse AGQA semantics: ")),
            "video_path": str(args.video_root / f"{row['video_id']}.mp4"),
            "rank_sha256": digest,
        } for _, digest, row in selected],
        "answers_read": False, "scene_graphs_read": False,
        "functional_program_visible_at_runtime": False,
    }
    public["cohort_sha256"] = stable_hash(public)
    private = {
        "schema_version": "agqa-layer-b-typed-temporal-private-audit-v1",
        "public_cohort_sha256": public["cohort_sha256"],
        "rows": [{
            "task_id": row["task_id"], "semantic_root": root,
            "gold_semantic_target_sha256": stable_hash(row["target"]),
        } for root, _, row in selected],
        "answers_read": False, "scene_graphs_read": False,
    }
    private["audit_sha256"] = stable_hash(private)
    selection_core = {
        "schema_version": "agqa-layer-b-typed-temporal-download-selection-v1",
        "status": "FROZEN_V99_QUALIFICATION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS",
        "samples": [dict(row) for row in public["rows"]],
        "public_cohort_sha256": public["cohort_sha256"],
        "source_compatibility_profile_sha256": profile["profile_sha256"],
        "raw_video_archive": {
            "url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
            "archive_prefix": "Charades_v1_480/"
        },
        "answers_read": False, "scene_graphs_read": False,
    }
    selection = selection_core | {"manifest_sha256": stable_hash(selection_core)}
    manifest = {
        "schema_version": "agqa-layer-b-typed-temporal-freeze-v1",
        "status": "FROZEN_BEFORE_VIDEO_DOWNLOAD_PARSER_GROUNDER_OR_OUTCOME",
        "rows": len(selected), "videos": len({row[2]["video_id"] for row in selected}),
        "per_semantic_root": args.per_root,
        "semantic_roots": sorted(ROOT_SIGNATURES),
        "rank_salt": args.rank_salt,
        "selection_algorithm": "DETERMINISTIC_BIPARTITE_CAPACITY_MATCHING_V1",
        "source_compatibility_profile_sha256": profile["profile_sha256"],
        "public_cohort_sha256": public["cohort_sha256"],
        "private_audit_sha256": private["audit_sha256"],
        "download_selection_sha256": selection["manifest_sha256"],
        "excluded_task_count": len(excluded_tasks),
        "excluded_video_count": len(excluded_videos),
        "answers_read": False, "scene_graphs_read": False, "provider_calls": 0,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    for name, value in (
        ("source_compatibility_profile.json", profile),
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
