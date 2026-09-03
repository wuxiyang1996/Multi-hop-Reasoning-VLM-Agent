#!/usr/bin/env python3
"""Audit real answer/program-blind oracle event graphs for both video cohorts.

This is a grounding-availability and authority-boundary audit, not a QA score.
It reads only frozen task manifests and official simulator/STSG artifacts.  It
never opens benchmark QA answers or functional programs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.official_video_event_graph import (  # noqa: E402
    OfficialEventGraphArtifact,
    load_builtin_only_pickle,
    load_clevrer_official_event_graph,
    normalize_agqa_stsg,
    sha256_file,
)


def _clevrer_ids(path: Path) -> list[str]:
    split = json.loads(path.read_text())["benchmarks"]["clevrer"]["family_roles"]
    return [task_id for family in sorted(split) for task_id in split[family]["reserve"]]


def _agqa_rows(path: Path) -> list[dict[str, str]]:
    manifest = json.loads(path.read_text())
    # Deliberately project only public identities.  Program hashes in the
    # manifest are not read into a controller state.
    return [{"task_id": str(row["task_id"]), "video_id": str(row["video_id"])}
            for row in manifest["samples"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clevrer-split", type=Path,
        default=REPO / "configs/clevrer_sokoban_proof_v14_splits.json",
    )
    parser.add_argument(
        "--clevrer-annotations", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official/"
                     "official_annotations/annotation_validation.zip"),
    )
    parser.add_argument(
        "--agqa-manifest", type=Path,
        default=REPO / "configs/agqa2_full_distribution_v62_manifest.json",
    )
    parser.add_argument(
        "--agqa-stsg", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/"
                     "scene_graphs/AGQA_scene_graphs/AGQA_train_stsgs.pkl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/official_video_event_graph_v1_audit.json",
    )
    args = parser.parse_args()

    clevrer_source_sha = sha256_file(args.clevrer_annotations)
    clevrer_tasks = _clevrer_ids(args.clevrer_split)
    clevrer_graphs: dict[int, dict] = {}
    clevrer_receipts = []
    for task_id in clevrer_tasks:
        match = re.fullmatch(r"video_(\d+)\.mp4\.Q\d+", task_id)
        if match is None:
            raise ValueError(f"invalid CLEVRER task identity: {task_id}")
        scene = int(match.group(1))
        if scene not in clevrer_graphs:
            clevrer_graphs[scene] = load_clevrer_official_event_graph(
                args.clevrer_annotations, scene
            )
        graph = clevrer_graphs[scene]
        artifact = OfficialEventGraphArtifact.create(
            benchmark="clevrer", task_id=task_id, split="reserve",
            graph=graph, source_artifact_sha256=clevrer_source_sha,
        )
        clevrer_receipts.append(artifact.shared_receipt())

    agqa_source_sha = sha256_file(args.agqa_stsg)
    agqa_rows = _agqa_rows(args.agqa_manifest)
    corpus = load_builtin_only_pickle(args.agqa_stsg)
    agqa_graphs = {
        video_id: normalize_agqa_stsg(video_id, corpus[video_id])
        for video_id in sorted({row["video_id"] for row in agqa_rows})
    }
    agqa_receipts = []
    for row in agqa_rows:
        artifact = OfficialEventGraphArtifact.create(
            benchmark="agqa2", task_id=row["task_id"], split="test",
            graph=agqa_graphs[row["video_id"]],
            source_artifact_sha256=agqa_source_sha,
        )
        agqa_receipts.append(artifact.shared_receipt())

    all_receipts = clevrer_receipts + agqa_receipts
    gates = {
        "all_frozen_tasks_have_official_event_graph": (
            len(clevrer_receipts) == 360 and len(agqa_receipts) == 900
        ),
        "all_receipts_are_answer_program_outcome_blind": all(
            not row.functional_program_read
            and not row.gold_answer_read
            and not row.target_outcome_read
            for row in all_receipts
        ),
        "all_receipts_disclose_oracle_graph_access": all(
            row.official_scene_graph_read for row in all_receipts
        ),
        "zero_tool_budget_in_static_oracle_track": all(
            row.tool_budget.max_tool_calls == 0
            and row.tool_budget.max_frames == 0
            and row.tool_budget.max_provider_calls == 0
            for row in all_receipts
        ),
        "task_receipts_are_unique": len({row.receipt_sha256 for row in all_receipts})
        == len(all_receipts),
    }
    body = {
        "schema_version": "official-video-event-graph-audit-v1",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "fresh_success_evidence": False,
        "claim": (
            "Grounding availability and authority-boundary audit only; no QA "
            "answer or transferred-controller success is measured here."
        ),
        "clevrer": {
            "tasks": len(clevrer_receipts), "unique_scenes": len(clevrer_graphs),
            "official_archive_sha256": clevrer_source_sha,
            "schemas": sorted({g["schema"] for g in clevrer_graphs.values()}),
            "receipt_set_sha256": stable_hash(
                [row.receipt_sha256 for row in clevrer_receipts]
            ),
        },
        "agqa2": {
            "tasks": len(agqa_receipts), "unique_videos": len(agqa_graphs),
            "official_stsg_sha256": agqa_source_sha,
            "schemas": sorted({g["schema"] for g in agqa_graphs.values()}),
            "receipt_set_sha256": stable_hash(
                [row.receipt_sha256 for row in agqa_receipts]
            ),
            "version_note": (
                "AGQA 2.0 QA manifest joined to the official supporting AGQA "
                "STSG release, as specified by the benchmark and STAIR setup."
            ),
        },
        "gates": gates,
        "runtime_authority": {
            "qa_answer_file_opened": False,
            "functional_program_opened": False,
            "program_derived_sg_grounding_opened": False,
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
