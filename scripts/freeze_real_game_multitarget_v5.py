#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.real_game_multitarget_manifest import (  # noqa: E402
    file_sha256,
    freeze_partition,
    freeze_round_robin_partition,
    stable_hash,
)


LEGACY_SOURCE_GAMES = (
    "candy_crush",
    "sokoban",
    "super_mario",
    "tetris",
    "twenty_forty_eight",
)
TIR_SPATIAL_RELATIONAL_FAMILIES = (
    "jigsaw",
    "maze",
    "refcoco",
    "rotation_game",
    "spot_difference",
    "visual_search",
    "word_search",
)


def _legacy_source_manifest(source_root: Path) -> dict:
    games = {}
    artifacts = {}
    for game in LEGACY_SOURCE_GAMES:
        paths = sorted((source_root / game).glob("episode_*.json"))
        if not paths:
            raise FileNotFoundError(f"no source episodes found for {game}: {source_root / game}")
        ids = [f"{game}/{path.name}" for path in paths]
        discovery = len(ids) * 3 // 5
        qualification = len(ids) // 5
        held_out = len(ids) - discovery - qualification
        games[game] = freeze_partition(
            ids,
            namespace=f"real-game-multitarget-v5:source:{game}",
            role_counts={
                "discovery": discovery,
                "qualification": qualification,
                "held_out": held_out,
            },
        )
        for item_id, path in zip(ids, paths, strict=True):
            artifacts[item_id] = {
                "path": str(path),
                "sha256": file_sha256(path),
            }
    return {
        "root": str(source_root),
        "games": games,
        "artifacts": artifacts,
        "compiler_field_allowlist": [
            "state",
            "action",
            "reward",
            "next_state",
            "done",
            "available_actions",
            "idx",
        ],
        "explicitly_forbidden_fields": [
            "intentions",
            "skills",
            "skill_candidates",
            "skill_chosen_idx",
            "skill_reasoning",
            "summary",
            "summary_state",
        ],
    }


def _thunder_manifest(evidence_root: Path) -> dict:
    episodes_path = evidence_root / "episodes.jsonl"
    manifest_path = evidence_root / "manifest.json"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    ids = [str(row["episode_id"]) for row in rows]
    if len(ids) != 12 or len(ids) != len(set(ids)):
        raise ValueError("Thunder source must contain exactly 12 unique episodes")
    partition = freeze_partition(
        ids,
        namespace="real-game-multitarget-v5:source:gymv_thunder_force_iii",
        role_counts={"discovery": 4, "qualification": 4, "held_out": 4},
    )
    return {
        "root": str(evidence_root),
        "partition": partition,
        "files": {
            name: file_sha256(evidence_root / name)
            for name in (
                "episodes.jsonl",
                "events.jsonl",
                "matched_policy_records.jsonl",
                "matched_policy_replays.jsonl",
                "replay_receipts.jsonl",
                "manifest.json",
            )
        },
        "upstream_manifest_sha256": file_sha256(manifest_path),
        "event_kind_allowlist": ["RESET", "OBSERVATION", "ACTION", "TRANSITION"],
        "forbidden_semantic_fields": [
            "selected_skill_id",
            "selected_skill_name",
            "reasoning",
            "intentions",
        ],
    }


def _tir_target(dataset_root: Path) -> dict:
    annotation = dataset_root / "TIR-Bench.json"
    rows = json.loads(annotation.read_text())
    eligible = [
        row for row in rows if str(row.get("task")) in TIR_SPATIAL_RELATIONAL_FAMILIES
    ]
    grouped_ids = {
        family: [str(row["id"]) for row in eligible if str(row.get("task")) == family]
        for family in TIR_SPATIAL_RELATIONAL_FAMILIES
    }
    family_counts = {
        family: sum(str(row.get("task")) == family for row in eligible)
        for family in TIR_SPATIAL_RELATIONAL_FAMILIES
    }
    return {
        "dataset_sha256": file_sha256(annotation),
        "split_contract": (
            "public test spatial/relational family allowlist then internal holdout; "
            "family and IDs only used for selection"
        ),
        "task_family_allowlist": list(TIR_SPATIAL_RELATIONAL_FAMILIES),
        "eligible_family_counts": family_counts,
        "partition": freeze_round_robin_partition(
            grouped_ids,
            namespace="real-game-multitarget-v5:target:tir-bench",
            role_counts={"adaptation": 8, "qualification": 8, "held_out": 24},
            excluded_ids=("668", "1001"),
        ),
        "prior_inspected_exclusions": ["668", "1001"],
    }


def _video_target(dataset_root: Path) -> dict:
    benchmark = dataset_root / "Benchmark"
    train_path = benchmark / "train_Video-Holmes.json"
    test_path = benchmark / "test_Video-Holmes.json"
    train_rows = json.loads(train_path.read_text())
    test_rows = json.loads(test_path.read_text())

    def sample_id(row: dict) -> str:
        return f"{row['video ID']}.Q{row['Question ID']}"

    return {
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
        "split_contract": "official train adaptation; official test qualification/held-out",
        "adaptation_partition": freeze_partition(
            [sample_id(row) for row in train_rows],
            namespace="real-game-multitarget-v5:target:video-holmes:train",
            role_counts={"adaptation": 8},
        ),
        "evaluation_partition": freeze_partition(
            [sample_id(row) for row in test_rows],
            namespace="real-game-multitarget-v5:target:video-holmes:test",
            role_counts={"qualification": 8, "held_out": 24},
            excluded_ids=("nT7w-T2aBOo.Q770",),
        ),
        "prior_inspected_exclusions": ["nT7w-T2aBOo.Q770"],
    }


def _webshop_target(browser_root: Path) -> dict:
    roots = sorted(browser_root.glob("webshop_50task_*"))
    ids = sorted({path.name for root in roots for path in root.glob("webshop.*") if path.is_dir()})
    if len(ids) != 50:
        raise ValueError(f"expected 50 WebShop task IDs, found {len(ids)}")
    return {
        "historical_rollout_roots": [str(root) for root in roots],
        "split_contract": "task-ID internal holdout; target adaptation may read only adaptation IDs",
        "partition": freeze_partition(
            ids,
            namespace="real-game-multitarget-v5:target:webshop",
            role_counts={"adaptation": 8, "qualification": 8, "held_out": 24},
            excluded_ids=("webshop.32", "webshop.40"),
        ),
        "prior_inspected_exclusions": ["webshop.32", "webshop.40"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-source-root",
        type=Path,
        default=WORKSPACE / "Multi-hop-Reasoning-VLM-Agent-github-main/labeling/gpt54_skill_labeled",
    )
    parser.add_argument(
        "--thunder-evidence-root",
        type=Path,
        default=REPO / "runs/fresh_source_execution_motif_v1/gymv_thunder_force_iii/evidence",
    )
    parser.add_argument("--datasets-root", type=Path, default=WORKSPACE / "datasets")
    parser.add_argument(
        "--browser-root",
        type=Path,
        default=WORKSPACE / "emnlp2026_download/workspace/main_project/Cold-start-out-browsergym",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "configs/real_game_multitarget_v5_manifest.json",
    )
    args = parser.parse_args()

    payload = {
        "schema_version": 1,
        "experiment": "real_game_multitarget_neurosymbolic_v5",
        "frozen_before_new_target_outcomes": True,
        "source": {
            "legacy_real_game_rollouts": _legacy_source_manifest(args.legacy_source_root),
            "fresh_thunder_causal_rollouts": _thunder_manifest(args.thunder_evidence_root),
        },
        "targets": {
            "tir_bench": _tir_target(args.datasets_root / "TIR-Bench"),
            "video_holmes": _video_target(args.datasets_root / "Video-Holmes"),
            "webshop": _webshop_target(args.browser_root),
        },
        "leakage_contract": {
            "source_candidate_discovery_reads_source_discovery_only": True,
            "source_qualification_cannot_update_candidate": True,
            "source_held_out_is_final_gate": True,
            "target_grounder_reads_target_adaptation_only": True,
            "target_qualification_cannot_update_weights": True,
            "target_held_out_is_unread_until_freeze": True,
        },
    }
    payload["manifest_sha256"] = stable_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "manifest_sha256": payload["manifest_sha256"],
        "source_games": list(payload["source"]["legacy_real_game_rollouts"]["games"]),
        "targets": list(payload["targets"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
