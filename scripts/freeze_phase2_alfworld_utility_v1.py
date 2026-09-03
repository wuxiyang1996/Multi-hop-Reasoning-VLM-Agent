#!/usr/bin/env python3
"""Freeze an outcome-blind, fresh 32-task ALFWorld causal-utility cohort."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    read_object,
    validate_manifest as validate_phase1_manifest,
)
from motif_transfer.phase2_alfworld_utility_v1 import (  # noqa: E402
    SCHEMA,
    STATUS,
    validate_manifest,
)
from motif_transfer.phase2_webshop_utility_v1 import file_sha256  # noqa: E402
from motif_transfer.webshop_search_automaton_v16 import CONDITIONS  # noqa: E402


OUTPUT = REPO / "configs/phase2_alfworld_utility_v1/manifest.json"
PHASE1_MANIFEST = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
DATA_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/"
    ".cache/alfworld_data"
)
ALFWORLD_CONFIG = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1/"
    "configs/alfworld_base_config.yaml"
)
TARGET_GROUNDER = REPO / "runs/procedural_game_alfworld_v1_development/frozen_candidate_artifact.json"
NAMESPACE = "phase2-alfworld-six-source-causal-utility-v1-valid-seen"
FAMILY_QUOTAS = {
    "look_at_obj_in_light": 6,
    "pick_and_place_simple": 6,
    "pick_clean_then_place_in_recep": 5,
    "pick_cool_then_place_in_recep": 5,
    "pick_heat_then_place_in_recep": 5,
    "pick_two_obj_and_place": 5,
}
TASK_PATTERN = re.compile(
    r"((?:look_at_obj_in_light|pick_and_place_simple|pick_clean_then_place_in_recep|"
    r"pick_cool_then_place_in_recep|pick_heat_then_place_in_recep|"
    r"pick_two_obj_and_place)-[^/\"\s]+/trial_[^/\"\s]+/game\.tw-pddl)"
)


def _historical_outcome_ids() -> set[str]:
    found: set[str] = set()
    roots = (REPO / "runs", REPO / "docs/results")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".log", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found.update(match.group(1) for match in TASK_PATTERN.finditer(text))
    return found


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    phase1 = read_object(PHASE1_MANIFEST)
    validate_phase1_manifest(phase1, repo=REPO)
    source_rows = dict(phase1["sources"])
    if set(source_rows) != set(SOURCE_GAMES):
        raise RuntimeError("Phase-1 source set changed")
    split_root = DATA_ROOT / "json_2.1.1" / "valid_seen"
    all_ids = sorted(
        str(path.relative_to(split_root)) for path in split_root.rglob("game.tw-pddl")
    )
    historical_all = _historical_outcome_ids()
    historical = sorted(set(all_ids).intersection(historical_all))
    eligible = sorted(set(all_ids).difference(historical))
    selected: list[str] = []
    for family, quota in FAMILY_QUOTAS.items():
        family_ids = [task_id for task_id in eligible if task_id.startswith(family + "-")]
        family_ids.sort(key=lambda task_id: stable_hash({
            "namespace": NAMESPACE, "family": family, "target_identity": task_id,
        }))
        if len(family_ids) < quota:
            raise RuntimeError(f"not enough fresh tasks for {family}: {len(family_ids)}")
        selected.extend(family_ids[:quota])
    selected.sort(key=lambda task_id: stable_hash({
        "namespace": NAMESPACE, "selected_target_identity": task_id,
    }))
    tasks = []
    for index, task_id in enumerate(selected):
        game = SOURCE_GAMES[index % len(SOURCE_GAMES)]
        source = source_rows[game]
        task_path = split_root / task_id
        family = next(key for key in FAMILY_QUOTAS if task_id.startswith(key + "-"))
        tasks.append({
            "target_identity": task_id,
            "target_file_sha256": file_sha256(task_path),
            "task_family": family,
            "selected_target_previously_executed": False,
            "selection_rank_sha256": stable_hash({
                "namespace": NAMESPACE, "family": family, "target_identity": task_id,
            }),
            "source_game": game,
            "source_artifact": source["artifact"],
            "source_artifact_sha256": source["artifact_sha256"],
            "source_artifact_file_sha256": file_sha256(REPO / source["artifact"]),
        })
    runtime_files = (
        "src/motif_transfer/phase2_alfworld_utility_v1.py",
        "src/motif_transfer/phase2_webshop_utility_v1.py",
        "src/motif_transfer/alfworld_env.py",
        "src/motif_transfer/alfworld_hierarchical_grounder.py",
        "src/motif_transfer/alfworld_search_automaton_v16.py",
        "src/motif_transfer/active_video_transfer.py",
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/direct_prospective_matrix_v1.py",
        "src/motif_transfer/search_automaton_transfer_v16.py",
        "src/motif_transfer/sokoban_search_automaton_v16.py",
        "scripts/freeze_phase2_alfworld_utility_v1.py",
        "scripts/run_phase2_alfworld_utility_v1.py",
        "scripts/run_alfworld_search_automaton_v16.py",
        "scripts/verify_phase2_alfworld_utility_v1.py",
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": (
            "Fresh valid_seen aggregate test of the shared policy instantiated from "
            "six independent Phase-1 game lineages; no per-game powered claim."
        ),
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "source_assignment_read_target_semantics": False,
        "source_assignment_rule": "SOURCE_GAMES order round-robin after hash ordering, before outcomes",
        "selection_namespace": NAMESPACE,
        "target_split": "eval_in_distribution",
        "seed": 88601,
        "max_steps": 70,
        "conditions": list(CONDITIONS),
        "family_quotas": FAMILY_QUOTAS,
        "dataset_task_count": len(all_ids),
        "historical_outcome_task_ids": historical,
        "eligible_task_count_before_selection": len(eligible),
        "tasks": tasks,
        "alfworld_data_root": str(DATA_ROOT),
        "alfworld_config": str(ALFWORLD_CONFIG),
        "alfworld_config_file_sha256": file_sha256(ALFWORLD_CONFIG),
        "target_grounder": str(TARGET_GROUNDER.relative_to(REPO)),
        "target_grounder_file_sha256": file_sha256(TARGET_GROUNDER),
        "parent_phase1_manifest": str(PHASE1_MANIFEST.relative_to(REPO)),
        "parent_phase1_manifest_file_sha256": file_sha256(PHASE1_MANIFEST),
        "runtime_file_sha256": {
            relative: file_sha256(REPO / relative) for relative in runtime_files
        },
        "formal_protocol": {
            "matrix": "32 target tasks x 5 matched conditions",
            "target_resets": 160,
            "success_measure": "ALFWorld official won signal only",
            "write_once_receipts": True,
            "no_result_dependent_retry": True,
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_once(OUTPUT, manifest)
    validate_manifest(manifest, repo=REPO)
    print(json.dumps({
        "status": manifest["status"],
        "tasks": len(tasks),
        "dataset_tasks": len(all_ids),
        "historical_outcome_tasks": len(historical),
        "eligible_tasks": len(eligible),
        "family_counts": dict(sorted({key: sum(row["task_family"] == key for row in tasks) for key in FAMILY_QUOTAS}.items())),
        "source_counts": dict(sorted({game: sum(row["source_game"] == game for row in tasks) for game in SOURCE_GAMES}.items())),
        "manifest_sha256": manifest["manifest_sha256"],
        "output": str(OUTPUT.relative_to(REPO)),
    }, indent=2))


if __name__ == "__main__":
    main()
