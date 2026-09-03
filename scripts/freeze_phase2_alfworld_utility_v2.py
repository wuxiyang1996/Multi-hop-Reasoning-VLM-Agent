#!/usr/bin/env python3
"""Freeze fresh V2 tasks after diagnosing V1 batch-order ambiguity."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES  # noqa: E402
from motif_transfer.phase2_alfworld_utility_v1 import validate_manifest as validate_v1  # noqa: E402
from motif_transfer.phase2_alfworld_utility_v2 import SCHEMA, STATUS, validate_manifest  # noqa: E402
from motif_transfer.phase2_webshop_utility_v1 import file_sha256, validate_self_hash  # noqa: E402
from motif_transfer.webshop_search_automaton_v16 import CONDITIONS  # noqa: E402
from scripts.freeze_phase2_alfworld_utility_v1 import (  # noqa: E402
    ALFWORLD_CONFIG, DATA_ROOT, FAMILY_QUOTAS, PHASE1_MANIFEST, TARGET_GROUNDER,
    _write_once,
)


OUTPUT = REPO / "configs/phase2_alfworld_utility_v2/manifest.json"
V1_MANIFEST = REPO / "configs/phase2_alfworld_utility_v1/manifest.json"
V1_PREFLIGHT = REPO / "docs/results/phase2_alfworld_utility_v1_preflight.json"
NAMESPACE = "phase2-alfworld-six-source-causal-utility-v2-single-task-env"


def main() -> None:
    v1 = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    validate_v1(v1, repo=REPO)
    failed = json.loads(V1_PREFLIGHT.read_text(encoding="utf-8"))
    validate_self_hash(failed, "preflight_sha256")
    expected = {str(row["target_identity"]) for row in v1["tasks"]}
    actual = {str(row["actual_target_identity"]) for row in failed["rows"]}
    if not (
        failed.get("status") == "PHASE2_ALFWORLD_PREFLIGHT_FAILED"
        and failed.get("gates", {}).get("frozen_task_order_matches_runtime") is False
        and failed.get("gates", {}).get("zero_actions") is True
        and failed.get("gates", {}).get("zero_outcomes_read") is True
        and expected == actual
    ):
        raise RuntimeError("V1 did not exhibit isolated batch-order ambiguity")
    phase1 = json.loads(PHASE1_MANIFEST.read_text(encoding="utf-8"))
    sources = dict(phase1["sources"])
    split_root = DATA_ROOT / "json_2.1.1" / "valid_seen"
    all_ids = sorted(str(path.relative_to(split_root)) for path in split_root.rglob("game.tw-pddl"))
    # Use V1's pre-reset historical snapshot.  The failed V1 preflight itself
    # contains reset observations but explicitly read no outcomes.
    historical = sorted(map(str, v1["historical_outcome_task_ids"]))
    excluded = sorted(expected)
    eligible = sorted(set(all_ids).difference(historical).difference(excluded))
    selected = []
    for family, quota in FAMILY_QUOTAS.items():
        rows = [task_id for task_id in eligible if task_id.startswith(family + "-")]
        rows.sort(key=lambda task_id: stable_hash({
            "namespace": NAMESPACE, "family": family, "target_identity": task_id,
        }))
        if len(rows) < quota:
            raise RuntimeError(f"not enough V2 tasks for {family}")
        selected.extend(rows[:quota])
    selected.sort(key=lambda task_id: stable_hash({"namespace": NAMESPACE, "selected": task_id}))
    tasks = []
    for index, task_id in enumerate(selected):
        game = SOURCE_GAMES[index % len(SOURCE_GAMES)]
        source = sources[game]
        family = next(key for key in FAMILY_QUOTAS if task_id.startswith(key + "-"))
        tasks.append({
            "target_identity": task_id,
            "target_file_sha256": file_sha256(split_root / task_id),
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
        "src/motif_transfer/phase2_alfworld_utility_v2.py",
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
        "scripts/freeze_phase2_alfworld_utility_v2.py",
        "scripts/run_phase2_alfworld_utility_v2.py",
        "scripts/run_alfworld_search_automaton_v16.py",
        "scripts/verify_phase2_alfworld_utility_v2.py",
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": "Fresh valid_seen aggregate six-lineage causal utility; no per-game powered claim.",
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "source_assignment_read_target_semantics": False,
        "source_assignment_rule": "SOURCE_GAMES order round-robin before V2 outcomes",
        "selection_namespace": NAMESPACE,
        "target_split": "eval_in_distribution",
        "environment_concurrency_policy": "one_task_per_environment",
        "python_executable": str(Path(sys.executable).resolve()),
        "seed": 88601,
        "max_steps": 70,
        "conditions": list(CONDITIONS),
        "family_quotas": FAMILY_QUOTAS,
        "dataset_task_count": len(all_ids),
        "historical_outcome_task_ids": historical,
        "excluded_v1_reset_task_ids": excluded,
        "eligible_task_count_before_selection": len(eligible),
        "tasks": tasks,
        "alfworld_data_root": str(DATA_ROOT),
        "alfworld_config": str(ALFWORLD_CONFIG),
        "alfworld_config_file_sha256": file_sha256(ALFWORLD_CONFIG),
        "target_grounder": str(TARGET_GROUNDER.relative_to(REPO)),
        "target_grounder_file_sha256": file_sha256(TARGET_GROUNDER),
        "parent_phase1_manifest": str(PHASE1_MANIFEST.relative_to(REPO)),
        "parent_phase1_manifest_file_sha256": file_sha256(PHASE1_MANIFEST),
        "v1_manifest": str(V1_MANIFEST.relative_to(REPO)),
        "v1_manifest_file_sha256": file_sha256(V1_MANIFEST),
        "v1_failed_preflight": str(V1_PREFLIGHT.relative_to(REPO)),
        "v1_failed_preflight_file_sha256": file_sha256(V1_PREFLIGHT),
        "v1_failed_preflight_sha256": failed["preflight_sha256"],
        "runtime_file_sha256": {relative: file_sha256(REPO / relative) for relative in runtime_files},
        "formal_protocol": {
            "matrix": "32 target tasks x 5 matched conditions", "target_resets": 160,
            "success_measure": "ALFWorld official won signal only",
            "write_once_receipts": True, "no_result_dependent_retry": True,
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_once(OUTPUT, manifest)
    validate_manifest(manifest, repo=REPO)
    print(json.dumps({
        "status": manifest["status"], "tasks": len(tasks),
        "historical_outcomes": len(historical), "excluded_v1_resets": len(excluded),
        "eligible": len(eligible),
        "source_counts": {game: sum(row["source_game"] == game for row in tasks) for game in SOURCE_GAMES},
        "manifest_sha256": manifest["manifest_sha256"], "output": str(OUTPUT.relative_to(REPO)),
    }, indent=2))


if __name__ == "__main__":
    main()
