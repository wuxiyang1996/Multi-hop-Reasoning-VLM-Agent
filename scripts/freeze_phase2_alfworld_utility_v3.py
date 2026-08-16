#!/usr/bin/env python3
"""Freeze every remaining ALFWorld task for an arity-selective V3 confirmation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES  # noqa: E402
from motif_transfer.phase2_alfworld_utility_v2 import (  # noqa: E402
    build_report as build_v2_report,
    validate_manifest as validate_v2_manifest,
)
from motif_transfer.phase2_alfworld_utility_v3 import (  # noqa: E402
    SCHEMA, STATUS, UNSUPPORTED_FAMILY, validate_manifest,
)
from motif_transfer.phase2_webshop_utility_v1 import file_sha256, validate_self_hash  # noqa: E402
from motif_transfer.webshop_search_automaton_v16 import AUTHENTIC, CONDITIONS, RAW  # noqa: E402


OUTPUT = REPO / "configs/phase2_alfworld_utility_v3/manifest.json"
V1_MANIFEST = REPO / "configs/phase2_alfworld_utility_v1/manifest.json"
V2_MANIFEST = REPO / "configs/phase2_alfworld_utility_v2/manifest.json"
V2_REPORT = REPO / "runs/phase2_alfworld_utility_v2/report.json"
PHASE1_MANIFEST = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
TARGET_GROUNDER = REPO / "runs/procedural_game_alfworld_v1_development/frozen_candidate_artifact.json"
DATA_ROOT = Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data")
ALFWORLD_CONFIG = Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1/configs/alfworld_base_config.yaml")
NAMESPACE = "phase2-alfworld-selective-utility-v3-all-remaining-valid-seen"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    v1, v2 = _read(V1_MANIFEST), _read(V2_MANIFEST)
    validate_v2_manifest(v2, repo=REPO)
    saved_v2 = _read(V2_REPORT)
    validate_self_hash(saved_v2, "report_sha256")
    receipts = [_read(path) for path in sorted((V2_REPORT.parent / "receipts").glob("*.json"))]
    if build_v2_report(v2, receipts) != saved_v2:
        raise RuntimeError("V2 report does not independently rebuild")
    by = {(row["target_identity"], row["condition"]): row for row in receipts}
    losses = []
    for task in v2["tasks"]:
        identity = task["target_identity"]
        if by[identity, RAW]["strict_success"] and not by[identity, AUTHENTIC]["strict_success"]:
            losses.append({"target_identity": identity, "task_family": task["task_family"]})
    unsupported_losses = sum(row["task_family"] == UNSUPPORTED_FAMILY for row in losses)
    if len(losses) != 4 or unsupported_losses != 3:
        raise RuntimeError("frozen V2 arity diagnosis changed")

    phase1 = _read(PHASE1_MANIFEST)
    sources = dict(phase1["sources"])
    split_root = DATA_ROOT / "json_2.1.1" / "valid_seen"
    all_ids = sorted(str(path.relative_to(split_root)) for path in split_root.rglob("game.tw-pddl"))
    excluded = sorted(
        set(v1["historical_outcome_task_ids"])
        | {row["target_identity"] for row in v1["tasks"]}
        | {row["target_identity"] for row in v2["tasks"]}
    )
    selected = sorted(
        set(all_ids).difference(excluded),
        key=lambda identity: stable_hash({"namespace": NAMESPACE, "target_identity": identity}),
    )
    if len(selected) != 75:
        raise RuntimeError(f"expected every remaining 75 tasks, got {len(selected)}")
    tasks = []
    for index, identity in enumerate(selected):
        family = identity.split("-", 1)[0]
        game = SOURCE_GAMES[index % len(SOURCE_GAMES)]
        source = sources[game]
        tasks.append({
            "target_identity": identity,
            "target_file_sha256": file_sha256(split_root / identity),
            "task_family": family,
            "selected_target_previously_executed": False,
            "transfer_applicable": family != UNSUPPORTED_FAMILY,
            "source_game": game,
            "source_artifact": source["artifact"],
            "source_artifact_sha256": source["artifact_sha256"],
            "source_artifact_file_sha256": file_sha256(REPO / source["artifact"]),
        })
    runtime_files = (
        "src/motif_transfer/phase2_alfworld_utility_v3.py",
        "src/motif_transfer/phase2_webshop_utility_v1.py",
        "src/motif_transfer/alfworld_env.py",
        "src/motif_transfer/alfworld_hierarchical_grounder.py",
        "src/motif_transfer/alfworld_search_automaton_v16.py",
        "src/motif_transfer/active_video_transfer.py",
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/direct_prospective_matrix_v1.py",
        "src/motif_transfer/search_automaton_transfer_v16.py",
        "src/motif_transfer/sokoban_search_automaton_v16.py",
        "scripts/freeze_phase2_alfworld_utility_v3.py",
        "scripts/run_phase2_alfworld_utility_v3.py",
        "scripts/run_alfworld_search_automaton_v16.py",
        "scripts/verify_phase2_alfworld_utility_v3.py",
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "selection_rule": "all remaining valid_seen identities after V1 reset and V2 outcome cohorts",
        "selection_namespace": NAMESPACE,
        "target_split": "eval_in_distribution",
        "environment_concurrency_policy": "one_task_per_environment",
        "python_executable": str(Path(sys.executable).resolve()),
        "seed": 88601,
        "max_steps": 70,
        "conditions": list(CONDITIONS),
        "formal_task_count": len(tasks),
        "dataset_task_count": len(all_ids),
        "excluded_prior_task_ids": excluded,
        "tasks": tasks,
        "transfer_applicability": {
            "criterion": "target_task_arity_equals_one",
            "abstain_family": UNSUPPORTED_FAMILY,
            "out_of_scope_behavior": "execute_matched_raw_target_only_policy",
            "development_evidence": {"v2_losses": 4, "v2_unsupported_family_losses": 3},
        },
        "gates": {"maximum_discordant_loss_rate": 0.25},
        "alfworld_data_root": str(DATA_ROOT),
        "alfworld_config": str(ALFWORLD_CONFIG),
        "alfworld_config_file_sha256": file_sha256(ALFWORLD_CONFIG),
        "target_grounder": str(TARGET_GROUNDER.relative_to(REPO)),
        "target_grounder_file_sha256": file_sha256(TARGET_GROUNDER),
        "v2_manifest": str(V2_MANIFEST.relative_to(REPO)),
        "v2_manifest_file_sha256": file_sha256(V2_MANIFEST),
        "v2_report": str(V2_REPORT.relative_to(REPO)),
        "v2_report_file_sha256": file_sha256(V2_REPORT),
        "runtime_file_sha256": {relative: file_sha256(REPO / relative) for relative in runtime_files},
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_once(OUTPUT, manifest)
    validate_manifest(manifest, repo=REPO)
    print(json.dumps({
        "status": manifest["status"], "tasks": len(tasks),
        "applicable": sum(row["transfer_applicable"] for row in tasks),
        "abstained": sum(not row["transfer_applicable"] for row in tasks),
        "family_counts": dict(sorted(__import__("collections").Counter(row["task_family"] for row in tasks).items())),
        "source_counts": {game: sum(row["source_game"] == game for row in tasks) for game in SOURCE_GAMES},
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
