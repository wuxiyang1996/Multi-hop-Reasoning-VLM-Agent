#!/usr/bin/env python3
"""Build a compact, hash-bound game-to-ALFWorld transfer receipt."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.train_procedural_game_alfworld_candidate import (  # noqa: E402
    build_candidate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def paired_counts(
    authentic: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    authentic_by_id = {str(row["task_id"]): bool(row["official_success"]) for row in authentic}
    comparator_by_id = {str(row["task_id"]): bool(row["official_success"]) for row in comparator}
    if authentic_by_id.keys() != comparator_by_id.keys():
        raise ValueError("paired task identities differ")
    wins = sum(authentic_by_id[key] and not comparator_by_id[key] for key in authentic_by_id)
    losses = sum(not authentic_by_id[key] and comparator_by_id[key] for key in authentic_by_id)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(authentic_by_id) - wins - losses,
    }


def exact_sign_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    from math import comb

    tail = sum(comb(n, index) for index in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def build_summary(
    *,
    source_config_path: Path,
    development_config_path: Path,
    final_config_path: Path,
    manifest_path: Path,
    artifact_path: Path,
    development_report_path: Path,
    final_report_path: Path,
) -> dict[str, Any]:
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    final_config = json.loads(final_config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    development = json.loads(development_report_path.read_text(encoding="utf-8"))
    final = json.loads(final_report_path.read_text(encoding="utf-8"))
    if artifact["status"] != "QUALIFICATION_AUTHORIZED":
        raise ValueError("source candidate was not authorized")
    rebuilt_artifact = build_candidate(source_config, config_path=source_config_path)
    normalized_rebuild = json.loads(json.dumps(rebuilt_artifact, sort_keys=True))
    if normalized_rebuild != artifact:
        raise ValueError("current source-game implementation does not reproduce artifact")
    if not artifact["source"]["gate_passed"]:
        raise ValueError("procedural source-game gate did not pass")
    if development["status"] != "QUALIFICATION_CANDIDATE_PASSED":
        raise ValueError("consumed target development gate did not pass")
    if final["status"] != "FINAL_HELDOUT_PASSED":
        raise ValueError("fresh final target gate did not pass")
    if not final["paired_task_order_verified"] or not final["heldout_read"]:
        raise ValueError("final report lacks a verified held-out pairing")
    expected_ids = manifest["cells"]["alfworld_valid_unseen"]["splits"]["held_out"]
    if set(expected_ids) != set(final["paired_task_order"]):
        raise ValueError("final report does not cover the frozen identities")
    evidence = final_config["development_evidence"]
    frozen_checks = {
        "origin_config": (
            development_config_path, evidence["origin_config_sha256"]
        ),
        "development_report": (
            development_report_path, evidence["qualification_report_sha256"]
        ),
        "candidate_artifact": (artifact_path, evidence["frozen_artifact_sha256"]),
        "runner": (
            REPO / "scripts/run_multisource_alfworld_v2_qualification.py",
            evidence["runner_sha256"],
        ),
    }
    mismatches = {
        name: {"expected": expected, "observed": _sha256(path)}
        for name, (path, expected) in frozen_checks.items()
        if _sha256(path) != expected
    }
    if mismatches:
        raise ValueError(f"frozen input mismatch: {mismatches}")

    authentic_rows = final["episodes"]["authentic_source_plus_target"]
    comparisons = {}
    for name, rows in final["episodes"].items():
        if name == "authentic_source_plus_target":
            continue
        counts = paired_counts(authentic_rows, rows)
        counts["exact_two_sided_sign_p"] = exact_sign_two_sided(
            counts["wins"], counts["losses"]
        )
        comparisons[name] = counts
    family_rows: dict[str, dict[str, int]] = {}
    for task_id in expected_ids:
        family = _family(task_id)
        family_rows.setdefault(family, {"tasks": 0})["tasks"] += 1
    for condition, rows in final["episodes"].items():
        successes = Counter(
            _family(str(row["task_id"]))
            for row in rows if row["official_success"]
        )
        for family in family_rows:
            family_rows[family][condition] = int(successes[family])

    source = artifact["source"]
    body = {
        "schema_version": "procedural-game-to-alfworld-v1-summary",
        "status": "PROCEDURAL_GAME_TO_ALFWORLD_FRESH_FORMAL_VALIDATED",
        "claim_boundary": (
            "Matched-intervention procedural workflow games to the exact 24-task "
            "ALFWorld valid_unseen reserve. This is controlled game-suite transfer, "
            "not Sokoban-only transfer and not zero-shot target grounding."
        ),
        "source": {
            "kind": source["kind"],
            "train_surfaces": source["train_surfaces"],
            "evaluation_surfaces": source["evaluation_surfaces"],
            "surface_overlap": source["surface_overlap"],
            "train_domains": source["train_domains"],
            "evaluation_domains": source["evaluation_domains"],
            "evaluation_intervention_receipts": source[
                "evaluation_intervention_receipts"
            ],
            "evaluation_receipts_sha256": source["evaluation_receipts_sha256"],
            "raw_action_tokens_transferred": source["raw_action_tokens_transferred"],
            "alpha_renamed_native_actions_per_domain": source[
                "alpha_renamed_native_actions_per_domain"
            ],
            "heldout_value_mse": source["heldout_value_mse"],
            "relative_mse_improvement_over_control": source[
                "relative_mse_improvement_over_control"
            ],
            "source_gate_passed": source["gate_passed"],
            "frozen_candidate_artifact_file_sha256": _sha256(artifact_path),
            "frozen_candidate_content_sha256": artifact["artifact_content_sha256"],
            "candidate_rebuilt_exactly_from_current_code": True,
        },
        "target": {
            "grounder_kind": artifact["target_grounder"]["kind"],
            "grounder_gate_passed": artifact["target_grounder_gate"]["passed"],
            "grounder_retrained_on_qualification_or_heldout": False,
            "native_action_authority": True,
            "manifest_status_before_first_attempt": manifest["status"],
            "manifest_parent_sha256": manifest["parent_manifest"]["manifest_sha256"],
            "tasks": len(expected_ids),
            "split": "valid_unseen",
        },
        "development": {
            "status": development["status"],
            "summaries": development["summaries"],
            "all_gates_passed": bool(
                development["nontriviality_gate"]["passed"]
                and development["qualification_superiority_gate"]["passed"]
                and development["efficiency_gate"]["passed"]
            ),
        },
        "formal": {
            "status": final["status"],
            "summaries": final["summaries"],
            "paired_comparisons": comparisons,
            "family_breakdown": family_rows,
            "nontriviality_gate_passed": final["nontriviality_gate"]["passed"],
            "superiority_gate_passed": final["qualification_superiority_gate"][
                "passed"
            ],
            "efficiency_gate_passed": final["efficiency_gate"]["passed"],
            "all_frozen_gates_passed": bool(final["cross_domain_transfer_supported"]),
        },
        "operational_disclosure": {
            "first_attempt_transport_interrupted_after_conditions": [
                "target_only", "authentic_source_plus_target"
            ],
            "first_attempt_partial_next_condition": {
                "condition": "shuffled_source_plus_target",
                "completed_tasks": 2
            },
            "first_attempt_final_report_written": False,
            "replay_used_identical_frozen_inputs": True,
            "scientific_configuration_changed_after_partial_outcomes": False,
            "interpretation": (
                "The exact frozen run was replayed after a transport interruption. "
                "The result is a deterministic frozen replay, not an uninterrupted "
                "one-shot execution."
            ),
        },
        "input_file_sha256": {
            "source_config": _sha256(source_config_path),
            "development_config": _sha256(development_config_path),
            "final_config": _sha256(final_config_path),
            "manifest": _sha256(manifest_path),
            "candidate_artifact": _sha256(artifact_path),
            "development_report": _sha256(development_report_path),
            "final_report": _sha256(final_report_path),
            "runner": _sha256(
                REPO / "scripts/run_multisource_alfworld_v2_qualification.py"
            ),
            "source_trainer": _sha256(
                REPO / "scripts/train_procedural_game_alfworld_candidate.py"
            ),
            "source_game_mdp": _sha256(
                REPO / "src/motif_transfer/procedural_workflow_game.py"
            ),
            "symbolic_value_model": _sha256(
                REPO / "src/motif_transfer/hierarchical_skill_transfer.py"
            ),
            "target_native_grounder": _sha256(
                REPO / "src/motif_transfer/alfworld_hierarchical_grounder.py"
            ),
        },
    }
    if source_config["target"]["artifact"] != final_config["target"]["artifact"]:
        raise ValueError("source and final configs do not bind the same artifact")
    body["summary_sha256"] = stable_hash(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config", type=Path,
        default=REPO / "configs/procedural_game_alfworld_v1_development.json",
    )
    parser.add_argument(
        "--development-config", type=Path,
        default=REPO / "configs/procedural_game_alfworld_v1_target_development.json",
    )
    parser.add_argument(
        "--final-config", type=Path,
        default=REPO / "configs/procedural_game_alfworld_v1_frozen.json",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/procedural_game_alfworld_v1_final_manifest.json",
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=REPO / "runs/procedural_game_alfworld_v1_development/frozen_candidate_artifact.json",
    )
    parser.add_argument(
        "--development-report", type=Path,
        default=REPO / "runs/procedural_game_alfworld_v1_development/target_development_report.json",
    )
    parser.add_argument(
        "--final-report", type=Path,
        default=REPO / "runs/procedural_game_alfworld_v1_frozen/heldout_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/procedural_game_alfworld_v1_summary.json",
    )
    args = parser.parse_args()
    summary = build_summary(
        source_config_path=args.source_config.resolve(),
        development_config_path=args.development_config.resolve(),
        final_config_path=args.final_config.resolve(),
        manifest_path=args.manifest.resolve(),
        artifact_path=args.artifact.resolve(),
        development_report_path=args.development_report.resolve(),
        final_report_path=args.final_report.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "summaries": summary["formal"]["summaries"],
        "paired": summary["formal"]["paired_comparisons"],
        "summary_sha256": summary["summary_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
