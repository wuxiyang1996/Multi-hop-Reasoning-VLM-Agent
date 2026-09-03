#!/usr/bin/env python3
"""Freeze a stratified, already-consumed ALFWorld development evaluation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_goal_relation_macro import CONDITIONS  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    plan_path = REPO / "configs/strict_subgoal_option_audit_plan_v15.json"
    output = REPO / "configs/alfworld_goal_relation_macro_v3_development.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen config: {output}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in map(str, plan["task_ids"]):
        match = re.search(r"-None-([^-/]+)-\d+/", task_id)
        if match is None:
            raise RuntimeError(f"cannot parse target stratum: {task_id}")
        grouped[match.group(1)].append(task_id)
    if len(grouped) != 8 or any(len(rows) < 3 for rows in grouped.values()):
        raise RuntimeError("V15 consumed pool lacks eight three-task strata")
    task_ids = [
        task_id
        for stratum in sorted(grouped)
        for task_id in grouped[stratum][:3]
    ]
    source = REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    confirmation = (
        REPO / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    )
    target = (
        REPO / "runs/procedural_game_alfworld_v1_development/"
        "frozen_candidate_artifact.json"
    )
    target_causal = REPO / "configs/real_source_relation_causal_candidate_v20.json"
    runner = REPO / "scripts/run_alfworld_goal_relation_macro_v3.py"
    runtime = REPO / "src/motif_transfer/alfworld_goal_relation_macro.py"
    body = {
        "schema_version": "alfworld-goal-relation-macro-development-config-v3",
        "status": "FROZEN_CONSUMED_DEVELOPMENT_BEFORE_OUTCOMES",
        "claim_boundary": (
            "24 STRATIFIED ALREADY-CONSUMED ALFWORLD TRAIN-DEVELOPMENT TASKS; "
            "SOURCE ARTIFACT FRESH-CONFIRMED; TARGET GROUNDERS TRAINED ONLY ON "
            "DISJOINT TARGET ADAPTATION; NOT CONFIRMATORY EVIDENCE; UNTOUCHED "
            "MULTIPLICITY RESERVE AND VALID_UNSEEN REMAIN UNREAD"
        ),
        "source_artifact": str(source.relative_to(REPO)),
        "source_artifact_file_sha256": _sha256(source),
        "source_confirmation": str(confirmation.relative_to(REPO)),
        "source_confirmation_file_sha256": _sha256(confirmation),
        "target_grounder": str(target.relative_to(REPO)),
        "target_grounder_file_sha256": _sha256(target),
        "target_causal_effect_artifact": str(target_causal.relative_to(REPO)),
        "target_causal_effect_file_sha256": _sha256(target_causal),
        "runner_file_sha256": _sha256(runner),
        "target_runtime_file_sha256": _sha256(runtime),
        "parent_consumed_plan": {
            "path": str(plan_path.relative_to(REPO)),
            "file_sha256": _sha256(plan_path),
            "plan_sha256": str(plan["plan_sha256"]),
        },
        "selection": {
            "authority": "FIRST_THREE_ALREADY_CONSUMED_IDENTITIES_PER_SORTED_STRATUM",
            "target_outcome_used": False,
            "strata": {name: 3 for name in sorted(grouped)},
        },
        "task_ids": task_ids,
        "conditions": list(CONDITIONS),
        "alfworld_config": (
            "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-"
            "source-fresh-v1/configs/alfworld_base_config.yaml"
        ),
        "alfworld_data": (
            "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-"
            "github-main/.cache/alfworld_data"
        ),
        "seed": 486301,
        "max_steps": 60,
        "thresholds": {
            "minimum_binding": 0.5,
            "minimum_realization": 0.1,
            "minimum_binding_margin": 0.0,
            "minimum_causal_effect": 0.5,
            "authority": (
                "BINDING_AND_REALIZATION_INHERITED_FROM_FROZEN_V8; "
                "CAUSAL_EFFECT_0P5_FROM_DISJOINT_V20_TARGET_CALIBRATION_GATE"
            ),
        },
        "gates": {
            "minimum_second_cycle_action_changes": 8,
            "require_success_gain_over_raw": True,
            "require_strict_source_control_superiority": True,
            "require_zero_negative_transfer": True,
            "require_zero_reopened_completed_slots": True,
        },
        "output": "runs/alfworld_goal_relation_macro_v3_development/report.json",
        "untouched_reserve_read_or_run": False,
        "valid_unseen_read_or_run": False,
    }
    config = body | {"config_sha256": stable_hash(body)}
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "task_count": len(task_ids),
        "strata": config["selection"]["strata"],
        "config_sha256": config["config_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
