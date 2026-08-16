#!/usr/bin/env python3
"""Freeze outcome-blind qualification of the Phase-3 typed grounder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_discoveryworld_transfer import (  # noqa: E402
    PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT,
)


FORK_ROOT = REPO / "runs/phase2_discoveryworld_utility_v1/frozen_forks"
SEEDS = tuple(seed for seed in range(51, 81) if seed != 70)
RUNTIME_PATHS = (
    "src/motif_transfer/contracts.py",
    "src/motif_transfer/discoveryworld_env.py",
    "src/motif_transfer/discoveryworld_policy.py",
    "src/motif_transfer/discoveryworld_sokoban_transfer.py",
    "src/motif_transfer/discoveryworld_applicability_grounder_v4.py",
    "src/motif_transfer/phase3_discoveryworld_transfer.py",
    "scripts/run_discoveryworld_commit_recovery_v1.py",
    "scripts/run_phase3_discoveryworld_typed_grounding_qualification_v3.py",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--binding-fix-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite frozen qualification manifest")

    development = _read(args.development_summary)
    _validate_self_hash(development, "summary_sha256")
    if development.get("status") != "PHASE3_DISCOVERYWORLD_DEVELOPMENT_RUNTIME_COMPLETE":
        raise SystemExit("typed development runtime is incomplete")
    if int(development.get("complete_tasks", 0)) != 6:
        raise SystemExit("typed development did not cover all six tasks")
    schema = development.get("neural_grounding_schema") or {}
    if not schema.get("qualification_gate_passed"):
        raise SystemExit("development typed-grounder schema gate did not pass")
    binding_fix = _read(args.binding_fix_result)
    _validate_self_hash(binding_fix, "result_sha256")
    if binding_fix.get("status") != "DEVELOPMENT_MECHANISM_COMPLETE":
        raise SystemExit("binding-catalog development smoke is incomplete")
    binding = binding_fix.get("target_binding") or {}
    if "statue" not in str(binding.get("target_name") or "").lower():
        raise SystemExit("binding-catalog smoke did not bind a statue")
    if (binding.get("commit_action") or {}).get("action") != "DROP":
        raise SystemExit("binding-catalog smoke did not bind DROP")
    if binding_fix.get("policy_runtime_saw_oracle_scorecard") is not False:
        raise SystemExit("binding-catalog smoke exposed the scorecard to policy")
    if not binding_fix.get("all_selection_receipts_valid"):
        raise SystemExit("binding-catalog smoke has invalid selection receipts")
    binder_attempts = binding_fix.get("target_binding_schema_attempts") or ()
    if not binder_attempts or any(
        row.get("formal_outcome_fields_visible") is not False
        for row in binder_attempts
    ):
        raise SystemExit("binding-catalog smoke lacks outcome-blind binder receipts")
    matched = binding_fix.get("conditions") or {}
    grounder_attempts = [
        attempt
        for condition in (
            "source_induced", "source_permuted", "generic_scaffold",
            "target_native_ceiling",
        )
        for step in (matched.get(condition, {}).get("recovery") or ())
        for attempt in (step.get("grounder_schema_attempts") or ())
    ]
    if not grounder_attempts or any(
        row.get("formal_outcome_fields_visible") is not False
        for row in grounder_attempts
    ):
        raise SystemExit("binding-catalog smoke lacks outcome-blind grounder receipts")

    tasks = []
    model = None
    for seed in SEEDS:
        fork_path = FORK_ROOT / f"proteomics.easy.seed{seed}.json"
        if not fork_path.is_file():
            raise SystemExit(f"missing consumed fork: {fork_path}")
        fork = _read(fork_path)
        reference_path = REPO / str(fork["reference_episode"])
        reference = _read(reference_path)
        reference_body = dict(reference)
        reference_sha = str(reference_body.pop("episode_sha256", ""))
        if not reference_sha or stable_hash(reference_body) != reference_sha:
            raise SystemExit(f"invalid reference episode self-hash: {reference_path}")
        if reference_sha != fork["reference_episode_sha256"]:
            raise SystemExit(f"fork/reference mismatch: {fork_path}")
        if model is None:
            model = dict(fork["model"])
        elif model != dict(fork["model"]):
            raise SystemExit("qualification forks do not share one model config")
        tasks.append({
            "task_id": f"proteomics.easy.seed{seed}",
            "seed": seed,
            "fork_config": _relative(fork_path),
            "fork_config_file_sha256": _file_sha256(fork_path),
            "reference_episode": _relative(reference_path),
            "reference_episode_file_sha256": _file_sha256(reference_path),
            "reference_episode_sha256": reference_sha,
            "fork_after_episode_step": int(fork["fork_after_episode_step"]),
            "target_state_previously_consumed": True,
            "phase3_typed_grounder_previously_called_on_state": False,
        })

    official_checkout = REPO.parent / "discoveryworld-official"
    official_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=official_checkout, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    body = {
        "schema_version": "phase3-discoveryworld-typed-grounding-freeze-v3",
        "status": "FROZEN_BEFORE_TYPED_GROUNDING_QUALIFICATION_CALLS",
        "target_domain": "DiscoveryWorld/Proteomics/Easy",
        "tasks": tasks,
        "task_count": len(tasks),
        "state_cohort": (
            "PREVIOUSLY_CONSUMED_PHASE2_FORK_STATES_SEEDS51_80_EXCEPT70;"
            "DISJOINT_FROM_TYPED_DEVELOPMENT_SEEDS45_50_AND_FORMAL_SEEDS97_120"
        ),
        "model": model,
        "runtime": {"task_workers": 4, "thread_id_base": 146000},
        "frozen_qualification_gates": {
            "required_complete_states": len(tasks),
            "maximum_schema_or_native_precondition_repair_rate": 0.10,
            "maximum_binder_repair_rate": 0.10,
            "maximum_accepted_bundle_candidate_parse_rejections": 0,
            "required_position_candidates": 4,
            "required_commit_candidates": 1,
            "maximum_post_fork_actions_executed": 0,
            "formal_target_outcomes_read": False,
        },
        "development_evidence": {
            "path": _relative(args.development_summary),
            "file_sha256": _file_sha256(args.development_summary),
            "summary_sha256": development["summary_sha256"],
            "repair_rate": schema["repair_rate"],
            "excluded_from_prospective_target_estimates": True,
        },
        "outcome_blind_binding_fix_evidence": {
            "path": _relative(args.binding_fix_result),
            "file_sha256": _file_sha256(args.binding_fix_result),
            "result_sha256": binding_fix["result_sha256"],
            "bound_target_name": binding["target_name"],
            "bound_commit_action": binding["commit_action"],
            "policy_runtime_saw_oracle_scorecard": False,
            "formal_outcome_fields_visible_to_binder_or_grounder": False,
            "excluded_from_prospective_target_estimates": True,
        },
        "grounder_prompt_sha256": stable_hash(PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT),
        "runtime_file_sha256": {
            path: _file_sha256(REPO / path) for path in RUNTIME_PATHS
        },
        "official_environment_commit": official_commit,
        "formal_seed_range": [97, 120],
        "formal_reserve_task_opened": False,
        "formal_target_outcome_read_for_freeze": False,
        "source_program_visible_to_grounder": False,
        "claim_boundary": (
            "Component-only qualification of the typed target-native neural "
            "grounder on 29 already-consumed fork states. Replay stops at the "
            "fork; no candidate is executed, no evaluator is finalized, no "
            "source artifact is supplied, and formal seeds97-120 remain unopened."
        ),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"], "tasks": len(tasks),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
