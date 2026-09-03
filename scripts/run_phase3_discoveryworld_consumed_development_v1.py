#!/usr/bin/env python3
"""Run Phase-3 selectors on consumed Phase-2 DiscoveryWorld forks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
DISCOVERYWORLD_CHECKOUT = REPO.parent / "discoveryworld-official"
if DISCOVERYWORLD_CHECKOUT.is_dir():
    sys.path.insert(0, str(DISCOVERYWORLD_CHECKOUT))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_discoveryworld_transfer import (  # noqa: E402
    CONDITIONS,
    MATCHED_CONDITIONS,
    Phase3DiscoveryWorldSelector,
    Phase3DiscoveryWorldPortfolioSelector,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    call_phase3_binder,
    call_phase3_grounder,
    extract_phase3_acquisition_evidence,
)
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    maximum_typed_program_contrast_derangement,
)


PROGRAM_DIR = REPO / "configs/phase3_source_induction_v3/frozen_reserve/programs"
CONSUMED_MANIFEST = REPO / "configs/phase2_discoveryworld_utility_v2/manifest.json"
DEVELOPMENT_TASKS = tuple(f"proteomics.easy.seed{seed}" for seed in range(45, 51))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")


def _artifacts() -> dict[str, dict[str, Any]]:
    return {
        path.stem: _read(path) for path in sorted(PROGRAM_DIR.glob("*.json"))
    }


def run_one(
    task_id: str, keys: Path, output_root: Path, source_mode: str,
) -> dict[str, Any]:
    import scripts.run_discoveryworld_commit_recovery_v1 as runner

    manifest = _read(CONSUMED_MANIFEST)
    task = next(row for row in manifest["tasks"] if row["task_id"] == task_id)
    source_game = str(task["source_game"])
    artifacts = _artifacts()
    permutation = maximum_typed_program_contrast_derangement(artifacts)
    original = _read(REPO / str(task["fork_config"]))
    reference = _read(REPO / str(original["reference_episode"]))
    acquisition_evidence = extract_phase3_acquisition_evidence(
        reference, int(original["fork_after_episode_step"]),
    )
    config = {
        **original,
        "status": "DEVELOPMENT_CONSUMED_PHASE2_FORKS_ONLY",
        "claim_boundary": (
            "Phase-3 runtime and multiplicity development on consumed Phase-2 "
            "DiscoveryWorld fork; excluded from prospective evidence."
        ),
        "conditions": list(MATCHED_CONDITIONS),
    }
    cell_dir = output_root / task_id
    config_path = cell_dir / "config.json"
    output_path = cell_dir / "matched_result.json"
    if output_path.is_file():
        return _read(output_path)
    _write(config_path, config)
    selector = (
        Phase3DiscoveryWorldPortfolioSelector(
            source_artifacts=list(artifacts.values()),
        )
        if source_mode == "portfolio" else
        Phase3DiscoveryWorldSelector(
            authentic_artifact=artifacts[source_game],
            permuted_artifact=artifacts[permutation[source_game]],
        )
    )
    old_grounder = runner.call_grounder
    old_binder = runner.call_binder
    old_selector = runner.select_candidate
    runner.call_grounder = call_phase3_grounder
    def outcome_blind_phase3_binder(
        backend, observation, *, memory, hypotheses, attempts,
    ):
        return call_phase3_binder(
            backend, observation, memory=memory, hypotheses=hypotheses,
            attempts=attempts, acquisition_evidence=acquisition_evidence,
        )

    runner.call_binder = outcome_blind_phase3_binder
    runner.select_candidate = selector.select
    transport_suffix = "\nReturn one valid json object."
    if not runner.TARGET_BINDER_SYSTEM_PROMPT.endswith(transport_suffix):
        runner.TARGET_BINDER_SYSTEM_PROMPT += transport_suffix
    old_argv = sys.argv
    try:
        sys.argv = [
            str(REPO / "scripts/run_discoveryworld_commit_recovery_v1.py"),
            "--config", str(config_path), "--keys", str(keys),
            "--output", str(output_path),
        ]
        runner.main()
    finally:
        sys.argv = old_argv
        runner.call_grounder = old_grounder
        runner.call_binder = old_binder
        runner.select_candidate = old_selector
    return _read(output_path)


def _worker(
    task_id: str, keys: str, output_root: str, source_mode: str,
) -> dict[str, Any]:
    try:
        result = run_one(task_id, Path(keys), Path(output_root), source_mode)
        conditions = result.get("conditions") or {}
        arm_errors = {
            name: conditions.get(name, {}).get("runtime_error")
            for name in MATCHED_CONDITIONS
            if conditions.get(name, {}).get("runtime_error") is not None
        }
        return {
            "task_id": task_id,
            "error": (
                f"matched arm runtime errors: {arm_errors}" if arm_errors else None
            ),
            "outcomes": {
                "neural_only": bool(
                    conditions.get("target_only_recorded", {}).get("official_success")
                ),
                **{
                    name: bool(conditions.get(name, {}).get("official_success"))
                    for name in MATCHED_CONDITIONS
                },
            },
        }
    except BaseException as exc:
        return {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--task-id", action="append")
    parser.add_argument(
        "--source-mode", choices=("assigned-pair", "portfolio"),
        default="assigned-pair",
    )
    args = parser.parse_args()
    task_ids = tuple(args.task_id or DEVELOPMENT_TASKS)
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(task_ids))) as pool:
        futures = {
            pool.submit(
                _worker, task, str(args.keys), str(args.output_root),
                args.source_mode,
            ): task
            for task in task_ids
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: row["task_id"])
    complete = [row for row in rows if row["error"] is None]
    counts = {
        condition: sum(row["outcomes"][condition] for row in complete)
        for condition in CONDITIONS
    }
    first_position_counts = []
    source_contrasts = 0
    admitted_source_contrasts = 0
    source_arm_audit = []
    unique_grounder_calls: dict[str, dict[str, Any]] = {}
    successes_by_horizon = {
        str(horizon): {condition: 0 for condition in CONDITIONS}
        for horizon in (1, 2, 4, 8)
    }
    recovery_steps_to_success = {condition: [] for condition in CONDITIONS}
    for row in complete:
        result = _read(args.output_root / row["task_id"] / "matched_result.json")
        conditions = result["conditions"]
        outcome_arms = {"neural_only": conditions["target_only_recorded"]} | {
            name: conditions[name] for name in MATCHED_CONDITIONS
        }
        for condition, arm_result in outcome_arms.items():
            transitions = [
                step.get("transition") or {}
                for step in arm_result.get("recovery") or ()
            ]
            first_success = next((
                index for index, transition in enumerate(transitions, start=1)
                if transition.get("official_success")
            ), None)
            if first_success is not None:
                recovery_steps_to_success[condition].append(first_success)
            for horizon in (1, 2, 4, 8):
                successes_by_horizon[str(horizon)][condition] += int(
                    first_success is not None and first_success <= horizon
                )
        source_step = conditions[SOURCE_INDUCED]["recovery"][0]
        permuted_step = conditions[SOURCE_PERMUTED]["recovery"][0]
        first_position_counts.append(sum(
            item["target_role"] == "POSITION"
            for item in source_step["candidate_bundle"]
        ))
        contrast = (
            source_step["selection"]["selected_candidate_sha256"]
            != permuted_step["selection"]["selected_candidate_sha256"]
        )
        source_contrasts += int(contrast)
        both_admitted = bool(
            source_step["selection"].get("source_admitted")
            and permuted_step["selection"].get("source_admitted")
        )
        admitted_source_contrasts += int(contrast and both_admitted)
        source_arm_audit.append({
            "task_id": row["task_id"],
            "source_initially_admitted": bool(
                source_step["selection"].get("source_admitted")
            ),
            "permuted_initially_admitted": bool(
                permuted_step["selection"].get("source_admitted")
            ),
            "source_fallback_reasons": sorted({
                str(step["selection"]["selection_reason"])
                for step in conditions[SOURCE_INDUCED]["recovery"]
                if "GROUNDER" in str(
                    step["selection"].get("selection_reason", "")
                )
            }),
            "permuted_fallback_reasons": sorted({
                str(step["selection"]["selection_reason"])
                for step in conditions[SOURCE_PERMUTED]["recovery"]
                if "GROUNDER" in str(
                    step["selection"].get("selection_reason", "")
                )
            }),
            "first_selection_contrast": bool(contrast),
            "selected_portfolio_program_sha256": source_step["selection"].get(
                "source_profile_sha256"
            ) if args.source_mode == "portfolio" else None,
            "portfolio_receipt_sha256": source_step["selection"].get(
                "portfolio_receipt_sha256"
            ),
        })
        for condition in MATCHED_CONDITIONS:
            for step in conditions[condition].get("recovery") or ():
                attempts = step.get("grounder_schema_attempts") or ()
                signature = []
                for attempt in attempts:
                    signature.append(str(
                        attempt.get("raw_sha256")
                        or (
                            step.get("grounder_response_sha256")
                            if attempt.get("accepted") else ""
                        )
                    ))
                call_sha = stable_hash(signature)
                unique_grounder_calls.setdefault(call_sha, {
                    "attempts": len(attempts),
                    "required_repair": any(
                        not bool(attempt.get("accepted")) for attempt in attempts
                    ),
                    "final_accepted": bool(
                        attempts and attempts[-1].get("accepted")
                    ),
                    "response_sha256": step.get("grounder_response_sha256"),
                })
    grounder_calls = list(unique_grounder_calls.values())
    repaired_calls = sum(row["required_repair"] for row in grounder_calls)
    body = {
        "schema_version": "phase3-discoveryworld-consumed-development-v1",
        "status": (
            "PHASE3_DISCOVERYWORLD_DEVELOPMENT_RUNTIME_COMPLETE"
            if len(complete) == len(rows) else
            "PHASE3_DISCOVERYWORLD_DEVELOPMENT_RUNTIME_INCOMPLETE"
        ),
        "tasks": len(rows),
        "complete_tasks": len(complete),
        "condition_successes": counts,
        "condition_successes_by_recovery_horizon": successes_by_horizon,
        "recovery_steps_to_success": recovery_steps_to_success,
        "initial_position_candidate_counts": first_position_counts,
        "source_vs_permuted_first_selection_contrasts": source_contrasts,
        "admitted_source_vs_permuted_first_selection_contrasts": (
            admitted_source_contrasts
        ),
        "source_arm_audit": source_arm_audit,
        "neural_grounding_schema": {
            "unique_calls": len(grounder_calls),
            "calls_requiring_schema_or_native_precondition_repair": repaired_calls,
            "repair_rate": (
                repaired_calls / len(grounder_calls) if grounder_calls else 0.0
            ),
            "all_calls_finally_accepted": all(
                row["final_accepted"] for row in grounder_calls
            ),
            "frozen_qualification_maximum_repair_rate": 0.10,
            "qualification_gate_passed": (
                bool(grounder_calls)
                and repaired_calls / len(grounder_calls) <= 0.10
                and all(row["final_accepted"] for row in grounder_calls)
            ),
            "development_outcomes_used_for_schema_gate": False,
        },
        "source_permutation": maximum_typed_program_contrast_derangement(_artifacts()),
        "source_mode": args.source_mode,
        "source_permuted_control": (
            "DETERMINISTIC_TARGET_CANDIDATE_EFFECT_BINDING_PERMUTATION"
            if args.source_mode == "portfolio" else
            "QUALIFICATION_STATUS_MATCHED_SOURCE_PROGRAM_DERANGEMENT"
        ),
        "rows": rows,
        "formal_target_outcome_included": False,
        "claim_boundary": (
            "Consumed seeds45-50 only; development diagnoses runtime and "
            "candidate multiplicity and is excluded from Phase-3 evidence."
        ),
    }
    summary = body | {"summary_sha256": stable_hash(body)}
    _write(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if len(complete) != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
