#!/usr/bin/env python3
"""Run frozen five-arm DiscoveryWorld structural-transfer forks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
DISCOVERYWORLD = REPO.parent / "discoveryworld-official"
if DISCOVERYWORLD.is_dir():
    sys.path.insert(0, str(DISCOVERYWORLD))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_env import (  # noqa: E402
    DETERMINISM_PROTOCOL, DiscoveryWorldEnvironment,
)
from motif_transfer.discoveryworld_policy import target_native_facts  # noqa: E402
from motif_transfer.discoveryworld_sokoban_transfer import (  # noqa: E402
    DiscoveryWorldGroundedCandidate, realize_localized_spatial_position,
)
from motif_transfer.discoveryworld_structural_runtime_v1 import (  # noqa: E402
    choose_structural_action, grounded_prefix_counts,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend, OpenAICompatibleBackend,
)
from motif_transfer.phase3_discoveryworld_transfer import (  # noqa: E402
    call_phase3_binder, extract_phase3_acquisition_evidence,
)
from motif_transfer.structural_delta_induction import (  # noqa: E402
    validate_structural_program,
)
from motif_transfer.target_structural_induction import (  # noqa: E402
    validate_mlp_grounder, validate_target_program,
)


CONDITIONS = (
    "neural_only", "source_induced", "source_permuted",
    "generic_scaffold", "target_native_ceiling",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _backend(manifest: Mapping[str, Any], key: str, cache: Path):
    model = manifest["binding_model"]
    os.environ["DISCOVERYWORLD_STRUCTURAL_BINDER_KEY"] = key
    raw = OpenAICompatibleBackend(
        str(model["base_url"]), {"binder": str(model["model"])},
        api_key_env="DISCOVERYWORLD_STRUCTURAL_BINDER_KEY", json_mode=True,
        temperature=float(model["temperature"]), timeout_seconds=180,
        request_overrides={"max_tokens": int(model["maximum_output_tokens"])},
        transport_attempts=3,
    )
    return MemoizedCompletionBackend(raw, cache_path=cache)


def _new_env(task, *, max_steps: int, thread_id: int, frame_dir: Path):
    return DiscoveryWorldEnvironment(
        scenario=str(task["scenario"]), difficulty=str(task["difficulty"]),
        seed=int(task["seed"]), max_steps=max_steps, thread_id=thread_id,
        include_vision=False, frame_dir=frame_dir,
    )


def _recorded_arm(
    *, env, observation, reference: Mapping[str, Any], fork_step: int,
    horizon: int,
) -> tuple[Any, list[dict[str, Any]], bool]:
    rows = []
    for reference_row in reference["steps"][fork_step:fork_step + horizon]:
        if observation.terminal:
            break
        action = dict(reference_row["action"])
        before = target_native_facts(observation)
        observation, transition = env.step(action)
        rows.append({
            "recovery_step": len(rows) + 1,
            "mode": "RECORDED_NEURAL_ONLY",
            "action": action,
            "before_target_native_facts": before,
            "after_target_native_facts": target_native_facts(observation),
            "transition": asdict(transition),
        })
    return observation, rows, False


def _structural_arm(
    *, condition: str, env, observation, target_binding,
    reference: Mapping[str, Any], fork_step: int, horizon: int,
    grounder: Mapping[str, Any], target_program: Mapping[str, Any],
    source_program: Mapping[str, Any] | None,
    prefix_counts: Mapping[str, int],
) -> tuple[Any, list[dict[str, Any]], bool]:
    rows = []
    localized = False
    used_recorded_fallback = False
    for recovery_index in range(horizon):
        if observation.terminal:
            break
        facts = target_native_facts(observation)
        position = {
            "action": "TELEPORT_TO_OBJECT",
            "arg1": int(target_binding.target_uuid),
        }
        decision, audit = choose_structural_action(
            condition=condition, facts=facts,
            target_uuid=int(target_binding.target_uuid),
            commit_action=dict(target_binding.commit_action),
            position_action=position, grounder=grounder,
            target_program=target_program, source_program=source_program,
            prefix_counts=prefix_counts,
        )
        if decision.kind == "ABSTAIN":
            # Fail closed to the exact neural continuation only if no target
            # intervention has changed the matched fork.
            if not rows:
                remaining = reference["steps"][
                    fork_step:fork_step + horizon
                ]
                for reference_row in remaining:
                    if observation.terminal:
                        break
                    action = dict(reference_row["action"])
                    before = target_native_facts(observation)
                    observation, transition = env.step(action)
                    rows.append({
                        "recovery_step": len(rows) + 1,
                        "mode": "FAIL_CLOSED_RECORDED_NEURAL_FALLBACK",
                        "runtime_decision": asdict(decision),
                        "runtime_audit": audit,
                        "action": action,
                        "before_target_native_facts": before,
                        "after_target_native_facts": target_native_facts(observation),
                        "transition": asdict(transition),
                    })
                used_recorded_fallback = True
            break

        action = dict(decision.action or {})
        realization = None
        if action.get("action") == "TELEPORT_TO_OBJECT" and localized:
            candidate = DiscoveryWorldGroundedCandidate(
                action=action, target_role="POSITION",
                prerequisite_probability=1.0,
                positive_effect_probability=1.0,
                information_gain_probability=0.0,
                expected_effect="Satisfy learned target-native commit guard.",
                evidence=(), reason="TARGET_NATIVE_SPATIAL_REALIZATION",
            )
            action, realization = realize_localized_spatial_position(
                candidate, observation, target_binding,
                target_was_localized=True,
            )
        before = target_native_facts(observation)
        observation, transition = env.step(action)
        if (
            action.get("action") == "TELEPORT_TO_OBJECT"
            and action.get("arg1") == target_binding.target_uuid
            and transition.action_succeeded
        ):
            localized = True
        rows.append({
            "recovery_step": len(rows) + 1,
            "mode": "STRUCTURAL_RUNTIME",
            "runtime_decision": asdict(decision),
            "runtime_audit": audit,
            "realization": realization,
            "action": action,
            "before_target_native_facts": before,
            "after_target_native_facts": target_native_facts(observation),
            "transition": asdict(transition),
        })
        # An irreversible commit is attempted once. Repeating it after the
        # carried operand has left inventory would fabricate recovery capacity.
        if action.get("action") in {"DROP", "PUT"}:
            break
    return observation, rows, used_recorded_fallback


def _load_frozen_components(manifest: Mapping[str, Any]):
    report_path = REPO / manifest["target_development_report"]["path"]
    if _sha(report_path) != manifest["target_development_report"]["file_sha256"]:
        raise ValueError("target development report changed")
    report = _read(report_path); _self_hash(report, "report_sha256")
    grounder = report["grounder"]
    target_program = report["target_program"]
    validate_mlp_grounder(grounder); validate_target_program(target_program)
    programs = {}
    for task, receipt in manifest["source_programs"].items():
        path = REPO / receipt["path"]
        if _sha(path) != receipt["file_sha256"]:
            raise ValueError(f"source program changed: {task}")
        program = _read(path); validate_structural_program(program)
        if program["program_sha256"] != receipt["program_sha256"]:
            raise ValueError(f"source program receipt mismatch: {task}")
        programs[task] = program
    return report, grounder, target_program, programs


def run_one(
    *, manifest_path: Path, reference_path: Path, key: str,
    output_path: Path, task_index: int,
) -> dict[str, Any]:
    if output_path.exists():
        result = _read(output_path); _self_hash(result, "result_sha256")
        return result
    manifest = _read(manifest_path); _self_hash(manifest, "manifest_sha256")
    report, grounder, target_program, programs = _load_frozen_components(manifest)
    reference = _read(reference_path); _self_hash(reference, "episode_sha256")
    if reference.get("formal_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("acquisition episode used the wrong manifest")
    fork_step = reference.get("fork_after_episode_step")
    if not isinstance(fork_step, int) or fork_step < 1:
        raise ValueError("acquisition did not produce an outcome-blind fork")
    task = dict(reference["task"])
    horizon = int(manifest["matched_runtime"]["recovery_horizon"])
    max_steps = fork_step + horizon
    prefix_steps = reference["steps"][:fork_step]
    prefix_actions = [dict(row["action"]) for row in prefix_steps]
    expected_policy = prefix_steps[-1]["transition"]["after_policy_state_sha256"]

    anchor_env = _new_env(
        task, max_steps=max_steps,
        thread_id=int(manifest["matched_runtime"]["thread_id_base"]) + task_index * 10,
        frame_dir=output_path.parent / "frames" / "anchor",
    )
    anchor, _ = anchor_env.replay_prefix(
        prefix_actions, expected_policy_state_sha256=expected_policy,
    )
    fork_policy = anchor.policy_state_sha256
    fork_audit = anchor_env.current_audit_hash
    evidence = extract_phase3_acquisition_evidence(
        reference, fork_step,
    )
    memory = str(prefix_steps[-1].get("memory") or "")
    hypotheses = tuple(prefix_steps[-1].get("running_hypotheses") or ())
    backend = _backend(
        manifest, key, output_path.parent / "binder_cache.json",
    )
    binding, binder_raw, binder_attempts = call_phase3_binder(
        backend, anchor, memory=memory, hypotheses=hypotheses,
        attempts=int(manifest["binding_model"]["schema_attempts"]),
        acquisition_evidence=evidence,
    )
    prefix_counts = grounded_prefix_counts(prefix_steps, grounder)
    results = {}
    for condition_index, condition in enumerate(CONDITIONS):
        env = _new_env(
            task, max_steps=max_steps,
            thread_id=(
                int(manifest["matched_runtime"]["thread_id_base"])
                + task_index * 10 + condition_index + 1
            ),
            frame_dir=output_path.parent / "frames" / condition,
        )
        observation, receipts = env.replay_prefix(
            prefix_actions, expected_policy_state_sha256=fork_policy,
            expected_audit_world_sha256=fork_audit,
        )
        if condition == "neural_only":
            observation, recovery, fallback = _recorded_arm(
                env=env, observation=observation, reference=reference,
                fork_step=fork_step, horizon=horizon,
            )
        else:
            source_program = None
            if condition == "source_induced":
                source_program = programs[report["selected_source_program"]]
            elif condition == "source_permuted":
                source_program = programs[report["source_permuted_control"]]
            observation, recovery, fallback = _structural_arm(
                condition=condition, env=env, observation=observation,
                target_binding=binding, reference=reference,
                fork_step=fork_step, horizon=horizon, grounder=grounder,
                target_program=target_program, source_program=source_program,
                prefix_counts=prefix_counts,
            )
        results[condition] = {
            "matched_fork_policy_state_sha256": (
                recovery[0]["transition"]["before_policy_state_sha256"]
                if recovery else fork_policy
            ),
            "matched_fork_audit_world_sha256": receipts[-1].after_audit_world_sha256,
            "recovery": recovery,
            "used_recorded_neural_fallback": fallback,
            "terminal": bool(observation.terminal),
            "official_success": bool(observation.official_success),
        }

    body = {
        "schema_version": "discoveryworld-structural-transfer-result-v1",
        "status": "DISCOVERYWORLD_STRUCTURAL_MATCHED_COMPLETE",
        "manifest_sha256": manifest["manifest_sha256"],
        "task": task,
        "task_id": reference["task_id"],
        "reference_episode_sha256": reference["episode_sha256"],
        "fork_after_episode_step": fork_step,
        "fork_policy_state_sha256": fork_policy,
        "fork_audit_world_sha256": fork_audit,
        "determinism_protocol": DETERMINISM_PROTOCOL,
        "target_binding": asdict(binding),
        "target_binding_response_sha256": hashlib.sha256(
            binder_raw.encode("utf-8")
        ).hexdigest(),
        "target_binding_attempts": binder_attempts,
        "prefix_grounded_operator_counts": prefix_counts,
        "conditions": results,
        "all_matched_forks": all(
            row["matched_fork_policy_state_sha256"] == fork_policy
            and row["matched_fork_audit_world_sha256"] == fork_audit
            for row in results.values()
        ),
        "policy_runtime_saw_oracle_scorecard": False,
        "formal_outcome_used_for_binding_or_selection": False,
    }
    result = body | {"result_sha256": stable_hash(body)}
    _write(output_path, result)
    return result


def _worker(manifest, reference, key, output, index):
    try:
        result = run_one(
            manifest_path=Path(manifest), reference_path=Path(reference),
            key=key, output_path=Path(output), task_index=index,
        )
        return {
            "task_id": result["task_id"], "error": None,
            "outcomes": {
                name: row["official_success"]
                for name, row in result["conditions"].items()
            },
        }
    except BaseException as exc:
        return {
            "task_id": Path(reference).stem,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _exact_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _pair(results, left: str, right: str):
    wins = sum(
        row["conditions"][left]["official_success"]
        and not row["conditions"][right]["official_success"]
        for row in results
    )
    losses = sum(
        not row["conditions"][left]["official_success"]
        and row["conditions"][right]["official_success"]
        for row in results
    )
    ties = len(results) - wins - losses
    return {
        "wins": wins, "losses": losses, "ties": ties,
        "exact_two_sided_sign_p": _exact_sign_p(wins, losses),
        "negative_transfer_rate": losses / (wins + losses) if wins + losses else 0.0,
    }


def _analyze(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]):
    success = {
        condition: sum(row["conditions"][condition]["official_success"] for row in results)
        for condition in CONDITIONS
    }
    source_neural = _pair(results, "source_induced", "neural_only")
    source_permuted = _pair(results, "source_induced", "source_permuted")
    source_generic = _pair(results, "source_induced", "generic_scaffold")
    applicable = sum(
        row["conditions"]["source_induced"]["recovery"]
        and row["conditions"]["source_induced"]["recovery"][0].get("mode")
        == "STRUCTURAL_RUNTIME"
        for row in results
    )
    behavior_contrasts = sum(
        [step.get("action") for step in row["conditions"]["source_induced"]["recovery"]]
        != [step.get("action") for step in row["conditions"]["source_permuted"]["recovery"]]
        for row in results
    )
    thresholds = manifest["preregistered_gates"]
    gates = {
        "all_tasks_complete": len(results) == int(manifest["task_count"]),
        "all_forks_matched": all(row["all_matched_forks"] for row in results),
        "minimum_applicable_tasks": applicable >= int(thresholds["minimum_applicable_tasks"]),
        "source_strictly_improves_neural": success["source_induced"] > success["neural_only"],
        "source_vs_neural_significance": source_neural["exact_two_sided_sign_p"] <= float(thresholds["source_vs_neural_sign_p_max"]),
        "source_vs_neural_negative_transfer": source_neural["negative_transfer_rate"] <= float(thresholds["negative_transfer_rate_max"]),
        "source_strictly_beats_permuted": success["source_induced"] > success["source_permuted"],
        "source_strictly_beats_generic": success["source_induced"] > success["generic_scaffold"],
        "ceiling_not_below_source": success["target_native_ceiling"] >= success["source_induced"],
        "source_permutation_behaviorally_distinct": behavior_contrasts >= int(thresholds["minimum_source_permutation_behavior_contrasts"]),
        "zero_oracle_use": all(
            row["formal_outcome_used_for_binding_or_selection"] is False
            and row["policy_runtime_saw_oracle_scorecard"] is False
            for row in results
        ),
    }
    body = {
        "schema_version": "discoveryworld-structural-transfer-report-v1",
        "status": "DISCOVERYWORLD_STRUCTURAL_TRANSFER_VALIDATED" if all(gates.values()) else "DISCOVERYWORLD_STRUCTURAL_TRANSFER_FAILED",
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": len(results), "applicable_tasks": applicable,
        "condition_successes": success,
        "source_vs_neural": source_neural,
        "source_vs_permuted": source_permuted,
        "source_vs_generic": source_generic,
        "source_permutation_behavior_contrasts": behavior_contrasts,
        "gates": gates,
        "claim_boundary": "PROSPECTIVE_FRESH_DISCOVERYWORLD_SEEDS;SHARED_IR_WITH_TARGET_INDUCED_DOMAIN_FUNCTION_AND_TARGET_NATIVE_NEURAL_GROUNDING",
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    manifest = _read(args.manifest); _self_hash(manifest, "manifest_sha256")
    for relative, expected in manifest["runtime_file_sha256"].items():
        if _sha(REPO / relative) != expected:
            raise SystemExit(f"frozen runtime changed: {relative}")
    values = runpy.run_path(str(args.keys))
    key = values.get(manifest["binding_model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    selected = set(args.task_id or ())
    tasks = [
        row for row in manifest["tasks"]
        if not selected or row["task_id"] in selected
    ]
    if selected and len(tasks) != len(selected):
        raise SystemExit("unknown task ID")
    workers = min(
        len(tasks), args.workers or int(manifest["matched_runtime"]["task_workers"]),
    )
    progress = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for task in tasks:
            index = manifest["tasks"].index(task)
            reference = args.acquisition_dir / f"{task['task_id']}.json"
            output = args.output_dir / task["task_id"] / "result.json"
            futures[pool.submit(
                _worker, str(args.manifest), str(reference), str(key),
                str(output), index,
            )] = task["task_id"]
        for future in as_completed(futures):
            row = future.result(); progress.append(row)
            print(json.dumps(row), flush=True)
    progress.sort(key=lambda row: row["task_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "progress.json", {"rows": progress})
    if any(row.get("error") for row in progress):
        raise SystemExit(2)
    all_results = []
    for task in manifest["tasks"]:
        path = args.output_dir / task["task_id"] / "result.json"
        if path.exists():
            value = _read(path); _self_hash(value, "result_sha256")
            all_results.append(value)
    if len(all_results) == int(manifest["task_count"]):
        report = _analyze(manifest, all_results)
        _write(args.output_dir / "report.json", report)
        print(json.dumps({
            "status": report["status"],
            "condition_successes": report["condition_successes"],
            "source_vs_neural": report["source_vs_neural"],
            "gates": report["gates"],
            "report_sha256": report["report_sha256"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
