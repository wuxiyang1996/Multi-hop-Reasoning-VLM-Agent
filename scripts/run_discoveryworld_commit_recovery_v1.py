#!/usr/bin/env python3
"""Run a matched development fork around an unsafe DiscoveryWorld COMMIT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.discoveryworld_env import (  # noqa: E402
    DETERMINISM_PROTOCOL,
    DiscoveryWorldEnvironment,
    stable_hash,
)
from motif_transfer.discoveryworld_policy import target_native_facts  # noqa: E402
from motif_transfer.discoveryworld_sokoban_transfer import (  # noqa: E402
    SOURCE_CONFIRMATION_SHA256,
    SOURCE_PROGRAM_SHA256,
    TARGET_BINDER_SYSTEM_PROMPT,
    TARGET_GROUNDER_SYSTEM_PROMPT,
    binder_prompt_payload,
    grounder_prompt_payload,
    parse_grounded_candidates,
    parse_target_binding,
    realize_localized_spatial_position,
    select_candidate,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def call_grounder(
    backend: MemoizedCompletionBackend,
    observation,
    *,
    memory: str,
    hypotheses: tuple[str, ...],
    recent: list[dict[str, Any]], target_binding,
    attempts: int,
):
    schema_error = None
    audit = []
    for attempt in range(attempts):
        payload = grounder_prompt_payload(
            observation, memory=memory, hypotheses=hypotheses,
            recent=recent, target_binding=target_binding, schema_error=schema_error,
        )
        raw = backend.complete("grounder", TARGET_GROUNDER_SYSTEM_PROMPT, payload)
        request_usage = dict(backend.last_usage)
        try:
            bundle, candidates = parse_grounded_candidates(raw, observation)
            audit.append({
                "attempt": attempt + 1,
                "accepted": True,
                "cache_hit": bool(request_usage.get("cache_hit")),
            })
            return bundle, candidates, raw, audit
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1,
                "accepted": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "cache_hit": bool(request_usage.get("cache_hit")),
            })
    raise RuntimeError(f"target grounder exhausted schema attempts: {audit}")


def call_binder(
    backend: MemoizedCompletionBackend,
    observation,
    *,
    memory: str,
    hypotheses: tuple[str, ...],
    attempts: int,
):
    schema_error = None
    audit = []
    for attempt in range(attempts):
        payload = binder_prompt_payload(
            observation, memory=memory, hypotheses=hypotheses, schema_error=schema_error,
        )
        raw = backend.complete("binder", TARGET_BINDER_SYSTEM_PROMPT, payload)
        request_usage = dict(backend.last_usage)
        try:
            binding = parse_target_binding(raw, observation)
            audit.append({
                "attempt": attempt + 1,
                "accepted": True,
                "cache_hit": bool(request_usage.get("cache_hit")),
            })
            return binding, raw, audit
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1, "accepted": False, "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "cache_hit": bool(request_usage.get("cache_hit")),
            })
    raise RuntimeError(f"target binder exhausted schema attempts: {audit}")


def new_env(task: Mapping[str, Any], max_steps: int, thread_id: int, frame_dir: Path):
    return DiscoveryWorldEnvironment(
        scenario=str(task["scenario"]), difficulty=str(task["difficulty"]),
        seed=int(task["seed"]), max_steps=max_steps, thread_id=thread_id,
        include_vision=False, frame_dir=frame_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spatial_realizer_config = dict(config.get("target_native_spatial_realizer") or {})
    spatial_realizer_enabled = bool(spatial_realizer_config.get("enabled", False))
    reference_path = REPO / config["reference_episode"]
    reference = json.loads(reference_path.read_text())
    stored_episode_hash = reference.get("episode_sha256")
    reference_body = dict(reference)
    reference_body.pop("episode_sha256", None)
    if stored_episode_hash != stable_hash(reference_body):
        raise SystemExit("reference episode self-hash mismatch")
    if stored_episode_hash != config["reference_episode_sha256"]:
        raise SystemExit("reference episode differs from frozen development config")
    source_receipt_path = REPO / config["source_contract"]["compact_receipt"]
    source_receipt = json.loads(source_receipt_path.read_text())
    if not source_receipt.get("fresh_confirmation", {}).get("source_gate_passed"):
        raise SystemExit("source program is not source-qualified")
    if config["source_contract"]["source_program_sha256"] != SOURCE_PROGRAM_SHA256:
        raise SystemExit("configured source artifact hash mismatch")
    if (
        config["source_contract"]["source_confirmation_sha256"]
        != SOURCE_CONFIRMATION_SHA256
    ):
        raise SystemExit("configured source confirmation hash mismatch")
    if source_receipt["artifact"]["artifact_sha256"] != SOURCE_PROGRAM_SHA256:
        raise SystemExit("source artifact hash mismatch")
    if source_receipt["fresh_confirmation"]["report_sha256"] != SOURCE_CONFIRMATION_SHA256:
        raise SystemExit("source confirmation hash mismatch")

    values = runpy.run_path(str(args.keys))
    key = values.get(config["model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    os.environ["DISCOVERYWORLD_OPENROUTER_KEY"] = str(key)
    raw_backend = OpenAICompatibleBackend(
            str(config["model"]["base_url"]),
            {
                "grounder": str(config["model"]["model"]),
                "binder": str(config["model"]["model"]),
            },
            api_key_env="DISCOVERYWORLD_OPENROUTER_KEY", json_mode=True,
            temperature=float(config["model"]["temperature"]), timeout_seconds=180,
            request_overrides={
                "max_tokens": int(config["model"]["maximum_output_tokens"]),
                "reasoning": {
                    "effort": str(config["model"].get("hidden_reasoning_effort") or "low"),
                    "exclude": True,
                },
            },
            transport_attempts=3,
    )
    cache_path = (
        args.output.parent / f"grounder_cache_{file_sha256(args.config)[:12]}.json"
    )
    cache_seed = dict(config.get("neural_cache_seed") or {})
    if cache_seed:
        seed_path = REPO / str(cache_seed["path"])
        expected_seed_hash = str(cache_seed["sha256"])
        if file_sha256(seed_path) != expected_seed_hash:
            raise SystemExit("frozen neural cache seed hash mismatch")
        seed_payload = json.loads(seed_path.read_text())
        if seed_payload.get("backend_identity_sha256") != stable_hash(raw_backend.identity):
            raise SystemExit("frozen neural cache seed backend mismatch")
        if cache_path.exists():
            current_payload = json.loads(cache_path.read_text())
            if current_payload.get("backend_identity_sha256") != stable_hash(
                raw_backend.identity
            ):
                raise SystemExit("current neural cache backend mismatch")
            current_entries = current_payload.get("entries") or {}
            for request_hash, row in (seed_payload.get("entries") or {}).items():
                if current_entries.get(request_hash) != row:
                    raise SystemExit(
                        "current neural cache does not preserve frozen seed entries"
                    )
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(seed_path, cache_path)
    backend = MemoizedCompletionBackend(raw_backend, cache_path=cache_path)

    fork_step = int(config["fork_after_episode_step"])
    prefix = [dict(row["action"]) for row in reference["steps"][:fork_step]]
    task = dict(reference["task"])
    max_steps = fork_step + int(config["recovery_horizon"])
    expected_policy = reference["steps"][fork_step - 1]["transition"]["after_policy_state_sha256"]
    # Recompute the hidden fork with the current deterministic wrapper; unlike
    # policy state, legacy audit hashes intentionally changed when RNG states
    # were added to the hash protocol.
    anchor_env = new_env(task, max_steps, 98100, args.output.parent / "frames" / "anchor")
    anchor, anchor_receipts = anchor_env.replay_prefix(
        prefix, expected_policy_state_sha256=expected_policy,
    )
    fork_audit = anchor_env.current_audit_hash
    fork_policy = anchor.policy_state_sha256
    initial_memory = str(reference["steps"][fork_step - 1].get("memory") or "")
    initial_hypotheses = tuple(
        reference["steps"][fork_step - 1].get("running_hypotheses") or ()
    )
    frozen_initial = config.get("frozen_initial_grounding")
    if frozen_initial is not None:
        if not isinstance(frozen_initial, Mapping):
            raise SystemExit("frozen_initial_grounding must be an object")
        applicability = frozen_initial.get("applicability_receipt")
        if not isinstance(applicability, Mapping) or not applicability.get("eligible"):
            raise SystemExit("frozen initial applicability receipt is absent or ineligible")
        applicability_body = dict(applicability)
        applicability_hash = applicability_body.pop(
            "applicability_receipt_sha256", None,
        )
        if applicability_hash != stable_hash(applicability_body):
            raise SystemExit("frozen applicability receipt self-hash mismatch")
        binder_payload = binder_prompt_payload(
            anchor, memory=initial_memory, hypotheses=initial_hypotheses,
        )
        if frozen_initial.get("binder_prompt_sha256") != stable_hash(binder_payload):
            raise SystemExit("frozen binder prompt differs at replayed fork")
        binder_raw = str(frozen_initial.get("binder_response") or "")
        if frozen_initial.get("binder_response_sha256") != hashlib.sha256(
            binder_raw.encode()
        ).hexdigest():
            raise SystemExit("frozen binder response hash mismatch")
        target_binding = parse_target_binding(binder_raw, anchor)
        if applicability.get("binding_sha256") != target_binding.binding_sha256:
            raise SystemExit("frozen applicability binding mismatch")
        binder_attempts = [{
            "attempt": 0,
            "accepted": True,
            "source": "FROZEN_OUTCOME_BLIND_APPLICABILITY_SCAN",
        }]
    else:
        target_binding, binder_raw, binder_attempts = call_binder(
            backend, anchor, memory=initial_memory, hypotheses=initial_hypotheses,
            attempts=int(config["model"]["schema_attempts"]),
        )

    target_only_env = new_env(
        task, max_steps, 98101, args.output.parent / "frames" / "target_only_recorded",
    )
    target_only_obs, _ = target_only_env.replay_prefix(
        prefix, expected_policy_state_sha256=fork_policy,
        expected_audit_world_sha256=fork_audit,
    )
    recorded_recovery = []
    target_only_after = target_only_obs
    for reference_row in reference["steps"][
        fork_step:fork_step + int(config["recovery_horizon"])
    ]:
        if target_only_after.terminal:
            break
        recorded_action = dict(reference_row["action"])
        before_facts = target_native_facts(target_only_after)
        target_only_after, target_only_transition = target_only_env.step(recorded_action)
        recorded_recovery.append({
            "recovery_step": len(recorded_recovery) + 1,
            "action": recorded_action,
            "transition": asdict(target_only_transition),
            "before_target_native_facts": before_facts,
            "after_target_native_facts": target_native_facts(target_only_after),
        })
    target_only_eval = (
        asdict(target_only_env.finalize_evaluation()) if target_only_after.terminal else None
    )
    results: dict[str, Any] = {
        "target_only_recorded": {
            "recovery": recorded_recovery,
            "terminal": target_only_after.terminal,
            "official_success": target_only_after.official_success,
            "evaluation": target_only_eval,
        },
    }

    selector_config = config["selector"]
    for condition_index, condition in enumerate(config["conditions"]):
        env = new_env(
            task, max_steps, 98200 + condition_index,
            args.output.parent / "frames" / str(condition),
        )
        observation, replay_receipts = env.replay_prefix(
            prefix, expected_policy_state_sha256=fork_policy,
            expected_audit_world_sha256=fork_audit,
        )
        memory = initial_memory
        hypotheses = initial_hypotheses
        recent: list[dict[str, Any]] = []
        localized_target_uuids: set[int] = set()
        recovery = []
        arm_error = None
        while not observation.terminal and len(recovery) < int(config["recovery_horizon"]):
            try:
                if not recovery and frozen_initial is not None:
                    grounder_payload = grounder_prompt_payload(
                        observation,
                        memory=memory,
                        hypotheses=hypotheses,
                        recent=recent,
                        target_binding=target_binding,
                    )
                    if frozen_initial.get("grounder_prompt_sha256") != stable_hash(
                        grounder_payload
                    ):
                        raise RuntimeError(
                            "frozen grounder prompt differs at replayed fork"
                        )
                    raw = str(frozen_initial.get("grounder_response") or "")
                    if frozen_initial.get(
                        "grounder_response_sha256"
                    ) != hashlib.sha256(raw.encode()).hexdigest():
                        raise RuntimeError("frozen grounder response hash mismatch")
                    bundle, candidates = parse_grounded_candidates(raw, observation)
                    if applicability.get("candidate_bundle_sha256") != stable_hash(
                        [row.candidate_sha256 for row in candidates]
                    ):
                        raise RuntimeError(
                            "frozen applicability candidate bundle mismatch"
                        )
                    schema_attempts = [{
                        "attempt": 0,
                        "accepted": True,
                        "source": "FROZEN_OUTCOME_BLIND_APPLICABILITY_SCAN",
                    }]
                else:
                    bundle, candidates, raw, schema_attempts = call_grounder(
                        backend, observation, memory=memory, hypotheses=hypotheses,
                        recent=recent, target_binding=target_binding,
                        attempts=int(config["model"]["schema_attempts"]),
                    )
            except RuntimeError as exc:
                arm_error = str(exc)
                print(json.dumps({
                    "condition": condition,
                    "recovery_step": len(recovery) + 1,
                    "runtime_error": arm_error,
                }), flush=True)
                break
            selected, selection = select_candidate(
                str(condition), candidates, observation,
                prerequisite_threshold=float(selector_config["prerequisite_threshold"]),
                positive_effect_threshold=float(selector_config["positive_effect_threshold"]),
                target_binding=target_binding,
            )
            realized_action, realization = realize_localized_spatial_position(
                selected, observation, target_binding,
                target_was_localized=(
                    spatial_realizer_enabled
                    and target_binding.target_uuid in localized_target_uuids
                ),
            )
            if not spatial_realizer_enabled:
                realization["reason"] = "DISABLED_BY_FROZEN_PROTOCOL"
                realization["receipt_sha256"] = stable_hash({
                    key: value for key, value in realization.items()
                    if key != "receipt_sha256"
                })
            before_facts = target_native_facts(observation)
            after, transition = env.step(realized_action)
            if (
                realized_action.get("action") == "TELEPORT_TO_OBJECT"
                and realized_action.get("arg1") == target_binding.target_uuid
                and transition.action_succeeded
            ):
                localized_target_uuids.add(target_binding.target_uuid)
            memory = str(bundle.get("memory") or memory)[-6000:]
            raw_hypotheses = bundle.get("running_hypotheses") or hypotheses
            hypotheses = tuple(str(row) for row in raw_hypotheses)[-24:]
            row = {
                "recovery_step": len(recovery) + 1,
                "candidate_bundle": [asdict(candidate) for candidate in candidates],
                "candidate_parse_rejections": bundle.get("candidate_parse_rejections", []),
                "choice_set_degenerate": bool(bundle.get("choice_set_degenerate")),
                "grounder_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "grounder_schema_attempts": schema_attempts,
                "selection": asdict(selection),
                "target_native_realization": realization,
                "transition": asdict(transition),
                "before_target_native_facts": before_facts,
                "after_target_native_facts": target_native_facts(after),
            }
            recovery.append(row)
            recent_row = {
                "action": dict(realized_action),
                "selected_role": selected.target_role,
                "action_succeeded": transition.action_succeeded,
                "last_action_message": str(after.ui.get("lastActionMessage") or "")[:1000],
                "expected_effect": selected.expected_effect,
            }
            if dict(realized_action) != dict(selected.action):
                recent_row["grounded_action"] = dict(selected.action)
            recent.append(recent_row)
            observation = after
            print(json.dumps({
                "condition": condition,
                "recovery_step": len(recovery),
                "role": selected.target_role,
                "grounded_action": selected.action,
                "action": realized_action,
                "spatial_realization_active": realization["active"],
                "terminal": observation.terminal,
                "official_success": observation.official_success,
            }), flush=True)
        evaluation = asdict(env.finalize_evaluation()) if observation.terminal else None
        results[str(condition)] = {
            "matched_fork_policy_state_sha256": observation.policy_state_sha256
            if not recovery else recovery[0]["transition"]["before_policy_state_sha256"],
            "matched_fork_audit_world_sha256": replay_receipts[-1].after_audit_world_sha256,
            "recovery": recovery,
            "terminal": observation.terminal,
            "official_success": observation.official_success,
            "evaluation": evaluation,
            "runtime_error": arm_error,
        }

    mechanism_complete = all(
        row.get("runtime_error") is None
        for name, row in results.items() if name != "target_only_recorded"
    )
    configured_status = str(config.get("status") or "")
    qualification = configured_status.startswith("QUALIFICATION")
    formal = configured_status.startswith("FORMAL_RESERVE")
    payload = {
        "schema_version": "discoveryworld-sokoban-commit-recovery-result-v1",
        "status": (
            ("FORMAL_MECHANISM_COMPLETE" if mechanism_complete
             else "FORMAL_MECHANISM_INCOMPLETE")
            if formal else
            ("QUALIFICATION_MECHANISM_COMPLETE" if mechanism_complete
             else "QUALIFICATION_MECHANISM_INCOMPLETE")
            if qualification else
            ("DEVELOPMENT_MECHANISM_COMPLETE" if mechanism_complete
             else "DEVELOPMENT_MECHANISM_INCOMPLETE")
        ),
        "claim_boundary": config["claim_boundary"],
        "task": task,
        "fork_after_episode_step": fork_step,
        "fork_policy_state_sha256": fork_policy,
        "fork_audit_world_sha256": fork_audit,
        "determinism_protocol": DETERMINISM_PROTOCOL,
        "source_program_sha256": SOURCE_PROGRAM_SHA256,
        "source_confirmation_sha256": SOURCE_CONFIRMATION_SHA256,
        "target_binding": asdict(target_binding),
        "target_binding_response_sha256": hashlib.sha256(binder_raw.encode()).hexdigest(),
        "target_binding_schema_attempts": binder_attempts,
        "target_native_spatial_realizer": spatial_realizer_config,
        "neural_cache_seed": cache_seed,
        "initial_grounding_mode": (
            "FROZEN_OUTCOME_BLIND_APPLICABILITY_SCAN"
            if frozen_initial is not None else "LIVE_AT_MATCHED_FORK"
        ),
        "frozen_initial_applicability_receipt_sha256": (
            applicability_hash if frozen_initial is not None else None
        ),
        "source_receipt_file_sha256": file_sha256(source_receipt_path),
        "runtime_hashes": {
            "config": file_sha256(args.config),
            "runner": file_sha256(Path(__file__)),
            "environment": file_sha256(REPO / "src/motif_transfer/discoveryworld_env.py"),
            "target_policy": file_sha256(REPO / "src/motif_transfer/discoveryworld_policy.py"),
            "transfer_selector": file_sha256(
                REPO / "src/motif_transfer/discoveryworld_sokoban_transfer.py"
            ),
        },
        "conditions": results,
        "all_matched_forks": all(
            row.get("matched_fork_policy_state_sha256", fork_policy) == fork_policy
            and row.get("matched_fork_audit_world_sha256", fork_audit) == fork_audit
            for name, row in results.items() if name != "target_only_recorded"
        ),
        "all_selection_receipts_valid": all(
            step["selection"]["receipt_sha256"]
            == stable_hash({k: v for k, v in step["selection"].items() if k != "receipt_sha256"})
            for name, row in results.items() if name != "target_only_recorded"
            for step in row["recovery"]
        ),
        "policy_runtime_saw_oracle_scorecard": False,
    }
    payload["result_sha256"] = stable_hash(payload)
    write_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "all_matched_forks": payload["all_matched_forks"],
        "all_selection_receipts_valid": payload["all_selection_receipts_valid"],
        "outcomes": {
            name: {
                "terminal": row.get("terminal", row.get("transition", {}).get("terminal")),
                "official_success": row.get(
                    "official_success", row.get("transition", {}).get("official_success"),
                ),
            }
            for name, row in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
    TARGET_BINDER_SYSTEM_PROMPT,
    binder_prompt_payload,
    parse_target_binding,
