#!/usr/bin/env python3
"""Freeze first source-effect-guard disagreement forks without task outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.discoveryworld_env import (  # noqa: E402
    DiscoveryWorldEnvironment,
    stable_hash,
)
from motif_transfer.discoveryworld_policy import target_native_facts  # noqa: E402
from motif_transfer.discoveryworld_qualification import (  # noqa: E402
    assess_effect_guard_applicability,
)
from motif_transfer.discoveryworld_sokoban_transfer import (  # noqa: E402
    SOURCE_CONFIRMATION_SHA256,
    SOURCE_PROGRAM_SHA256,
    TARGET_BINDER_SYSTEM_PROMPT,
    TARGET_GROUNDER_SYSTEM_PROMPT,
    binder_prompt_payload,
    grounder_prompt_payload,
    parse_grounded_candidates,
    parse_target_binding,
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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def complete_binder(
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
            observation,
            memory=memory,
            hypotheses=hypotheses,
            schema_error=schema_error,
        )
        raw = backend.complete("binder", TARGET_BINDER_SYSTEM_PROMPT, payload)
        try:
            binding = parse_target_binding(raw, observation)
            audit.append({"attempt": attempt + 1, "accepted": True})
            return binding, raw, payload, audit
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1,
                "accepted": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            })
    raise RuntimeError(json.dumps(audit, sort_keys=True))


def complete_grounder(
    backend: MemoizedCompletionBackend,
    observation,
    *,
    memory: str,
    hypotheses: tuple[str, ...],
    target_binding,
    attempts: int,
):
    schema_error = None
    audit = []
    for attempt in range(attempts):
        payload = grounder_prompt_payload(
            observation,
            memory=memory,
            hypotheses=hypotheses,
            recent=[],
            target_binding=target_binding,
            schema_error=schema_error,
        )
        raw = backend.complete("grounder", TARGET_GROUNDER_SYSTEM_PROMPT, payload)
        try:
            bundle, candidates = parse_grounded_candidates(raw, observation)
            audit.append({"attempt": attempt + 1, "accepted": True})
            return bundle, candidates, raw, payload, audit
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1,
                "accepted": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            })
    raise RuntimeError(json.dumps(audit, sort_keys=True))


def validate_episode(episode: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if episode.get("status") != "TARGET_ONLY_EPISODE_COMPLETE":
        raise ValueError("reference target-only episode is incomplete")
    stored = episode.get("episode_sha256")
    body = dict(episode)
    body.pop("episode_sha256", None)
    if stored != stable_hash(body):
        raise ValueError("reference target-only episode self-hash mismatch")
    expected = config["reference_runtime_hashes"]
    observed = episode.get("runtime_hashes") or {}
    for name, digest in expected.items():
        if observed.get(name) != digest:
            raise ValueError(f"reference runtime hash mismatch: {name}")
    if episode.get("policy_runtime_saw_oracle_scorecard") is not False:
        raise ValueError("reference policy saw the oracle scorecard")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    baseline_dir = REPO / str(config["reference_baseline_dir"])

    source_receipt_path = REPO / config["source_contract"]["compact_receipt"]
    source_receipt = json.loads(source_receipt_path.read_text())
    if not source_receipt.get("fresh_confirmation", {}).get("source_gate_passed"):
        raise SystemExit("source program is not source-qualified")
    if config["source_contract"]["source_program_sha256"] != SOURCE_PROGRAM_SHA256:
        raise SystemExit("configured source program hash mismatch")
    if (
        config["source_contract"]["source_confirmation_sha256"]
        != SOURCE_CONFIRMATION_SHA256
    ):
        raise SystemExit("configured source confirmation hash mismatch")

    values = runpy.run_path(str(args.keys))
    key = values.get(config["model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    os.environ["DISCOVERYWORLD_V18_OPENROUTER_KEY"] = str(key)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            str(config["model"]["base_url"]),
            {
                "grounder": str(config["model"]["model"]),
                "binder": str(config["model"]["model"]),
            },
            api_key_env="DISCOVERYWORLD_V18_OPENROUTER_KEY",
            json_mode=True,
            temperature=float(config["model"]["temperature"]),
            timeout_seconds=180,
            request_overrides={
                "max_tokens": int(config["model"]["maximum_output_tokens"]),
                "reasoning": {
                    "effort": str(
                        config["model"].get("hidden_reasoning_effort") or "low"
                    ),
                    "exclude": True,
                },
            },
            transport_attempts=3,
        ),
        cache_path=args.run_dir / f"scanner_cache_{file_sha256(args.config)[:12]}.json",
    )

    scan_receipts = []
    generated_configs = []
    eligibility = config["eligibility"]
    selector = config["selector"]
    minimum_fork = int(eligibility["minimum_fork_after_episode_step"])
    schema_attempts = int(config["model"].get("eligibility_schema_attempts", 2))

    for task_index, task_id in enumerate(config["task_ids"]):
        episode_path = baseline_dir / f"{task_id}.json"
        episode = json.loads(episode_path.read_text())
        validate_episode(episode, config)
        if episode.get("task_id") != task_id:
            raise SystemExit(f"reference task identity mismatch: {task_id}")
        steps = episode.get("steps")
        if not isinstance(steps, list) or not steps:
            raise SystemExit(f"reference has no policy steps: {task_id}")
        task = dict(episode["task"])
        env = DiscoveryWorldEnvironment(
            scenario=str(task["scenario"]),
            difficulty=str(task["difficulty"]),
            seed=int(task["seed"]),
            max_steps=len(steps),
            thread_id=99100 + task_index,
            include_vision=False,
            frame_dir=args.run_dir / "frames" / str(task_id),
        )
        observation = env.reset()
        if observation.policy_state_sha256 != episode["initial_policy_state_sha256"]:
            raise SystemExit(f"initial policy state mismatch: {task_id}")
        if env.current_audit_hash != episode["initial_audit_world_sha256"]:
            raise SystemExit(f"initial hidden state mismatch: {task_id}")

        attempts_log = []
        selected = None
        for index, reference_row in enumerate(steps):
            # At loop entry, `index` recorded actions have been replayed.  We
            # assess only noninitial states that precede another recorded
            # action; task terminal/outcome fields are never consulted.
            if index >= minimum_fork:
                memory = str(steps[index - 1].get("memory") or "")
                hypotheses = tuple(
                    str(value)
                    for value in (steps[index - 1].get("running_hypotheses") or ())
                )
                facts = target_native_facts(observation)
                if not facts["inventory"]:
                    attempts_log.append({
                        "fork_after_episode_step": index,
                        "reason": "NO_HELD_SUBJECT_FOR_DROP_OR_PUT",
                    })
                else:
                    try:
                        binding, binder_raw, binder_payload, binder_audit = complete_binder(
                            backend,
                            observation,
                            memory=memory,
                            hypotheses=hypotheses,
                            attempts=schema_attempts,
                        )
                    except RuntimeError as exc:
                        attempts_log.append({
                            "fork_after_episode_step": index,
                            "reason": "BINDER_SCHEMA_UNAVAILABLE",
                            "audit_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                        })
                    else:
                        try:
                            bundle, candidates, grounder_raw, grounder_payload, grounder_audit = complete_grounder(
                                backend,
                                observation,
                                memory=memory,
                                hypotheses=hypotheses,
                                target_binding=binding,
                                attempts=schema_attempts,
                            )
                        except RuntimeError as exc:
                            attempts_log.append({
                                "fork_after_episode_step": index,
                                "reason": "GROUNDER_SCHEMA_UNAVAILABLE",
                                "binding_sha256": binding.binding_sha256,
                                "audit_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                            })
                        else:
                            assessment = assess_effect_guard_applicability(
                                observation,
                                binding,
                                candidates,
                                allowed_commit_actions=eligibility[
                                    "allowed_commit_actions"
                                ],
                                minimum_binding_confidence=float(
                                    eligibility["minimum_binding_confidence"]
                                ),
                                prerequisite_threshold=float(
                                    selector["prerequisite_threshold"]
                                ),
                                positive_effect_threshold=float(
                                    selector["positive_effect_threshold"]
                                ),
                            )
                            attempts_log.append({
                                "fork_after_episode_step": index,
                                "reason": assessment["reason"],
                                "eligible": assessment["eligible"],
                                "binding_sha256": binding.binding_sha256,
                                "binding_response_sha256": hashlib.sha256(
                                    binder_raw.encode()
                                ).hexdigest(),
                                "grounder_response_sha256": hashlib.sha256(
                                    grounder_raw.encode()
                                ).hexdigest(),
                                "applicability_receipt_sha256": assessment[
                                    "applicability_receipt_sha256"
                                ],
                                "binder_schema_attempts": binder_audit,
                                "grounder_schema_attempts": grounder_audit,
                                "candidate_parse_rejection_count": len(
                                    bundle.get("candidate_parse_rejections", [])
                                ),
                            })
                            if assessment["eligible"]:
                                selected = {
                                    "fork_after_episode_step": index,
                                    "memory": memory,
                                    "hypotheses": hypotheses,
                                    "binding": binding,
                                    "binder_raw": binder_raw,
                                    "binder_payload": binder_payload,
                                    "grounder_raw": grounder_raw,
                                    "grounder_payload": grounder_payload,
                                    "assessment": assessment,
                                }
                                break

            transition_ref = reference_row.get("transition")
            if not isinstance(transition_ref, Mapping):
                raise SystemExit(f"missing transition receipt: {task_id}/{index + 1}")
            if observation.policy_state_sha256 != transition_ref.get(
                "before_policy_state_sha256"
            ):
                raise SystemExit(f"before policy replay mismatch: {task_id}/{index + 1}")
            if env.current_audit_hash != transition_ref.get("before_audit_world_sha256"):
                raise SystemExit(f"before hidden replay mismatch: {task_id}/{index + 1}")
            observation, transition = env.step(dict(reference_row["action"]))
            if transition.after_policy_state_sha256 != transition_ref.get(
                "after_policy_state_sha256"
            ):
                raise SystemExit(f"after policy replay mismatch: {task_id}/{index + 1}")
            if transition.after_audit_world_sha256 != transition_ref.get(
                "after_audit_world_sha256"
            ):
                raise SystemExit(f"after hidden replay mismatch: {task_id}/{index + 1}")

        eligible = selected is not None
        task_receipt = {
            "schema_version": "discoveryworld-applicability-scan-task-v18",
            "task_id": str(task_id),
            "task": task,
            "reference_episode_sha256": episode["episode_sha256"],
            "eligible": eligible,
            "reason": (
                "FIRST_SOURCE_EFFECT_GUARD_DISAGREEMENT"
                if eligible else "NO_SOURCE_EFFECT_GUARD_DISAGREEMENT"
            ),
            "fork_after_episode_step": (
                selected["fork_after_episode_step"] if selected else None
            ),
            "scan_attempts": attempts_log,
            "policy_visible_state_read": True,
            "reward_or_evaluator_fields_read": False,
            "recorded_action_success_read": False,
            "recorded_terminal_or_official_success_read": False,
        }
        task_receipt["scan_receipt_sha256"] = stable_hash(task_receipt)
        scan_receipts.append(task_receipt)

        if selected is None:
            print(json.dumps({
                "task_id": task_id,
                "eligible": False,
                "states_assessed": len(attempts_log),
            }), flush=True)
            continue

        baseline_relative = episode_path.resolve().relative_to(REPO.resolve())
        generated = {
            "schema_version": "discoveryworld-sokoban-frozen-fork-v3",
            "status": "DEVELOPMENT_FIRST_APPLICABILITY_FROZEN",
            "claim_boundary": config["claim_boundary"],
            "reference_episode": str(baseline_relative),
            "reference_episode_sha256": episode["episode_sha256"],
            "fork_after_episode_step": selected["fork_after_episode_step"],
            "recovery_horizon": config["recovery_horizon"],
            "conditions": list(config["conditions"]),
            "source_contract": dict(config["source_contract"]),
            "selector": dict(config["selector"]),
            "model": dict(config["model"]),
            "applicability_scan_config_sha256": file_sha256(args.config),
            "applicability_scan_receipt_sha256": task_receipt["scan_receipt_sha256"],
            "frozen_initial_grounding": {
                "binder_prompt_sha256": stable_hash(selected["binder_payload"]),
                "binder_response": selected["binder_raw"],
                "binder_response_sha256": hashlib.sha256(
                    selected["binder_raw"].encode()
                ).hexdigest(),
                "grounder_prompt_sha256": stable_hash(selected["grounder_payload"]),
                "grounder_response": selected["grounder_raw"],
                "grounder_response_sha256": hashlib.sha256(
                    selected["grounder_raw"].encode()
                ).hexdigest(),
                "applicability_receipt": selected["assessment"],
            },
        }
        generated_path = args.output_dir / f"{task_id}.json"
        write_json(generated_path, generated)
        generated_configs.append(
            str(generated_path.resolve().relative_to(REPO.resolve()))
        )
        print(json.dumps({
            "task_id": task_id,
            "eligible": True,
            "fork_after_episode_step": selected["fork_after_episode_step"],
            "states_assessed": len(attempts_log),
        }), flush=True)

    summary = {
        "schema_version": "discoveryworld-applicability-scan-summary-v18",
        "status": "POST_FORMAL_ADAPTATION_SCAN_COMPLETE",
        "claim_boundary": config["claim_boundary"],
        "config_file_sha256": file_sha256(args.config),
        "scanner_file_sha256": file_sha256(Path(__file__)),
        "source_program_sha256": SOURCE_PROGRAM_SHA256,
        "source_confirmation_sha256": SOURCE_CONFIRMATION_SHA256,
        "tasks": len(scan_receipts),
        "eligible_forks": sum(bool(row["eligible"]) for row in scan_receipts),
        "outcome_fields_read_for_eligibility": False,
        "receipts": scan_receipts,
        "generated_configs": generated_configs,
    }
    summary["summary_sha256"] = stable_hash(summary)
    write_json(args.output_dir / "applicability_scan_receipt.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "tasks": summary["tasks"],
        "eligible_forks": summary["eligible_forks"],
        "summary_sha256": summary["summary_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
