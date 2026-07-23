#!/usr/bin/env python3
"""Evaluate a target policy with a separate advisory-only Harness Agent.

The target policy is the only component allowed to propose environment actions.
The Harness Agent may initialize receipt-bound hypotheses, review a hashed policy
proposal, request policy re-planning, and diagnose a real transition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_wrappers.alfworld_nl_wrapper import make_alfworld_env  # noqa: E402
from harness.advisory_reasoning_harness import (  # noqa: E402
    MatchedHarnessCallLedger,
    admit_policy_proposal,
    parse_adaptation_binding_set,
    policy_execution_identity_receipt,
    policy_proposal_id,
)
from harness.agent_reasoning_cycle import (  # noqa: E402
    parse_agent_post_transition_verdict,
)
from harness.frozen_transfer_policy import (  # noqa: E402
    StrictOpenAIClient,
    native_target_action_prompt,
    parse_native_target_plan_reply,
)
from harness.provider_clients import (  # noqa: E402
    StrictOpenAIResponsesClient,
    load_literal_secret,
)


TREATMENTS = (
    # Receipt-grounded causal-motif experiment.
    "authentic", "generic_protocol", "shuffled_topology", "receipt_null",
    "other_source", "target_only",
    # Backward-compatible architecture-smoke conditions.
    "correct", "renamed", "randomized",
)


def _hash(value: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _clean(value: str) -> str:
    return str(value).split("\n\nAdmissible actions:", 1)[0].strip()


def _won(info: Mapping[str, Any]) -> bool:
    value = info.get("won", False)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    return bool(value)


def _source_payload(path: Path, treatment: str) -> tuple[list[str], list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") == "causal_motif_conditioning_v1":
        if str(payload.get("treatment")) != treatment:
            raise SystemExit("causal motif conditioning treatment mismatch")
        if str(payload.get("status")) != "READY":
            raise SystemExit("causal motif conditioning artifact is not ready")
        unsigned = dict(payload)
        claimed = str(unsigned.pop("artifact_hash", ""))
        if not claimed or _hash(unsigned) != claimed:
            raise SystemExit("causal motif conditioning artifact hash mismatch")
        refs = [str(item) for item in payload.get("source_refs") or ()]
        contexts = [dict(item) for item in payload.get("source_contexts") or ()]
        if treatment == "target_only" and (refs or contexts):
            raise SystemExit("target-only causal motif artifact leaked source context")
        if treatment != "target_only" and (not refs or not contexts):
            raise SystemExit("source causal motif artifact has no conditioning")
        return refs, contexts, claimed
    recorded_treatment = str(payload.get("source_treatment"))
    expected_recorded = "empty" if treatment == "target_only" else treatment
    if recorded_treatment != expected_recorded:
        raise SystemExit("conditioning artifact treatment mismatch")
    if str(payload.get("status")) not in {"READY", "ADMITTED", "PARTIAL"}:
        raise SystemExit("conditioning artifact is not admitted")
    refs, contexts = [], []
    if treatment != "target_only":
        for candidate in payload.get("candidates") or []:
            ref = str(candidate.get("source_hypothesis_hash") or "")
            if not ref or ref in refs:
                continue
            refs.append(ref)
            contexts.append({
                "source_ref": ref,
                "nodes": candidate.get("nodes") or [],
                "source_edges": candidate.get("source_edges") or [],
            })
    return refs, contexts, str(payload.get("artifact_hash") or _hash(payload))


def _call_harness(
    client: StrictOpenAIResponsesClient, *, model: str, prompt: str,
    max_tokens: int, reasoning_effort: str, reserve: int,
) -> tuple[str, dict[str, Any]]:
    reply, usage = client.complete(
        model=model, prompt=prompt, max_tokens=max_tokens + reserve,
        reasoning_effort=reasoning_effort,
    )
    usage = dict(usage)
    usage["harness_visible_token_budget"] = max_tokens
    usage["api_reasoning_token_reserve"] = reserve
    return reply, usage


def _policy_action(
    client: StrictOpenAIClient, *, model: str, goal: str, observation: str,
    actions: Sequence[str], history: Sequence[Mapping[str, Any]],
    advisory: Sequence[Mapping[str, Any]], replan_note: str = "",
    generation_seed: int | None = None,
) -> tuple[str, str, str, Mapping[str, Any]]:
    augmented_history = list(history)
    if replan_note:
        augmented_history.append({
            "event": "HARNESS_REQUESTED_POLICY_REPLAN",
            "untrusted_advisory_reason": replan_note,
        })
    prompt = native_target_action_prompt(
        domain="alfworld", goal=goal, observation=observation,
        actions=actions, interaction_history=augmented_history,
        source_conditioning=advisory,
    )
    reply, usage = client.complete(
        model=model, prompt=prompt, max_tokens=512, reasoning_effort="none",
        seed=generation_seed,
    )
    plan = parse_native_target_plan_reply(reply, n=len(actions))
    return actions[plan.action_index], prompt, reply, usage


def _adaptation_prompt(
    *, treatment: str, source_refs: Sequence[str], source_contexts: Sequence[Mapping[str, Any]],
    demo: Mapping[str, Any], goal: str, observation: str,
) -> str:
    return (
        "You are an advisory Harness Agent, not the target policy. Initialize at most "
        "three one-shot reasoning-binding hypotheses. Do not propose or select an "
        "environment action and do not assert source-to-target semantic equivalence. "
        "Every hypothesis must cite exactly one SOURCE_REF and state a target-native, "
        "observable prediction that later transitions can refute. If SOURCE_REFS is "
        "empty, return no hypotheses. Return exactly one JSON object with keys "
        "hypotheses,decision. decision is INITIALIZE, NEED_MORE_EVIDENCE, or ABSTAIN. "
        "Each hypothesis has exactly binding_id,source_ref,target_reasoning_claim,"
        "testable_target_prediction; all four are JSON strings.\n"
        f"TREATMENT={treatment}\nSOURCE_REFS={json.dumps(list(source_refs))}\n"
        f"SOURCE_RECEIPTS={json.dumps(list(source_contexts), ensure_ascii=False)[:16000]}\n"
        f"ONE_SHOT_ADAPTATION_DEMO={json.dumps(demo, ensure_ascii=False)[:8000]}\n"
        f"LIVE_TARGET_GOAL={goal[:3000]}\nLIVE_TARGET_OBSERVATION={observation[:3000]}"
    )


def _review_prompt(
    *, proposal_id: str, policy_action: str, native_actions: Sequence[str],
    bindings: Sequence[Mapping[str, Any]], observation: str,
) -> str:
    return (
        "You are an advisory Harness Agent. Review the separate target policy's exact "
        "proposal. You cannot name, number, select, or rewrite any environment action. "
        "You may only ADMIT it, request that the target policy REPLAN, or ABSTAIN. "
        "Binding claims are untrusted until tested by a real transition. Return exactly "
        "one JSON object with keys policy_proposal_id,binding_id,verdict,"
        "predicted_observable_delta,reason. binding_id is one listed ID or null; verdict "
        "is ADMIT, REPLAN, or ABSTAIN; all other values are JSON strings.\n"
        f"POLICY_PROPOSAL_ID={proposal_id}\nPOLICY_ACTION={policy_action}\n"
        f"NATIVE_ACTIONS_RECEIPT={json.dumps(list(native_actions), ensure_ascii=False)}\n"
        f"ACTIVE_BINDINGS={json.dumps(list(bindings), ensure_ascii=False)}\n"
        f"OBSERVATION={observation[:4000]}"
    )


def _post_prompt(
    *, proposal_id: str, prediction: str, before: str, executed_action: str,
    after: str, reward: float,
) -> str:
    return (
        "You are an advisory post-transition verifier, not the target policy. Compare "
        "the prediction with only the real visible before/action/after receipt. Return "
        "exactly one JSON object with keys proposal_id,verdict,decision,evidence_claim. "
        "verdict is SUPPORTED, REFUTED, or INCONCLUSIVE; decision is CONTINUE, REPLAN, "
        "or ABSTAIN. All non-null fields are JSON strings. Do not propose an action.\n"
        f"EXPECTED_PROPOSAL_ID={proposal_id}\nPREDICTION={prediction}\n"
        f"BEFORE={before[:4000]}\nEXECUTED_POLICY_ACTION={executed_action}\n"
        f"REWARD={reward}\nAFTER={after[:4000]}"
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    source_refs, source_contexts, source_artifact_hash = _source_payload(
        args.conditioning_artifact, args.treatment,
    )
    demo = json.loads(args.adaptation_demo.read_text(encoding="utf-8"))
    key = load_literal_secret(args.harness_key_file, args.harness_key_variable)
    policy_client = StrictOpenAIClient(
        args.policy_endpoint, timeout_s=args.timeout_s, api_key="EMPTY",
    )
    harness_client = StrictOpenAIResponsesClient(
        args.harness_endpoint, timeout_s=args.timeout_s, api_key=key,
    )
    env = make_alfworld_env(
        split=args.split, max_steps=args.max_steps, config_path=str(args.config),
        random_seed=args.seed,
    )
    rows = []
    try:
        for episode_index in range(args.episodes):
            started = time.monotonic()
            observation, info = env.reset()
            goal = str((info.get("structured_state") or {}).get("task_goal") or "").strip()
            if not goal:
                goal = _clean(observation)
            matched_identity = {
                "comparison_id": f"seed-{args.seed}:episode-{episode_index}",
                "initial_state_sha256": _hash({
                    "goal": goal,
                    "observation": _clean(observation),
                    "native_actions": [str(item) for item in info.get("action_names") or ()],
                }),
                "prefix_sha256": _hash([]),
                "policy_identity_sha256": _hash({
                    "endpoint": args.policy_endpoint,
                    "model": args.policy_model,
                }),
                "budget_sha256": _hash({
                    "max_steps": args.max_steps,
                    "max_harness_calls": args.max_harness_calls,
                    "audit_interval": args.audit_interval,
                    "policy_max_tokens": 512,
                }),
            }
            ledger = MatchedHarnessCallLedger(
                max_calls=args.max_harness_calls,
                audit_interval=args.audit_interval,
            )
            adaptation_prompt = _adaptation_prompt(
                treatment=args.treatment, source_refs=source_refs,
                source_contexts=source_contexts, demo=demo, goal=goal,
                observation=_clean(observation),
            )
            adaptation_raw, adaptation_usage = _call_harness(
                harness_client, model=args.harness_model, prompt=adaptation_prompt,
                max_tokens=700, reasoning_effort=args.harness_reasoning_effort,
                reserve=args.harness_reasoning_token_reserve,
            )
            adaptation_error = None
            adaptation = None
            try:
                adaptation = parse_adaptation_binding_set(
                    adaptation_raw, allowed_source_refs=source_refs,
                    target_only=args.treatment == "target_only",
                )
            except Exception as exc:
                adaptation_error = f"{type(exc).__name__}:{exc}"
            ledger.record(
                phase="ADAPTATION", step=None, effective=True,
                prompt_sha256=_hash(adaptation_prompt),
                generation_id=str(adaptation_usage.get("generation_id") or ""),
            )
            bindings = (
                [item.__dict__ for item in adaptation.hypotheses]
                if adaptation is not None and adaptation.decision == "INITIALIZE" else []
            )
            source_active = bool(bindings)
            actions_taken: list[str] = []
            history: list[dict[str, Any]] = []
            traces: list[dict[str, Any]] = []
            success = _won(info)
            terminated = truncated = False
            error = None
            cumulative_reward = 0.0
            while not (success or terminated or truncated) and len(actions_taken) < args.max_steps:
                step = len(actions_taken)
                native_actions = [str(item) for item in info.get("action_names") or ()]
                if not native_actions:
                    error = "NO_NATIVE_ACTIONS"
                    break
                advisory_for_policy = bindings if source_active else []
                try:
                    policy_action, policy_prompt, policy_reply, policy_usage = _policy_action(
                        policy_client, model=args.policy_model, goal=goal,
                        observation=_clean(observation), actions=native_actions,
                        history=history, advisory=advisory_for_policy,
                        generation_seed=args.seed + episode_index * 10000 + step * 2,
                    )
                except Exception as exc:
                    error = f"POLICY_FAILURE:{type(exc).__name__}:{exc}"
                    break
                proposal_id = policy_proposal_id(
                    target_state=_clean(observation), native_actions=native_actions,
                    policy_prompt_sha256=_hash(policy_prompt), policy_reply=policy_reply,
                    policy_action=policy_action,
                )
                trace: dict[str, Any] = {
                    "step": step, "policy_model": args.policy_model,
                    "policy_prompt_sha256": _hash(policy_prompt),
                    "policy_reply_sha256": _hash(policy_reply),
                    "policy_usage": dict(policy_usage),
                    "initial_policy_proposal_id": proposal_id,
                    "initial_policy_action": policy_action,
                    "source_active_before": source_active,
                    "harness_agent_can_emit_action": False,
                }
                admission = None
                review_usage = None
                reviewed_proposal_id = proposal_id
                should_audit = ledger.scheduled_pre_action(step) and ledger.remaining >= 2
                if should_audit:
                    prompt = _review_prompt(
                        proposal_id=proposal_id, policy_action=policy_action,
                        native_actions=native_actions,
                        bindings=advisory_for_policy,
                        observation=_clean(observation),
                    )
                    raw, review_usage = _call_harness(
                        harness_client, model=args.harness_model, prompt=prompt,
                        max_tokens=450, reasoning_effort=args.harness_reasoning_effort,
                        reserve=args.harness_reasoning_token_reserve,
                    )
                    ledger.record(
                        phase="PRE_ACTION", step=step, effective=True,
                        prompt_sha256=_hash(prompt),
                        generation_id=str(review_usage.get("generation_id") or ""),
                    )
                    admission = admit_policy_proposal(
                        treatment=args.treatment, target_state=_clean(observation),
                        native_actions=native_actions,
                        policy_prompt_sha256=_hash(policy_prompt),
                        policy_reply=policy_reply, policy_action=policy_action,
                        advisory_prompt_sha256=_hash(prompt), advisory_reply=raw,
                        allowed_binding_ids=[row["binding_id"] for row in bindings],
                    )
                    trace["advisory_admission"] = asdict(admission)
                    review = admission.advisory_review or {}
                    if admission.status == "POLICY_REPLAN_REQUESTED":
                        try:
                            policy_action, policy_prompt, policy_reply, policy_usage = _policy_action(
                                policy_client, model=args.policy_model, goal=goal,
                                observation=_clean(observation), actions=native_actions,
                                history=history, advisory=advisory_for_policy,
                                replan_note=str(review.get("reason") or "replan requested"),
                                generation_seed=(
                                    args.seed + episode_index * 10000 + step * 2 + 1
                                ),
                            )
                            proposal_id = policy_proposal_id(
                                target_state=_clean(observation), native_actions=native_actions,
                                policy_prompt_sha256=_hash(policy_prompt),
                                policy_reply=policy_reply, policy_action=policy_action,
                            )
                            trace["replanned_policy_proposal_id"] = proposal_id
                            trace["replanned_policy_action"] = policy_action
                        except Exception as exc:
                            error = f"POLICY_REPLAN_FAILURE:{type(exc).__name__}:{exc}"
                            break
                    elif admission.status != "POLICY_ACTION_ADMITTED":
                        # A malformed/abstaining GPT review is diagnostic only. It
                        # cannot refute a binding or mutate its evidence state.
                        trace["advisory_review_effect"] = "NO_BINDING_STATE_CHANGE"
                before = _clean(observation)
                observation, reward, terminated, truncated, info = env.step(policy_action)
                actions_taken.append(policy_action)
                cumulative_reward += float(reward)
                success = _won(info)
                identity = policy_execution_identity_receipt(
                    policy_proposal_id_value=proposal_id, policy_action=policy_action,
                    executed_action=policy_action, native_actions=native_actions,
                )
                if not identity.execution_matches_policy:
                    raise AssertionError("environment action escaped target policy proposal")
                trace["policy_execution_identity"] = asdict(identity)
                trace["reward"] = float(reward)
                if should_audit:
                    prediction = str(
                        (admission.advisory_review or {}).get(
                            "predicted_observable_delta", "no admitted prediction",
                        ) if admission is not None else "no admitted prediction"
                    )
                    post_prompt = _post_prompt(
                        proposal_id=reviewed_proposal_id, prediction=prediction,
                        before=before, executed_action=policy_action,
                        after=_clean(observation), reward=float(reward),
                    )
                    post_raw, post_usage = _call_harness(
                        harness_client, model=args.harness_model, prompt=post_prompt,
                        max_tokens=350, reasoning_effort=args.harness_reasoning_effort,
                        reserve=args.harness_reasoning_token_reserve,
                    )
                    ledger.record(
                        phase="POST_TRANSITION", step=step, effective=True,
                        prompt_sha256=_hash(post_prompt),
                        generation_id=str(post_usage.get("generation_id") or ""),
                    )
                    post_error = None
                    post = None
                    try:
                        post = parse_agent_post_transition_verdict(
                            post_raw, expected_proposal_id=reviewed_proposal_id,
                        )
                    except Exception as exc:
                        post_error = f"{type(exc).__name__}:{exc}"
                    trace["post_transition"] = {
                        "prompt_sha256": _hash(post_prompt),
                        "reply_sha256": _hash(post_raw), "usage": dict(post_usage),
                        "verdict": post.to_dict() if post is not None else None,
                        "parse_error": post_error,
                        "binding_state_authority": "MATCHED_OFFICIAL_ENVIRONMENT_OUTCOME_ONLY",
                        "changes_binding_state": False,
                    }
                history.append({
                    "step": step, "action": policy_action,
                    "observation_after": _clean(observation)[:3000],
                    "reward": float(reward), "official_success": success,
                })
                trace["source_active_after"] = source_active
                traces.append(trace)
            while ledger.remaining:
                padding_prompt = (
                    "Matched-compute padding call. It cannot affect any action or state. "
                    "Return exactly {\"ack\":\"PADDING\"}."
                )
                _, padding_usage = _call_harness(
                    harness_client, model=args.harness_model, prompt=padding_prompt,
                    max_tokens=32, reasoning_effort=args.harness_reasoning_effort,
                    reserve=args.harness_reasoning_token_reserve,
                )
                ledger.record(
                    phase="PADDING", step=None, effective=False,
                    prompt_sha256=_hash(padding_prompt),
                    generation_id=str(padding_usage.get("generation_id") or ""),
                )
            rows.append({
                "episode_index": episode_index, "treatment": args.treatment,
                "matched_identity": matched_identity,
                "success": success, "steps": len(actions_taken),
                "actions": actions_taken, "cumulative_reward": cumulative_reward,
                "error": error, "adaptation": {
                    "prompt_sha256": _hash(adaptation_prompt),
                    "reply_sha256": _hash(adaptation_raw),
                    "usage": dict(adaptation_usage),
                    "binding_set": adaptation.to_dict() if adaptation is not None else None,
                    "parse_error": adaptation_error,
                },
                "call_ledger": ledger.to_dict(), "traces": traces,
                "wall_time_s": time.monotonic() - started,
            })
    finally:
        env.close()
        policy_client.close()
        harness_client.close()
    valid = [row for row in rows if row["error"] is None]
    return {
        "schema_version": 1,
        "protocol": "separate_target_policy_advisory_harness_v1",
        "treatment": args.treatment,
        "source_artifact_hash": source_artifact_hash,
        "policy_model": args.policy_model,
        "harness_model": args.harness_model,
        "harness_agent_can_emit_or_execute_actions": False,
        "harness_agent_can_confirm_or_refute_binding": False,
        "binding_state_authority": "MATCHED_OFFICIAL_ENVIRONMENT_OUTCOME_ONLY",
        "matched_call_cap": args.max_harness_calls,
        "audit_interval": args.audit_interval,
        "target_gradient_updates": 0,
        "summary": {
            "episodes": len(rows), "valid": len(valid),
            "errors": len(rows) - len(valid),
            "success_rate": sum(row["success"] for row in valid) / len(valid) if valid else 0.0,
            "mean_steps": mean(row["steps"] for row in valid) if valid else 0.0,
            "all_executions_match_policy": all(
                trace["policy_execution_identity"]["execution_matches_policy"]
                for row in valid for trace in row["traces"]
            ),
            "all_call_budgets_exact": all(
                row["call_ledger"]["used_calls"] == args.max_harness_calls
                for row in rows
            ),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", choices=TREATMENTS, required=True)
    parser.add_argument("--conditioning-artifact", type=Path, required=True)
    parser.add_argument("--adaptation-demo", type=Path, required=True)
    parser.add_argument("--policy-endpoint", required=True)
    parser.add_argument("--policy-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--harness-endpoint", default="https://us.api.openai.com/v1")
    parser.add_argument("--harness-model", default="gpt-5-mini")
    parser.add_argument("--harness-key-file", type=Path, required=True)
    parser.add_argument("--harness-key-variable", default="OPENAI_API_KEY")
    parser.add_argument("--harness-reasoning-effort", choices=("minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--harness-reasoning-token-reserve", type=int, default=512)
    parser.add_argument("--max-harness-calls", type=int, default=6)
    parser.add_argument("--audit-interval", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--split", choices=("train", "eval_in_distribution", "eval_out_of_distribution"), default="eval_in_distribution")
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/alfworld_pick_and_place_config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.conditioning_artifact, args.adaptation_demo, args.harness_key_file, args.config):
        if not path.is_file():
            parser.error(f"missing required file: {path}")
    if args.max_harness_calls < 3:
        parser.error("max-harness-calls must allow adaptation plus one audit pair")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
