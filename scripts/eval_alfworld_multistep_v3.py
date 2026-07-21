#!/usr/bin/env python3
"""Small/frozen ALFWorld v3 candidate-set consensus evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_wrappers.alfworld_nl_wrapper import make_alfworld_env  # noqa: E402
from harness.candidate_set_runtime import (  # noqa: E402
    CandidateActionProposal,
    FrozenCandidateSetRuntime,
)
from harness.frozen_transfer_policy import (  # noqa: E402
    StrictOpenAIClient,
    action_prompt,
    parse_exact_numbered_response,
)
from harness.multistep_binding import multistep_artifact_from_dict  # noqa: E402
from harness.online_rebinding import (  # noqa: E402
    OnlineRebindingAdmission,
    action_evidence_contract_prompt,
    build_action_contract_scope,
    build_rebind_scope,
    online_rebind_prompt,
    parse_action_evidence_contract_reply,
    parse_online_rebind_reply,
    qualify_action_evidence_contract,
    verify_action_evidence_contract,
    verify_rebind_evidence,
)
from harness.online_transfer_runtime import (  # noqa: E402
    NativeTransitionEvidence,
    OnlineTransferController,
    OnlineTransferState,
)
from harness.reasoning_event_log import ReasoningEventKind, ReasoningEventRecorder  # noqa: E402
from harness.source_conditioning_controls import rotate_source_conditioning  # noqa: E402


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _won(info) -> bool:
    value = info.get("won", False)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    return bool(value)


def _clean(value: str) -> str:
    return str(value).split("\n\nAdmissible actions:", 1)[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--condition", choices=("source", "target_only"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--split", default="train", choices=("train", "eval_in_distribution", "eval_out_of_distribution"))
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--online-source-control", action="store_true",
        help="Enable episode-local source verification and live target-only fallback.",
    )
    parser.add_argument(
        "--shadow-source-control", action="store_true",
        help=(
            "Make and verify the same pre-action contracts without gating execution; "
            "use as the compute-matched Harness-off development control."
        ),
    )
    parser.add_argument(
        "--fallback-artifact", type=Path,
        help="Optional legacy same-demo receipt; live fallback no longer depends on its prefix.",
    )
    parser.add_argument("--max-rebind-requests", type=int, default=1)
    parser.add_argument("--max-consecutive-no-delta", type=int, default=2)
    parser.add_argument(
        "--source-conditioning-control", choices=("none", "rotate"), default="none",
        help="Explicit untrusted-conditioning ablation; never changes the frozen artifact.",
    )
    parser.add_argument("--conditioning-control-seed", type=int, default=1729)
    parser.add_argument(
        "--require-binding-source-control", action="store_true",
        help="Reject legacy artifacts lacking a pre-binding E/S/W/R control receipt.",
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/alfworld_pick_and_place_config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    artifact = multistep_artifact_from_dict(artifact_payload)
    expected_origin = "SOURCE_HYPOTHESIS" if args.condition == "source" else "TARGET_NATIVE_SAME_DEMO"
    if any(item.candidate.origin.value != expected_origin for item in artifact.candidates):
        raise SystemExit("condition/artifact origin mismatch")
    if args.require_binding_source_control:
        expected_treatments = (
            {"correct", "wrong", "renamed"}
            if args.condition == "source" else {"empty"}
        )
        if (
            artifact.source_treatment not in expected_treatments
            or not artifact.source_control_receipt_sha256
        ):
            raise SystemExit("artifact lacks required pre-binding source control provenance")
    if args.condition != "source" and args.source_conditioning_control != "none":
        raise SystemExit("source-conditioning controls require source condition")
    if args.online_source_control and args.shadow_source_control:
        raise SystemExit("online and shadow source control are mutually exclusive")
    if args.shadow_source_control and args.condition != "source":
        raise SystemExit("shadow source control requires source condition")
    contract_control_enabled = (
        args.online_source_control or args.shadow_source_control
    )
    fallback_artifact = None
    if args.online_source_control:
        if args.condition != "source":
            raise SystemExit("--online-source-control is only valid for source condition")
        if args.fallback_artifact is not None:
            fallback_artifact = multistep_artifact_from_dict(json.loads(
                args.fallback_artifact.read_text(encoding="utf-8")
            ))
            if any(
                item.candidate.origin.value != "TARGET_NATIVE_SAME_DEMO"
                for item in fallback_artifact.candidates
            ):
                raise SystemExit("fallback artifact must be TARGET_NATIVE_SAME_DEMO")
            if fallback_artifact.demo_hash != artifact.demo_hash:
                raise SystemExit("source and fallback artifacts must use the same target demo")
    key = os.environ.get(args.api_key_env, "").strip()
    if not key and "openrouter.ai" in args.endpoint.lower():
        try:
            from API_func import open_router_api_key
            key = str(open_router_api_key or "").strip()
        except Exception:
            key = ""
    if "openrouter.ai" in args.endpoint.lower() and not key:
        raise SystemExit("OpenRouter API key unavailable")

    client = StrictOpenAIClient(args.endpoint, timeout_s=180, api_key=key or "EMPTY")
    env = make_alfworld_env(
        split=args.split, max_steps=args.max_steps, config_path=str(args.config),
        random_seed=args.seed,
    )
    rows = []
    try:
        for episode_index in range(args.episodes):
            started = time.monotonic()
            observation, info = env.reset()
            goal = _clean(observation)
            initial_actions = [str(item) for item in info.get("action_names") or ()]
            target_instance_identity = {
                "split": args.split,
                "seed": args.seed,
                "episode_index": episode_index,
                "goal_sha256": _hash(goal),
                "initial_observation_sha256": _hash(_clean(observation)),
                "initial_native_actions_sha256": _hash(initial_actions),
                "environment_task_id": str(
                    info.get("gamefile") or info.get("task_id") or "UNAVAILABLE"
                ),
            }
            target_instance_identity["identity_sha256"] = _hash(
                target_instance_identity
            )
            runtime = FrozenCandidateSetRuntime(artifact)
            online = (
                OnlineTransferController(
                    max_rebind_requests=args.max_rebind_requests,
                    max_consecutive_no_delta=args.max_consecutive_no_delta,
                )
                if args.online_source_control else None
            )
            rebind_admission = OnlineRebindingAdmission()
            recorder = ReasoningEventRecorder(
                f"v3-{args.condition}-{args.split}-seed{args.seed}-ep{episode_index}"
            )
            recorder.append(ReasoningEventKind.RESET, {
                "seed": args.seed, "split": args.split,
                "artifact_hash": artifact.artifact_hash,
                "target_instance_identity": target_instance_identity,
            })
            actions = []
            traces = []
            error = None
            abstain_reason = None
            terminated = truncated = False
            success = _won(info)
            max_native_step_reward = 0.0
            cumulative_native_reward = 0.0
            while not (success or terminated or truncated) and len(actions) < args.max_steps:
                if online is not None and online.state == OnlineTransferState.REBIND_REQUIRED:
                    admissible = [str(item) for item in info.get("action_names") or ()]
                    active_contexts = runtime.active_source_conditioning()
                    control_receipt = None
                    control_gap = None
                    if (
                        len(active_contexts) >= 2
                        and args.source_conditioning_control == "rotate"
                    ):
                        active_contexts, control_receipt = rotate_source_conditioning(
                            active_contexts,
                            seed=args.conditioning_control_seed,
                            step=len(actions),
                        )
                    elif active_contexts and args.source_conditioning_control == "rotate":
                        control_gap = "CONTROL_UNAVAILABLE_SINGLE_ACTIVE_CONTEXT"
                    trace = {
                        "step": len(actions), "treatment": "online_rebind",
                        "status": "PENDING", "reason": None, "command": None,
                        "cursors_before": runtime.cursors, "actor_rows": [],
                        "conditioning_control_receipt": (
                            control_receipt.to_dict() if control_receipt is not None else None
                        ),
                        "conditioning_control_gap": control_gap,
                    }
                    traces.append(trace)
                    recorder.append(ReasoningEventKind.OBSERVATION, {
                        "step": len(actions),
                        "observation_sha256": _hash(_clean(observation)),
                        "native_actions_sha256": _hash(admissible),
                        "phase": "online_rebind",
                    })
                    if not active_contexts:
                        event = online.fallback_to_target_only(
                            step=len(actions), reason="NO_ACTIVE_SOURCE_CONTEXT_FOR_REBIND",
                        )
                        trace.update(status="REJECTED", reason=event.reason)
                        recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                            "event_sha256": event.event_sha256,
                            "verdict": event.verdict.value,
                            "reason": event.reason,
                            "state_after": event.state_after.value,
                        })
                        continue
                    scope = build_rebind_scope(
                        artifact_hash=artifact.artifact_hash,
                        demo_hash=artifact.demo_hash,
                        step=len(actions),
                        observation_sha256=_hash(_clean(observation)),
                        admissible_actions=admissible,
                        active_contexts=active_contexts,
                    )
                    prompt = online_rebind_prompt(
                        goal=goal, observation=_clean(observation),
                        admissible_actions=admissible,
                        active_contexts=active_contexts,
                        scope=scope,
                        failure_reason=online.events[-1].reason,
                    )
                    raw_reply, rebind_usage, rebind_error = "", {}, None
                    endpoint_failure = False
                    proposal = None
                    qualified_rebind = None
                    admission_failures = ()
                    try:
                        raw_reply, rebind_usage = client.complete(
                            model=args.model, prompt=prompt, max_tokens=1000,
                        )
                        proposal = parse_online_rebind_reply(raw_reply)
                    except Exception as exc:
                        rebind_error = f"{type(exc).__name__}:{exc}"
                        endpoint_failure = type(exc).__module__.startswith("httpx")
                    proposal_receipt_payload = {
                        "proposal_scope_hash": scope["proposal_scope_hash"],
                        "model": args.model,
                        "prompt_sha256": _hash(prompt),
                        "raw_reply": raw_reply,
                        "usage": dict(rebind_usage),
                    }
                    proposal_receipt_sha256 = _hash(proposal_receipt_payload)
                    if proposal is not None:
                        qualified_rebind, admission_failures = rebind_admission.admit(
                            proposal=proposal,
                            proposal_source="online_rebinding_agent",
                            proposal_receipt_sha256=proposal_receipt_sha256,
                            artifact_hash=artifact.artifact_hash,
                            demo_hash=artifact.demo_hash,
                            step=len(actions),
                            observation_sha256=_hash(_clean(observation)),
                            admissible_actions=admissible,
                            active_contexts=active_contexts,
                        )
                    trace["rebind_agent"] = {
                        "proposal_receipt_sha256": proposal_receipt_sha256,
                        "prompt_sha256": _hash(prompt),
                        "raw_reply": raw_reply,
                        "usage": dict(rebind_usage),
                        "parse_or_endpoint_error": rebind_error,
                        "endpoint_failure": endpoint_failure,
                        "admission_failures": list(admission_failures),
                        "qualified_receipt": (
                            qualified_rebind.to_dict() if qualified_rebind is not None else None
                        ),
                    }
                    recorder.append(ReasoningEventKind.AGENT_PROPOSAL_SET, {
                        "step": len(actions), "phase": "online_rebind",
                        "proposal_receipt_sha256": proposal_receipt_sha256,
                        "qualified_rebind_receipt_sha256": (
                            qualified_rebind.receipt_sha256
                            if qualified_rebind is not None else None
                        ),
                    })
                    recorder.append(ReasoningEventKind.NATIVE_ADMISSIBILITY, {
                        "step": len(actions), "phase": "online_rebind",
                        "native_actions": admissible,
                    })
                    if endpoint_failure:
                        trace.update(status="ERROR", reason="ONLINE_REBIND_ENDPOINT_ERROR")
                        error = "ONLINE_REBIND_ENDPOINT_ERROR"
                        break
                    if qualified_rebind is None:
                        reason = (
                            "ONLINE_REBIND_AGENT_ABSTAINED"
                            if proposal is not None and proposal.abstain
                            else "ONLINE_REBIND_NOT_ADMITTED"
                        )
                        event = online.fallback_to_target_only(
                            step=len(actions), reason=reason,
                        )
                        trace.update(status="REJECTED", reason=reason)
                        recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                            "event_sha256": event.event_sha256,
                            "verdict": event.verdict.value,
                            "reason": event.reason,
                            "state_after": event.state_after.value,
                        })
                        continue
                    event = online.accept_rebind(
                        step=len(actions),
                        binding_receipt_sha256=qualified_rebind.receipt_sha256,
                        known_binding_receipt_sha256s=(
                            rebind_admission.known_receipt_sha256s
                        ),
                    )
                    recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                        "event_sha256": event.event_sha256,
                        "binding_receipt_sha256": qualified_rebind.receipt_sha256,
                        "verdict": event.verdict.value,
                        "reason": event.reason,
                        "state_after": event.state_after.value,
                    })
                    action_scope_hash = _hash({
                        "binding_receipt_sha256": qualified_rebind.receipt_sha256,
                        "common_actions": list(qualified_rebind.common_actions),
                    })
                    action_prompt_text = action_prompt(
                        domain="alfworld", goal=goal, observation=_clean(observation),
                        actions=qualified_rebind.common_actions,
                        recent_actions=actions,
                        source_conditioning=active_contexts,
                    )
                    action_reply, action_usage, action_error = "", {}, None
                    rebound_command = None
                    try:
                        action_reply, action_usage = client.complete(
                            model=args.model, prompt=action_prompt_text, max_tokens=48,
                        )
                        selected = parse_exact_numbered_response(
                            action_reply, kind="action",
                            n=len(qualified_rebind.common_actions),
                        )
                        rebound_command = qualified_rebind.common_actions[selected]
                    except Exception as exc:
                        action_error = f"{type(exc).__name__}:{exc}"
                        if type(exc).__module__.startswith("httpx"):
                            trace.update(status="ERROR", reason="REBIND_ACTION_ENDPOINT_ERROR")
                            error = "REBIND_ACTION_ENDPOINT_ERROR"
                            break
                    trace["actor_rows"].append({
                        "candidate_hashes": [
                            item.candidate_hash for item in artifact.candidates
                        ],
                        "proposal_scope_hash": action_scope_hash,
                        "prompt_sha256": _hash(action_prompt_text),
                        "reply": action_reply[:500],
                        "usage": dict(action_usage),
                        "allowed_actions": list(qualified_rebind.common_actions),
                        "command": rebound_command,
                        "parse_error": action_error,
                        "n_source_conditioning": len(active_contexts),
                    })
                    if rebound_command is None:
                        event = online.fallback_to_target_only(
                            step=len(actions), reason="REBIND_ACTION_ACTOR_ABSTAINED_OR_INVALID",
                        )
                        trace.update(status="REJECTED", reason=event.reason)
                        recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                            "event_sha256": event.event_sha256,
                            "verdict": event.verdict.value,
                            "reason": event.reason,
                            "state_after": event.state_after.value,
                        })
                        continue
                    before_obs, before_actions = _clean(observation), list(admissible)
                    observation, reward, terminated, truncated, info = env.step(rebound_command)
                    actions.append(rebound_command)
                    max_native_step_reward = max(max_native_step_reward, float(reward))
                    cumulative_native_reward += float(reward)
                    success = _won(info)
                    after_actions = [str(item) for item in info.get("action_names") or ()]
                    native_receipt = NativeTransitionEvidence.build(
                        step=len(actions) - 1,
                        command=rebound_command,
                        before_observation_sha256=_hash(before_obs),
                        after_observation_sha256=_hash(_clean(observation)),
                        before_actions_sha256=_hash(before_actions),
                        after_actions_sha256=_hash(after_actions),
                        reward=float(reward), official_success=success,
                        command_was_admissible=rebound_command in before_actions,
                        executed_action_admissible_after=rebound_command in after_actions,
                        terminated=terminated, truncated=truncated,
                    )
                    verification = verify_rebind_evidence(
                        binding=qualified_rebind, transition=native_receipt,
                    )
                    runtime.observe_admitted_rebind_executed(
                        binding_receipt_sha256=qualified_rebind.receipt_sha256,
                        known_binding_receipt_sha256s=(
                            rebind_admission.known_receipt_sha256s
                        ),
                        covered_candidate_hashes=[
                            item.candidate_hash
                            for item in qualified_rebind.proposal.candidate_bindings
                        ],
                        common_actions=qualified_rebind.common_actions,
                        executed_command=rebound_command,
                        candidate_results={
                            row.candidate_hash: row.all_satisfied
                            for row in verification.candidate_results
                        },
                        verification_receipt_sha256=verification.receipt_sha256,
                    )
                    event = online.observe_rebind_transition(
                        native_receipt,
                        evidence_contract_satisfied=verification.any_satisfied,
                    )
                    trace.update(
                        status="EXECUTE", command=rebound_command,
                        cursors_after=runtime.cursors,
                        candidate_statuses_after=runtime.statuses,
                        online_transition_receipt=native_receipt.to_dict(),
                        rebind_evidence_verification=verification.to_dict(),
                        online_verdict=event.verdict.value,
                    )
                    recorder.append(ReasoningEventKind.AGENT_DECISION, {
                        "step": len(actions) - 1, "phase": "online_rebind_action",
                        "status": "EXECUTE", "command": rebound_command,
                    })
                    recorder.append(ReasoningEventKind.ENVIRONMENT_STEP, {
                        "step": len(actions) - 1, "command": rebound_command,
                        "reward": float(reward), "terminated": terminated,
                        "truncated": truncated,
                    })
                    recorder.append(ReasoningEventKind.NATIVE_DELTA, {
                        "step": len(actions) - 1,
                        "before_observation_sha256": _hash(before_obs),
                        "after_observation_sha256": _hash(_clean(observation)),
                        "before_actions_sha256": _hash(before_actions),
                        "after_actions_sha256": _hash(after_actions),
                    })
                    recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                        "event_sha256": event.event_sha256,
                        "transition_receipt_sha256": native_receipt.receipt_sha256,
                        "verification_receipt_sha256": verification.receipt_sha256,
                        "verdict": event.verdict.value,
                        "reason": event.reason,
                        "state_after": event.state_after.value,
                    })
                    continue

                if online is not None and online.state == OnlineTransferState.TARGET_ONLY:
                    admissible = [str(item) for item in info.get("action_names") or ()]
                    recorder.append(ReasoningEventKind.OBSERVATION, {
                        "step": len(actions),
                        "observation_sha256": _hash(_clean(observation)),
                        "native_actions_sha256": _hash(admissible),
                        "phase": "live_target_only_fallback",
                    })
                    prompt = action_prompt(
                        domain="alfworld", goal=goal, observation=_clean(observation),
                        actions=admissible, recent_actions=actions, source_conditioning=(),
                    )
                    reply, usage, live_error = "", {}, None
                    command = None
                    try:
                        reply, usage = client.complete(
                            model=args.model, prompt=prompt, max_tokens=48,
                        )
                        selected = parse_exact_numbered_response(
                            reply, kind="action", n=len(admissible),
                        )
                        command = admissible[selected]
                    except Exception as exc:
                        live_error = f"{type(exc).__name__}:{exc}"
                        if type(exc).__module__.startswith("httpx"):
                            error = "LIVE_TARGET_ONLY_ENDPOINT_ERROR"
                    trace = {
                        "step": len(actions),
                        "treatment": "live_target_only_fallback",
                        "status": "EXECUTE" if command is not None else "ABSTAIN",
                        "reason": live_error,
                        "command": command,
                        "actor_rows": [{
                            "prompt_sha256": _hash(prompt), "reply": reply[:500],
                            "usage": dict(usage), "allowed_actions": list(admissible),
                            "command": command, "parse_error": live_error,
                            "n_source_conditioning": 0,
                        }],
                    }
                    traces.append(trace)
                    recorder.append(ReasoningEventKind.AGENT_DECISION, {
                        "step": len(actions), "phase": "live_target_only_fallback",
                        "status": trace["status"], "command": command,
                    })
                    if command is None:
                        abstain_reason = "LIVE_TARGET_ONLY_ACTOR_ABSTAINED_OR_INVALID"
                        break
                    before_obs, before_actions = _clean(observation), list(admissible)
                    observation, reward, terminated, truncated, info = env.step(command)
                    actions.append(command)
                    max_native_step_reward = max(max_native_step_reward, float(reward))
                    cumulative_native_reward += float(reward)
                    success = _won(info)
                    recorder.append(ReasoningEventKind.ENVIRONMENT_STEP, {
                        "step": len(actions) - 1, "command": command,
                        "reward": float(reward), "terminated": terminated,
                        "truncated": truncated, "phase": "live_target_only_fallback",
                    })
                    recorder.append(ReasoningEventKind.NATIVE_DELTA, {
                        "step": len(actions) - 1,
                        "before_observation_sha256": _hash(before_obs),
                        "after_observation_sha256": _hash(_clean(observation)),
                        "before_actions_sha256": _hash(before_actions),
                        "after_actions_sha256": _hash(list(info.get("action_names") or ())),
                    })
                    continue
                current_runtime = runtime
                using_source = args.condition == "source" and current_runtime is runtime
                admissible = [str(item) for item in info.get("action_names") or ()]
                source_conditioning = current_runtime.active_source_conditioning()
                control_receipt = None
                control_gap = None
                if (
                    using_source and len(source_conditioning) >= 2
                    and args.source_conditioning_control == "rotate"
                ):
                    source_conditioning, control_receipt = rotate_source_conditioning(
                        source_conditioning,
                        seed=args.conditioning_control_seed,
                        step=len(actions),
                    )
                elif (
                    using_source and source_conditioning
                    and args.source_conditioning_control == "rotate"
                ):
                    control_gap = "CONTROL_UNAVAILABLE_SINGLE_ACTIVE_CONTEXT"
                recorder.append(ReasoningEventKind.OBSERVATION, {
                    "step": len(actions), "observation_sha256": _hash(_clean(observation)),
                    "native_actions_sha256": _hash(admissible),
                })
                actor_rows = []

                def actor(qualified_set, allowed, scope_hash):
                    candidate_hashes = [item.candidate_hash for item in qualified_set]
                    if not allowed:
                        proposal = CandidateActionProposal(
                            scope_hash, None, abstain=True,
                        )
                        actor_rows.append({
                            "candidate_hashes": candidate_hashes,
                            "allowed_actions": [], "abstained": True,
                            "reason": "NO_OPERATOR_MATCHING_NATIVE_ACTION",
                        })
                        return proposal
                    prompt = action_prompt(
                        domain="alfworld", goal=goal, observation=_clean(observation),
                        actions=allowed, recent_actions=actions,
                        source_conditioning=source_conditioning,
                    )
                    try:
                        reply, usage = client.complete(
                            model=args.model, prompt=prompt, max_tokens=48,
                        )
                    except Exception as exc:
                        proposal = CandidateActionProposal(
                            scope_hash, None,
                            endpoint_error=f"{type(exc).__name__}:{exc}",
                        )
                        actor_rows.append({
                            "candidate_hashes": candidate_hashes,
                            "prompt_sha256": _hash(prompt), "endpoint_error": proposal.endpoint_error,
                        })
                        return proposal
                    try:
                        selected = parse_exact_numbered_response(reply, kind="action", n=len(allowed))
                        command = allowed[selected]
                        proposal = CandidateActionProposal(scope_hash, command)
                        parse_error = None
                    except ValueError as exc:
                        proposal = CandidateActionProposal(scope_hash, None, abstain=True)
                        command = None
                        parse_error = f"{type(exc).__name__}:{exc}"
                    actor_rows.append({
                        "candidate_hashes": candidate_hashes,
                        "prompt_sha256": _hash(prompt), "reply": reply[:500],
                        "usage": dict(usage), "allowed_actions": list(allowed),
                        "command": command, "parse_error": parse_error,
                        "source_conditioning_sha256": _hash(source_conditioning),
                        "n_source_conditioning": len(source_conditioning),
                    })
                    return proposal

                decision = current_runtime.choose(admissible=admissible, actor=actor)
                trace = {
                    "step": len(actions), "status": decision.status,
                    "reason": decision.reason, "command": decision.command,
                    "treatment": "source" if using_source else "target_only_fallback",
                    "cursors_before": current_runtime.cursors, "actor_rows": actor_rows,
                    "conditioning_control_receipt": (
                        control_receipt.to_dict() if control_receipt is not None else None
                    ),
                    "conditioning_control_gap": control_gap,
                }
                traces.append(trace)
                recorder.append(ReasoningEventKind.AGENT_PROPOSAL_SET, {
                    "step": len(actions), "actor_rows_sha256": _hash(actor_rows),
                    "n_candidates": len(artifact.candidates),
                })
                recorder.append(ReasoningEventKind.NATIVE_ADMISSIBILITY, {
                    "step": len(actions), "native_actions": admissible,
                })
                recorder.append(ReasoningEventKind.AGENT_DECISION, {
                    "step": len(actions), "status": decision.status,
                    "reason": decision.reason, "command": decision.command,
                })
                if decision.status == "ERROR":
                    error = decision.reason
                    break
                if decision.status != "EXECUTE" or decision.command is None:
                    if online is not None and using_source:
                        event = online.observe_source_abstention(
                            step=len(actions), reason=decision.reason or "CONSENSUS_ABSTENTION",
                        )
                        recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                            "event_sha256": event.event_sha256,
                            "verdict": event.verdict.value,
                            "reason": event.reason,
                            "state_after": event.state_after.value,
                        })
                        trace["online_rebind_requested"] = (
                            online.state == OnlineTransferState.REBIND_REQUIRED
                        )
                        continue
                    abstain_reason = decision.reason or "CONSENSUS_ABSTENTION"
                    break
                action_contract = None
                if contract_control_enabled and using_source:
                    contract_scope = build_action_contract_scope(
                        artifact_hash=artifact.artifact_hash,
                        step=len(actions), command=decision.command,
                        observation_sha256=_hash(_clean(observation)),
                        admissible_actions=admissible,
                        active_contexts=source_conditioning,
                    )
                    contract_prompt = action_evidence_contract_prompt(
                        goal=goal, observation=_clean(observation),
                        command=decision.command, admissible_actions=admissible,
                        active_contexts=source_conditioning, scope=contract_scope,
                    )
                    contract_reply, contract_usage, contract_error = "", {}, None
                    try:
                        contract_reply, contract_usage = client.complete(
                            model=args.model, prompt=contract_prompt, max_tokens=768,
                        )
                        contract_proposal = parse_action_evidence_contract_reply(
                            contract_reply
                        )
                        contract_receipt_sha256 = _hash({
                            "model": args.model,
                            "prompt_sha256": _hash(contract_prompt),
                            "raw_reply": contract_reply,
                            "usage": dict(contract_usage),
                        })
                        action_contract = qualify_action_evidence_contract(
                            proposal=contract_proposal,
                            proposal_receipt_sha256=contract_receipt_sha256,
                            scope=contract_scope,
                        )
                    except Exception as exc:
                        contract_error = f"{type(exc).__name__}:{exc}"
                        if type(exc).__module__.startswith("httpx"):
                            error = "ACTION_CONTRACT_ENDPOINT_ERROR"
                    trace["contract_agent"] = {
                        "prompt_sha256": _hash(contract_prompt),
                        "raw_reply": contract_reply,
                        "usage": dict(contract_usage),
                        "error": contract_error,
                        "qualified_contract": (
                            action_contract.to_dict() if action_contract is not None else None
                        ),
                    }
                    if error is not None:
                        trace.update(status="ERROR", reason=error)
                        break
                    if action_contract is None:
                        if online is not None:
                            event = online.observe_source_abstention(
                                step=len(actions),
                                reason="MISSING_PREDECLARED_ACTION_EVIDENCE_CONTRACT",
                            )
                            trace.update(status="ABSTAIN", reason=event.reason, command=None)
                            recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                                "event_sha256": event.event_sha256,
                                "verdict": event.verdict.value,
                                "reason": event.reason,
                                "state_after": event.state_after.value,
                            })
                            continue
                        trace["shadow_contract_missing_fail_open"] = True
                before_obs, before_actions = _clean(observation), list(admissible)
                observation, reward, terminated, truncated, info = env.step(decision.command)
                actions.append(decision.command)
                if online is None or not using_source:
                    current_runtime.observe_executed(
                        decision, executed_command=decision.command,
                    )
                max_native_step_reward = max(max_native_step_reward, float(reward))
                cumulative_native_reward += float(reward)
                success = _won(info)
                recorder.append(ReasoningEventKind.ENVIRONMENT_STEP, {
                    "step": len(actions) - 1, "command": decision.command,
                    "reward": float(reward), "terminated": terminated, "truncated": truncated,
                })
                recorder.append(ReasoningEventKind.NATIVE_DELTA, {
                    "step": len(actions) - 1,
                    "before_observation_sha256": _hash(before_obs),
                    "after_observation_sha256": _hash(_clean(observation)),
                    "before_actions_sha256": _hash(before_actions),
                    "after_actions_sha256": _hash(list(info.get("action_names") or ())),
                })
                if contract_control_enabled and using_source:
                    native_receipt = NativeTransitionEvidence.build(
                        step=len(actions) - 1,
                        command=decision.command,
                        before_observation_sha256=_hash(before_obs),
                        after_observation_sha256=_hash(_clean(observation)),
                        before_actions_sha256=_hash(before_actions),
                        after_actions_sha256=_hash(list(info.get("action_names") or ())),
                        reward=float(reward),
                        official_success=success,
                        # The command came from the exact native admissible set.
                        command_was_admissible=decision.command in before_actions,
                        executed_action_admissible_after=(
                            decision.command in list(info.get("action_names") or ())
                        ),
                        terminated=terminated,
                        truncated=truncated,
                    )
                    trace["online_transition_receipt"] = native_receipt.to_dict()
                    trace["online_transition_receipt_sha256"] = native_receipt.receipt_sha256
                    if action_contract is not None:
                        verification = verify_action_evidence_contract(
                            contract=action_contract, transition=native_receipt,
                        )
                        trace["action_evidence_verification"] = verification.to_dict()
                        if online is not None:
                            current_runtime.observe_evidence_contract(
                                decision, executed_command=decision.command,
                                candidate_results={
                                    row.candidate_hash: row.all_satisfied
                                    for row in verification.candidate_results
                                },
                                verification_receipt_sha256=verification.receipt_sha256,
                            )
                            event = online.observe_contract_transition(
                                native_receipt,
                                evidence_contract_satisfied=verification.any_satisfied,
                                contract_kind=(
                                    "SOURCE_ACTION:"
                                    f"{sum(row.all_satisfied for row in verification.candidate_results)}"
                                    f"/{len(verification.candidate_results)}_SURVIVED"
                                ),
                            )
                            trace["online_verdict"] = event.verdict.value
                            recorder.append(ReasoningEventKind.ONLINE_TRANSFER_VERDICT, {
                                "event_sha256": event.event_sha256,
                                "transition_receipt_sha256": native_receipt.receipt_sha256,
                                "verdict": event.verdict.value,
                                "reason": event.reason,
                                "state_after": event.state_after.value,
                            })
                        else:
                            trace["shadow_contract_not_enforced"] = True
                trace["cursors_after"] = current_runtime.cursors
                trace["candidate_statuses_after"] = current_runtime.statuses
            recorder.append(ReasoningEventKind.OFFICIAL_STOP, {
                "official_success": success,
                "max_native_step_reward": max_native_step_reward,
                "cumulative_native_reward": cumulative_native_reward,
                "terminated": terminated, "truncated": truncated,
                "abstain_reason": abstain_reason, "error": error,
            })
            rows.append({
                "episode_index": episode_index, "condition": args.condition,
                "split": args.split, "seed": args.seed,
                "artifact_hash": artifact.artifact_hash,
                "target_instance_identity": target_instance_identity,
                "success": success, "official_success": success,
                "max_native_step_reward": max_native_step_reward,
                "cumulative_native_reward": cumulative_native_reward,
                "steps": len(actions), "actions": actions,
                "abstain_reason": abstain_reason, "error": error,
                "traces": traces, "reasoning_event_log": recorder.to_dict(),
                "online_transfer_log": online.to_dict() if online is not None else None,
                "fallback_mode": (
                    "LIVE_TARGET_ONLY_FROM_CURRENT_STATE"
                    if online is not None else None
                ),
                "wall_time_s": time.monotonic() - started,
            })
    finally:
        env.close()
        client.close()
    output = {
        "schema_version": 4,
        "pilot_only": args.split == "train",
        "condition": args.condition, "artifact_hash": artifact.artifact_hash,
        "source_treatment": artifact.source_treatment,
        "source_control_receipt_sha256": artifact.source_control_receipt_sha256,
        "source_control_applied_before_binding_generation": bool(
            artifact.source_control_receipt_sha256
        ),
        "model": args.model, "target_gradient_updates": 0,
        "online_source_control": args.online_source_control,
        "shadow_source_control": args.shadow_source_control,
        "source_control_mode": (
            "enforce" if args.online_source_control
            else "shadow" if args.shadow_source_control
            else "off"
        ),
        "online_rebinding_agent": args.online_source_control,
        "max_rebind_requests": args.max_rebind_requests,
        "max_consecutive_no_delta": args.max_consecutive_no_delta,
        "max_steps": args.max_steps,
        "config_sha256": _hash(args.config.read_text(encoding="utf-8")),
        "registered_call_caps": {
            "action_max_completion_tokens": 48,
            "action_contract_max_completion_tokens": 768,
            "online_rebind_max_completion_tokens": 1000,
            "max_rebind_requests": args.max_rebind_requests,
        },
        "source_conditioning_control": args.source_conditioning_control,
        "conditioning_control_seed": (
            args.conditioning_control_seed
            if args.source_conditioning_control != "none" else None
        ),
        "fallback_artifact_hash": (
            fallback_artifact.artifact_hash if fallback_artifact is not None else None
        ),
        "rows": rows,
        "summary": {
            "n": len(rows), "n_success": sum(row["success"] for row in rows),
            "n_abstain": sum(row["abstain_reason"] is not None for row in rows),
            "n_error": sum(row["error"] is not None for row in rows),
            "total_actor_calls": sum(len(t["actor_rows"]) for row in rows for t in row["traces"]),
            "total_rebind_agent_calls": sum(
                "rebind_agent" in trace for row in rows for trace in row["traces"]
            ),
            "total_contract_agent_calls": sum(
                "contract_agent" in trace for row in rows for trace in row["traces"]
            ),
            "n_rebind_admitted": sum(
                bool(trace.get("rebind_agent", {}).get("qualified_receipt"))
                for row in rows for trace in row["traces"]
            ),
            "n_rebind_executed": sum(
                trace.get("treatment") == "online_rebind"
                and trace.get("status") == "EXECUTE"
                for row in rows for trace in row["traces"]
            ),
            "n_source_disabled": sum(
                bool(row["online_transfer_log"])
                and row["online_transfer_log"]["state"] == "TARGET_ONLY"
                for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    return 1 if output["summary"]["n_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
