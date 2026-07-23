"""Compile compact instrumented source batches into immutable TracePrograms."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.reasoning_event_log import (
    reasoning_event_log_from_dict,
    validate_reasoning_protocol,
)
from harness.replay_fork import ForkInterventionReceipt
from skill_agents.evidence_query import ContentAddressedEvidenceSession
from skill_bank.trace_program_ir import (
    BackboneCoverage,
    NativeTransitionReceipt,
    ObservedOrderEdge,
    TraceProgram,
)


def _hash(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class InstrumentedProgramEvidence:
    program: TraceProgram
    evidence_session: ContentAddressedEvidenceSession
    intervention_receipts: Sequence[Mapping[str, Any]]
    reasoning_log_sha256: str


def _group(rows: Sequence[Mapping[str, Any]], kind: str) -> Mapping[int, Mapping[str, Any]]:
    selected = {}
    for row in rows:
        if row["kind"] != kind:
            continue
        step = int(row["payload"]["step"])
        if step in selected:
            raise ValueError(f"duplicate {kind} event at step {step}")
        selected[step] = row
    return selected


def load_instrumented_source_batch(root: str | Path) -> Sequence[InstrumentedProgramEvidence]:
    """Verify a four-file batch and compile Agent-origin contiguous programs.

    Policy-postprocessed/fallback actions remain in the immutable reasoning log,
    but cannot be promoted into source reasoning.  They mechanically split an
    episode into maximal contiguous Agent-origin spans; spans shorter than two
    transitions cannot form a multi-step program and are retained only in the
    source log.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, receipt in manifest.get("files", {}).items():
        if _file_hash(root / name) != receipt["sha256"]:
            raise ValueError(f"batch file hash mismatch: {name}")
    replay_meta = manifest.get("replay_forks") or {}
    replay_path = root / str(replay_meta.get("file") or "replay_receipts.jsonl")
    if _file_hash(replay_path) != replay_meta.get("sha256"):
        raise ValueError("replay receipt file hash mismatch")
    episode_rows = {
        row["episode_id"]: row for row in (
            json.loads(line) for line in (root / "episodes.jsonl").read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        )
    }
    event_rows: dict[str, list[Mapping[str, Any]]] = {}
    for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event_rows.setdefault(str(row["episode_id"]), []).append(row)
    replays = [
        json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in replays:
        payload = {key: row[key] for key in (
            "intervention_id", "seed", "prefix_actions", "expected_fork_state_sha256",
            "replayed_fork_state_sha256", "alternative_action",
            "admissible_actions_sha256", "alternative_next_state_sha256", "status",
            "failure_codes",
        )}
        receipt = ForkInterventionReceipt(**payload)
        if receipt.content_hash() != row.get("receipt_sha256"):
            raise ValueError("intervention receipt content hash mismatch")
    compiled = []
    for episode_id, episode in sorted(episode_rows.items()):
        rows = sorted(event_rows.get(episode_id, ()), key=lambda item: item["sequence"])
        unsigned_log = {"schema_version": 1, "episode_id": episode_id, "events": rows}
        unsigned_log["log_sha256"] = _hash(unsigned_log)
        if unsigned_log["log_sha256"] != episode["reasoning_log_sha256"]:
            raise ValueError(f"episode reasoning log hash mismatch: {episode_id}")
        events = reasoning_event_log_from_dict(unsigned_log)
        has_reasoning_v2 = any(
            row["kind"] in {
                "AGENT_ACTION_PROPOSAL_SET", "AGENT_POST_TRANSITION_VERDICT",
            }
            for row in rows
        )
        failures = validate_reasoning_protocol(
            events, profile="source_agent_v2" if has_reasoning_v2 else "source_agent",
        )
        if failures:
            raise ValueError(f"incomplete source protocol {episode_id}: {failures}")
        observations = _group(rows, "OBSERVATION")
        admissibility = _group(rows, "NATIVE_ADMISSIBILITY")
        decisions = _group(rows, "AGENT_DECISION")
        responses = _group(rows, "AGENT_RESPONSE")
        environment_steps = _group(rows, "ENVIRONMENT_STEP")
        native_deltas = _group(rows, "NATIVE_DELTA")
        action_proposals = _group(rows, "AGENT_ACTION_PROPOSAL_SET")
        post_verdicts = _group(rows, "AGENT_POST_TRANSITION_VERDICT")
        n_steps = int(episode["steps"])
        verified_steps = []
        for step in range(n_steps):
            required = (
                observations.get(step), observations.get(step + 1),
                admissibility.get(step), decisions.get(step), responses.get(step),
                environment_steps.get(step), native_deltas.get(step),
            )
            if has_reasoning_v2:
                required += (action_proposals.get(step), post_verdicts.get(step))
            if any(item is None for item in required):
                raise ValueError(f"missing source event at {episode_id} step {step}")
            before, after, allowed, decision, response, env_step, delta = required[:7]
            if decision["payload"]["executed_action"] != env_step["payload"]["executed_action"]:
                raise ValueError(f"decision/environment action mismatch at step {step}")
            if before["payload"]["observable_state_sha256"] != delta["payload"]["before_observable_sha256"]:
                raise ValueError(f"before-state delta mismatch at step {step}")
            if after["payload"]["observable_state_sha256"] != delta["payload"]["after_observable_sha256"]:
                raise ValueError(f"after-state delta mismatch at step {step}")
            transition_id = _hash({
                "episode_id": episode_id,
                "step": step,
                "decision_event_sha256": decision["event_sha256"],
                "environment_event_sha256": env_step["event_sha256"],
            })
            action = str(env_step["payload"]["executed_action"])
            receipt = NativeTransitionReceipt(
                transition_id=transition_id,
                step_index=step,
                state_sha256=str(before["payload"]["observable_state_sha256"]),
                next_state_sha256=str(after["payload"]["observable_state_sha256"]),
                available_actions_sha256=str(allowed["payload"]["native_actions_sha256"]),
                action=action,
                reward=float(env_step["payload"]["reward"]),
                done=bool(env_step["payload"]["terminated"] or env_step["payload"]["truncated"]),
            )
            native_row = {
                "state": before["payload"]["observable_state"],
                "next_state": after["payload"]["observable_state"],
                "available_actions": list(allowed["payload"]["native_actions"]),
                "executed_action": action,
                "reward": receipt.reward,
                "done": receipt.done,
                "raw_agent_response": response["payload"]["raw_response"],
                "agent_response_sha256": response["payload"]["raw_response_sha256"],
            }
            if has_reasoning_v2:
                proposal_event = action_proposals[step]
                verdict_event = post_verdicts[step]
                native_row.update({
                    "action_proposal_receipt": dict(proposal_event["payload"]),
                    "action_proposal_event_sha256": proposal_event["event_sha256"],
                    "post_transition_verdict_receipt": dict(verdict_event["payload"]),
                    "post_transition_verdict_event_sha256": verdict_event["event_sha256"],
                })
            can_support = bool(
                decision["payload"]["can_support_agent_reasoning_induction"]
            )
            if has_reasoning_v2:
                can_support = can_support and bool(
                    post_verdicts[step]["payload"].get(
                        "can_support_closed_loop_reasoning_induction"
                    )
                )
            verified_steps.append((
                step,
                can_support,
                receipt,
                native_row,
            ))

        spans = []
        current = []
        for item in verified_steps:
            if item[1]:
                current.append(item)
            else:
                if len(current) >= 2:
                    spans.append(current)
                current = []
        if len(current) >= 2:
            spans.append(current)

        for span_index, span in enumerate(spans):
            is_full_episode = len(spans) == 1 and len(span) == n_steps
            span_episode_id = (
                episode_id if is_full_episode
                else f"{episode_id}.agent_segment_{span_index:03d}"
            )
            transition_receipts = [item[2] for item in span]
            native = {item[2].transition_id: item[3] for item in span}
            span_steps = {item[0] for item in span}
            ids = [item.transition_id for item in transition_receipts]
            program = TraceProgram(
                program_id="instrumented." + _hash(span_episode_id)[:24],
                game=str(episode["game"]),
                episode_id=span_episode_id,
                source_file_sha256=str(episode["reasoning_log_sha256"]),
                transitions=transition_receipts,
                observed_order=[ObservedOrderEdge(a, b) for a, b in zip(ids, ids[1:])],
                coverage=BackboneCoverage(
                    True, True, True, True, True, True,
                    span[-1][0] == n_steps - 1,
                ),
                full_reset_to_stop_trace=is_full_episode,
                official_success_verified=False,
                metadata=(
                    {
                        "compiler": (
                            "instrumented_reasoning_events_v3_closed_loop"
                            if has_reasoning_v2 else "instrumented_reasoning_events_v1"
                        ),
                        "full_path_partition_required_for_source_hypotheses": True,
                        "explicit_replan_abstain_supported": has_reasoning_v2,
                        "reasoning_backbone_protocol": (
                            "agent_native_v2" if has_reasoning_v2 else "legacy_action_justification"
                        ),
                    }
                    if is_full_episode else {
                        "compiler": (
                            "instrumented_reasoning_events_v3_closed_loop"
                            if has_reasoning_v2 else
                            "instrumented_reasoning_events_v2_agent_spans"
                        ),
                        "full_path_partition_required_for_source_hypotheses": True,
                        "explicit_replan_abstain_supported": has_reasoning_v2,
                        "parent_episode_id": episode_id,
                        "source_step_start": span[0][0],
                        "source_step_end_inclusive": span[-1][0],
                        "non_agent_transitions_excluded": n_steps - sum(
                            int(item[1]) for item in verified_steps
                        ),
                        "segmentation_rule": (
                            "maximal_contiguous_closed_loop_agent_span_min_length_2"
                            if has_reasoning_v2 else
                            "maximal_contiguous_agent_origin_span_min_length_2"
                        ),
                        "reasoning_backbone_protocol": (
                            "agent_native_v2" if has_reasoning_v2 else "legacy_action_justification"
                        ),
                    }
                ),
            )
            program.validate_structure()
            episode_replays = []
            for row in replays:
                intervention_id = str(row["intervention_id"])
                if not intervention_id.startswith(episode_id + "."):
                    continue
                match = re.search(r"\.fork_step_(\d+)\.", intervention_id)
                if match is not None and int(match.group(1)) in span_steps:
                    episode_replays.append(row)
            compiled.append(InstrumentedProgramEvidence(
                program=program,
                evidence_session=ContentAddressedEvidenceSession(
                    program, native_evidence_by_transition_id=native,
                ),
                intervention_receipts=episode_replays,
                reasoning_log_sha256=str(episode["reasoning_log_sha256"]),
            ))
    return tuple(compiled)


__all__ = ["InstrumentedProgramEvidence", "load_instrumented_source_batch"]
