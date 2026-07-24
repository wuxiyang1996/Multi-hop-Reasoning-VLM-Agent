from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import (
    Advisory,
    AdvisoryVerdict,
    ContinuationDecision,
    DecisionCycleReceipt,
    DecisionCycleRecord,
    DecisionProposal,
    DecisionProposalSet,
    EvidenceVerdict,
    Observation,
    PostTransitionAssessment,
    ReplayForkReceipt,
    SourcePolicyStepRecord,
    SourceTransitionReceipt,
    TransitionReceipt,
    stable_hash,
)
from .phase1_assets import read_jsonl


@dataclass(frozen=True)
class ImportedEpisode:
    episode_id: str
    game: str
    records: tuple[DecisionCycleRecord, ...]
    replay_forks: tuple[ReplayForkReceipt, ...]
    total_reward: float
    official_success: bool | None
    gaps: tuple[str, ...]


@dataclass(frozen=True)
class ImportedSourceEpisode:
    episode_id: str
    game: str
    records: tuple[SourcePolicyStepRecord, ...]
    replay_forks: tuple[ReplayForkReceipt, ...]
    total_reward: float
    official_success: bool | None
    gaps: tuple[str, ...]


def _index_events(events: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        payload = event.get("payload") or {}
        step = payload.get("step")
        if isinstance(step, int):
            result[str(event.get("kind"))][step] = event
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_replay_hash(raw: dict[str, Any]) -> str:
    payload = {
        key: raw[key]
        for key in (
            "intervention_id", "seed", "prefix_actions",
            "expected_fork_state_sha256", "replayed_fork_state_sha256",
            "alternative_action", "admissible_actions_sha256",
            "alternative_next_state_sha256", "status", "failure_codes",
        )
    }
    return stable_hash(payload)


def _load_supplemental_replays(
    evidence_root: Path,
    supplemental_root: Path,
) -> list[dict[str, Any]]:
    manifest_path = supplemental_root / "manifest.json"
    receipt_path = supplemental_root / "replay_receipts.jsonl"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ValueError("supplemental replay bundle is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("authority") != "SUPPLEMENTAL_SWITCH_BOUNDARY_REPLAY_ONLY":
        raise ValueError("unsupported supplemental replay authority")
    if manifest.get("boundary_rule") != "EXACT_RECORDED_SELECTED_SKILL_ID_CHANGE_V1":
        raise ValueError("unsupported supplemental replay boundary rule")
    expected_source_hashes = manifest.get("source_files_sha256") or {}
    for name in ("manifest.json", "events.jsonl", "episodes.jsonl"):
        if expected_source_hashes.get(name) != _file_sha256(evidence_root / name):
            raise ValueError(f"supplemental replay source hash mismatch: {name}")
    if manifest.get("receipt_file_sha256") != _file_sha256(receipt_path):
        raise ValueError("supplemental replay file hash mismatch")
    rows = read_jsonl(receipt_path)
    if manifest.get("receipt_count") != len(rows):
        raise ValueError("supplemental replay count mismatch")
    for raw in rows:
        if raw.get("status") != "INTERVENTION_OBSERVED":
            raise ValueError("supplemental replay is not an observed intervention")
        if raw.get("receipt_sha256") != _raw_replay_hash(raw):
            raise ValueError("supplemental replay content hash mismatch")
    return rows


def _observation(event: dict[str, Any], native_actions: tuple[str, ...], *, terminal=False, success=False, score=0.0):
    payload = event.get("payload") or {}
    return Observation(
        {
            "observable_state": payload.get("observable_state", ""),
            "structured_state": payload.get("structured_state"),
        },
        native_actions,
        terminal,
        success,
        score,
    )


def import_instrumented_batch(evidence_dir: str | Path) -> tuple[ImportedEpisode, ...]:
    root = Path(evidence_dir)
    events = read_jsonl(root / "events.jsonl")
    episode_rows = read_jsonl(root / "episodes.jsonl")
    replay_rows = read_jsonl(root / "replay_receipts.jsonl") if (root / "replay_receipts.jsonl").exists() else []
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_episode[str(event.get("episode_id"))].append(event)

    imported: list[ImportedEpisode] = []
    for episode_row in episode_rows:
        episode_id = str(episode_row["episode_id"])
        indexed = _index_events(by_episode[episode_id])
        records: list[DecisionCycleRecord] = []
        transition_by_source_step: dict[int, TransitionReceipt] = {}
        gaps: list[str] = []
        decisions = indexed.get("AGENT_DECISION", {})
        for step in sorted(decisions):
            plan_event = indexed.get("AGENT_ACTION_PROPOSAL_SET", {}).get(step)
            verdict_event = indexed.get("AGENT_POST_TRANSITION_VERDICT", {}).get(step)
            before_event = indexed.get("OBSERVATION", {}).get(step)
            after_event = indexed.get("OBSERVATION", {}).get(step + 1)
            admissibility_event = indexed.get("NATIVE_ADMISSIBILITY", {}).get(step)
            env_event = indexed.get("ENVIRONMENT_STEP", {}).get(step)
            required = (plan_event, verdict_event, before_event, after_event, admissibility_event, env_event)
            if any(event is None for event in required):
                gaps.append(f"STEP_{step}_INCOMPLETE_CYCLE")
                continue
            plan_payload = plan_event["payload"]
            verdict_payload = verdict_event["payload"]
            if plan_payload.get("schema_valid") is not True or verdict_payload.get("schema_valid") is not True:
                gaps.append(f"STEP_{step}_INVALID_AGENT_SCHEMA")
                continue
            raw_plan = plan_payload.get("proposal_set") or {}
            if raw_plan.get("decision") != "EXECUTE":
                gaps.append(f"STEP_{step}_AGENT_ABSTAINED")
                continue
            native_actions = tuple(str(value) for value in admissibility_event["payload"].get("native_actions", []))
            proposals: list[DecisionProposal] = []
            try:
                for raw in raw_plan.get("proposals", []):
                    action_number = int(raw["action_number"])
                    action = native_actions[action_number - 1]
                    proposals.append(
                        DecisionProposal(
                            str(raw["proposal_id"]),
                            action,
                            str(raw.get("predicted_observable_delta", "")),
                            str(raw.get("rationale", "")),
                        )
                    )
                proposal_set = DecisionProposalSet(
                    str(plan_payload.get("proposal_set_sha256") or stable_hash(raw_plan)),
                    tuple(proposals),
                    str(raw_plan["selected_proposal_id"]),
                )
                selected = proposal_set.selected
            except (KeyError, IndexError, TypeError, ValueError):
                gaps.append(f"STEP_{step}_INVALID_PROPOSAL_REFERENCE")
                continue
            executed = str(env_event["payload"].get("executed_action", ""))
            if selected.action != executed:
                gaps.append(f"STEP_{step}_SELECTED_EXECUTED_MISMATCH")
                continue
            raw_verdict = verdict_payload.get("verdict") or {}
            try:
                assessment = PostTransitionAssessment(
                    EvidenceVerdict(str(raw_verdict["verdict"])),
                    ContinuationDecision(str(raw_verdict["decision"])),
                    str(raw_verdict.get("evidence_claim", "")),
                )
            except (KeyError, ValueError):
                gaps.append(f"STEP_{step}_INVALID_VERDICT_REFERENCE")
                continue
            next_native_event = indexed.get("NATIVE_ADMISSIBILITY", {}).get(step + 1)
            next_native = tuple(
                str(value) for value in (next_native_event or {}).get("payload", {}).get("native_actions", [])
            )
            terminal = bool(env_event["payload"].get("terminated") or env_event["payload"].get("truncated"))
            total_reward = float(episode_row.get("total_reward", 0.0) or 0.0)
            official = episode_row.get("official_success")
            if not isinstance(official, bool):
                official = episode_row.get("outcome") if isinstance(episode_row.get("outcome"), bool) else False
            before = _observation(before_event, native_actions)
            after = _observation(
                after_event,
                next_native,
                terminal=terminal,
                success=bool(official and terminal),
                score=total_reward if terminal else 0.0,
            )
            reward = float(env_event["payload"].get("reward", 0.0) or 0.0)
            transition = TransitionReceipt.create(before, selected, after, reward)
            cycle_receipt = DecisionCycleReceipt.create(proposal_set, transition, assessment)
            advisory = Advisory(AdvisoryVerdict.ADMIT, "source collection without transferred motif")
            record = DecisionCycleRecord(
                before, proposal_set, advisory, after, reward, transition, assessment, cycle_receipt
            )
            if not record.validate():
                gaps.append(f"STEP_{step}_RECOMPUTE_FAILED")
                continue
            records.append(record)
            transition_by_source_step[step] = transition

        forks: list[ReplayForkReceipt] = []
        marker = f"{episode_id}.fork_step_"
        for raw in replay_rows:
            intervention_id = str(raw.get("intervention_id", ""))
            if not intervention_id.startswith(marker):
                continue
            try:
                step = int(intervention_id[len(marker):].split(".", 1)[0])
                source_transition = transition_by_source_step[step]
            except (ValueError, KeyError):
                gaps.append("REPLAY_FORK_SOURCE_TRANSITION_UNRESOLVED")
                continue
            fork = ReplayForkReceipt.create(
                source_transition_id=source_transition.receipt_id,
                prefix_hash=stable_hash(raw.get("prefix_actions", [])),
                fork_state_hash=str(raw.get("replayed_fork_state_sha256", "")),
                admissible_actions_hash=str(raw.get("admissible_actions_sha256", "")),
                alternative_action=str(raw.get("alternative_action", "")),
                alternative_after_hash=str(raw.get("alternative_next_state_sha256", "")),
            )
            forks.append(fork)
        imported.append(
            ImportedEpisode(
                episode_id,
                str(episode_row.get("game", "unknown")),
                tuple(records),
                tuple(forks),
                float(episode_row.get("total_reward", 0.0) or 0.0),
                episode_row.get("official_success") if isinstance(episode_row.get("official_success"), bool) else None,
                tuple(gaps),
            )
        )
    return tuple(imported)


def import_native_source_batch(
    evidence_dir: str | Path,
    supplemental_replay_dir: str | Path | None = None,
    *,
    include_base_replays: bool = True,
) -> tuple[ImportedSourceEpisode, ...]:
    """Import unchanged source-policy steps without inventing proposal/verdict fields."""

    root = Path(evidence_dir)
    events = read_jsonl(root / "events.jsonl")
    episode_rows = read_jsonl(root / "episodes.jsonl")
    replay_rows = (
        read_jsonl(root / "replay_receipts.jsonl")
        if include_base_replays and (root / "replay_receipts.jsonl").exists()
        else []
    )
    if supplemental_replay_dir is not None:
        replay_rows += _load_supplemental_replays(
            root, Path(supplemental_replay_dir)
        )
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_episode[str(event.get("episode_id"))].append(event)

    imported: list[ImportedSourceEpisode] = []
    for episode_row in episode_rows:
        episode_id = str(episode_row["episode_id"])
        indexed = _index_events(by_episode[episode_id])
        records: list[SourcePolicyStepRecord] = []
        transition_by_source_step: dict[int, SourceTransitionReceipt] = {}
        gaps: list[str] = []
        decisions = indexed.get("AGENT_DECISION", {})
        for step in sorted(decisions):
            decision_event = decisions[step]
            skill_event = indexed.get("AGENT_PROPOSAL_SET", {}).get(step)
            response_event = indexed.get("AGENT_RESPONSE", {}).get(step)
            parsed_event = indexed.get("PARSED_DECISION", {}).get(step)
            before_event = indexed.get("OBSERVATION", {}).get(step)
            after_event = indexed.get("OBSERVATION", {}).get(step + 1)
            admissibility_event = indexed.get("NATIVE_ADMISSIBILITY", {}).get(step)
            env_event = indexed.get("ENVIRONMENT_STEP", {}).get(step)
            required = (
                skill_event, response_event, parsed_event, before_event,
                after_event, admissibility_event, env_event,
            )
            if any(event is None for event in required):
                gaps.append(f"STEP_{step}_INCOMPLETE_NATIVE_POLICY_CYCLE")
                continue
            decision_payload = decision_event["payload"]
            response_payload = response_event["payload"]
            env_payload = env_event["payload"]
            action_origin = str(decision_payload.get("decision_origin", ""))
            if action_origin not in {"AGENT", "POLICY_POSTPROCESSOR", "FALLBACK"}:
                gaps.append(f"STEP_{step}_UNKNOWN_ACTION_ORIGIN")
                continue
            if response_payload.get("adapter") != "action_taking":
                gaps.append(f"STEP_{step}_ACTION_ADAPTER_MISMATCH")
                continue
            action = str(env_payload.get("executed_action", ""))
            if action != str(decision_payload.get("executed_action", "")):
                gaps.append(f"STEP_{step}_DECISION_EXECUTION_MISMATCH")
                continue
            native_actions = tuple(str(value) for value in admissibility_event["payload"].get("native_actions", []))
            if action not in native_actions:
                gaps.append(f"STEP_{step}_ACTION_OUTSIDE_NATIVE_LIST")
                continue
            next_native_event = indexed.get("NATIVE_ADMISSIBILITY", {}).get(step + 1)
            next_native = tuple(
                str(value) for value in (next_native_event or {}).get("payload", {}).get("native_actions", [])
            )
            terminal = bool(env_payload.get("terminated") or env_payload.get("truncated"))
            official = episode_row.get("official_success")
            if not isinstance(official, bool):
                official = episode_row.get("outcome") if isinstance(episode_row.get("outcome"), bool) else False
            before = _observation(before_event, native_actions)
            after = _observation(
                after_event,
                next_native,
                terminal=terminal,
                success=bool(official and terminal),
                score=float(episode_row.get("total_reward", 0.0) or 0.0) if terminal else 0.0,
            )
            skill_payload = skill_event["payload"]
            skill_id = skill_payload.get("selected_skill_id")
            skill_hash = skill_payload.get("selected_skill_sha256")
            reward = float(env_payload.get("reward", 0.0) or 0.0)
            transition = SourceTransitionReceipt.create(
                before,
                episode_id=episode_id,
                step=step,
                selected_skill_hash=str(skill_hash) if skill_hash else None,
                action_response_hash=str(response_payload.get("raw_response_sha256", "")),
                action=action,
                action_origin=action_origin,
                policy_adapter="action_taking",
                after=after,
                reward=reward,
            )
            record = SourcePolicyStepRecord(
                episode_id=episode_id,
                step=step,
                before=before,
                selected_skill_id=str(skill_id) if skill_id else None,
                selected_skill_hash=str(skill_hash) if skill_hash else None,
                action_reasoning=str(parsed_event["payload"].get("reasoning", "")),
                action_response_hash=str(response_payload.get("raw_response_sha256", "")),
                action=action,
                action_origin=action_origin,
                policy_adapter="action_taking",
                after=after,
                reward=reward,
                transition=transition,
            )
            if not record.validate():
                gaps.append(f"STEP_{step}_SOURCE_RECEIPT_RECOMPUTE_FAILED")
                continue
            records.append(record)
            transition_by_source_step[step] = transition

        forks: list[ReplayForkReceipt] = []
        seen_fork_ids: set[str] = set()
        marker = f"{episode_id}.fork_step_"
        for raw in replay_rows:
            intervention_id = str(raw.get("intervention_id", ""))
            if not intervention_id.startswith(marker):
                continue
            try:
                step = int(intervention_id[len(marker):].split(".", 1)[0])
                source_transition = transition_by_source_step[step]
            except (ValueError, KeyError):
                gaps.append("REPLAY_FORK_SOURCE_TRANSITION_UNRESOLVED")
                continue
            fork = ReplayForkReceipt.create(
                source_transition_id=source_transition.receipt_id,
                prefix_hash=stable_hash(raw.get("prefix_actions", [])),
                fork_state_hash=str(raw.get("replayed_fork_state_sha256", "")),
                admissible_actions_hash=str(raw.get("admissible_actions_sha256", "")),
                alternative_action=str(raw.get("alternative_action", "")),
                alternative_after_hash=str(raw.get("alternative_next_state_sha256", "")),
            )
            if fork.receipt_id not in seen_fork_ids:
                forks.append(fork)
                seen_fork_ids.add(fork.receipt_id)
        imported.append(ImportedSourceEpisode(
            episode_id=episode_id,
            game=str(episode_row.get("game", "unknown")),
            records=tuple(records),
            replay_forks=tuple(forks),
            total_reward=float(episode_row.get("total_reward", 0.0) or 0.0),
            official_success=(
                episode_row.get("official_success")
                if isinstance(episode_row.get("official_success"), bool) else None
            ),
            gaps=tuple(gaps),
        ))
    return tuple(imported)
