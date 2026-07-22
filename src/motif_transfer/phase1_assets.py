from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class EvidenceBatchAudit:
    game: str
    batch_path: str
    protocol_profile: str
    episodes: int
    steps: int
    events: int
    agent_origin_steps: int
    policy_postprocessor_steps: int
    fallback_steps: int
    skill_candidate_sets: int
    selected_skill_receipts: int
    native_action_policy_steps: int
    action_proposal_attempts: int
    action_proposal_sets: int
    action_adapter_grounded_proposal_sets: int
    skill_conditioned_proposal_sets: int
    post_transition_verdict_attempts: int
    post_transition_verdicts: int
    explicit_replan_abstain: int
    episodes_with_official_outcome: int
    replay_fork_receipts: int
    content_hashes_valid: bool
    event_chains_valid: bool
    motif_ready: bool
    gaps: tuple[str, ...]
    warnings: tuple[str, ...]


def _has_official_outcome(row: dict[str, Any]) -> bool:
    score = row.get("total_reward")
    return (
        isinstance(row.get("official_success"), bool)
        or isinstance(row.get("outcome"), bool)
        or (isinstance(score, (int, float)) and not isinstance(score, bool))
    )


def audit_evidence_batch(evidence_dir: str | Path) -> EvidenceBatchAudit:
    root = Path(evidence_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol_profile = str(manifest.get("protocol_profile", "unknown"))
    events = read_jsonl(root / "events.jsonl")
    episodes = read_jsonl(root / "episodes.jsonl")
    game = str(manifest.get("metadata", {}).get("game") or episodes[0].get("game", "unknown"))

    hashes_valid = True
    for filename, metadata in manifest.get("files", {}).items():
        expected = metadata.get("sha256")
        path = root / filename
        if expected and (not path.exists() or sha256_file(path) != expected):
            hashes_valid = False

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_episode[str(event.get("episode_id", ""))].append(event)
    chains_valid = True
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row.get("sequence", -1)))
        previous = None
        for expected_sequence, event in enumerate(rows):
            if event.get("sequence") != expected_sequence or event.get("previous_event_sha256") != previous:
                chains_valid = False
            previous = event.get("event_sha256")

    kinds = Counter(str(event.get("kind", "")) for event in events)
    action_proposals = 0
    action_proposal_attempts = 0
    action_adapter_grounded = 0
    skill_conditioned = 0
    skill_candidate_sets = 0
    selected_skill_receipts = 0
    post_verdicts = 0
    post_verdict_attempts = 0
    replan_abstain = 0
    official_episode_ids = {str(row.get("episode_id")) for row in episodes if _has_official_outcome(row)}
    for event in events:
        payload = event.get("payload") or {}
        kind = str(event.get("kind", ""))
        if kind == "AGENT_PROPOSAL_SET":
            if isinstance(payload.get("action_proposals"), list):
                action_proposals += 1
            if isinstance(payload.get("skill_candidates"), list):
                skill_candidate_sets += 1
            if payload.get("selected_skill_sha256"):
                selected_skill_receipts += 1
        if (
            kind == "AGENT_ACTION_PROPOSAL_SET"
        ):
            action_proposal_attempts += 1
            if (
                payload.get("schema_valid") is True
                and isinstance((payload.get("proposal_set") or {}).get("proposals"), list)
            ):
                action_proposals += 1
                if (
                    payload.get("policy_adapter_requested") == "action_taking"
                    and payload.get("policy_adapter_used") == "action_taking"
                ):
                    action_adapter_grounded += 1
                if payload.get("conditioning_skill_sha256"):
                    skill_conditioned += 1
        if kind in {"POST_TRANSITION_AGENT_VERDICT", "AGENT_POST_TRANSITION_VERDICT"}:
            post_verdict_attempts += 1
        if kind in {"POST_TRANSITION_AGENT_VERDICT", "AGENT_POST_TRANSITION_VERDICT"} or "post_transition_verdict" in payload:
            if payload.get("schema_valid", True):
                post_verdicts += 1
        nested_verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
        decision = str(
            payload.get("continuation_decision")
            or nested_verdict.get("decision")
            or payload.get("decision_type")
            or ""
        ).upper()
        if decision in {"REPLAN", "ABSTAIN"}:
            replan_abstain += 1
        if kind == "OFFICIAL_STOP" and _has_official_outcome(payload):
            official_episode_ids.add(str(event.get("episode_id")))

    replay_path = root / str(manifest.get("replay_forks", {}).get("file", "replay_receipts.jsonl"))
    replay_count = len(read_jsonl(replay_path)) if replay_path.exists() else 0
    steps = sum(int(row.get("steps", 0)) for row in episodes)
    agent_steps = sum(
        1
        for event in events
        if event.get("kind") == "AGENT_DECISION"
        and (event.get("payload") or {}).get("decision_origin") == "AGENT"
    )
    postprocessor_steps = sum(
        1
        for event in events
        if event.get("kind") == "AGENT_DECISION"
        and (event.get("payload") or {}).get("decision_origin") == "POLICY_POSTPROCESSOR"
    )
    fallback_steps = sum(
        1
        for event in events
        if event.get("kind") == "AGENT_DECISION"
        and (event.get("payload") or {}).get("decision_origin") == "FALLBACK"
    )
    response_adapter_by_step = {
        (str(event.get("episode_id")), int((event.get("payload") or {}).get("step"))):
        (event.get("payload") or {}).get("adapter")
        for event in events
        if event.get("kind") == "AGENT_RESPONSE"
        and isinstance((event.get("payload") or {}).get("step"), int)
    }
    native_action_policy_steps = sum(
        1
        for event in events
        if event.get("kind") == "AGENT_DECISION"
        and (event.get("payload") or {}).get("decision_origin")
        in {"AGENT", "POLICY_POSTPROCESSOR", "FALLBACK"}
        and response_adapter_by_step.get((
            str(event.get("episode_id")), int((event.get("payload") or {}).get("step", -1))
        )) == "action_taking"
    )

    gaps: list[str] = []
    protocol_checks = (
        (
            native_action_policy_steps > 0
            and native_action_policy_steps
            == agent_steps + postprocessor_steps + fallback_steps,
            "NATIVE_ACTIONS_NOT_GROUNDED_IN_CHECKPOINT_POLICY",
        ),
    ) if protocol_profile == "source_agent" else (
        (action_proposals > 0, "NO_VALID_ACTION_PROPOSAL_SETS"),
        (
            action_proposals > 0 and action_adapter_grounded == action_proposals,
            "ACTION_PROPOSALS_NOT_GROUNDED_IN_CHECKPOINT_POLICY",
        ),
        (post_verdicts > 0, "NO_VALID_POST_TRANSITION_AGENT_VERDICTS"),
    )
    checks = protocol_checks + (
        (len(official_episode_ids) == len(episodes), "OFFICIAL_OUTCOME_INCOMPLETE"),
        (replay_count > 0, "NO_REPLAY_FORK_RECEIPTS"),
        (hashes_valid, "CONTENT_HASH_MISMATCH"),
        (chains_valid, "EVENT_CHAIN_INVALID"),
    )
    for passed, code in checks:
        if not passed:
            gaps.append(code)
    warnings = (
        [] if protocol_profile == "source_agent" or replan_abstain > 0
        else ["NO_OBSERVED_REPLAN_OR_ABSTAIN"]
    )
    return EvidenceBatchAudit(
        game=game,
        batch_path=str(root),
        protocol_profile=protocol_profile,
        episodes=len(episodes),
        steps=steps,
        events=len(events),
        agent_origin_steps=agent_steps,
        policy_postprocessor_steps=postprocessor_steps,
        fallback_steps=fallback_steps,
        skill_candidate_sets=skill_candidate_sets,
        selected_skill_receipts=selected_skill_receipts,
        native_action_policy_steps=native_action_policy_steps,
        action_proposal_attempts=action_proposal_attempts,
        action_proposal_sets=action_proposals,
        action_adapter_grounded_proposal_sets=action_adapter_grounded,
        skill_conditioned_proposal_sets=skill_conditioned,
        post_transition_verdict_attempts=post_verdict_attempts,
        post_transition_verdicts=post_verdicts,
        explicit_replan_abstain=replan_abstain,
        episodes_with_official_outcome=len(official_episode_ids),
        replay_fork_receipts=replay_count,
        content_hashes_valid=hashes_valid,
        event_chains_valid=chains_valid,
        motif_ready=not gaps,
        gaps=tuple(gaps),
        warnings=tuple(warnings),
    )


def discover_evidence_batches(root: str | Path) -> tuple[Path, ...]:
    base = Path(root)
    return tuple(sorted(path.parent for path in base.rglob("evidence/manifest.json")))


def audit_batches(paths: Iterable[str | Path]) -> dict[str, Any]:
    audits = tuple(audit_evidence_batch(path) for path in paths)
    return {
        "schema_version": 1,
        "claim_limit": "logging audit only; no source attribution or transfer claim",
        "batches": [asdict(row) for row in audits],
        "totals": {
            "games": len({row.game for row in audits}),
            "episodes": sum(row.episodes for row in audits),
            "steps": sum(row.steps for row in audits),
            "agent_origin_steps": sum(row.agent_origin_steps for row in audits),
            "policy_postprocessor_steps": sum(row.policy_postprocessor_steps for row in audits),
            "fallback_steps": sum(row.fallback_steps for row in audits),
            "native_action_policy_steps": sum(row.native_action_policy_steps for row in audits),
            "selected_skill_receipts": sum(row.selected_skill_receipts for row in audits),
            "action_proposal_attempts": sum(row.action_proposal_attempts for row in audits),
            "action_proposal_sets": sum(row.action_proposal_sets for row in audits),
            "action_adapter_grounded_proposal_sets": sum(
                row.action_adapter_grounded_proposal_sets for row in audits
            ),
            "skill_conditioned_proposal_sets": sum(
                row.skill_conditioned_proposal_sets for row in audits
            ),
            "post_transition_verdict_attempts": sum(
                row.post_transition_verdict_attempts for row in audits
            ),
            "post_transition_verdicts": sum(row.post_transition_verdicts for row in audits),
            "explicit_replan_abstain": sum(row.explicit_replan_abstain for row in audits),
            "episodes_with_official_outcome": sum(row.episodes_with_official_outcome for row in audits),
            "replay_fork_receipts": sum(row.replay_fork_receipts for row in audits),
            "motif_ready_games": len({row.game for row in audits if row.motif_ready}),
        },
    }


def audit_checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_game: dict[str, dict[str, Any]] = {}
    for row in payload.get("files", []):
        game = str(row.get("game", "unknown"))
        entry = per_game.setdefault(game, {"files": 0, "bytes": 0, "adapter_roles": set(), "hash_failures": 0})
        entry["files"] += 1
        entry["bytes"] += int(row.get("size", 0))
        local = manifest_path.parent / str(row.get("local", ""))
        parts = local.parts
        if "skillbank" in parts:
            index = parts.index("skillbank")
            if index + 1 < len(parts):
                entry["adapter_roles"].add(parts[index + 1])
        expected = row.get("local_sha256")
        if expected and (not local.exists() or sha256_file(local) != expected):
            entry["hash_failures"] += 1
    return {
        "manifest": str(manifest_path),
        "games": {
            game: {**entry, "adapter_roles": sorted(entry["adapter_roles"])}
            for game, entry in sorted(per_game.items())
        },
    }
