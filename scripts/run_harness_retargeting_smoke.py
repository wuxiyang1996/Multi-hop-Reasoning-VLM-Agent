#!/usr/bin/env python3
"""Run the deterministic permission-bounded Harness retargeting smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.retargeting_harness import (  # noqa: E402
    ExperimentVerdict,
    FrozenSkillArtifact,
    GroundingDraft,
    NativeActionCandidate,
    OptionGrounding,
    PermissionBoundTargetHarness,
    RetargetingCondition,
    SkillArtifactKind,
    SourceQualification,
    SymbolicOptionRule,
    TargetHarnessArtifact,
    TargetObservation,
    TargetStep,
    evaluate_retargeting_experiment,
    run_retargeting_episode,
)


OPTIONS = ("SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "VERIFY")
PREDICATES = (
    "location_unknown",
    "subject_visible",
    "evidence_acquired",
    "transformation_required",
    "transformation_not_required",
    "subject_ready",
    "result_placed",
    "result_verified",
)
ROLES = ("subject", "destination")


def _rules() -> tuple[SymbolicOptionRule, ...]:
    return (
        SymbolicOptionRule(
            "locate-subject", "SEARCH", ("location_unknown",),
            ("subject_visible",), ("subject_visible",), ("location_unknown",),
            recovery_option="SEARCH", priority=50,
        ),
        SymbolicOptionRule(
            "acquire-evidence", "ACQUIRE", ("subject_visible",),
            ("evidence_acquired",), ("evidence_acquired",), (),
            recovery_option="SEARCH", priority=40,
        ),
        SymbolicOptionRule(
            "transform-subject", "TRANSFORM",
            ("evidence_acquired", "transformation_required"),
            ("subject_ready",), ("subject_ready",), ("transformation_required",),
            recovery_option="ACQUIRE", priority=30,
        ),
        SymbolicOptionRule(
            "place-transformed", "PLACE", ("subject_ready",),
            ("result_placed",), ("result_placed",), (),
            recovery_option="TRANSFORM", priority=20,
        ),
        SymbolicOptionRule(
            "place-direct", "PLACE",
            ("evidence_acquired", "transformation_not_required"),
            ("result_placed",), ("result_placed",), (),
            recovery_option="ACQUIRE", priority=20,
        ),
        SymbolicOptionRule(
            "verify-result", "VERIFY", ("result_placed",),
            ("result_verified",), ("result_verified",), ("result_placed",),
            recovery_option="PLACE", priority=10,
        ),
    )


def authentic_source_skill() -> FrozenSkillArtifact:
    source_receipts = tuple(
        stable_hash({"source": "controlled-puzzle-game", "fork": index})
        for index in range(24)
    )
    qualification = SourceQualification.create(
        authentic_value=0.92,
        shuffled_value=0.17,
        marginal_value=0.25,
        receipt_ids=source_receipts,
    )
    return FrozenSkillArtifact.create(
        skill_id="controlled-game-relational-workflow-v1",
        artifact_kind=SkillArtifactKind.AUTHENTIC_SOURCE,
        source_domain="controlled-puzzle-game",
        option_vocabulary=OPTIONS,
        predicate_vocabulary=PREDICATES,
        role_vocabulary=ROLES,
        rules=_rules(),
        termination_facts=("result_verified",),
        source_lineage=source_receipts,
        qualification=qualification,
    )


def target_oracle_skill() -> FrozenSkillArtifact:
    return FrozenSkillArtifact.create(
        skill_id="incident-response-target-oracle-v1",
        artifact_kind=SkillArtifactKind.TARGET_ORACLE,
        source_domain="incident-response-target",
        option_vocabulary=OPTIONS,
        predicate_vocabulary=PREDICATES,
        role_vocabulary=ROLES,
        rules=_rules(),
        termination_facts=("result_verified",),
    )


class IncidentGrounder:
    grounder_hash = stable_hash({
        "kind": "incident-target-native-grounder",
        "version": 1,
        "inputs": ["telemetry", "packet", "decode", "ticket", "audit", "native_actions"],
        "outputs": ["facts", "roles", "option-local-action-scores"],
    })

    _ACTION_OPTIONS = {
        "scan telemetry": ("SEARCH", 0.95),
        "reboot display": ("SEARCH", 0.05),
        "capture packet": ("ACQUIRE", 0.95),
        "archive screenshot": ("ACQUIRE", 0.05),
        "decode packet": ("TRANSFORM", 0.95),
        "compress packet": ("TRANSFORM", 0.05),
        "submit incident": ("PLACE", 0.95),
        "close empty ticket": ("PLACE", 0.05),
        "audit closure": ("VERIFY", 0.95),
        "browse dashboard": ("VERIFY", 0.05),
    }

    def ground(self, observation: TargetObservation) -> GroundingDraft:
        payload = observation.payload
        facts: set[str] = set()
        if payload["telemetry"] == "unlocated":
            facts.add("location_unknown")
        else:
            facts.add("subject_visible")
        if payload["packet"] == "captured":
            facts.add("evidence_acquired")
        if payload["requires_decode"] and not payload["decoded"]:
            facts.add("transformation_required")
        elif not payload["requires_decode"]:
            facts.add("transformation_not_required")
        if payload["decoded"]:
            facts.add("subject_ready")
        if payload["ticket_submitted"] and not payload["audit_complete"]:
            facts.add("result_placed")
        if payload["audit_complete"]:
            facts.add("result_verified")
        grouped: dict[str, list[NativeActionCandidate]] = {option: [] for option in OPTIONS}
        for action in observation.native_actions:
            option, score = self._ACTION_OPTIONS[action]
            grouped[option].append(NativeActionCandidate(action, score))
        return GroundingDraft(
            facts=frozenset(facts),
            role_bindings=(
                ("subject", str(payload["case_name"])),
                ("destination", "incident-ledger"),
            ),
            option_groundings=tuple(
                OptionGrounding(option, tuple(grouped[option])) for option in OPTIONS
            ),
        )


class WeakTargetPolicy:
    policy_hash = stable_hash({
        "kind": "frozen-weak-target-policy",
        "option_rule": "prefer-verify",
        "raw_rule": "prefer-dashboard",
    })

    def choose_option(self, frame) -> str:
        return "VERIFY" if "VERIFY" in frame.available_options else frame.available_options[0]

    def choose_raw_action(self, observation: TargetObservation) -> str:
        if "browse dashboard" in observation.native_actions:
            return "browse dashboard"
        return observation.native_actions[0]


class IncidentResponseEnvironment:
    ACTIONS = tuple(IncidentGrounder._ACTION_OPTIONS)

    def __init__(self, episodes: Mapping[str, Mapping[str, Any]]):
        self._episodes = dict(episodes)
        self._state: dict[str, Any] = {}
        self._step_index = 0
        self.environment_hash = stable_hash({
            "kind": "incident-response-controlled-target",
            "version": 1,
            "episode_specs": self._episodes,
            "actions": list(self.ACTIONS),
        })

    def _observation(self) -> TargetObservation:
        terminal = bool(self._state["audit_complete"])
        return TargetObservation(
            observation_id=f"{self._state['episode_id']}:{self._step_index}",
            payload=dict(self._state),
            native_actions=() if terminal else self.ACTIONS,
            terminal=terminal,
        )

    def reset(self, episode_id: str) -> TargetObservation:
        spec = self._episodes[episode_id]
        self._step_index = 0
        self._state = {
            "episode_id": episode_id,
            "case_name": f"case-{episode_id}",
            "telemetry": "unlocated",
            "packet": "missing",
            "requires_decode": bool(spec["requires_decode"]),
            "decoded": False,
            "ticket_submitted": False,
            "audit_complete": False,
        }
        return self._observation()

    def step(self, action: str) -> TargetStep:
        if action not in self.ACTIONS:
            raise ValueError(f"inadmissible incident action: {action}")
        changed = False
        if action == "scan telemetry" and self._state["telemetry"] == "unlocated":
            self._state["telemetry"] = "located"
            changed = True
        elif action == "capture packet" and self._state["telemetry"] == "located" \
                and self._state["packet"] == "missing":
            self._state["packet"] = "captured"
            changed = True
        elif action == "decode packet" and self._state["packet"] == "captured" \
                and self._state["requires_decode"] and not self._state["decoded"]:
            self._state["decoded"] = True
            changed = True
        elif action == "submit incident" and self._state["packet"] == "captured" \
                and (not self._state["requires_decode"] or self._state["decoded"]) \
                and not self._state["ticket_submitted"]:
            self._state["ticket_submitted"] = True
            changed = True
        elif action == "audit closure" and self._state["ticket_submitted"]:
            self._state["audit_complete"] = True
            changed = True
        self._step_index += 1
        success = bool(self._state["audit_complete"])
        return TargetStep(
            observation=self._observation(),
            reward=1.0 if success else (0.05 if changed else -0.05),
            official_success=success,
            official_score=1.0 if success else 0.0,
        )


def build_harness() -> PermissionBoundTargetHarness:
    grounder = IncidentGrounder()
    artifact = TargetHarnessArtifact.create(
        harness_id="incident-target-native-harness-v1",
        target_domain="incident-response-target",
        option_vocabulary=OPTIONS,
        predicate_vocabulary=PREDICATES,
        role_vocabulary=ROLES,
        adaptation_receipt_ids=tuple(
            stable_hash({"target_adaptation_receipt": index}) for index in range(10)
        ),
        grounder_hash=grounder.grounder_hash,
        realizer_hash=stable_hash("within-option-score-only-v1"),
        verifier_hash=stable_hash("observable-canonical-fact-delta-v1"),
    )
    return PermissionBoundTargetHarness(artifact, grounder)


def run_smoke(config: Mapping[str, Any]):
    episode_specs = {
        str(row["episode_id"]): row for row in config["episodes"]
    }
    environment = IncidentResponseEnvironment(episode_specs)
    harness = build_harness()
    fallback = WeakTargetPolicy()
    authentic = authentic_source_skill()
    shuffled = authentic.shuffled_control()
    oracle = target_oracle_skill()
    artifacts = {
        RetargetingCondition.RAW_TARGET_ONLY: None,
        RetargetingCondition.NULL_SKILL_SAME_HARNESS: None,
        RetargetingCondition.SHUFFLED_SOURCE_SKILL: shuffled,
        RetargetingCondition.AUTHENTIC_SOURCE_SKILL: authentic,
        RetargetingCondition.TARGET_ORACLE_SKILL: oracle,
    }
    outcomes = []
    for episode_id in episode_specs:
        for condition in RetargetingCondition:
            outcomes.append(run_retargeting_episode(
                environment=environment,
                episode_id=episode_id,
                condition=condition,
                harness=harness,
                fallback_policy=fallback,
                maximum_steps=int(config["maximum_steps"]),
                skill_artifact=artifacts[condition],
            ))
    report = evaluate_retargeting_experiment(
        outcomes,
        minimum_authentic_intervention_rate=float(
            config["minimum_authentic_intervention_rate"]
        ),
    )
    return report, {
        "authentic_skill_hash": authentic.artifact_hash,
        "shuffled_skill_hash": shuffled.artifact_hash,
        "target_oracle_hash": oracle.artifact_hash,
        "target_harness_hash": harness.artifact.artifact_hash,
        "target_fallback_policy_hash": fallback.policy_hash,
        "environment_hash": environment.environment_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/harness_retargeting_smoke_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "CONTROLLED_MECHANISM_SMOKE":
        raise SystemExit("refusing to run a config outside the controlled smoke boundary")
    report, artifacts = run_smoke(config)
    payload = {
        "schema_version": "harness-retargeting-smoke-v1",
        "claim_boundary": config["claim_boundary"],
        "config_path": (
            config_path.relative_to(REPO).as_posix()
            if config_path.is_relative_to(REPO) else str(config_path)
        ),
        "config_sha256": stable_hash(config),
        "artifacts": artifacts,
        "verdict": report.verdict.value,
        "reason": report.reason,
        "metrics": report.metrics,
        "episode_outcomes": [
            {
                "episode_id": row.pair_id,
                "condition": row.condition.value,
                "valid": row.valid,
                "status": row.status,
                "official_success": row.official_success,
                "official_score": row.official_score,
                "cumulative_reward": row.cumulative_reward,
                "steps": row.steps,
                "initial_observation_hash": row.initial_observation_hash,
                "harness_hash": row.harness_hash,
                "skill_artifact_hash": row.skill_artifact_hash,
                "receipt_chain_sha256": stable_hash([
                    receipt.receipt_hash for receipt in row.receipts
                ]),
            }
            for row in report.outcomes
        ],
    }
    output = args.output or (REPO / config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "verdict": report.verdict.value,
        "reason": report.reason,
        "metrics": report.metrics,
        "artifacts": artifacts,
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.verdict == ExperimentVerdict.MECHANISM_SUPPORTED else 2


if __name__ == "__main__":
    raise SystemExit(main())
