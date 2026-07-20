"""Compile evidence-index rows into conservative source skill programs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from skill_bank.program_ir import (
    ActionSchema,
    CanonicalSkillProgram,
    EffectKind,
    Operator,
    ProgramStatus,
    ProgramStep,
    SourceStepKey,
    TransitionEvidenceRef,
    TypedEffect,
)


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def compile_source_programs(
    invocation_rows: Iterable[Mapping[str, Any]],
    *,
    min_invocations: int = 2,
) -> List[CanonicalSkillProgram]:
    """Compile only directly observed skill/action transitions.

    This first compiler intentionally performs no semantic clustering.  It
    groups exact ``(game, chosen_skill_id)`` identities and exposes the set of
    actually observed primitive actions.  Later anti-unification may consume
    these verified programs, but it may not manufacture their evidence.
    """
    groups: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in invocation_rows:
        game = str(row.get("game") or "").strip()
        skill_id = str(row.get("chosen_skill_id") or "").strip()
        if game and skill_id and row.get("source_only", True):
            groups[(game, skill_id)].append(row)

    programs: List[CanonicalSkillProgram] = []
    for (game, skill_id), rows in sorted(groups.items()):
        if len(rows) < min_invocations:
            continue
        evidence: List[TransitionEvidenceRef] = []
        for row in rows:
            key = SourceStepKey(
                game=game,
                episode_id=str(row["episode_id"]),
                step_index=int(row["step_index"]),
                provider_or_run=str(row.get("provider_or_run") or "unknown"),
            )
            evidence.append(
                TransitionEvidenceRef(
                    key=key,
                    source_file_sha256=str(row["source_file_sha256"]),
                    state_sha256=str(row["state_sha256"]),
                    next_state_sha256=str(row["next_state_sha256"]),
                    action=str(row["action"]),
                    reward=float(row.get("reward") or 0.0),
                    done=bool(row.get("done")),
                )
            )
        evidence_ids = [item.key.stable_id() for item in evidence]
        effect_evidence: Dict[str, List[str]] = defaultdict(list)
        for row, evidence_id in zip(rows, evidence_ids):
            for effect in row.get("source_effects") or []:
                effect_evidence[str(effect)].append(evidence_id)
        observed_actions = sorted({item.action for item in evidence})
        effects = [
            TypedEffect(
                kind=EffectKind.CHANGE,
                predicate="source_state_transition_observed",
                value_type="sha256_pair",
                evidence_step_ids=evidence_ids,
            )
        ]
        if all(item.reward >= 0.0 for item in evidence):
            effects.append(
                TypedEffect(
                    kind=EffectKind.EVENT,
                    predicate="source_reward_nonnegative",
                    value_type="float",
                    evidence_step_ids=evidence_ids,
                )
            )
        for predicate, supporting_ids in sorted(effect_evidence.items()):
            effects.append(
                TypedEffect(
                    kind=EffectKind.CHANGE,
                    predicate=predicate,
                    value_type="typed_source_state_delta",
                    evidence_step_ids=supporting_ids,
                )
            )
        program = CanonicalSkillProgram(
            program_id=f"source.{game}.{_safe_id(skill_id)}",
            name=skill_id,
            source_skill_ids=[skill_id],
            source_games=[game],
            status=ProgramStatus.SOURCE_VERIFIED,
            evidence=evidence,
            steps=[
                ProgramStep(
                    step_id="commit-observed-action",
                    operator=Operator.COMMIT,
                    action=ActionSchema(
                        name="source_primitive_action",
                        argument_types={},
                        observed_source_actions=observed_actions,
                    ),
                    effects=effects,
                    evidence_step_ids=evidence_ids,
                )
            ],
            metadata={
                "compiler": "exact_source_identity_typed_effects_v2",
                "n_invocations": len(evidence),
                "no_semantic_clustering": True,
                "verified_effect_counts": {
                    key: len(value) for key, value in sorted(effect_evidence.items())
                },
            },
        )
        program.validate()
        programs.append(program)
    return programs


__all__ = ["compile_source_programs"]
