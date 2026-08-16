"""Apply the unchanged Phase-3 AttemptLedger to ALFWorld native actions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .alfworld_hierarchical_grounder import action_option
from .contracts import stable_hash
from .phase3_attempt_runtime import AnonymousAttemptRuntime
from .phase3_source_portfolio import (
    permute_selected_effect_binding,
    select_source_program_portfolio,
)


CONDITIONS = (
    "neural_only",
    "source_induced",
    "source_permuted",
    "generic_scaffold",
    "target_native_ceiling",
)

EFFECT_OBSERVATION_HORIZONS = {
    "EFFECT_BY_TRANSITION_1": 1,
    "EFFECT_BY_TRANSITION_4": 4,
    "EFFECT_BY_TRANSITION_8": 8,
    "EXECUTABLE_TRANSITION_PERSISTENCE": 8,
}


def effect_observation_horizon(effect_type: str) -> int:
    try:
        return EFFECT_OBSERVATION_HORIZONS[str(effect_type)]
    except KeyError as error:
        raise ValueError(f"unknown typed effect horizon: {effect_type}") from error


def _candidate_view(
    grounded: Mapping[str, Mapping[str, Any]], history: Sequence[str],
    binding_level: str,
) -> tuple[
    list[str], list[str], list[dict[str, float]], list[float], dict[str, str],
]:
    actions = sorted(map(str, grounded))
    action_policy = {
        action: (
        float(grounded[action]["target_policy_probability"])
        / (1.0 + history.count(action))
        ) for action in actions
    }
    if binding_level == "target_native_action":
        units = actions
        realization = {action: action for action in actions}
        ids = [str(grounded[action]["action_sha256"]) for action in actions]
        effects = [
            dict(grounded[action]["typed_effect_probabilities"])
            for action in actions
        ]
        policy = [action_policy[action] for action in actions]
        return units, ids, effects, policy, realization
    if binding_level != "target_native_option":
        raise ValueError(f"unsupported ALFWorld binding level: {binding_level}")
    grouped: dict[str, list[str]] = {}
    for action in actions:
        grouped.setdefault(action_option(action), []).append(action)
    units = sorted(grouped)
    realization = {
        option: max(
            grouped[option], key=lambda action: (action_policy[action], action),
        )
        for option in units
    }
    ids = [stable_hash({"target_native_option": option}) for option in units]
    effect_names = tuple(next(iter(
        grounded.values()
    ))["typed_effect_probabilities"])
    # A target-native option is realized by the frozen target policy.  Its
    # effect must therefore be the prediction for that same concrete action;
    # taking a maximum over other actions would silently bind one action's
    # effect to a different action's execution.
    effects = [dict(
        grounded[realization[option]]["typed_effect_probabilities"]
    ) for option in units]
    policy = [action_policy[realization[option]] for option in units]
    return units, ids, effects, policy, realization


class Phase3ALFWorldSelector:
    """One condition-specific stateful selector with a target-free source IR."""

    def __init__(
        self, *, condition: str,
        source_artifacts: Sequence[Mapping[str, Any]],
        minimum_source_policy_ratio: float = 0.0,
        binding_level: str = "target_native_action",
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unsupported Phase-3 ALFWorld condition: {condition}")
        self.condition = str(condition)
        self.source_artifacts = tuple(source_artifacts)
        self.minimum_source_policy_ratio = float(minimum_source_policy_ratio)
        if not 0.0 <= self.minimum_source_policy_ratio <= 1.0:
            raise ValueError("source policy-support ratio must be in [0,1]")
        self.binding_level = str(binding_level)
        if self.binding_level not in {
            "target_native_action", "target_native_option",
        }:
            raise ValueError("unsupported target binding level")
        self.runtime: AnonymousAttemptRuntime | None = None
        self.selected_artifact: Mapping[str, Any] | None = None
        self.pending_effect: str | None = None
        self.selected_programs: Counter[str] = Counter()
        self.selected_effects: Counter[str] = Counter()
        self.portfolio_abstentions = 0
        self.runtime_abstentions = 0

    def observe_transition(
        self, *, progress_delta: float, transition_changed: bool = False,
        persistence_fraction: float | None = None,
    ) -> None:
        effect_type = (
            str(self.selected_artifact["typed_effect_program"]["selected_effect_type"])
            if self.selected_artifact is not None else ""
        )
        if effect_type == "EXECUTABLE_TRANSITION_PERSISTENCE":
            high = (
                float(persistence_fraction) > 0.5
                if persistence_fraction is not None else bool(transition_changed)
            )
        else:
            high = float(progress_delta) > 0.0
        self.pending_effect = "HIGH" if high else "LOW"

    @staticmethod
    def _neural_action(
        actions: Sequence[str], policy: Sequence[float],
    ) -> str:
        return actions[max(
            range(len(actions)), key=lambda index: (policy[index], actions[index]),
        )]

    @staticmethod
    def _generic_action(
        actions: Sequence[str], effects: Sequence[Mapping[str, float]],
        policy: Sequence[float],
    ) -> str:
        return actions[max(range(len(actions)), key=lambda index: (
            sum(map(float, effects[index].values())) / len(effects[index]),
            policy[index],
            actions[index],
        ))]

    def _new_runtime(
        self, *, ids: Sequence[str], effects: Sequence[Mapping[str, float]],
        grounding_sha256: str,
    ) -> tuple[AnonymousAttemptRuntime | None, dict[str, Any]]:
        receipt = select_source_program_portfolio(
            self.source_artifacts,
            candidate_ids=ids,
            candidate_effects=effects,
            target_grounding_sha256=grounding_sha256,
        )
        selected_sha = receipt["selected_artifact_sha256"]
        if selected_sha is None:
            self.portfolio_abstentions += 1
            self.selected_artifact = None
            return None, receipt
        artifact = next(
            row for row in self.source_artifacts
            if row["artifact_sha256"] == selected_sha
        )
        bound_effects = list(effects)
        control_receipt = None
        if self.condition == "source_permuted":
            bound_effects, control_receipt = self._permuted_effects(
                artifact=artifact, ids=ids, effects=effects,
            )
        runtime = AnonymousAttemptRuntime(
            artifact=artifact,
            candidate_ids=ids,
            candidate_effects=bound_effects,
            target_grounding_sha256=grounding_sha256,
        )
        self.selected_artifact = artifact
        program = artifact["typed_effect_program"]
        self.selected_programs[str(program["program_sha256"])] += 1
        self.selected_effects[str(program["selected_effect_type"])] += 1
        return runtime, receipt | {
            "effect_binding_control_receipt": control_receipt,
        }

    def _bound_effects(
        self, *, ids: Sequence[str], effects: Sequence[Mapping[str, float]],
    ) -> tuple[list[Mapping[str, float]], Mapping[str, Any] | None]:
        if self.condition != "source_permuted":
            return list(effects), None
        if self.selected_artifact is None:
            raise RuntimeError("permuted arm has no selected source artifact")
        return self._permuted_effects(
            artifact=self.selected_artifact, ids=ids, effects=effects,
        )

    @staticmethod
    def _permuted_effects(
        *, artifact: Mapping[str, Any], ids: Sequence[str],
        effects: Sequence[Mapping[str, float]],
    ) -> tuple[list[Mapping[str, float]], Mapping[str, Any]]:
        if len(ids) <= 1:
            body = {
                "schema_version": "phase3-singleton-effect-control-v1",
                "status": "IDENTITY_CONTROL_SINGLETON_NOT_PERMUTABLE",
                "candidate_ids": list(ids),
                "target_outcome_read": False,
            }
            return list(effects), body | {"receipt_sha256": stable_hash(body)}
        values, receipt = permute_selected_effect_binding(
            artifact["typed_effect_program"],
            candidate_ids=ids,
            candidate_effects=effects,
        )
        return list(values), receipt

    def select(
        self, *, grounded: Mapping[str, Mapping[str, Any]],
        history: Sequence[str], expert_action: str | None = None,
    ) -> dict[str, Any]:
        units, ids, effects, policy, realization = _candidate_view(
            grounded, history, self.binding_level,
        )
        if not units:
            raise ValueError("ALFWorld target grounder returned no native action")
        fallback_unit = self._neural_action(units, policy)
        fallback = realization[fallback_unit]
        if self.condition == "neural_only":
            return {
                "action": fallback,
                "fallback_action": fallback,
                "source_admitted": False,
                "reason": "TARGET_NATIVE_NEURAL_POLICY",
            }
        if self.condition == "generic_scaffold":
            selected_unit = self._generic_action(units, effects, policy)
            selected = realization[selected_unit]
            return {
                "action": selected,
                "fallback_action": fallback,
                "selected_target_unit": selected_unit,
                "source_admitted": False,
                "reason": "SOURCE_FREE_MEAN_TYPED_EFFECT",
            }
        if self.condition == "target_native_ceiling":
            if expert_action not in grounded:
                raise ValueError("target-native expert action is not admissible")
            return {
                "action": str(expert_action),
                "fallback_action": fallback,
                "source_admitted": False,
                "reason": "TARGET_NATIVE_EXPERT_CEILING",
            }

        grounding_sha = stable_hash({
            "grounded": {action: grounded[action] for action in sorted(grounded)},
            "binding_level": self.binding_level,
        })
        events: list[dict[str, Any]] = []
        for _ in range(3):
            portfolio_receipt = None
            control_receipt = None
            if self.runtime is None:
                self.runtime, portfolio_receipt = self._new_runtime(
                    ids=ids, effects=effects, grounding_sha256=grounding_sha,
                )
                if self.runtime is None:
                    return {
                        "action": fallback,
                        "fallback_action": fallback,
                        "source_admitted": False,
                        "reason": "SOURCE_PORTFOLIO_ABSTAINED",
                        "portfolio_receipt": portfolio_receipt,
                        "runtime_events": events,
                    }
                decision = self.runtime.start()
            else:
                bound_effects, control_receipt = self._bound_effects(
                    ids=ids, effects=effects,
                )
                self.runtime.rebind_candidates(
                    candidate_ids=ids,
                    candidate_effects=bound_effects,
                    target_grounding_sha256=grounding_sha,
                )
                decision = self.runtime.observe(self.pending_effect or "UNKNOWN")
                self.pending_effect = None
            events.append({
                "kind": decision.kind,
                "candidate_id": decision.candidate_id,
                "reason": decision.reason,
                "receipt_sha256": decision.receipt_sha256,
            })
            if decision.kind == "TERMINATE":
                self.runtime = None
                self.selected_artifact = None
                continue
            if decision.kind == "ABSTAIN":
                self.runtime_abstentions += 1
                self.runtime = None
                self.selected_artifact = None
                return {
                    "action": fallback,
                    "fallback_action": fallback,
                    "source_admitted": False,
                    "reason": "SOURCE_RUNTIME_ABSTAINED",
                    "portfolio_receipt": portfolio_receipt,
                    "effect_binding_control_receipt": control_receipt,
                    "runtime_events": events,
                }
            selected_unit = next((
                unit for unit, candidate_id in zip(units, ids)
                if candidate_id == decision.candidate_id
            ), None)
            if selected_unit is None:
                self.runtime_abstentions += 1
                self.runtime = None
                self.selected_artifact = None
                return {
                    "action": fallback,
                    "fallback_action": fallback,
                    "source_admitted": False,
                    "reason": "SOURCE_TRIAL_NOT_REGROUNDED",
                    "runtime_events": events,
                }
            selected = realization[selected_unit]
            selected_index = units.index(selected_unit)
            fallback_index = units.index(fallback_unit)
            support_ratio = policy[selected_index] / max(
                policy[fallback_index], 1e-300,
            )
            if support_ratio < self.minimum_source_policy_ratio:
                # The target-native qualification threshold owns grounding
                # admission.  A rejected source trial was never executed, so
                # its ledger cannot legitimately consume a target effect.
                self.runtime = None
                self.selected_artifact = None
                return {
                    "action": fallback,
                    "fallback_action": fallback,
                    "source_admitted": False,
                    "reason": "TARGET_NATIVE_POLICY_SUPPORT_ABSTENTION",
                    "source_candidate_action": selected,
                    "source_candidate_target_unit": selected_unit,
                    "source_policy_support_ratio": support_ratio,
                    "minimum_source_policy_support_ratio": (
                        self.minimum_source_policy_ratio
                    ),
                    "runtime_events": events,
                }
            program = self.selected_artifact["typed_effect_program"]
            return {
                "action": selected,
                "fallback_action": fallback,
                "source_admitted": True,
                "reason": "SOURCE_INDUCED_ANONYMOUS_TRIAL_DELTA",
                "selected_program_sha256": str(program["program_sha256"]),
                "selected_effect_type": str(program["selected_effect_type"]),
                "selected_target_unit": selected_unit,
                "source_policy_support_ratio": support_ratio,
                "minimum_source_policy_support_ratio": (
                    self.minimum_source_policy_ratio
                ),
                "portfolio_receipt": portfolio_receipt,
                "effect_binding_control_receipt": control_receipt,
                "runtime_events": events,
            }
        raise RuntimeError("source runtime emitted too many non-action decisions")


__all__ = [
    "CONDITIONS", "EFFECT_OBSERVATION_HORIZONS", "Phase3ALFWorldSelector",
    "effect_observation_horizon",
]
