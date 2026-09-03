"""Target-native ALFWorld bindings for the V16 search automaton."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .alfworld_hierarchical_grounder import infer_required_option, workflow_status
from .search_automaton_transfer_v16 import (
    OUTCOME_NONTERMINAL_EFFECT,
    OUTCOME_REFUTED,
    OUTCOME_TERMINAL_VERIFIED,
)


RAW = "raw_target_only"
AUTHENTIC = "authentic_search_automaton_plus_target"
PERMUTED = "event_binding_permuted_control"
LEDGER_BLIND = "ledger_blind_control"
CEILING = "target_native_search_ceiling"
CONDITIONS = (RAW, AUTHENTIC, PERMUTED, LEDGER_BLIND, CEILING)

_REFUTATION_MARKERS = (
    "nothing happens",
    "can't",
    "cannot",
    "not carrying",
    "not holding",
    "not possible",
    "not visible",
    "you don't see",
)


def target_policy_rank(
    grounded: Mapping[str, Mapping[str, float | str]],
    history: Sequence[str],
    *,
    discount_repeats: bool,
    structured: bool = False,
) -> list[str]:
    """Rank with target-trained heads and optional target workflow grounding."""

    def score(action: str) -> tuple[Any, ...]:
        row = grounded[action]
        probability = float(
            row.get("policy", row["applicability"])
        )
        denominator = 1.0 + history.count(action) if discount_repeats else 1.0
        if not structured:
            return probability / denominator, action
        option = str(row["option"])
        binding_factor = (
            1.0 if option == "SEARCH" else 0.25 + 0.75 * float(row["binding"])
        )
        effect_score = (
            float(row["applicability"])
            * (0.20 + 0.80 * float(row["completion"]))
            * binding_factor
            / denominator
        )
        return (
            float(option == str(row["required_option"])),
            effect_score,
            probability / denominator,
            action,
        )

    return sorted(grounded, key=score, reverse=True)


def target_scope_id(
    *,
    goal: str,
    native_actions: Sequence[str],
    history: Sequence[str],
) -> str:
    """Hash a target workflow/affordance state, excluding response wording."""

    status = workflow_status(goal, history)
    return stable_hash({
        "goal": goal,
        "native_action_set": sorted(map(str, native_actions)),
        "workflow": {
            "held": status.held,
            "transformed": status.transformed,
            "placed_count": status.placed_count,
        },
    })


def classify_target_outcome(
    *,
    goal: str,
    selected_action: str,
    selected_grounding: Mapping[str, float | str],
    effect_history: Sequence[str],
    before_observation: str,
    after_observation: str,
    before_native_actions: Sequence[str],
    after_native_actions: Sequence[str],
    official_success_after: bool,
) -> str:
    """Classify feedback without consulting an evaluator reward or expert."""

    if official_success_after:
        return OUTCOME_TERMINAL_VERIFIED
    normalized = after_observation.lower()
    if any(marker in normalized for marker in _REFUTATION_MARKERS):
        return OUTCOME_REFUTED
    required_before = str(selected_grounding["required_option"])
    selected_option = str(selected_grounding["option"])
    required_after = infer_required_option(
        goal=goal,
        native_actions=after_native_actions,
        action_history=(*effect_history, selected_action),
    )
    # A target candidate is supported only by goal-relevant workflow progress,
    # not by arbitrary response text or locomotion.  SEARCH actions that do
    # not expose the next affordance are therefore refuted and replanned.
    if selected_option == required_before and required_after != required_before:
        return OUTCOME_NONTERMINAL_EFFECT
    return OUTCOME_REFUTED


def summarize_episodes(
    episodes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    output = {}
    for condition, rows in episodes.items():
        output[condition] = {
            "tasks": len(rows),
            "successes": sum(bool(row["official_success"]) for row in rows),
            "success_rate": (
                sum(bool(row["official_success"]) for row in rows) / len(rows)
            ),
            "mean_steps": sum(int(row["steps"]) for row in rows) / len(rows),
            "changed_actions_vs_raw": sum(
                int(row.get("changed_actions_vs_raw", 0)) for row in rows
            ),
            "source_decisions": sum(
                int(row.get("source_decisions", 0)) for row in rows
            ),
        }
    return output


__all__ = [
    "AUTHENTIC",
    "CEILING",
    "CONDITIONS",
    "LEDGER_BLIND",
    "PERMUTED",
    "RAW",
    "classify_target_outcome",
    "summarize_episodes",
    "target_policy_rank",
    "target_scope_id",
]
