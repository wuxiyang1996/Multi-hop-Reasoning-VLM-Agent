"""Shared outcome labelling for canonical cross-domain source episodes.

The three published memory baselines disagree about what outcome supervision they
are entitled to: ExpeL contrasts environment-reported successes against failures,
offline AWM abstracts workflows from successful or otherwise canonical
trajectories, and only ReasoningBank natively runs an LLM judge.  Letting each
method bring its own labeller would hand ReasoningBank supervision the others were
denied, so the main experiment resolves every episode once, here, and hands the
identical labels to all four arms.

Resolution order is fixed and recorded per episode:

``official predicate > benchmark-defined predicate > shared frozen evaluator > UNKNOWN``

A missing outcome is ``UNKNOWN``.  It is never rewritten as a failure, and an
``UNKNOWN`` episode is withheld from every method rather than inflating one
method's failure pool.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import stable_hash
from .cross_domain_memory_baselines import (
    OutcomeAuthority,
    OutcomeLabel,
    resolve_source_outcome,
)
from .frozen_motif_agent import CompletionBackend

JUDGE_ROLE = "source_outcome_judge"

JUDGE_SYSTEM = (
    "Judge whether one recorded game episode achieved that game's own objective. "
    "You are given the ordered observations, actions, and rewards of a single "
    "source episode. Decide only about this episode's own domain; no other task, "
    "benchmark, or domain exists for this judgement. Answer SUCCESS only when the "
    "trajectory shows the game's objective was met, FAILURE only when it shows the "
    "objective was definitively not met, and UNKNOWN whenever the record is "
    "insufficient to decide. Prefer UNKNOWN over guessing. Return exactly one JSON "
    'object {"verdict": "SUCCESS"|"FAILURE"|"UNKNOWN", "reason": "<one sentence>"}.'
)


class OutcomeRuleError(ValueError):
    """A declared benchmark predicate is malformed or unknown."""


@dataclass(frozen=True)
class OutcomeDecision:
    episode_id: str
    label: str
    authority: str
    reason: str


COHORT_RULE = "WITHIN_GAME_SCORE_TERCILE"


def _rule_verdict(
    rule: Mapping[str, Any],
    episode: Mapping[str, Any],
    cohort: Callable[[Mapping[str, Any]], bool | None] | None = None,
) -> bool | None:
    """Evaluate one declared benchmark predicate, or abstain."""
    kind = str(rule.get("kind") or "")
    if kind == "ABSTAIN":
        return None
    if kind == COHORT_RULE:
        if cohort is None:
            raise OutcomeRuleError(
                f"{COHORT_RULE} ranks an episode against its game's cohort, so the "
                "full episode list must be supplied when building the predicate"
            )
        return cohort(episode)
    if kind == "TOTAL_REWARD_AT_LEAST":
        if "threshold" not in rule:
            raise OutcomeRuleError("TOTAL_REWARD_AT_LEAST needs a declared threshold")
        return float(episode.get("total_reward", 0.0)) >= float(rule["threshold"])
    if kind == "TERMINATED_IS_FAILURE":
        # Reaching a terminal state is a loss in games whose only stop is defeat;
        # surviving to the step budget is not thereby a win, so that case abstains.
        return False if bool(episode.get("terminated")) else None
    raise OutcomeRuleError(f"unknown benchmark predicate kind: {kind!r}")


def within_game_quality_predicate(
    episodes: Sequence[Mapping[str, Any]],
    *,
    metric: str = "total_reward",
    high_quantile: float = 2.0 / 3.0,
    low_quantile: float = 1.0 / 3.0,
    minimum_cohort: int = 6,
) -> Callable[[Mapping[str, Any]], bool | None]:
    """Rank each episode against the same game's own cohort.

    Score-maximisation games have no win condition, so "did this episode succeed"
    is unanswerable and any absolute cut-off would be invented here.  What the
    game *does* define is its own score, and what this study transfers is skill,
    not victory.  So the positive class is a high-quality demonstration relative
    to the same policy on the same game, and the negative class is a poor one.

    This is deliberately not ``reward > 0``: the comparison is within-game and
    relative, never a fixed number applied across domains.  The middle band
    abstains, and a game whose cohort is too small or whose scores are tied
    abstains entirely rather than manufacturing a split.
    """
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise OutcomeRuleError("quantiles must satisfy 0 < low < high < 1")
    cohorts: dict[str, list[float]] = {}
    for episode in episodes:
        cohorts.setdefault(str(episode.get("source_domain") or ""), []).append(
            float(episode.get(metric, 0.0))
        )

    bounds: dict[str, tuple[float, float]] = {}
    for domain, scores in cohorts.items():
        ordered = sorted(scores)
        if len(ordered) < minimum_cohort or ordered[0] == ordered[-1]:
            continue
        def _at(fraction: float) -> float:
            position = fraction * (len(ordered) - 1)
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
        low, high = _at(low_quantile), _at(high_quantile)
        # Heavy ties can collapse both quantiles onto one value, which would erase
        # the abstention band and force every episode into a class on a tie-break.
        if low >= high:
            continue
        bounds[domain] = (low, high)

    def predicate(episode: Mapping[str, Any]) -> bool | None:
        domain = str(episode.get("source_domain") or "")
        if domain not in bounds:
            return None
        low, high = bounds[domain]
        score = float(episode.get(metric, 0.0))
        if score >= high:
            return True
        if score <= low:
            return False
        return None

    predicate.cohort_bounds = dict(bounds)  # type: ignore[attr-defined]
    return predicate


def benchmark_predicate_from_config(
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]] | None = None,
) -> Callable[[Mapping[str, Any]], bool | None]:
    """Build the per-domain benchmark predicate declared in a frozen config.

    ``episodes`` is required when any domain declares a cohort rule, which ranks
    an episode against the same game's other episodes instead of against a fixed
    number.
    """
    declared = config.get("predicates")
    if not isinstance(declared, Mapping) or not declared:
        raise OutcomeRuleError("outcome config needs a non-empty predicates map")
    for domain, rule in declared.items():
        if not isinstance(rule, Mapping) or "kind" not in rule:
            raise OutcomeRuleError(f"predicate for {domain} must declare a kind")

    cohort = None
    if any(str(rule.get("kind")) == COHORT_RULE for rule in declared.values()):
        if episodes is None:
            raise OutcomeRuleError(
                f"{COHORT_RULE} is declared but no episode cohort was supplied"
            )
        settings = dict(config.get("cohort_rule") or {})
        cohort = within_game_quality_predicate(
            episodes,
            metric=str(settings.get("metric", "total_reward")),
            high_quantile=float(settings.get("high_quantile", 2.0 / 3.0)),
            low_quantile=float(settings.get("low_quantile", 1.0 / 3.0)),
            minimum_cohort=int(settings.get("minimum_cohort", 6)),
        )

    def predicate(episode: Mapping[str, Any]) -> bool | None:
        domain = str(episode.get("source_domain") or "")
        rule = declared.get(domain)
        if rule is None:
            raise OutcomeRuleError(
                f"no benchmark predicate declared for source domain {domain!r}; "
                "declare an explicit ABSTAIN rather than leaving it undeclared"
            )
        return _rule_verdict(rule, episode, cohort)

    predicate.cohort_bounds = getattr(cohort, "cohort_bounds", {})  # type: ignore[attr-defined]
    return predicate


def _episode_view(episode: Mapping[str, Any], *, maximum_steps: int) -> dict[str, Any]:
    steps = list(episode.get("steps") or ())
    if len(steps) > maximum_steps:
        # Keep both ends: the objective is usually visible in the opening state and
        # in whatever the episode finally reached.
        head = maximum_steps // 2
        steps = steps[:head] + steps[len(steps) - (maximum_steps - head):]
    return {
        "source_domain": episode.get("source_domain"),
        "terminated": bool(episode.get("terminated", False)),
        "truncated": bool(episode.get("truncated", False)),
        "total_reward": float(episode.get("total_reward", 0.0)),
        "step_count": len(episode.get("steps") or ()),
        "steps": [
            {
                "step": row.get("step"),
                "observation": row.get("observation"),
                "action": row.get("action"),
                "next_observation": row.get("next_observation"),
                "reward": row.get("reward"),
                "terminal": row.get("terminal"),
            }
            for row in steps
        ],
    }


class FrozenJudgeEvaluator:
    """One frozen LLM judge, applied identically to every method's episodes."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        maximum_steps_shown: int = 24,
    ) -> None:
        self.backend = backend
        self.maximum_steps_shown = maximum_steps_shown
        self.receipts: list[dict[str, Any]] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "judge": "frozen-source-outcome-judge-v1",
            "system_sha256": stable_hash(JUDGE_SYSTEM),
            "maximum_steps_shown": self.maximum_steps_shown,
            "backend": dict(self.backend.identity),
        }

    def __call__(self, episode: Mapping[str, Any]) -> bool | None:
        view = _episode_view(episode, maximum_steps=self.maximum_steps_shown)
        raw = self.backend.complete(JUDGE_ROLE, JUDGE_SYSTEM, view)
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("source outcome judge returned a non-object")
        verdict = OutcomeLabel(str(value.get("verdict") or ""))
        self.receipts.append({
            "episode_id": str(episode.get("episode_id") or ""),
            "input_sha256": stable_hash(view),
            "response_sha256": stable_hash(raw),
            "verdict": verdict.value,
            "reason": str(value.get("reason") or "")[:500],
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
        })
        if verdict is OutcomeLabel.UNKNOWN:
            return None
        return verdict is OutcomeLabel.SUCCESS


def label_source_payload(
    payload: Mapping[str, Any],
    *,
    benchmark_predicate: Callable[[Mapping[str, Any]], bool | None] | None = None,
    shared_evaluator: Callable[[Mapping[str, Any]], bool | None] | None = None,
    attribution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate every episode with a resolved outcome and its deciding authority.

    ``attribution`` carries the config hash and evaluator identity that produced
    these labels; it is folded into the signed body so the labels cannot be
    replayed under a different evaluator without breaking the hash.
    """
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("source payload needs a non-empty episodes list")
    decisions: list[OutcomeDecision] = []
    labelled: list[dict[str, Any]] = []
    for raw_episode in raw_episodes:
        if str(raw_episode.get("outcome") or ""):
            raise ValueError(
                f"episode {raw_episode.get('episode_id')} is already labelled; "
                "relabelling a frozen payload would break its lineage"
            )
        label, authority = resolve_source_outcome(
            raw_episode,
            benchmark_predicate=benchmark_predicate,
            shared_evaluator=shared_evaluator,
        )
        decisions.append(OutcomeDecision(
            episode_id=str(raw_episode.get("episode_id") or ""),
            label=label.value,
            authority=authority.value,
            reason=f"resolved by {authority.value}",
        ))
        labelled.append(dict(raw_episode) | {
            "outcome": label.value,
            "outcome_authority": authority.value,
        })
    census: dict[str, int] = {row.value: 0 for row in OutcomeLabel}
    by_authority: dict[str, int] = {row.value: 0 for row in OutcomeAuthority}
    for decision in decisions:
        census[decision.label] += 1
        by_authority[decision.authority] += 1
    body = dict(payload) | {
        "episodes": labelled,
        "outcome_census": census,
        "outcome_authority_census": by_authority,
        "outcome_decisions": [vars(row) for row in decisions],
        "outcome_attribution": dict(attribution or {}),
    }
    body.pop("source_labelled_sha256", None)
    return body | {"source_labelled_sha256": stable_hash(body)}


_KNOWN_RULE_KINDS = frozenset({
    "ABSTAIN", "TOTAL_REWARD_AT_LEAST", "TERMINATED_IS_FAILURE", COHORT_RULE,
})


def load_outcome_config(path: str | Path) -> dict[str, Any]:
    """Load a frozen outcome config and check its declarations are well formed.

    Structure only: cohort rules cannot be instantiated until the episodes they
    rank against are known.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise OutcomeRuleError("outcome config must be one JSON object")
    declared = config.get("predicates")
    if not isinstance(declared, Mapping) or not declared:
        raise OutcomeRuleError("outcome config needs a non-empty predicates map")
    for domain, rule in declared.items():
        if not isinstance(rule, Mapping) or "kind" not in rule:
            raise OutcomeRuleError(f"predicate for {domain} must declare a kind")
        kind = str(rule["kind"])
        if kind not in _KNOWN_RULE_KINDS:
            raise OutcomeRuleError(f"unknown benchmark predicate kind: {kind!r}")
    return config


def declared_domains(config: Mapping[str, Any]) -> Sequence[str]:
    return sorted(map(str, (config.get("predicates") or {})))


__all__ = [
    "FrozenJudgeEvaluator", "JUDGE_SYSTEM", "OutcomeDecision", "OutcomeRuleError",
    "benchmark_predicate_from_config", "declared_domains", "label_source_payload",
    "load_outcome_config",
]
