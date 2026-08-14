"""Intervention-grounded procedural games for typed workflow transfer.

The source environment is deliberately small, but it is an actual finite-horizon
MDP rather than a table of supervised workflow labels.  At every frozen state we
fork the same state/RNG stream across all native actions, execute the first
intervention, and then follow a source-native optimal continuation.  Only the
resulting token-free option/effect features and matched returns are exported.

The five option names are canonical aliases for audit readability.  Native action
tokens are alpha-renamed independently in every domain and are never model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np

from .hierarchical_skill_transfer import (
    HierarchicalValueExample,
    OPTION_NAMES,
    option_features,
)


WORKFLOW_PATTERNS = (
    ("SEARCH", "ACQUIRE", "PLACE"),
    ("SEARCH", "ACQUIRE", "TRANSFORM", "PLACE"),
    ("SEARCH", "TRANSFORM", "VERIFY"),
    ("SEARCH", "ACQUIRE", "PLACE", "SEARCH", "ACQUIRE", "PLACE"),
    ("SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "VERIFY"),
)


def stable_seed(*values: object) -> int:
    payload = "\0".join(map(str, values)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)


@dataclass(frozen=True)
class WorkflowGameDomain:
    surface: str
    domain_index: int
    workflow: tuple[str, ...]
    native_action_by_option: Mapping[str, str]
    completion_probability: Mapping[str, float]
    failure_cost: Mapping[str, float]
    progress_reward: float
    invalid_option_cost: float
    maximum_budget: int

    def __post_init__(self) -> None:
        if not self.workflow or any(option not in OPTION_NAMES for option in self.workflow):
            raise ValueError("workflow contains an unknown option")
        if set(self.native_action_by_option) != set(OPTION_NAMES):
            raise ValueError("every canonical option needs one native action")
        if len(set(self.native_action_by_option.values())) != len(OPTION_NAMES):
            raise ValueError("native action tokens must be unique")
        if self.maximum_budget < 2:
            raise ValueError("maximum budget must be at least two")


@dataclass(frozen=True)
class InterventionReceipt:
    state_id: str
    surface: str
    domain_index: int
    stage: int
    budget: int
    native_action: str
    canonical_option: str
    required_option: str
    replicate: int
    first_transition_advanced: bool
    first_transition_noop: bool
    completed_workflow: bool
    discounted_return: float


@dataclass(frozen=True)
class SourceGameCollection:
    examples: tuple[HierarchicalValueExample, ...]
    receipts: tuple[InterventionReceipt, ...]
    domains: int
    states: int
    alpha_renamed_native_actions: bool = True
    raw_action_tokens_exported: bool = False


def make_domain(
    *,
    surface: str,
    domain_index: int,
    seed: int,
    maximum_budget: int,
    completion_probability_range: Sequence[float],
    failure_cost_range: Sequence[float],
    progress_reward: float,
    invalid_option_cost: float,
) -> WorkflowGameDomain:
    rng = np.random.default_rng(stable_seed(seed, surface, domain_index, "domain"))
    workflow = WORKFLOW_PATTERNS[int(rng.integers(0, len(WORKFLOW_PATTERNS)))]
    # Tokens deliberately carry no stable semantics across domains.
    token_ids = rng.permutation(np.arange(1000, 1000 + len(OPTION_NAMES)))
    native = {
        option: f"a{int(token_id):04d}_{stable_seed(surface, domain_index, option) % 997:03d}"
        for option, token_id in zip(OPTION_NAMES, token_ids)
    }
    completion = {
        option: float(rng.uniform(*completion_probability_range))
        for option in OPTION_NAMES
    }
    costs = {
        option: float(rng.uniform(*failure_cost_range)) for option in OPTION_NAMES
    }
    return WorkflowGameDomain(
        surface=surface,
        domain_index=domain_index,
        workflow=tuple(workflow),
        native_action_by_option=native,
        completion_probability=completion,
        failure_cost=costs,
        progress_reward=float(progress_reward),
        invalid_option_cost=float(invalid_option_cost),
        maximum_budget=int(maximum_budget),
    )


def _continuation_values(
    domain: WorkflowGameDomain,
) -> np.ndarray:
    """Exact source-native continuation values used after the fork action."""

    stages = len(domain.workflow)
    values = np.zeros((stages + 1, domain.maximum_budget + 1), dtype=np.float64)
    values[stages, :] = 1.0
    for budget in range(1, domain.maximum_budget + 1):
        for stage in range(stages - 1, -1, -1):
            candidates = []
            required = domain.workflow[stage]
            for option in OPTION_NAMES:
                cost = float(domain.failure_cost[option])
                if option == required:
                    probability = float(domain.completion_probability[option])
                    q_value = (
                        -cost
                        + probability
                        * (
                            domain.progress_reward
                            + values[stage + 1, budget - 1]
                        )
                        + (1.0 - probability) * values[stage, budget - 1]
                    )
                else:
                    q_value = (
                        -cost
                        - domain.invalid_option_cost
                        + values[stage, budget - 1]
                    )
                candidates.append(q_value)
            values[stage, budget] = max(candidates)
    return values


def _fork_return(
    *,
    domain: WorkflowGameDomain,
    continuation: np.ndarray,
    stage: int,
    budget: int,
    option: str,
    completion_probability: float,
    uniform_draw: float,
) -> tuple[float, bool, bool]:
    cost = float(domain.failure_cost[option])
    required = domain.workflow[stage]
    if option != required:
        return (
            -cost - domain.invalid_option_cost + continuation[stage, budget - 1],
            False,
            False,
        )
    advanced = uniform_draw < completion_probability
    next_stage = stage + int(advanced)
    reward = -cost + (domain.progress_reward if advanced else 0.0)
    return reward + continuation[next_stage, budget - 1], advanced, (
        next_stage == len(domain.workflow)
    )


def collect_intervention_examples(
    *,
    surfaces: Sequence[str],
    domains_per_surface: int,
    states_per_domain: int,
    replicates_per_action: int,
    seed: int,
    minimum_budget: int,
    maximum_budget: int,
    completion_probability_range: Sequence[float],
    failure_cost_range: Sequence[float],
    progress_reward: float,
    invalid_option_cost: float,
    retain_receipts: bool = True,
) -> SourceGameCollection:
    if not surfaces or len(set(surfaces)) != len(surfaces):
        raise ValueError("source surfaces must be unique and non-empty")
    if domains_per_surface < 1 or states_per_domain < 1:
        raise ValueError("source domain and state counts must be positive")
    if replicates_per_action < 2:
        raise ValueError("matched intervention estimates require at least two replicates")
    if not 2 <= minimum_budget <= maximum_budget:
        raise ValueError("invalid source-game budget range")

    examples: list[HierarchicalValueExample] = []
    receipts: list[InterventionReceipt] = []
    state_count = 0
    for surface in surfaces:
        for domain_index in range(domains_per_surface):
            domain = make_domain(
                surface=str(surface),
                domain_index=domain_index,
                seed=seed,
                maximum_budget=maximum_budget,
                completion_probability_range=completion_probability_range,
                failure_cost_range=failure_cost_range,
                progress_reward=progress_reward,
                invalid_option_cost=invalid_option_cost,
            )
            continuation = _continuation_values(domain)
            state_rng = np.random.default_rng(
                stable_seed(seed, surface, domain_index, "states")
            )
            for state_index in range(states_per_domain):
                stage = int(state_rng.integers(0, len(domain.workflow)))
                budget = int(state_rng.integers(minimum_budget, maximum_budget + 1))
                state_id = f"{surface}:{domain_index}:{state_index}"
                state_count += 1
                common_draws = np.random.default_rng(
                    stable_seed(seed, state_id, "matched-forks")
                ).random(replicates_per_action)
                required = domain.workflow[stage]
                for option in OPTION_NAMES:
                    precondition = float(state_rng.beta(5, 3))
                    binding = float(state_rng.beta(4, 3))
                    completion = float(
                        domain.completion_probability[option]
                        * (0.5 + 0.5 * precondition)
                    )
                    fork_returns = []
                    for replicate, draw in enumerate(common_draws):
                        value, advanced, terminated = _fork_return(
                            domain=domain,
                            continuation=continuation,
                            stage=stage,
                            budget=budget,
                            option=option,
                            completion_probability=completion,
                            uniform_draw=float(draw),
                        )
                        fork_returns.append(value)
                        if retain_receipts:
                            receipts.append(InterventionReceipt(
                                state_id=state_id,
                                surface=str(surface),
                                domain_index=domain_index,
                                stage=stage,
                                budget=budget,
                                native_action=domain.native_action_by_option[option],
                                canonical_option=option,
                                required_option=required,
                                replicate=replicate,
                                first_transition_advanced=advanced,
                                first_transition_noop=not advanced,
                                completed_workflow=terminated,
                                discounted_return=float(value),
                            ))
                    repeat = float(state_rng.integers(0, 5)) / 4.0
                    features = option_features(
                        option=option,
                        required_option=required,
                        precondition_satisfied=precondition,
                        completion_probability=completion,
                        goal_binding_probability=binding,
                        remaining_budget_fraction=budget / maximum_budget,
                        workflow_progress_fraction=stage / max(1, len(domain.workflow) - 1),
                        action_repeat_fraction=repeat,
                        noop_probability=1.0 - completion,
                        stage_urgency=(len(domain.workflow) - stage) / budget,
                        failure_cost=domain.failure_cost[option],
                    )
                    examples.append(HierarchicalValueExample(
                        state_id=state_id,
                        option=option,
                        features=features,
                        value=float(np.mean(fork_returns)),
                    ))
    return SourceGameCollection(
        examples=tuple(examples),
        receipts=tuple(receipts),
        domains=len(surfaces) * domains_per_surface,
        states=state_count,
    )


__all__ = [
    "InterventionReceipt",
    "SourceGameCollection",
    "WorkflowGameDomain",
    "collect_intervention_examples",
    "make_domain",
    "stable_seed",
]
