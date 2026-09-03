"""Shared matched-fork protocol for structured video benchmarks.

The neural target model proposes world/dynamics particles and typed predicate
probes from question text plus a low-bandwidth video scout.  Matched probe
receipts are then collected without asking for the final answer again.  This
module validates those target-native objects and evaluates frozen source
TEST-value models on the anonymous nine-dimensional interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from .video_dynamics_mdp import (
    PREDICATE_KINDS,
    PredicateProbe,
    PredicateProbeReceipt,
    WorldParticle,
    answer_belief,
    apply_probe_receipt,
    initial_state,
    source_compatible_action_features,
    probe_statistics,
)


FIXED_TEST_CONDITIONS = (
    "target_native_information_gain",
    "target_native_expected_accuracy",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
    "deterministic_random_probe",
)


@dataclass(frozen=True)
class ParsedTargetWorldModel:
    particles: tuple[WorldParticle, ...]
    particle_summaries: tuple[str, ...]
    probes: tuple[PredicateProbe, ...]
    probe_rationales: tuple[str, ...]


def _answer_is_valid(
    answer: str,
    *,
    valid_answers: Sequence[str] | None,
    binary_vector_length: int | None,
) -> bool:
    if valid_answers is not None:
        return answer in set(map(str, valid_answers))
    if binary_vector_length is not None:
        return (
            len(answer) == binary_vector_length
            and set(answer) <= {"0", "1"}
        )
    raise ValueError("an explicit native answer contract is required")


def parse_target_world_model(
    payload: Mapping[str, Any],
    *,
    duration_seconds: float,
    particle_count: int | None,
    probe_count: int,
    valid_answers: Sequence[str] | None = None,
    binary_vector_length: int | None = None,
    required_answers: Sequence[str] | None = None,
    maximum_object_present_probes: int | None = None,
    minimum_non_presence_probes: int = 0,
    minimum_likelihood_span: float = 0.0,
    require_bind_relate_roles: bool = False,
    minimum_bind_probes: int = 0,
    minimum_relate_probes: int = 0,
    minimum_relate_likelihood_span: float = 0.0,
) -> ParsedTargetWorldModel:
    """Fail closed on malformed or answer-reprompt-like target outputs."""

    if duration_seconds <= 0 or (
        particle_count is not None and particle_count < 2
    ) or probe_count < 2:
        raise ValueError("invalid duration or frozen model cardinalities")
    raw_particles = list(payload.get("world_particles") or ())
    raw_probes = list(payload.get("typed_probes") or ())
    if particle_count is not None and len(raw_particles) != particle_count:
        raise ValueError(
            "target model did not return the frozen particle count: "
            f"expected {particle_count}, got {len(raw_particles)}"
        )
    required = tuple(map(str, required_answers or ()))
    if required:
        if len(required) < 2 or len(set(required)) != len(required):
            raise ValueError("required_answers must contain unique native answers")
        if particle_count is not None and particle_count != len(required):
            raise ValueError("particle count must equal the complete answer space")
        if len(raw_particles) != len(required):
            raise ValueError("target model did not cover the complete answer space")
    if len(raw_probes) != probe_count:
        raise ValueError("target model did not return the frozen probe count")

    particles = []
    summaries = []
    for index, row in enumerate(raw_particles):
        answer = str(row.get("native_answer") or "").strip().upper()
        if not _answer_is_valid(
            answer,
            valid_answers=valid_answers,
            binary_vector_length=binary_vector_length,
        ):
            raise ValueError(f"particle has invalid native answer: {answer!r}")
        particle = WorldParticle(
            particle_id=f"W{index}",
            native_answer=answer,
            prior_weight=float(row.get("prior_weight", 0.0)),
        )
        particle.validate()
        summary = str(row.get("event_graph_summary") or "").strip()
        if not summary:
            raise ValueError("world particle requires an event/dynamics summary")
        particles.append(particle)
        summaries.append(summary)
    if len({particle.native_answer for particle in particles}) < 2:
        raise ValueError("world particles must express answer uncertainty")
    if required and tuple(particle.native_answer for particle in particles) != required:
        raise ValueError(
            "world particles must contain every required answer exactly once and "
            "in the frozen order"
        )

    entity_catalog: dict[str, str] = {}
    for row in payload.get("entity_catalog") or ():
        entity_id = str(row.get("entity_id") or "").strip().upper()
        description = str(row.get("visual_description") or "").strip()
        if not entity_id or not description or entity_id in entity_catalog:
            raise ValueError("entity catalog requires unique IDs and descriptions")
        entity_catalog[entity_id] = f"{entity_id}: {description}"

    probes = []
    rationales = []
    for index, row in enumerate(raw_probes):
        kind = str(row.get("predicate_kind") or "").strip().upper()
        if kind not in PREDICATE_KINDS:
            raise ValueError(f"target model emitted unsupported predicate: {kind}")
        window = list(row.get("window_fraction") or ())
        if len(window) != 2:
            raise ValueError("typed probe needs a two-value normalized window")
        start_fraction = float(window[0])
        end_fraction = float(window[1])
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError(
                "probe window_fraction must satisfy 0 <= start < end <= 1; "
                f"got [{start_fraction}, {end_fraction}]"
            )
        start = start_fraction * duration_seconds
        end = end_fraction * duration_seconds
        likelihood = tuple(map(float, row.get("true_probability_by_particle") or ()))
        raw_entity_ids = tuple(
            str(value).strip().upper() for value in row.get("entity_ids") or ()
        )
        if raw_entity_ids:
            if not entity_catalog or any(
                value not in entity_catalog for value in raw_entity_ids
            ):
                raise ValueError("typed probe references an unknown entity ID")
            entity_refs = tuple(entity_catalog[value] for value in raw_entity_ids)
        else:
            entity_refs = tuple(map(str, row.get("entity_refs") or ()))
        probe = PredicateProbe(
            probe_id=f"P{index}",
            predicate_kind=kind,
            entity_refs=entity_refs,
            start_sec=start,
            end_sec=end,
            latent_true_probability_by_particle=likelihood,
            target_tool="sample_frames",
            expected_sensor_reliability=float(
                row.get("expected_sensor_reliability", 0.0)
            ),
            target_event_role=str(row.get("target_event_role") or "").upper(),
        )
        probe.validate(len(particles))
        rationale = str(row.get("rationale") or "").strip()
        if not rationale:
            raise ValueError("typed probe requires a target-native rationale")
        probes.append(probe)
        rationales.append(rationale)
    if len({
        (probe.predicate_kind, probe.entity_refs, probe.start_sec, probe.end_sec)
        for probe in probes
    }) != len(probes):
        raise ValueError("target model emitted duplicate typed probes")
    object_present_count = sum(
        probe.predicate_kind == "OBJECT_PRESENT" for probe in probes
    )
    if maximum_object_present_probes is not None and (
        object_present_count > maximum_object_present_probes
    ):
        raise ValueError("too many option-entity OBJECT_PRESENT probes")
    if len(probes) - object_present_count < minimum_non_presence_probes:
        raise ValueError("too few relation/attribute/dynamics probes")
    for probe in probes:
        likelihood = probe.latent_true_probability_by_particle
        if max(likelihood) - min(likelihood) < minimum_likelihood_span:
            raise ValueError("typed probe does not distinguish answer hypotheses")
    if require_bind_relate_roles:
        if not entity_catalog:
            raise ValueError("BIND->RELATE target grounding requires entity_catalog")
        bind = [probe for probe in probes if probe.target_event_role == "BIND"]
        relate = [probe for probe in probes if probe.target_event_role == "RELATE"]
        if len(bind) < minimum_bind_probes or len(relate) < minimum_relate_probes:
            raise ValueError("target model did not return the frozen BIND/RELATE counts")
        if len(bind) + len(relate) != len(probes):
            raise ValueError("every typed probe requires a BIND or RELATE role")
        for probe in bind:
            if probe.predicate_kind not in {
                "OBJECT_PRESENT", "OBJECT_ATTRIBUTE", "OBJECT_TRACK",
            }:
                raise ValueError("BIND probe uses a non-binding predicate kind")
        for probe in relate:
            if probe.predicate_kind not in {
                "OBJECT_MOTION", "COLLISION", "ENTRY", "EXIT",
                "EVENT_ORDER", "CAUSAL_ANCESTOR",
            }:
                raise ValueError("RELATE probe uses a non-relational predicate kind")
            if not any(set(probe.entity_refs) & set(bound.entity_refs) for bound in bind):
                raise ValueError("RELATE probe has no source-guarded BIND entity")
            likelihood = probe.latent_true_probability_by_particle
            if max(likelihood) - min(likelihood) < minimum_relate_likelihood_span:
                raise ValueError("RELATE probe does not distinguish answer hypotheses")
    return ParsedTargetWorldModel(
        tuple(particles), tuple(summaries), tuple(probes), tuple(rationales),
    )


def parse_typed_probe_receipt(
    payload: Mapping[str, Any],
    *,
    probe: PredicateProbe,
    evidence_sha256: Sequence[str],
) -> PredicateProbeReceipt:
    value = payload.get("observed_true")
    if not isinstance(value, bool):
        raise ValueError("typed grounder must return a boolean observed_true")
    receipt = PredicateProbeReceipt(
        probe_id=probe.probe_id,
        predicate_kind=probe.predicate_kind,
        entity_refs=probe.entity_refs,
        start_sec=probe.start_sec,
        end_sec=probe.end_sec,
        observed_true=value,
        sensor_reliability=float(payload.get("sensor_reliability", 0.0)),
        evidence_sha256=tuple(map(str, evidence_sha256)),
    )
    receipt.validate_against(probe)
    return receipt


def _stable_index(sample_id: str, count: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % count


def evaluate_fixed_test_budget(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    probe_receipts: Mapping[str, PredicateProbeReceipt],
    source_models: Mapping[str, Any],
    test_budget: int,
    conditions: Sequence[str] = FIXED_TEST_CONDITIONS,
    action_contrast_reference: str = "target_native_expected_accuracy",
) -> dict[str, Any]:
    """Evaluate sequential typed-probe selection at an equal fixed budget."""

    particles = world_model.particles
    probes = world_model.probes
    if not 1 <= test_budget <= len(probes):
        raise ValueError("test_budget must be between one and the probe count")
    if set(probe_receipts) != {probe.probe_id for probe in probes}:
        raise ValueError("matched receipts must cover every frozen typed probe")
    initial = initial_state(particles, probes, max_tests=test_budget)
    _, _, answer_space = source_compatible_action_features(
        particles, initial, probes, max_tests=test_budget,
    )
    initial_belief = answer_belief(particles, initial, answer_space=answer_space)
    baseline_index = int(np.argmax(initial_belief))
    baseline_answer = answer_space[baseline_index]

    condition_rows: dict[str, Any] = {}
    for condition in conditions:
        state = initial
        trajectory: list[int] = []
        for step in range(test_budget):
            tests, _, _ = source_compatible_action_features(
                particles, state, probes, max_tests=test_budget,
                answer_space=answer_space,
            )
            available = [index for index in range(len(probes)) if index not in trajectory]
            if condition == "target_native_information_gain":
                index = max(available, key=lambda value: tests[value][1])
            elif condition == "target_native_expected_accuracy":
                index = max(available, key=lambda value: tests[value][2])
            elif condition == "deterministic_random_probe":
                index = available[_stable_index(f"{sample_id}|{step}", len(available))]
            else:
                means, _ = source_models[condition].predict(tests)
                index = max(available, key=lambda value: means[value])
            trajectory.append(index)
            probe = probes[index]
            state = apply_probe_receipt(
                particles, state, probes, probe_receipts[probe.probe_id],
            )
        posterior = answer_belief(
            particles, state, answer_space=answer_space,
        )
        committed = answer_space[int(np.argmax(posterior))]
        condition_rows[condition] = {
            "selected_probe_ids": [probes[index].probe_id for index in trajectory],
            "predicate_kinds": [probes[index].predicate_kind for index in trajectory],
            "committed_answer": committed,
            "correct": committed == gold_answer,
            "gold_probability_before": (
                float(initial_belief[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            ),
            "gold_probability_after": (
                float(posterior[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            ),
        }

    oracle_rows = []
    for trajectory in itertools.permutations(range(len(probes)), test_budget):
        state = initial
        for index in trajectory:
            probe = probes[index]
            state = apply_probe_receipt(
                particles, state, probes, probe_receipts[probe.probe_id],
            )
        posterior = answer_belief(particles, state, answer_space=answer_space)
        probability = (
            float(posterior[answer_space.index(gold_answer)])
            if gold_answer in answer_space else 0.0
        )
        oracle_rows.append((probability, trajectory, posterior))
    _, oracle_trajectory, oracle_belief = max(oracle_rows, key=lambda row: row[0])
    oracle_answer = answer_space[int(np.argmax(oracle_belief))]
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "answer_space": list(answer_space),
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_answer == gold_answer,
        "conditions": condition_rows,
        "test_budget": test_budget,
        "oracle_probe_ids": [probes[index].probe_id for index in oracle_trajectory],
        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_answer == gold_answer,
        "authentic_action_contrast": (
            condition_rows["authentic_source_plus_target"]["selected_probe_ids"]
            != condition_rows[action_contrast_reference]["selected_probe_ids"]
        ),
        "action_contrast_reference": action_contrast_reference,
    }


def evaluate_fixed_one_test(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    probe_receipts: Mapping[str, PredicateProbeReceipt],
    source_models: Mapping[str, Any],
) -> dict[str, Any]:
    """Backward-compatible one-TEST specialization."""

    return evaluate_fixed_test_budget(
        sample_id=sample_id,
        gold_answer=gold_answer,
        world_model=world_model,
        probe_receipts=probe_receipts,
        source_models=source_models,
        test_budget=1,
    )


def _target_exact_plan_value(
    particles,
    probes,
    state,
    *,
    answer_space: Sequence[str],
    selected: frozenset[int],
    test_cost: float,
) -> tuple[float, tuple[str, int]]:
    belief = answer_belief(particles, state, answer_space=answer_space)
    best = (float(np.max(belief)), ("COMMIT", int(np.argmax(belief))))
    if state.remaining_tests <= 0:
        return best
    for index, probe in enumerate(probes):
        if index in selected:
            continue
        latent = np.asarray(
            probe.latent_true_probability_by_particle, dtype=np.float64,
        )
        reliability = probe.expected_sensor_reliability
        measured_true = (
            reliability * latent + (1.0 - reliability) * (1.0 - latent)
        )
        probability_true = float(np.dot(state.world_weights, measured_true))
        expected = -test_cost
        for observed_true, probability in (
            (True, probability_true), (False, 1.0 - probability_true),
        ):
            if probability <= 1e-12:
                continue
            from .video_dynamics_mdp import _posterior, VideoDynamicsBeliefState
            posterior = _posterior(
                state.world_weights, probe, observed_true=observed_true,
                sensor_reliability=reliability,
            )
            counts = list(state.test_counts)
            counts[index] += 1
            next_state = VideoDynamicsBeliefState(
                tuple(map(float, posterior)), state.remaining_tests - 1,
                tuple(counts),
            )
            continuation, _ = _target_exact_plan_value(
                particles, probes, next_state, answer_space=answer_space,
                selected=selected | {index}, test_cost=test_cost,
            )
            expected += probability * continuation
        if expected > best[0]:
            best = (expected, ("TEST", index))
    return best


def evaluate_adaptive_test_commit(
    *,
    sample_id: str,
    gold_answer: str,
    world_model: ParsedTargetWorldModel,
    probe_receipts: Mapping[str, PredicateProbeReceipt],
    source_models: Mapping[str, Any],
    max_tests: int,
    test_cost: float,
) -> dict[str, Any]:
    """Evaluate learned TEST/COMMIT control with a variable evidence budget."""

    if not 0 <= test_cost < 1:
        raise ValueError("test_cost must be in [0,1)")
    particles, probes = world_model.particles, world_model.probes
    initial = initial_state(particles, probes, max_tests=max_tests)
    _, _, answer_space = source_compatible_action_features(
        particles, initial, probes, max_tests=max_tests,
    )
    initial_belief = answer_belief(particles, initial, answer_space=answer_space)
    baseline_answer = answer_space[int(np.argmax(initial_belief))]
    condition_names = (
        "target_native_greedy_test_commit",
        "target_native_exact_dp_test_commit",
        "authentic_source_plus_target",
        "shuffled_source_plus_target",
        "source_marginal_plus_target",
    )
    rows: dict[str, Any] = {}
    for condition in condition_names:
        state = initial
        selected: set[int] = set()
        actions: list[str] = []
        committed: str | None = None
        while committed is None:
            tests, commits, _ = source_compatible_action_features(
                particles, state, probes, max_tests=max_tests,
                answer_space=answer_space,
            )
            belief = answer_belief(particles, state, answer_space=answer_space)
            if state.remaining_tests <= 0 or len(selected) == len(probes):
                commit_index = int(np.argmax(belief))
                committed = answer_space[commit_index]
                actions.append(f"COMMIT:{committed}")
                break
            if condition == "target_native_greedy_test_commit":
                available = [index for index in range(len(probes)) if index not in selected]
                index = max(
                    available,
                    key=lambda value: probe_statistics(
                        particles, state, probes[value], answer_space=answer_space,
                    ).expected_map_confidence_gain,
                )
                gain = probe_statistics(
                    particles, state, probes[index], answer_space=answer_space,
                ).expected_map_confidence_gain
                choice = (
                    ("TEST", index) if gain > test_cost
                    else ("COMMIT", int(np.argmax(belief)))
                )
            elif condition == "target_native_exact_dp_test_commit":
                _, choice = _target_exact_plan_value(
                    particles, probes, state, answer_space=answer_space,
                    selected=frozenset(selected), test_cost=test_cost,
                )
            else:
                features = tests + commits
                means, _ = source_models[condition].predict(features)
                allowed = [
                    index for index in range(len(probes)) if index not in selected
                ] + list(range(len(probes), len(probes) + len(commits)))
                flat_index = max(allowed, key=lambda value: means[value])
                choice = (
                    ("TEST", flat_index) if flat_index < len(probes)
                    else ("COMMIT", flat_index - len(probes))
                )
            kind, index = choice
            if kind == "COMMIT":
                committed = answer_space[index]
                actions.append(f"COMMIT:{committed}")
                break
            selected.add(index)
            probe = probes[index]
            actions.append(f"TEST:{probe.probe_id}")
            state = apply_probe_receipt(
                particles, state, probes, probe_receipts[probe.probe_id],
            )
        posterior = answer_belief(particles, state, answer_space=answer_space)
        rows[condition] = {
            "actions": actions,
            "selected_probe_ids": [
                action.split(":", 1)[1] for action in actions
                if action.startswith("TEST:")
            ],
            "test_count": sum(action.startswith("TEST:") for action in actions),
            "committed_answer": committed,
            "correct": committed == gold_answer,
            "net_utility": float(committed == gold_answer) - test_cost * sum(
                action.startswith("TEST:") for action in actions
            ),
            "gold_probability_after": (
                float(posterior[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            ),
        }

    oracle_rows = []
    for length in range(max_tests + 1):
        for trajectory in itertools.permutations(range(len(probes)), length):
            state = initial
            for index in trajectory:
                state = apply_probe_receipt(
                    particles, state, probes, probe_receipts[probes[index].probe_id],
                )
            posterior = answer_belief(particles, state, answer_space=answer_space)
            probability = (
                float(posterior[answer_space.index(gold_answer)])
                if gold_answer in answer_space else 0.0
            )
            oracle_rows.append((probability, trajectory, posterior))
    _, oracle_trajectory, oracle_belief = max(oracle_rows, key=lambda row: row[0])
    oracle_answer = answer_space[int(np.argmax(oracle_belief))]
    return {
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "answer_space": list(answer_space),
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_answer == gold_answer,
        "conditions": rows,
        "max_tests": max_tests,
        "test_cost": test_cost,
        "oracle_probe_ids": [probes[index].probe_id for index in oracle_trajectory],
        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_answer == gold_answer,
        "authentic_action_contrast": (
            rows["authentic_source_plus_target"]["actions"]
            != rows["target_native_exact_dp_test_commit"]["actions"]
        ),
    }


__all__ = [
    "FIXED_TEST_CONDITIONS",
    "ParsedTargetWorldModel",
    "evaluate_fixed_test_budget",
    "evaluate_fixed_one_test",
    "evaluate_adaptive_test_commit",
    "parse_target_world_model",
    "parse_typed_probe_receipt",
]
