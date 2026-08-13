"""Target-native video dynamics belief MDP.

The transferable controller is allowed to see only the same nine anonymous
TEST/COMMIT features used by the controlled active-identification source.  It
must not treat a temporal crop as a new answer.  A video-native model first
represents uncertainty with world/dynamics particles, compiles a proposed
visual intervention into a typed predicate probe, and estimates how likely
each probe outcome is under every particle.  The symbolic transition then
updates the particle posterior and derives the native answer belief.

This module deliberately does not parse pixels, questions, or CLEVRER oracle
programs.  Those are responsibilities of target-native neural grounders and a
target-native dynamics/program executor.  Keeping that boundary explicit
prevents an answer re-prompt from being mislabeled as a neural-symbolic state
transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .controlled_exploration_transfer import FEATURE_NAMES


PREDICATE_KINDS = frozenset({
    "OBJECT_PRESENT",
    "OBJECT_ATTRIBUTE",
    "OBJECT_MOTION",
    "OBJECT_TRACK",
    "COLLISION",
    "ENTRY",
    "EXIT",
    "EVENT_ORDER",
    "CAUSAL_ANCESTOR",
})


def _normalize(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not len(vector):
        raise ValueError("belief must be a non-empty vector")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError("belief values must be finite and nonnegative")
    total = float(np.sum(vector))
    if total <= 0:
        raise ValueError("belief must have positive mass")
    return vector / total


def _entropy(values: Sequence[float]) -> float:
    vector = np.clip(_normalize(values), 1e-12, 1.0)
    return float(-np.sum(vector * np.log(vector)))


@dataclass(frozen=True)
class WorldParticle:
    """One target-native trajectory/event-graph/dynamics hypothesis.

    ``native_answer`` may be a normal MCQ slot (for example ``"C"``) or a
    complete CLEVRER choice-label vector (for example ``"1010"``).  Multiple
    world particles may execute to the same answer.  This is the crucial
    difference from TIR, where an answer slot itself is an adequate hidden
    hypothesis.
    """

    particle_id: str
    native_answer: str
    prior_weight: float

    def validate(self) -> None:
        if not self.particle_id or not self.native_answer:
            raise ValueError("particle_id and native_answer must be non-empty")
        if not math.isfinite(self.prior_weight) or self.prior_weight <= 0:
            raise ValueError("particle prior_weight must be positive and finite")


@dataclass(frozen=True)
class PredicateProbe:
    """A typed sensing action and its target-native observation model.

    ``latent_true_probability_by_particle[i]`` is the dynamics model's
    probability that the typed predicate is true under particle ``i``.  It is
    not a probability of the final answer.  Entity descriptions, timestamps,
    and tool names stay on the target side and are never source-model inputs.
    """

    probe_id: str
    predicate_kind: str
    entity_refs: tuple[str, ...]
    start_sec: float
    end_sec: float
    latent_true_probability_by_particle: tuple[float, ...]
    target_tool: str
    expected_sensor_reliability: float = 1.0

    def validate(self, particle_count: int) -> None:
        if not self.probe_id or not self.target_tool:
            raise ValueError("probe_id and target_tool must be non-empty")
        if self.predicate_kind not in PREDICATE_KINDS:
            raise ValueError(f"unknown predicate kind: {self.predicate_kind}")
        if not self.entity_refs or any(not value for value in self.entity_refs):
            raise ValueError("a probe requires non-empty target entity references")
        if not (
            math.isfinite(self.start_sec)
            and math.isfinite(self.end_sec)
            and 0 <= self.start_sec < self.end_sec
        ):
            raise ValueError("probe window must be finite, nonnegative, and non-empty")
        likelihood = np.asarray(
            self.latent_true_probability_by_particle, dtype=np.float64,
        )
        if likelihood.shape != (particle_count,):
            raise ValueError("probe likelihood length must match world particles")
        if not np.all(np.isfinite(likelihood)) or np.any(
            (likelihood < 0) | (likelihood > 1)
        ):
            raise ValueError("latent predicate likelihoods must be in [0, 1]")
        if not 0.5 <= self.expected_sensor_reliability <= 1.0:
            raise ValueError("expected sensor reliability must be in [0.5, 1]")


@dataclass(frozen=True)
class PredicateProbeReceipt:
    """Typed evidence returned by a target-native neural video grounder."""

    probe_id: str
    predicate_kind: str
    entity_refs: tuple[str, ...]
    start_sec: float
    end_sec: float
    observed_true: bool
    sensor_reliability: float
    evidence_sha256: tuple[str, ...]

    def validate_against(self, probe: PredicateProbe) -> None:
        if (
            self.probe_id != probe.probe_id
            or self.predicate_kind != probe.predicate_kind
            or self.entity_refs != probe.entity_refs
            or not math.isclose(self.start_sec, probe.start_sec, abs_tol=1e-6)
            or not math.isclose(self.end_sec, probe.end_sec, abs_tol=1e-6)
        ):
            raise ValueError("typed receipt does not match the selected probe")
        if not 0.5 <= self.sensor_reliability <= 1.0:
            raise ValueError("sensor reliability must be calibrated in [0.5, 1]")
        if not self.evidence_sha256 or any(not value for value in self.evidence_sha256):
            raise ValueError("receipt requires immutable evidence hashes")


@dataclass(frozen=True)
class VideoDynamicsBeliefState:
    """Posterior over native world hypotheses plus the intervention budget."""

    world_weights: tuple[float, ...]
    remaining_tests: int
    test_counts: tuple[int, ...]

    def validate(self, *, particle_count: int, probe_count: int) -> None:
        if len(self.world_weights) != particle_count:
            raise ValueError("world belief length must match particles")
        normalized = _normalize(self.world_weights)
        if not np.allclose(normalized, self.world_weights, atol=1e-8):
            raise ValueError("world belief must already be normalized")
        if self.remaining_tests < 0:
            raise ValueError("remaining_tests cannot be negative")
        if len(self.test_counts) != probe_count or any(
            value < 0 for value in self.test_counts
        ):
            raise ValueError("test_counts must align with probes and be nonnegative")


@dataclass(frozen=True)
class ProbeStatistics:
    expected_information_gain: float
    expected_map_confidence_gain: float
    predicted_outcome_balance: float
    probability_true: float


def native_answer_space(
    particles: Sequence[WorldParticle],
) -> tuple[str, ...]:
    """Return target answer tokens in stable first-occurrence order."""

    output: list[str] = []
    for particle in particles:
        particle.validate()
        if particle.native_answer not in output:
            output.append(particle.native_answer)
    if len(output) < 2:
        raise ValueError("world particles must support at least two native answers")
    if len({particle.particle_id for particle in particles}) != len(particles):
        raise ValueError("particle IDs must be unique")
    return tuple(output)


def initial_state(
    particles: Sequence[WorldParticle],
    probes: Sequence[PredicateProbe],
    *,
    max_tests: int,
) -> VideoDynamicsBeliefState:
    if max_tests <= 0:
        raise ValueError("max_tests must be positive")
    native_answer_space(particles)
    for probe in probes:
        probe.validate(len(particles))
    if len({probe.probe_id for probe in probes}) != len(probes):
        raise ValueError("probe IDs must be unique")
    state = VideoDynamicsBeliefState(
        world_weights=tuple(map(float, _normalize([
            particle.prior_weight for particle in particles
        ]))),
        remaining_tests=max_tests,
        test_counts=(0,) * len(probes),
    )
    state.validate(particle_count=len(particles), probe_count=len(probes))
    return state


def answer_belief(
    particles: Sequence[WorldParticle],
    state: VideoDynamicsBeliefState,
    *,
    answer_space: Sequence[str] | None = None,
) -> np.ndarray:
    space = tuple(answer_space or native_answer_space(particles))
    if len(set(space)) != len(space) or set(space) != {
        particle.native_answer for particle in particles
    }:
        raise ValueError("answer_space must exactly cover native particle answers")
    state.validate(particle_count=len(particles), probe_count=len(state.test_counts))
    indices = {answer: index for index, answer in enumerate(space)}
    output = np.zeros(len(space), dtype=np.float64)
    for particle, weight in zip(particles, state.world_weights):
        output[indices[particle.native_answer]] += weight
    return _normalize(output)


def _posterior(
    world_weights: Sequence[float],
    probe: PredicateProbe,
    *,
    observed_true: bool,
    sensor_reliability: float = 1.0,
) -> np.ndarray:
    if not 0.5 <= sensor_reliability <= 1.0:
        raise ValueError("sensor reliability must be in [0.5, 1]")
    latent = np.asarray(
        probe.latent_true_probability_by_particle, dtype=np.float64,
    )
    # A reliability of .5 makes the sensor uninformative; 1.0 preserves the
    # target dynamics model's native predicate likelihood.
    measured_true = (
        sensor_reliability * latent
        + (1.0 - sensor_reliability) * (1.0 - latent)
    )
    factors = measured_true if observed_true else 1.0 - measured_true
    return _normalize(_normalize(world_weights) * factors)


def probe_statistics(
    particles: Sequence[WorldParticle],
    state: VideoDynamicsBeliefState,
    probe: PredicateProbe,
    *,
    answer_space: Sequence[str] | None = None,
) -> ProbeStatistics:
    """Compute answer-level value-of-information from world-level dynamics."""

    space = tuple(answer_space or native_answer_space(particles))
    current = answer_belief(particles, state, answer_space=space)
    latent = np.asarray(probe.latent_true_probability_by_particle, dtype=np.float64)
    reliability = probe.expected_sensor_reliability
    measured_true = reliability * latent + (1.0 - reliability) * (1.0 - latent)
    probability_true = float(np.dot(state.world_weights, measured_true))
    expected_entropy = 0.0
    expected_confidence = 0.0
    for observed_true, probability in (
        (True, probability_true), (False, 1.0 - probability_true),
    ):
        if probability <= 1e-12:
            continue
        posterior_state = VideoDynamicsBeliefState(
            tuple(map(float, _posterior(
                state.world_weights,
                probe,
                observed_true=observed_true,
                sensor_reliability=reliability,
            ))),
            state.remaining_tests,
            state.test_counts,
        )
        posterior_answer = answer_belief(
            particles, posterior_state, answer_space=space,
        )
        expected_entropy += probability * _entropy(posterior_answer)
        expected_confidence += probability * float(np.max(posterior_answer))
    entropy_scale = math.log(len(space))
    return ProbeStatistics(
        expected_information_gain=max(
            0.0, (_entropy(current) - expected_entropy) / entropy_scale,
        ),
        expected_map_confidence_gain=(
            expected_confidence - float(np.max(current))
        ),
        predicted_outcome_balance=1.0 - 2.0 * abs(probability_true - 0.5),
        probability_true=probability_true,
    )


def source_compatible_action_features(
    particles: Sequence[WorldParticle],
    state: VideoDynamicsBeliefState,
    probes: Sequence[PredicateProbe],
    *,
    max_tests: int,
    answer_space: Sequence[str] | None = None,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[str, ...],
]:
    """Compile target dynamics uncertainty into the frozen source interface."""

    if max_tests <= 0 or state.remaining_tests > max_tests:
        raise ValueError("invalid max_tests for current state")
    state.validate(particle_count=len(particles), probe_count=len(probes))
    space = tuple(answer_space or native_answer_space(particles))
    belief = answer_belief(particles, state, answer_space=space)
    entropy = _entropy(belief) / math.log(len(space))
    current_map = float(np.max(belief))
    remaining_fraction = state.remaining_tests / max_tests
    tests = []
    for index, probe in enumerate(probes):
        probe.validate(len(particles))
        statistics = probe_statistics(
            particles, state, probe, answer_space=space,
        )
        tests.append((
            1.0,
            statistics.expected_information_gain,
            statistics.expected_map_confidence_gain,
            statistics.predicted_outcome_balance,
            current_map,
            entropy,
            remaining_fraction,
            0.0,
            state.test_counts[index] / max_tests,
        ))
    commits = tuple((
        0.0, 0.0, 0.0, 0.0, current_map, entropy, remaining_fraction,
        float(probability), 0.0,
    ) for probability in belief)
    output_tests = tuple(tests)
    if any(len(row) != len(FEATURE_NAMES) for row in output_tests + commits):
        raise AssertionError("source feature contract drift")
    return output_tests, commits, space


def apply_probe_receipt(
    particles: Sequence[WorldParticle],
    state: VideoDynamicsBeliefState,
    probes: Sequence[PredicateProbe],
    receipt: PredicateProbeReceipt,
) -> VideoDynamicsBeliefState:
    """Apply one real typed observation and consume one TEST transition."""

    if state.remaining_tests <= 0:
        raise ValueError("cannot TEST after the intervention budget is exhausted")
    matches = [
        (index, probe) for index, probe in enumerate(probes)
        if probe.probe_id == receipt.probe_id
    ]
    if len(matches) != 1:
        raise ValueError("receipt probe_id must identify exactly one probe")
    index, probe = matches[0]
    receipt.validate_against(probe)
    posterior = _posterior(
        state.world_weights,
        probe,
        observed_true=receipt.observed_true,
        sensor_reliability=receipt.sensor_reliability,
    )
    counts = list(state.test_counts)
    counts[index] += 1
    output = VideoDynamicsBeliefState(
        tuple(map(float, posterior)),
        state.remaining_tests - 1,
        tuple(counts),
    )
    output.validate(particle_count=len(particles), probe_count=len(probes))
    return output


__all__ = [
    "PREDICATE_KINDS",
    "PredicateProbe",
    "PredicateProbeReceipt",
    "ProbeStatistics",
    "VideoDynamicsBeliefState",
    "WorldParticle",
    "answer_belief",
    "apply_probe_receipt",
    "initial_state",
    "native_answer_space",
    "probe_statistics",
    "source_compatible_action_features",
]
