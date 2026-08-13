import numpy as np
import pytest

from motif_transfer.controlled_exploration_transfer import FEATURE_NAMES
from motif_transfer.video_dynamics_mdp import (
    PredicateProbe,
    PredicateProbeReceipt,
    WorldParticle,
    answer_belief,
    apply_probe_receipt,
    initial_state,
    probe_statistics,
    source_compatible_action_features,
)


def _particles():
    # Particles 0/1 are different event graphs that execute to the same
    # whole-question answer vector.  Particle 2 executes to another vector.
    return (
        WorldParticle("w0", "101", 0.25),
        WorldParticle("w1", "101", 0.25),
        WorldParticle("w2", "010", 0.50),
    )


def _probes():
    return (
        PredicateProbe(
            "collision:red-sphere:blue-cube",
            "COLLISION",
            ("red sphere", "blue cube"),
            1.0,
            2.0,
            (0.9, 0.8, 0.1),
            "detect_event",
        ),
        PredicateProbe(
            "entry:yellow-cylinder",
            "ENTRY",
            ("yellow cylinder",),
            2.0,
            3.0,
            (0.55, 0.55, 0.55),
            "detect_event",
        ),
    )


def test_world_particles_aggregate_to_native_whole_question_answers():
    particles = _particles()
    probes = _probes()
    state = initial_state(particles, probes, max_tests=2)
    assert np.allclose(answer_belief(particles, state), (0.5, 0.5))

    tests, commits, answers = source_compatible_action_features(
        particles, state, probes, max_tests=2,
    )
    assert answers == ("101", "010")
    assert len(tests) == 2
    assert len(commits) == 2
    assert all(len(row) == len(FEATURE_NAMES) for row in tests + commits)
    assert commits[0][7] == pytest.approx(0.5)
    assert commits[1][7] == pytest.approx(0.5)


def test_typed_collision_probe_has_answer_information_but_flat_probe_does_not():
    particles = _particles()
    probes = _probes()
    state = initial_state(particles, probes, max_tests=2)
    collision = probe_statistics(particles, state, probes[0])
    flat = probe_statistics(particles, state, probes[1])
    assert collision.expected_information_gain > 0.3
    assert collision.expected_map_confidence_gain > 0.25
    assert flat.expected_information_gain == pytest.approx(0.0, abs=1e-12)
    assert flat.expected_map_confidence_gain == pytest.approx(0.0, abs=1e-12)


def test_expected_sensor_reliability_is_part_of_probe_value():
    particles = _particles()
    probe = PredicateProbe(
        "unreliable-collision",
        "COLLISION",
        ("red sphere", "blue cube"),
        1.0,
        2.0,
        (0.9, 0.8, 0.1),
        "detect_event",
        expected_sensor_reliability=0.5,
    )
    state = initial_state(particles, (probe,), max_tests=1)
    statistics = probe_statistics(particles, state, probe)
    assert statistics.expected_information_gain == pytest.approx(0.0, abs=1e-12)
    assert statistics.expected_map_confidence_gain == pytest.approx(0.0, abs=1e-12)


def test_receipt_updates_world_then_answer_belief_and_budget():
    particles = _particles()
    probes = _probes()
    state = initial_state(particles, probes, max_tests=2)
    receipt = PredicateProbeReceipt(
        probe_id=probes[0].probe_id,
        predicate_kind="COLLISION",
        entity_refs=("red sphere", "blue cube"),
        start_sec=1.0,
        end_sec=2.0,
        observed_true=True,
        sensor_reliability=1.0,
        evidence_sha256=("frame-hash-1", "frame-hash-2"),
    )
    updated = apply_probe_receipt(particles, state, probes, receipt)
    assert updated.remaining_tests == 1
    assert updated.test_counts == (1, 0)
    assert answer_belief(particles, updated)[0] > 0.8

    tests, _, _ = source_compatible_action_features(
        particles, updated, probes, max_tests=2,
    )
    assert tests[0][8] == pytest.approx(0.5)
    assert tests[0][6] == pytest.approx(0.5)


def test_uncalibrated_coin_flip_sensor_cannot_change_belief():
    particles = _particles()
    probes = _probes()
    state = initial_state(particles, probes, max_tests=2)
    receipt = PredicateProbeReceipt(
        probe_id=probes[0].probe_id,
        predicate_kind="COLLISION",
        entity_refs=("red sphere", "blue cube"),
        start_sec=1.0,
        end_sec=2.0,
        observed_true=True,
        sensor_reliability=0.5,
        evidence_sha256=("frame-hash",),
    )
    updated = apply_probe_receipt(particles, state, probes, receipt)
    assert np.allclose(updated.world_weights, state.world_weights)


def test_receipt_cannot_be_rebound_to_a_different_typed_probe():
    particles = _particles()
    probes = _probes()
    state = initial_state(particles, probes, max_tests=2)
    mismatched = PredicateProbeReceipt(
        probe_id=probes[0].probe_id,
        predicate_kind="ENTRY",
        entity_refs=("yellow cylinder",),
        start_sec=2.0,
        end_sec=3.0,
        observed_true=True,
        sensor_reliability=0.9,
        evidence_sha256=("frame-hash",),
    )
    with pytest.raises(ValueError, match="does not match"):
        apply_probe_receipt(particles, state, probes, mismatched)
