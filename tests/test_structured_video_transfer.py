import numpy as np
import pytest

from motif_transfer.structured_video_transfer import (
    evaluate_fixed_test_budget,
    evaluate_fixed_one_test,
    parse_target_world_model,
    parse_typed_probe_receipt,
)


class _Model:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, features):
        assert len(features) == len(self.values)
        return self.values, np.zeros_like(self.values)


def _payload():
    return {
        "world_particles": [
            {"native_answer": "A", "prior_weight": 0.35, "event_graph_summary": "x"},
            {"native_answer": "A", "prior_weight": 0.15, "event_graph_summary": "y"},
            {"native_answer": "B", "prior_weight": 0.30, "event_graph_summary": "z"},
            {"native_answer": "B", "prior_weight": 0.20, "event_graph_summary": "q"},
        ],
        "typed_probes": [
            {
                "predicate_kind": "COLLISION",
                "entity_refs": ["person", "door"],
                "window_fraction": [0.0, 0.33],
                "true_probability_by_particle": [0.9, 0.8, 0.1, 0.2],
                "expected_sensor_reliability": 0.9,
                "rationale": "distinguishes the answer worlds",
            },
            {
                "predicate_kind": "ENTRY",
                "entity_refs": ["person"],
                "window_fraction": [0.33, 0.67],
                "true_probability_by_particle": [0.5, 0.5, 0.5, 0.5],
                "expected_sensor_reliability": 0.8,
                "rationale": "checks a weak alternative",
            },
            {
                "predicate_kind": "EXIT",
                "entity_refs": ["person"],
                "window_fraction": [0.67, 1.0],
                "true_probability_by_particle": [0.2, 0.3, 0.7, 0.8],
                "expected_sensor_reliability": 0.8,
                "rationale": "checks the late event",
            },
        ],
    }


def test_world_model_and_probe_receipts_are_typed_and_bound():
    model = parse_target_world_model(
        _payload(), duration_seconds=3, particle_count=4, probe_count=3,
        valid_answers=("A", "B"),
    )
    receipt = parse_typed_probe_receipt(
        {"observed_true": True, "sensor_reliability": 0.9},
        probe=model.probes[0], evidence_sha256=("hash",),
    )
    assert receipt.probe_id == "P0"
    with pytest.raises(ValueError, match="boolean"):
        parse_typed_probe_receipt(
            {"observed_true": "yes", "sensor_reliability": 0.9},
            probe=model.probes[0], evidence_sha256=("hash",),
        )


def test_fixed_budget_conditions_update_world_not_reanswer():
    model = parse_target_world_model(
        _payload(), duration_seconds=3, particle_count=4, probe_count=3,
        valid_answers=("A", "B"),
    )
    receipts = {
        probe.probe_id: parse_typed_probe_receipt(
            {"observed_true": index == 0, "sensor_reliability": 1.0},
            probe=probe, evidence_sha256=(f"hash-{index}",),
        )
        for index, probe in enumerate(model.probes)
    }
    source = {
        "authentic_source_plus_target": _Model((3, 1, 2)),
        "shuffled_source_plus_target": _Model((1, 3, 2)),
        "source_marginal_plus_target": _Model((1, 2, 3)),
    }
    report = evaluate_fixed_one_test(
        sample_id="sample", gold_answer="A", world_model=model,
        probe_receipts=receipts, source_models=source,
    )
    assert report["baseline_answer"] == "A"
    assert report["conditions"]["authentic_source_plus_target"][
        "selected_probe_ids"
    ] == ["P0"]
    assert report["conditions"]["shuffled_source_plus_target"][
        "selected_probe_ids"
    ] == ["P1"]
    assert report["conditions"]["authentic_source_plus_target"][
        "gold_probability_after"
    ] > 0.8


def test_two_test_budget_replans_after_symbolic_transition():
    model = parse_target_world_model(
        _payload(), duration_seconds=3, particle_count=4, probe_count=3,
        valid_answers=("A", "B"),
    )
    receipts = {
        probe.probe_id: parse_typed_probe_receipt(
            {"observed_true": index == 0, "sensor_reliability": 0.9},
            probe=probe, evidence_sha256=(f"h-{index}",),
        )
        for index, probe in enumerate(model.probes)
    }
    source = {
        "authentic_source_plus_target": _Model((3, 1, 2)),
        "shuffled_source_plus_target": _Model((1, 3, 2)),
        "source_marginal_plus_target": _Model((1, 2, 3)),
    }
    report = evaluate_fixed_test_budget(
        sample_id="two", gold_answer="A", world_model=model,
        probe_receipts=receipts, source_models=source, test_budget=2,
    )
    for condition in report["conditions"].values():
        assert len(condition["selected_probe_ids"]) == 2
        assert len(set(condition["selected_probe_ids"])) == 2


def test_clevrer_binary_vector_answer_contract():
    payload = _payload()
    for index, particle in enumerate(payload["world_particles"]):
        particle["native_answer"] = "10" if index < 2 else "01"
    model = parse_target_world_model(
        payload, duration_seconds=3, particle_count=4, probe_count=3,
        binary_vector_length=2,
    )
    assert {particle.native_answer for particle in model.particles} == {"10", "01"}


def test_probe_windows_fail_closed_outside_observed_clip():
    payload = _payload()
    payload["typed_probes"][0]["window_fraction"] = [0.9, 1.2]
    with pytest.raises(ValueError, match="0 <= start < end <= 1"):
        parse_target_world_model(
            payload, duration_seconds=3, particle_count=4, probe_count=3,
            valid_answers=("A", "B"),
        )
