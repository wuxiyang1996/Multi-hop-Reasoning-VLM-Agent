import pytest

from motif_transfer.structured_video_transfer import parse_target_world_model, parse_typed_probe_receipt
from motif_transfer.video_target_residual import build_target_test_value_examples


def test_target_examples_exclude_gold_from_features_and_cover_replanning_states():
    payload = {
        "world_particles": [
            {"native_answer": "A", "prior_weight": 0.5, "event_graph_summary": "a"},
            {"native_answer": "B", "prior_weight": 0.5, "event_graph_summary": "b"},
        ],
        "typed_probes": [
            {"predicate_kind": "ENTRY", "entity_refs": ["x"], "window_fraction": [0, .3], "true_probability_by_particle": [.9, .1], "expected_sensor_reliability": .9, "rationale": "r0"},
            {"predicate_kind": "EXIT", "entity_refs": ["x"], "window_fraction": [.3, .6], "true_probability_by_particle": [.1, .9], "expected_sensor_reliability": .9, "rationale": "r1"},
            {"predicate_kind": "OBJECT_PRESENT", "entity_refs": ["x"], "window_fraction": [.6, 1], "true_probability_by_particle": [.5, .5], "expected_sensor_reliability": .9, "rationale": "r2"},
        ],
    }
    model = parse_target_world_model(
        payload, duration_seconds=3, particle_count=2, probe_count=3,
        valid_answers=("A", "B"), required_answers=("A", "B"),
    )
    receipts = {
        probe.probe_id: parse_typed_probe_receipt(
            {"observed_true": index == 0, "sensor_reliability": 1.0},
            probe=probe, evidence_sha256=(f"h{index}",),
        )
        for index, probe in enumerate(model.probes)
    }
    rows = build_target_test_value_examples(
        sample_id="held", gold_answer="A", world_model=model,
        probe_receipts=receipts, test_budget=2,
    )
    assert len(rows) == 3 + 3 * 2
    assert {len(row.features) for row in rows} == {9}
    assert max(row.value for row in rows if row.state_id.endswith("tested=")) > .8
    assert all("A" not in row.state_id and "B" not in row.state_id for row in rows)
    assert all(0 <= row.value <= 1 for row in rows)
