from motif_transfer.structured_video_transfer import parse_target_world_model, parse_typed_probe_receipt
from motif_transfer.typed_video_ir import evaluate_typed_bind_relate_transfer, probes_share_bound_entity


def test_bind_relate_edge_uses_shared_target_native_entity_and_controls_differ():
    payload = {
        "world_particles": [
            {"native_answer": "A", "prior_weight": .5, "event_graph_summary": "a"},
            {"native_answer": "B", "prior_weight": .5, "event_graph_summary": "b"},
        ],
        "typed_probes": [
            {"predicate_kind": "OBJECT_TRACK", "entity_refs": ["red ball"], "window_fraction": [0, .3], "true_probability_by_particle": [.9, .1], "expected_sensor_reliability": .9, "rationale": "bind red"},
            {"predicate_kind": "EVENT_ORDER", "entity_refs": ["red ball", "door"], "window_fraction": [.3, .7], "true_probability_by_particle": [.9, .1], "expected_sensor_reliability": .9, "rationale": "relate red"},
            {"predicate_kind": "OBJECT_ATTRIBUTE", "entity_refs": ["blue cube"], "window_fraction": [0, .4], "true_probability_by_particle": [.5, .5], "expected_sensor_reliability": .9, "rationale": "bind blue"},
            {"predicate_kind": "COLLISION", "entity_refs": ["blue cube", "wall"], "window_fraction": [.4, 1], "true_probability_by_particle": [.5, .5], "expected_sensor_reliability": .9, "rationale": "relate blue"},
        ],
    }
    model = parse_target_world_model(
        payload, duration_seconds=3, particle_count=2, probe_count=4,
        valid_answers=("A", "B"), required_answers=("A", "B"),
    )
    assert probes_share_bound_entity(model.probes[0], model.probes[1])
    assert not probes_share_bound_entity(model.probes[0], model.probes[3])
    receipts = {
        probe.probe_id: parse_typed_probe_receipt(
            {"observed_true": index < 2, "sensor_reliability": 1.0},
            probe=probe, evidence_sha256=(str(index),),
        )
        for index, probe in enumerate(model.probes)
    }
    result = evaluate_typed_bind_relate_transfer(
        sample_id="typed", gold_answer="A", world_model=model,
        probe_receipts=receipts,
    )
    authentic = result["conditions"]["authentic_bind_relate_ir"]
    reversed_edge = result["conditions"]["reversed_relate_bind_ir"]
    assert authentic["predicate_kinds"] == ["OBJECT_TRACK", "EVENT_ORDER"]
    assert authentic["shared_entity_guard"]
    assert reversed_edge["predicate_kinds"] == ["EVENT_ORDER", "OBJECT_TRACK"]
