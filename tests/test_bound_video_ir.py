from dataclasses import asdict

from motif_transfer.bound_video_ir import evaluate_bound_bind_relate_transfer
from motif_transfer.structured_video_transfer import parse_target_world_model, parse_typed_probe_receipt


def test_bound_edge_changes_receipt_kernel_and_obeys_guard():
    payload = {
        "entity_catalog": [
            {"entity_id": "E0", "visual_description": "red ball"},
            {"entity_id": "E1", "visual_description": "blue cube"},
            {"entity_id": "E2", "visual_description": "door"},
        ],
        "world_particles": [
            {"native_answer": "A", "prior_weight": .5, "event_graph_summary": "a"},
            {"native_answer": "B", "prior_weight": .5, "event_graph_summary": "b"},
        ],
        "typed_probes": [
            {"target_event_role": "BIND", "predicate_kind": "OBJECT_TRACK", "entity_ids": ["E0"], "window_fraction": [0, .4], "true_probability_by_particle": [.5, .5], "expected_sensor_reliability": .9, "rationale": "bind red"},
            {"target_event_role": "BIND", "predicate_kind": "OBJECT_ATTRIBUTE", "entity_ids": ["E1"], "window_fraction": [0, .4], "true_probability_by_particle": [.5, .5], "expected_sensor_reliability": .9, "rationale": "bind blue"},
            {"target_event_role": "RELATE", "predicate_kind": "EVENT_ORDER", "entity_ids": ["E0", "E2"], "window_fraction": [.2, .8], "true_probability_by_particle": [.9, .1], "expected_sensor_reliability": .9, "rationale": "red order"},
            {"target_event_role": "RELATE", "predicate_kind": "COLLISION", "entity_ids": ["E1", "E2"], "window_fraction": [.3, .9], "true_probability_by_particle": [.1, .9], "expected_sensor_reliability": .9, "rationale": "blue collision"},
            {"target_event_role": "RELATE", "predicate_kind": "OBJECT_MOTION", "entity_ids": ["E0"], "window_fraction": [.1, 1], "true_probability_by_particle": [.7, .3], "expected_sensor_reliability": .8, "rationale": "red motion"},
        ],
    }
    model = parse_target_world_model(
        payload, duration_seconds=3, particle_count=2, probe_count=5,
        valid_answers=("A", "B"), required_answers=("A", "B"),
        require_bind_relate_roles=True, minimum_bind_probes=2,
        minimum_relate_probes=3, minimum_relate_likelihood_span=.2,
    )

    def receipt(probe, observed):
        return parse_typed_probe_receipt(
            {"observed_true": observed, "sensor_reliability": 1.0},
            probe=probe, evidence_sha256=(f"{probe.probe_id}-{observed}",),
        )

    global_receipts = {
        probe.probe_id: receipt(probe, False) for probe in model.probes
    }
    tracks = {}
    matrix = {}
    for bind_index in (0, 1):
        bind_probe = model.probes[bind_index]
        bind_id = bind_probe.probe_id
        tracks[bind_id] = {
            "primary_entity_ref": bind_probe.entity_refs[0],
            "bind_receipt": asdict(receipt(bind_probe, True)),
        }
        matrix[bind_id] = {}
        for relate_index in (2, 3, 4):
            relate_probe = model.probes[relate_index]
            shared = bool(set(bind_probe.entity_refs) & set(relate_probe.entity_refs))
            matrix[bind_id][relate_probe.probe_id] = {
                "shared_primary_entity": shared,
                "receipt": asdict(receipt(relate_probe, shared)),
            }
    report = evaluate_bound_bind_relate_transfer(
        sample_id="bound", gold_answer="A", world_model=model,
        global_receipts=global_receipts,
        fork_receipt={"tracks": tracks, "bound_relation_receipts": matrix},
    )
    authentic = report["conditions"]["authentic_bound_bind_relate_ir"]
    assert authentic["actions"][0].startswith("BIND:")
    assert authentic["actions"][1].startswith("BOUND_RELATE:")
    assert report["authentic_guard_obeyed"]
    assert authentic["receipts_used"][1].startswith("bound:")
    assert "target_native_exact_dp" in report["conditions"]
    assert report["oracle_correct"] or not report["baseline_correct"]
