from motif_transfer.source_operator_capability_induction import induce_capabilities


def fixtures():
    artifact = {
        "artifact_sha256": "artifact", "target_data_read": False,
        "named_controller_template_used": False,
    }
    confirmation = {
        "status": "SOURCE_GOAL_ACQUISITION_FRESH_VALIDATED",
        "source_gate_passed": True,
        "claim_boundary": "FRESH_SOURCE_ONLY;NO_TARGET_EVIDENCE",
        "artifact_sha256": "artifact", "source_dataset_sha256": "dataset",
        "metrics": {"binding_onsets": 7, "binding_to_relation_transitions": 7,
                    "binding_to_relation_precision": 1.0, "shuffled_effect_bindings": 0},
        "gates": {name: True for name in (
            "heldout_transition_conformance", "authentic_beats_shuffled_effect_conformance",
            "binding_onset_predicts_relation", "authentic_effect_binding_exact",
            "shuffled_effect_binding_rejected", "single_positive_binding_cardinality_induced",
            "source_only_lineage", "no_named_controller_template",
        )},
    }
    manifest = {"claim_boundary": "SOURCE_ONLY;NO_TARGET_DATA"}
    wrapper = {"source_function_program": {
        "status": "SOURCE_DOMAIN_FUNCTION_QUALIFIED", "target_data_read": False,
        "source_identity_used_as_feature": False,
        "qualification_gates": {"a": True},
        "qualification_metrics": {"examples": 10, "correct": 8, "accuracy": .8},
        "qualification_shuffled_effect_metrics": {"correct": 1, "accuracy": .1},
        "program_sha256": "p",
        "shared_ir": {"effect_endpoints": {"near": 1, "far": 8}},
        "operators": [{"preconditions": {"observed": True, "unique": True}}],
        "transition_graph": {"transitions": [
            {"guard": "OBSERVED_EFFECT_HIGH"}, {"guard": "OBSERVED_EFFECT_LOW"},
            {"guard": "OBSERVED_EFFECT_UNKNOWN"},
        ]},
    }}
    return artifact, confirmation, manifest, [(wrapper, "file1"), (wrapper, "file2"), (wrapper, "file3")]


def test_induces_capabilities_without_target_input():
    artifact, confirmation, manifest, programs = fixtures()
    result = induce_capabilities(
        acquisition_artifact=artifact, acquisition_confirmation=confirmation,
        temporal_manifest=manifest, temporal_programs=programs,
    )
    assert result["status"] == "SOURCE_CAPABILITIES_INDUCED"
    assert result["target_data_read"] is False
    assert {"EXISTS", "FIRST", "LAST", "XOR", "ARGMAX"} <= set(result["authorized_operators"])
    assert "UNION" not in result["authorized_operators"]


def test_shuffled_control_failure_revokes_temporal_capabilities():
    artifact, confirmation, manifest, programs = fixtures()
    for wrapper, _ in programs:
        wrapper["source_function_program"]["qualification_shuffled_effect_metrics"]["accuracy"] = .9
    result = induce_capabilities(
        acquisition_artifact=artifact, acquisition_confirmation=confirmation,
        temporal_manifest=manifest, temporal_programs=programs,
    )
    assert result["status"] == "SOURCE_CAPABILITIES_ABSTAINED"
    assert "ARGMAX" not in result["authorized_operators"]
    assert "CARDINALITY" in result["authorized_operators"]


def test_confirmation_hash_mismatch_revokes_binding_capabilities():
    artifact, confirmation, manifest, programs = fixtures()
    confirmation["artifact_sha256"] = "wrong"
    result = induce_capabilities(
        acquisition_artifact=artifact, acquisition_confirmation=confirmation,
        temporal_manifest=manifest, temporal_programs=programs,
    )
    assert "CARDINALITY" not in result["authorized_operators"]
    assert "ARGMAX" in result["authorized_operators"]
