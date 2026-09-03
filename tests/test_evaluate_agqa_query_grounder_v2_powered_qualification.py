from scripts.evaluate_agqa_query_grounder_v2_powered_qualification import (
    _qualification_gates, _validate_preoutcome_amendment,
)
import hashlib
import json
from pathlib import Path

import pytest


REQUIRED = {
    "provider_and_contract_success_fraction_minimum": 0.95,
    "entity_candidate_pool_recall_minimum": 0.75,
    "typed_role_binding_fidelity_minimum": 0.8,
    "cross_frame_dedup_fidelity_minimum": 0.9,
    "unique_supported_count_minimum": 128,
    "unique_supported_precision_wilson_95_lower_bound_minimum": 0.6,
    "unique_supported_coverage_fraction_minimum": 0.2,
    "residual_duplicate_typed_event_key_count_maximum": 0,
    "authority_safe_receipt_fraction": 1.0,
}


def _metrics(**updates):
    values = {
        "provider_and_contract_success_fraction": 1.0,
        "entity_candidate_pool_recall": 0.9,
        "typed_role_binding_fidelity": 1.0,
        "cross_frame_dedup_fidelity": 1.0,
        "unique_supported_count": 128,
        "unique_supported_precision_wilson_95_lower": 0.61,
        "unique_supported_coverage_fraction": 0.2,
        "residual_duplicate_typed_event_key_count": 0,
        "authority_safe_receipt_fraction": 1.0,
        "expected_query_tasks": 640,
        "fixed_threshold_verified": True,
        "grounder_contract_verified": True,
    }
    values.update(updates)
    return values


def test_all_frozen_powered_qualification_gates_pass_at_boundaries():
    gates = _qualification_gates(_metrics(), REQUIRED, rows=640)
    assert gates and all(gates.values())


def test_wilson_or_coverage_failure_cannot_be_hidden_by_other_metrics():
    gates = _qualification_gates(
        _metrics(
            unique_supported_precision_wilson_95_lower=0.599,
            unique_supported_coverage_fraction=0.199,
        ),
        REQUIRED,
        rows=640,
    )
    assert not gates[
        "unique_supported_precision_wilson_95_lower_bound_minimum"
    ]
    assert not gates["unique_supported_coverage_fraction_minimum"]


def test_preoutcome_amendment_is_hash_bound_and_outcome_blind(tmp_path: Path):
    protocol_path = tmp_path / "protocol.json"
    component_path = tmp_path / "compiler.py"
    protocol_path.write_text("{}")
    component_path.write_text("fixed")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {"frozen_grounder": {"component_sha256s": {
        "query_grounder_compiler": "a" * 64,
    }}}
    amendment = {
        "status": "FROZEN_AFTER_OUTCOME_BLIND_INFRASTRUCTURE_FAILURE_BEFORE_QUALIFICATION_OUTCOMES",
        "protocol_file": "protocol.json",
        "protocol_file_sha256": digest(protocol_path),
        "component_path": "compiler.py",
        "original_component_sha256": "a" * 64,
        "amended_component_sha256": digest(component_path),
    }
    for key in (
        "qualification_outcomes_opened_before_amendment",
        "authoritative_grounding_output_existed_before_amendment",
        "selection_changed", "videos_or_tasks_changed", "model_or_checkpoint_changed",
        "frame_budget_changed", "ontology_changed", "candidate_ranking_changed",
        "support_threshold_changed", "target_answer_read", "official_scene_graph_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
    ):
        amendment[key] = False
    _validate_preoutcome_amendment(amendment, protocol, root=tmp_path)
    amendment["support_threshold_changed"] = True
    with pytest.raises(ValueError, match="repair boundary"):
        _validate_preoutcome_amendment(amendment, protocol, root=tmp_path)
