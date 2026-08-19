import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _load_sources
from scripts.collect_agqa2_active_grounding_v3 import (
    _evaluation_protocol_core,
    _grounder_semantic_core,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads((ROOT / path).read_text())


def _verified(path, field):
    value = _read(path)
    body = dict(value)
    claimed = body.pop(field)
    assert stable_hash(body) == claimed
    return value


def test_v55_v56_summary_and_frozen_pools_are_consistent():
    summary = _verified(
        "docs/results/agqa2_asymmetric_support_v55_v56_summary.json",
        "summary_sha256",
    )
    qualification = _verified(
        "configs/agqa2_asymmetric_support_v55_qualification_selection.json",
        "manifest_sha256",
    )
    formal = _verified(
        "configs/agqa2_asymmetric_support_v56_formal_selection.json",
        "manifest_sha256",
    )
    assert summary["formal"]["confirmatory_claim"] is True
    assert summary["formal"]["wins"] == 14
    assert summary["formal"]["losses"] == 0
    assert summary["formal"]["net_gain"] == 14
    assert summary["controls"]["formal_effect_shuffled_abstentions"] == 300
    assert summary["controls"]["formal_wrong_source_abstentions"] == 300
    assert summary["controls"]["source_provenance_necessity_validated"] is False
    qualification_ids = {row["video_id"] for row in qualification["samples"]}
    formal_ids = {row["video_id"] for row in formal["samples"]}
    assert len(qualification_ids) == len(formal_ids) == 300
    assert qualification_ids.isdisjoint(formal_ids)


def test_v56_completion_changes_only_administrative_dependencies():
    original = _read("configs/agqa2_asymmetric_support_v56_formal.json")
    completion = _read(
        "configs/agqa2_asymmetric_support_v56_formal_completion_v2.json"
    )
    original_sources, _ = _load_sources(original)
    completion_sources, _ = _load_sources(completion)
    assert stable_hash(
        _grounder_semantic_core(original, original_sources)
    ) == stable_hash(_grounder_semantic_core(completion, completion_sources))
    assert stable_hash(_evaluation_protocol_core(original)) == stable_hash(
        _evaluation_protocol_core(completion)
    )
    assert original["postground"]["evaluation_protocol_sha256"] == completion[
        "postground"
    ]["evaluation_protocol_sha256"]
    assert original["postground"]["formal_gates"] == completion["postground"][
        "formal_gates"
    ]
    changed = {
        key for key in set(original) | set(completion)
        if original.get(key) != completion.get(key)
    }
    assert changed == {
        "development_qualification_report",
        "development_qualification_file_sha256",
        "preregistration",
        "preregistration_file_sha256",
    }


def test_v56_completion_audits_are_self_validating():
    dependency = _verified(
        "runs/agqa2_asymmetric_support_v56_formal/dependency_alias_completion.json",
        "audit_sha256",
    )
    evaluator = _verified(
        "runs/agqa2_asymmetric_support_v56_formal/evaluator_alias_completion.json",
        "audit_sha256",
    )
    assert dependency["runtime_receipt_count"] == 300
    assert evaluator["runtime_receipt_count"] == 300
    assert dependency["samples_prompts_models_predictions_gates_changed"] is False
    assert evaluator["samples_prompts_models_predictions_gates_changed"] is False
    assert dependency["original_grounder_sha256"] == dependency[
        "completion_grounder_sha256"
    ]
    assert evaluator["original_grounder_sha256"] == evaluator[
        "completion_grounder_sha256"
    ]
