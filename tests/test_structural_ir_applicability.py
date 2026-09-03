from pathlib import Path
import json

from motif_transfer.contracts import stable_hash
from motif_transfer.structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
    TargetIRRequirement,
    contract_matches,
    goal_acquisition_artifact_contract,
    goal_relation_artifact_contract,
    relational_artifact_contract,
    select_source_contract,
    structural_program_contract,
    temporal_function_artifact_contract,
)


REPO = Path(__file__).resolve().parents[1]


def _signature(operation="ADD", family="ENTITY_SLOT"):
    return OperatorSignature(operation, family, 1, "ENTITY_REFERENCE")


def _source(*, sha="a" * 64, qualified=True, kind="FINITE"):
    return SourceIRContract.create(
        program_sha256=sha,
        ir_kind=kind,
        operator_sequence=(_signature(),),
        recurrent=False,
        terminal_predicate_families=(),
        source_intervention_qualified=qualified,
        source_confirmation_sha256="c" * 64,
    )


def _target(*, kind="FINITE", outcome=False, qualified=True):
    return TargetIRRequirement.create(
        task_id="target.1",
        target_domain="target",
        target_interface="native.v1",
        target_grounder_sha256="g" * 64,
        ir_kind=kind,
        operator_sequence=(_signature(),),
        recurrent=False,
        terminal_predicate_families=(),
        grounder_qualified=qualified,
        formal_outcome_read=outcome,
    )


def test_exact_anonymous_contract_selects_without_action_or_identity():
    receipt = select_source_contract((_source(),), _target())
    assert receipt["status"] == "UNIQUE_SOURCE_CONTRACT_SELECTED"
    assert receipt["selected_program_sha256"] == "a" * 64
    assert receipt["source_identity_used_as_feature"] is False
    assert receipt["target_outcome_read"] is False
    assert receipt["target_action_emitted"] is False


def test_kind_mismatch_abstains():
    receipt = select_source_contract((_source(),), _target(kind="OTHER"))
    assert receipt["status"] == "SOURCE_CONTRACT_SELECTION_ABSTAINED"
    assert receipt["source_contracts"][0]["reason"] == "IR_KIND_MISMATCH"


def test_unconfirmed_source_and_unqualified_grounder_fail_closed():
    assert contract_matches(_source(qualified=False), _target()) == (
        False, "SOURCE_PROGRAM_NOT_FRESH_CONFIRMED",
    )
    assert contract_matches(_source(), _target(qualified=False)) == (
        False, "TARGET_GROUNDER_NOT_QUALIFIED",
    )


def test_current_target_outcome_is_rejected_before_matching():
    assert contract_matches(_source(), _target(outcome=True)) == (
        False, "CURRENT_TARGET_OUTCOME_EXPOSED",
    )


def test_multiple_content_identical_programs_abstain():
    receipt = select_source_contract((
        _source(sha="a" * 64), _source(sha="b" * 64),
    ), _target())
    assert receipt["reason"] == "MULTIPLE_SOURCE_CONTRACTS_MATCH"
    assert receipt["selected_program_sha256"] is None


def test_real_minigrid_and_sokoban_contracts_are_structurally_distinct():
    minigrid = json.loads((
        REPO / "configs/source_structural_v5c_frozen/programs/put_near.json"
    ).read_text())
    sokoban = json.loads((
        REPO / "runs/sokoban_relational_structural_v2/artifact.json"
    ).read_text())
    left = structural_program_contract(
        minigrid, source_confirmation_sha256="m" * 64,
        source_intervention_qualified=True,
    )
    right = relational_artifact_contract(
        sokoban, source_confirmation_sha256="s" * 64,
        source_intervention_qualified=True,
    )
    assert left.ir_kind == "FINITE_STRUCTURAL_DELTA_SEQUENCE"
    assert [row.operation for row in left.operator_sequence] == ["ADD", "REMOVE"]
    assert right.ir_kind == "RECURRENT_RELATIONAL_TRANSITION_PROGRAM"
    assert right.recurrent is True
    assert right.terminal_predicate_families == ("ENTITY_GOAL_RELATION",)
    assert left.contract_sha256 != right.contract_sha256


def test_real_arcade_function_is_a_third_ir_kind():
    artifact = json.loads((
        REPO / "configs/phase3_source_function_v4/frozen_reserve/programs/"
        "candy_crush.json"
    ).read_text())
    contract = temporal_function_artifact_contract(
        artifact, source_confirmation_sha256="r" * 64,
        source_intervention_qualified=True,
    )
    assert contract.ir_kind == "SPARSE_TEMPORAL_EFFECT_FUNCTION"
    assert contract.operator_sequence[0].operation == "SCORE"
    assert contract.source_intervention_qualified is True


def test_goal_acquisition_contract_preserves_cardinality_and_relation_types():
    artifact = json.loads((
        REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"
    ).read_text())
    confirmation = json.loads((
        REPO / "runs/sokoban_goal_acquisition_v1/fresh_confirmation_report.json"
    ).read_text())
    contract = goal_acquisition_artifact_contract(
        artifact, confirmation=confirmation,
    )
    assert contract.ir_kind == "RECURRENT_GOAL_ACQUISITION_RELATION_PROGRAM"
    assert contract.recurrent is True
    assert [row.predicate_family for row in contract.operator_sequence] == [
        "ENTITY_RELATION",
        "CONTROL_STATE",
        "POSITIVE_EFFECT_BINDING",
        "ENTITY_GOAL_RELATION",
    ]
    assert contract.operator_sequence[2].value_kind == "CANDIDATE_CARDINALITY"
    assert contract.terminal_predicate_families == ("ENTITY_GOAL_RELATION",)
    assert contract.source_intervention_qualified is True


def test_goal_relation_contract_covers_video_extension_ir():
    artifact = json.loads((
        REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    ).read_text())
    confirmation = json.loads((
        REPO / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    ).read_text())
    contract = goal_relation_artifact_contract(
        artifact, confirmation=confirmation,
    )
    assert contract.ir_kind == "RECURRENT_GOAL_RELATION_PROGRAM"
    assert contract.recurrent is True
    assert [row.predicate_family for row in contract.operator_sequence] == [
        "ENTITY_GOAL_RELATION",
    ]
    assert contract.terminal_predicate_families == ("ENTITY_GOAL_RELATION",)
    assert contract.source_intervention_qualified is True


def test_contract_hash_is_content_sensitive_and_stable():
    assert _source().contract_sha256 == _source().contract_sha256
    assert _source().contract_sha256 == stable_hash({
        "program_sha256": "a" * 64,
        "ir_kind": "FINITE",
        "operator_sequence": [{
            "operation": "ADD", "predicate_family": "ENTITY_SLOT",
            "arity": 1, "value_kind": "ENTITY_REFERENCE",
        }],
        "recurrent": False,
        "terminal_predicate_families": [],
        "source_intervention_qualified": True,
        "source_confirmation_sha256": "c" * 64,
    })
