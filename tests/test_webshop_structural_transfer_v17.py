from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motif_transfer.contracts import stable_hash
from motif_transfer.webshop_structural_transfer_v17 import (
    GENERIC_SCAFFOLD,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    TARGET_NATIVE_CEILING,
    WebShopStructuralController,
    induce_webshop_relational_function,
    permute_target_terminal,
    structural_compatibility_receipt,
)


REPO = Path(__file__).resolve().parents[1]


def _receipt(*, success: bool, relation: bool, changes: int = 2) -> dict:
    required = ["size:large"] if relation else []
    steps = [
        {
            "before_hash": f"b{index}",
            "after_hash": f"a{index}",
            "terminated": False,
        }
        for index in range(changes)
    ]
    body = {
        "strict_success": success,
        "failure": None,
        "steps": steps,
        "commit_audit": [{
            "authorized": relation,
            "coverage": {"required": required},
        }],
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _source_pair() -> tuple[dict, dict]:
    source = json.loads((
        REPO / "runs/sokoban_relational_structural_v2/artifact.json"
    ).read_text())
    confirmation = json.loads((
        REPO / "runs/sokoban_relational_structural_v2/"
        "fresh_confirmation_report.json"
    ).read_text())
    return source, confirmation


def test_target_induction_is_development_only_and_family_selective() -> None:
    rows = [
        _receipt(success=True, relation=True),
        _receipt(success=True, relation=True, changes=3),
        _receipt(success=True, relation=False),
        _receipt(success=False, relation=True),
    ]
    program = induce_webshop_relational_function(
        rows,
        development_receipts_sha256=stable_hash(
            [row["receipt_sha256"] for row in rows]
        ),
        target_grounder_sha256="grounder",
    )
    assert program["status"] == "TARGET_RELATIONAL_FUNCTION_QUALIFIED"
    assert program["qualification_metrics"]["relation_family_successful_receipts"] == 2
    assert program["qualification_metrics"]["out_of_family_successful_receipts"] == 1
    assert program["source_program_read_during_induction"] is False
    assert program["formal_target_data_read"] is False


def test_structural_content_not_identity_controls_admission() -> None:
    source, confirmation = _source_pair()
    rows = [_receipt(success=True, relation=True) for _ in range(2)]
    target = induce_webshop_relational_function(
        rows, development_receipts_sha256="development",
        target_grounder_sha256="grounder",
    )
    authentic = structural_compatibility_receipt(source, confirmation, target)
    permuted = structural_compatibility_receipt(
        source, confirmation, permute_target_terminal(target),
    )
    assert authentic["status"] == "STRUCTURAL_TRANSFER_ADMITTED"
    assert permuted["status"] == "STRUCTURAL_TRANSFER_ABSTAINED"
    assert authentic["source_identity_used_as_feature"] is False
    assert authentic["target_outcome_read_at_routing"] is False


def test_controller_fails_closed_by_condition_and_episode_schema() -> None:
    source, confirmation = _source_pair()
    rows = [_receipt(success=True, relation=True) for _ in range(2)]
    target = induce_webshop_relational_function(
        rows, development_receipts_sha256="development",
        target_grounder_sha256="grounder",
    )
    kwargs = dict(
        source=source, source_confirmation=confirmation,
        target_function=target, goal_options={"size": "large"},
    )
    authentic = WebShopStructuralController(condition=SOURCE_INDUCED, **kwargs)
    permuted = WebShopStructuralController(condition=SOURCE_PERMUTED, **kwargs)
    generic = WebShopStructuralController(condition=GENERIC_SCAFFOLD, **kwargs)
    ceiling = WebShopStructuralController(condition=TARGET_NATIVE_CEILING, **kwargs)
    out_of_family = WebShopStructuralController(
        condition=SOURCE_INDUCED, source=source,
        source_confirmation=confirmation, target_function=target,
        goal_options={},
    )
    assert authentic.source_admitted and authentic.target_function_enabled
    assert not permuted.source_admitted and not permuted.target_function_enabled
    assert not generic.source_admitted and not generic.target_function_enabled
    assert not ceiling.source_admitted and ceiling.target_function_enabled
    assert not out_of_family.source_admitted
    assert not out_of_family.target_function_enabled


def test_induced_terminal_binds_unique_native_commit() -> None:
    source, confirmation = _source_pair()
    rows = [_receipt(success=True, relation=True) for _ in range(2)]
    target = induce_webshop_relational_function(
        rows, development_receipts_sha256="development",
        target_grounder_sha256="grounder",
    )
    controller = WebShopStructuralController(
        condition=SOURCE_INDUCED, source=source,
        source_confirmation=confirmation, target_function=target,
        goal_options={"size": "large"},
    )
    controller.controller.ledger.required.add("size:large")
    controller.controller.ledger.verified.add("size:large")
    decision = controller(
        condition=SOURCE_INDUCED,
        predictions=np.asarray([
            [0.9, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0],
        ]),
        semantics=[
            {"is_commit": False, "is_noop": False, "verb": "click"},
            {"is_commit": True, "is_noop": False, "verb": "click"},
        ],
        source_models={}, visible_satisfied=True, visible_unsatisfied=False,
        prior_no_effect=False, remaining_fraction=0.5,
        previous_action=None, candidates=["click('tab')", "click('buy')"],
        uncertainty_scale=0.0, decision_margin=0.0,
    )
    assert decision.selected_index == 1
    assert decision.abstract_kind == "STRUCTURAL_IR"
    assert "unique_commit_after_relation_coverage" in decision.reason
    assert controller.terminal_bindings == 1


def test_budget_infeasible_fallback_is_not_source_authorized() -> None:
    source, confirmation = _source_pair()
    rows = [_receipt(success=True, relation=True) for _ in range(2)]
    target = induce_webshop_relational_function(
        rows, development_receipts_sha256="development",
        target_grounder_sha256="grounder",
    )
    controller = WebShopStructuralController(
        condition=SOURCE_INDUCED, source=source,
        source_confirmation=confirmation, target_function=target,
        goal_options={"size": "large"}, maximum_steps=12,
    )
    controller.controller.ledger.required.add("size:large")
    decision = controller(
        condition=SOURCE_INDUCED,
        predictions=np.asarray([
            [0.9, 0.9, 0.9, 0.0],
            [0.9, 0.0, 0.0, 0.0],
        ]),
        semantics=[
            {"is_commit": True, "is_noop": False, "verb": "click"},
            {"is_commit": False, "is_noop": False, "verb": "click"},
        ],
        source_models={}, visible_satisfied=False, visible_unsatisfied=False,
        prior_no_effect=False, remaining_fraction=1 / 12,
        previous_action=None, candidates=["click('buy')", "click('tab')"],
        uncertainty_scale=0.0, decision_margin=0.0,
    )
    assert decision.selected_index == 0
    assert decision.source_abstained
    assert controller.source_authorized_decisions == 0


def test_successful_relational_update_may_repeat_same_native_binding() -> None:
    source, confirmation = _source_pair()
    rows = [_receipt(success=True, relation=True) for _ in range(2)]
    target = induce_webshop_relational_function(
        rows, development_receipts_sha256="development",
        target_grounder_sha256="grounder",
    )
    controller = WebShopStructuralController(
        condition=SOURCE_INDUCED, source=source,
        source_confirmation=confirmation, target_function=target,
        goal_options={"size": "large"},
    )
    kwargs = dict(
        condition=SOURCE_INDUCED,
        predictions=np.zeros((3, 4)),
        semantics=[
            {"verb": "fill", "is_commit": False, "is_noop": False},
            {"verb": "fill", "is_commit": False, "is_noop": False},
            {"verb": "noop", "is_commit": False, "is_noop": True},
        ],
        source_models={}, visible_satisfied=False, visible_unsatisfied=False,
        prior_no_effect=False, remaining_fraction=1.0,
        previous_action=None,
        candidates=["fill('q','one')", "fill('q','two')", "noop()"],
        uncertainty_scale=0.0, decision_margin=0.0,
    )
    first = controller(**kwargs)
    second = controller(**kwargs)
    assert first.selected_index == 0
    assert second.selected_index == 0
    assert controller.source_authorized_decisions == 2


def test_no_effect_refutes_native_binding_and_retries_without_outcome() -> None:
    source, confirmation = _source_pair()
    rows = [_receipt(success=True, relation=True) for _ in range(2)]
    target = induce_webshop_relational_function(
        rows, development_receipts_sha256="development",
        target_grounder_sha256="grounder",
    )
    controller = WebShopStructuralController(
        condition=SOURCE_INDUCED, source=source,
        source_confirmation=confirmation, target_function=target,
        goal_options={"color": "blue"},
    )
    candidates = ("click('10')", "click('11')", "noop()")
    semantics = (
        {"verb": "click", "url_phase": "product", "element_role": "button",
         "element_text": "[10] button 'details'", "is_commit": False,
         "is_constraint": False, "is_noop": False},
        {"verb": "click", "url_phase": "product", "element_role": "button",
         "element_text": "[11] button 'more'", "is_commit": False,
         "is_constraint": False, "is_noop": False},
        {"verb": "noop", "url_phase": "product", "element_role": "other",
         "element_text": "", "is_commit": False, "is_constraint": False,
         "is_noop": True},
    )
    kwargs = dict(
        condition=SOURCE_INDUCED,
        predictions=np.zeros((3, 4)), semantics=semantics,
        source_models={}, visible_satisfied=False, visible_unsatisfied=False,
        remaining_fraction=1.0, candidates=candidates,
        uncertainty_scale=0.0, decision_margin=0.0,
    )
    first = controller(
        **kwargs, prior_no_effect=False, previous_action=None,
    )
    second = controller(
        **kwargs, prior_no_effect=True,
        previous_action=candidates[first.selected_index],
    )
    assert first.selected_index == 0
    assert second.selected_index == 1
    assert second.reason.startswith("source_induced_relational_function:")
    assert controller.effect_refutation_retries == 1
    assert len(controller.refuted_binding_signatures) == 1


def test_missing_exact_target_relation_binding_preserves_source_abstention() -> None:
    source, confirmation = _source_pair()
    rows = [_receipt(success=True, relation=True) for _ in range(2)]
    target = induce_webshop_relational_function(
        rows, development_receipts_sha256="development",
        target_grounder_sha256="grounder",
    )
    controller = WebShopStructuralController(
        condition=SOURCE_INDUCED, source=source,
        source_confirmation=confirmation, target_function=target,
        goal_options={"size": "15.7 x 1.25"},
    )
    decision = controller(
        condition=SOURCE_INDUCED,
        predictions=np.zeros((2, 4)),
        semantics=[
            {"verb": "click", "url_phase": "product", "element_role": "radio",
             "element_text": "[28] radio \"15.7'' x 1.25''\", checked='false'",
             "is_goal_constraint": True, "is_constraint": True,
             "is_selected": False, "is_commit": False, "is_noop": False},
            {"verb": "click", "url_phase": "product", "element_role": "button",
             "element_text": "[55] button 'Buy Now'", "is_goal_constraint": False,
             "is_constraint": False, "is_selected": False,
             "is_commit": True, "is_noop": False},
        ],
        source_models={}, visible_satisfied=False, visible_unsatisfied=True,
        prior_no_effect=False, remaining_fraction=1.0,
        previous_action=None, candidates=["click('28')", "click('55')"],
        uncertainty_scale=0.0, decision_margin=0.0,
    )
    assert decision.selected_index == 0
    assert decision.source_abstained
    assert controller.source_authorized_decisions == 0
    followup = controller(
        condition=SOURCE_INDUCED,
        predictions=np.zeros((2, 4)),
        semantics=[
            {"verb": "click", "url_phase": "product", "element_role": "button",
             "element_text": "[55] button 'Buy Now'", "is_goal_constraint": False,
             "is_constraint": False, "is_selected": False,
             "is_commit": True, "is_noop": False},
            {"verb": "go_back", "url_phase": "product", "element_role": "other",
             "element_text": "", "is_goal_constraint": False,
             "is_constraint": False, "is_selected": False,
             "is_commit": False, "is_noop": False},
        ],
        source_models={}, visible_satisfied=False, visible_unsatisfied=True,
        prior_no_effect=False, remaining_fraction=0.5,
        previous_action="click('28')", candidates=["click('55')", "go_back()"],
        uncertainty_scale=0.0, decision_margin=0.0,
    )
    assert followup.selected_index == 0
    assert followup.source_abstained
    assert followup.reason == "target_runtime_binding_abstained_episode"
    assert controller.source_authorized_decisions == 0
