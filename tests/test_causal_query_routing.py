from __future__ import annotations

from motif_transfer.causal_query_routing import (
    CausalQueryState,
    build_source_models,
    select_action,
    source_gate_report,
)


CONFIG = {
    "seed": 901,
    "control_seed": 902,
    "heldout_seed": 903,
    "train_states": 4096,
    "heldout_states": 1024,
    "ridge_alpha": 0.1,
    "minimum_authentic_accuracy": 0.95,
    "minimum_control_margin": 0.15,
}


def test_source_causal_routing_gate_passes_controls() -> None:
    report = source_gate_report(CONFIG)
    assert report["status"] == "SOURCE_CAUSAL_ROUTING_GATE_PASSED"
    accuracy = report["selection_accuracy"]
    assert accuracy["authentic_source_router"] > accuracy["shuffled_source_router"]
    assert accuracy["authentic_source_router"] > accuracy["source_marginal_router"]


def test_authentic_router_changes_representation_after_intervention() -> None:
    model = build_source_models(CONFIG)["authentic_source_router"]
    factual = CausalQueryState(
        "factual", False, True, 0.86, 0.68, 0.55,
    )
    counterfactual = CausalQueryState(
        "counterfactual", True, True, 0.86, 0.68, 0.55,
    )
    assert select_action(factual, model) == "USE_EXPLICIT_RELATION"
    assert select_action(counterfactual, model) == "DERIVE_FROM_TRAJECTORY"
