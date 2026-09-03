from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.neurosymbolic_skill_library import (
    DispatchVerdict,
    EvidenceTier,
    FrozenNeurosymbolicSkillLibrary,
    TargetRequest,
    validate_dispatch_receipt,
)


REPO = Path(__file__).resolve().parents[1]


REQUESTS = (
    ("webshop", "product_search_and_option_commit", (
        "candidate_outcomes", "effect_verification", "native_commit", "native_search",
    )),
    ("alfworld", "text_household_workflow", (
        "admissible_actions", "effect_prediction", "goal_binding", "workflow_state",
    )),
    ("discoveryworld", "scientific_spatial_commit", (
        "commit_effect_prediction", "inventory_state", "object_relation_state",
        "spatial_realization",
    )),
    ("tir", "single_image_maze", (
        "direction_binding", "pixel_graph", "sequence_execution",
        "unique_goal_verification",
    )),
)


def test_all_four_exact_routes_are_fresh_formal_and_target_native() -> None:
    library = FrozenNeurosymbolicSkillLibrary.load(
        REPO / "configs/neurosymbolic_skill_library_v1.json", repo=REPO,
    )
    receipts = [
        library.dispatch(
            TargetRequest.create(domain, interface, capabilities),
            minimum_evidence=EvidenceTier.FRESH_FORMAL,
        )
        for domain, interface, capabilities in REQUESTS
    ]
    for receipt in receipts:
        validate_dispatch_receipt(receipt)
        assert receipt.verdict == DispatchVerdict.SELECT_SKILL
        assert receipt.evidence_tier == EvidenceTier.FRESH_FORMAL
        assert receipt.action_authority == "TARGET_NATIVE_GROUNDER_AND_EXECUTOR"
        assert not hasattr(receipt, "action")
    assert {receipt.request.domain for receipt in receipts} == {
        "webshop", "alfworld", "discoveryworld", "tir",
    }
    by_domain = {receipt.request.domain: receipt for receipt in receipts}
    assert by_domain["alfworld"].skill_id == (
        "procedural-game-typed-workflow-value-v1"
    )


def test_alfworld_route_is_bound_to_procedural_game_sources() -> None:
    registry = json.loads(
        (REPO / "configs/neurosymbolic_skill_library_v1.json").read_text()
    )
    skill = next(
        row for row in registry["skills"]
        if row["skill_id"] == "procedural-game-typed-workflow-value-v1"
    )
    assert set(skill["source_domains"]) == {
        "grid_quest", "factory_quest", "dungeon_quest", "space_quest",
        "island_quest", "circuit_quest",
    }
    assert "raw action" not in skill["symbolic_payload"].lower()
    assert skill["routes"][0]["action_authority"] == (
        "TARGET_NATIVE_GROUNDER_AND_EXECUTOR"
    )


def test_unsupported_video_and_broad_tir_routes_abstain() -> None:
    library = FrozenNeurosymbolicSkillLibrary.load(
        REPO / "configs/neurosymbolic_skill_library_v1.json", repo=REPO,
    )
    for request in (
        TargetRequest.create("video", "causal_video_qa", ["event_graph"]),
        TargetRequest.create("tir", "rotation", ["direction_binding"]),
    ):
        receipt = library.dispatch(request, minimum_evidence=EvidenceTier.MECHANISM)
        validate_dispatch_receipt(receipt)
        assert receipt.verdict == DispatchVerdict.ABSTAIN
