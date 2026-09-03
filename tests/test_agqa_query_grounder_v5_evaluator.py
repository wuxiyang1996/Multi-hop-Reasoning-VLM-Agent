from __future__ import annotations

import pytest

from scripts.evaluate_agqa_query_grounder_v5_development import (
    _expected_grounding_status,
    _gate_requirement,
)


def test_gate_requirement_accepts_frozen_canonical_role_key() -> None:
    assert _gate_requirement(
        {"typed_role_fidelity_minimum": 0.8},
        "typed_role_fidelity_minimum",
        "typed_role_binding_fidelity_minimum",
    ) == 0.8


def test_gate_requirement_keeps_legacy_protocol_compatibility() -> None:
    assert _gate_requirement(
        {"typed_role_binding_fidelity_minimum": 0.75},
        "typed_role_fidelity_minimum",
        "typed_role_binding_fidelity_minimum",
    ) == 0.75


def test_gate_requirement_fails_closed_when_threshold_is_absent() -> None:
    with pytest.raises(KeyError, match="missing frozen grounding gate"):
        _gate_requirement({}, "typed_role_fidelity_minimum")


def test_consumed_development_status_cannot_be_confused_with_fresh() -> None:
    assert _expected_grounding_status(True).startswith("CONSUMED_DEVELOPMENT")
    assert _expected_grounding_status(False) == (
        "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME"
    )
