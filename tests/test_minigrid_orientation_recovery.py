from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.minigrid_orientation_recovery import (
    TOKENS,
    ground_truth_binding,
    parse_neural_binding,
    select_recovery,
    task_spec,
)


REPO = Path(__file__).resolve().parents[1]


def _program():
    report = json.loads(
        (REPO / "docs/results/tetris_cyclic_source_induction_v28.json").read_text()
    )
    return report["development"]["first_qualified"]["program"]


def test_task_specs_are_nonidentity_and_bijective():
    for seed in range(710001, 710101):
        spec = task_spec(seed, "unit-orientation-target")
        assert spec.probe_effect in {1, 2, 3}
        assert set(spec.token_effects) == {0, 1, 2, 3}
        assert 7 <= len(spec.probe_effects) <= 14


def test_source_program_selects_unique_native_inverse():
    program = _program()
    for seed in range(720001, 720041):
        spec = task_spec(seed, "unit-orientation-target")
        binding = ground_truth_binding(spec, initial_direction=seed % 4)
        token = select_recovery(program, binding, condition="source_induced")
        assert token in TOKENS
        assert (spec.probe_effect + spec.token_to_effect[token]) % 4 == 0
        assert select_recovery(
            program, binding, condition="alpha_renamed_source",
        ) == token
        assert select_recovery(
            program, binding, condition="target_written_isomorphic",
        ) == token


def test_grounder_fails_closed_on_low_confidence_or_ambiguity():
    payload = {
        "directions": {
            "I": "right", "P": "down", "C0": "up",
            "A": "right", "B": "down", "C": "left", "D": "up",
        },
        "confidences": {
            "I": 0.99, "P": 0.99, "C0": 0.99,
            "A": 0.99, "B": 0.2, "C": 0.99, "D": 0.99,
        },
        "direct_recovery": "A",
    }
    binding = parse_neural_binding(payload, minimum_confidence=0.8)
    assert binding["qualified"] is False
    assert binding["direct_recovery"] == "ABSTAIN"
    assert select_recovery(
        _program(), binding, condition="source_induced",
    ) == "ABSTAIN"


def test_copy_effect_is_a_destructive_relation_except_half_turns():
    program = _program()
    observed_non_half_turn = 0
    for seed in range(730001, 730101):
        spec = task_spec(seed, "unit-orientation-target")
        binding = ground_truth_binding(spec, initial_direction=0)
        source = select_recovery(program, binding, condition="source_induced")
        copied = select_recovery(program, binding, condition="copy_effect_control")
        if spec.probe_effect != 2:
            observed_non_half_turn += 1
            assert copied != source
    assert observed_non_half_turn > 0
