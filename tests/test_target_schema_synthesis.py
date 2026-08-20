from __future__ import annotations

from motif_transfer.target_schema_synthesis import (
    expected_program,
    parse_program_response,
    score_program,
    synthesis_prompt,
)


def test_exact_target_program_is_canonicalized() -> None:
    expected = expected_program("tir_rotation")
    response = {
        **expected,
        "operators": list(reversed(expected["operators"])),
        "abstention": list(reversed(expected["abstention"])),
    }
    parsed = parse_program_response(
        "```json\n" + __import__("json").dumps(response) + "\n```"
    )
    assert score_program("tir_rotation", parsed)["exact_program_match"]


def test_wrong_isomorphic_family_does_not_pass() -> None:
    wrong = expected_program("discoveryworld")
    score = score_program("alfworld", wrong)
    assert score["exact_program_match"] is False
    assert score["field_matches"]["program_family"] is False


def test_prompt_explicitly_forbids_target_outcomes_and_source_program() -> None:
    prompt = synthesis_prompt(
        "alfworld", "Actions and typed observations are available.",
        variant=0,
    )
    assert "NO successful trajectory" in prompt
    assert "source program" in prompt
    assert "Return exactly one JSON object" in prompt
