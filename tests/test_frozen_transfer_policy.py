from __future__ import annotations

import pytest

from harness.frozen_transfer_policy import (
    action_prompt,
    parse_exact_numbered_response,
)


@pytest.mark.parametrize("value", ["ACTION: 2", " action: 2\n"])
def test_exact_action_response(value: str) -> None:
    assert parse_exact_numbered_response(value, kind="action", n=3) == 1


@pytest.mark.parametrize(
    "value",
    ["I choose ACTION: 2", "ACTION: go north", "ACTION: 0", "ACTION: 4", "2"],
)
def test_hallucinated_or_extracted_action_fails_closed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_exact_numbered_response(value, kind="action", n=3)


def test_prompt_preserves_exact_action_strings() -> None:
    prompt = action_prompt(
        domain="alfworld",
        goal="put the mug away",
        observation="at the table",
        actions=["look", "take mug 1 from table 1"],
    )
    assert "1. look" in prompt
    assert "2. take mug 1 from table 1" in prompt
    assert "Return exactly `ACTION: N`" in prompt
