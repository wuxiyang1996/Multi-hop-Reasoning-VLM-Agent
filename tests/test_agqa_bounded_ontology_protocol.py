import pytest

from motif_transfer.agqa_bounded_ontology_protocol import (
    bounded_response_format,
    bounded_system_prompt,
)


def _base():
    return {"json_schema": {"schema": {"properties": {
        "decision": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_frames": {"type": "array"},
        "visual_description": {"type": "string"},
        "uncertainty": {"type": "string"},
    }}}}


def test_bounded_protocol_only_changes_explanatory_strings():
    base = _base()
    result = bounded_response_format(base, max_characters=160)
    properties = result["json_schema"]["schema"]["properties"]
    assert properties["visual_description"]["maxLength"] == 160
    assert properties["uncertainty"]["maxLength"] == 160
    assert properties["decision"] == {"type": "string"}
    assert properties["confidence"] == {"type": "number"}
    assert properties["evidence_frames"] == {"type": "array"}
    assert "maxLength" not in base["json_schema"]["schema"]["properties"][
        "visual_description"
    ]


def test_bounded_prompt_and_limit_fail_closed():
    assert "160 characters" in bounded_system_prompt("base")
    with pytest.raises(ValueError):
        bounded_response_format(_base(), max_characters=16)
    with pytest.raises(ValueError):
        bounded_system_prompt("base", max_characters=200)
