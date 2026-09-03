from scripts.collect_agqa2_query_object_v20 import _ontology_response_format


def test_ontology_schema_is_closed_without_per_question_candidates():
    schema = _ontology_response_format()["json_schema"]["schema"]
    decisions = schema["properties"]["decision"]["enum"]
    assert "unknown" in decisions
    assert "chair" in decisions
    assert schema["additionalProperties"] is False
    assert "answer_candidates" not in schema["properties"]
