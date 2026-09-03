from scripts.collect_candidate_qualification_base import _parse_base_payload


def test_parse_base_payload_preserves_order_and_normalizes_priors():
    particles, entities = _parse_base_payload(
        {
            "answer_priors": [
                {"native_answer": "A", "prior_weight": 2, "event_graph_summary": "a"},
                {"native_answer": "B", "prior_weight": 1, "event_graph_summary": "b"},
            ],
            "entity_catalog": [
                {"entity_id": "e0", "visual_description": "person in blue"},
            ],
        },
        answer_space=("A", "B"),
    )
    assert [row["native_answer"] for row in particles] == ["A", "B"]
    assert sum(row["prior_weight"] for row in particles) == 1.0
    assert entities == [{"entity_id": "e0", "visual_description": "person in blue"}]


def test_parse_base_payload_rejects_reordered_answers():
    try:
        _parse_base_payload(
            {
                "answer_priors": [
                    {"native_answer": "B", "prior_weight": 1, "event_graph_summary": "b"},
                    {"native_answer": "A", "prior_weight": 1, "event_graph_summary": "a"},
                ],
                "entity_catalog": [
                    {"entity_id": "e0", "visual_description": "object"},
                ],
            },
            answer_space=("A", "B"),
        )
    except ValueError as exc:
        assert "frozen answer order" in str(exc)
    else:
        raise AssertionError("reordered answer prior should fail closed")
