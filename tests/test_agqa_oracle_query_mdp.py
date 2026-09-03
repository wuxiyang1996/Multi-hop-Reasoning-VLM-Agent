import pytest

from motif_transfer.agqa_oracle_query_mdp import (
    AGQAOracleQueryBackend,
    AGQAOracleToolBudget,
    compose_localized_with_generic,
    execute_temporal_object_query,
)
from motif_transfer.agqa_temporal_localized_query import (
    parse_temporal_localized_object_question,
)
from motif_transfer.contracts import stable_hash


ONTOLOGY = {
    "o1": "person", "o4": "blanket", "o26": "pillow",
    "r15": "holding", "r22": "touching", "c075": "tidying up a blanket",
    "c000": "holding some clothes", "r1": "looking at",
}


def _graph():
    graph = {}
    for frame in (10, 20, 30, 40, 50):
        key = f"{frame:06d}"
        graph[key] = {"id": key, "type": "frame"}
    graph["c075/1"] = {
        "id": "c075/1", "charades": "c075", "type": "action",
        "phrase": "tidying up a blanket", "all_f": ["000020", "000030"],
    }
    pillow = {"id": "o26/000040", "type": "object", "class": "o26"}
    graph[pillow["id"]] = pillow
    graph["r22/000040"] = {
        "id": "r22/000040", "type": "contact", "class": "r22",
        "frame": "000040", "objects": [pillow],
    }
    return graph


def _backend(max_calls=2):
    graph = _graph()
    return AGQAOracleQueryBackend(
        video_id="v", graph=graph, id_to_text=ONTOLOGY,
        graph_sha256=stable_hash("graph"),
        budget=AGQAOracleToolBudget(max_calls=max_calls),
    )


def test_answer_blind_oracle_query_executes_temporal_object_binding():
    plan = parse_temporal_localized_object_question(
        "After tidying up a blanket, which object did the person touch?"
    )
    assert plan is not None
    result = execute_temporal_object_query(plan, _backend())
    assert result.prediction == "pillow"
    assert result.status == "COMMITTED"
    assert [row.tool for row in result.receipts] == [
        "LOCATE_ACTION", "QUERY_RELATION_IN_WINDOW",
    ]
    assert all(not row.answer_read and not row.functional_program_read
               for row in result.receipts)


def test_oracle_query_fails_closed_on_nonunique_object_binding():
    graph = _graph()
    blanket = {"id": "o4/000050", "type": "object", "class": "o4"}
    graph[blanket["id"]] = blanket
    graph["r22/000050"] = {
        "id": "r22/000050", "type": "contact", "class": "r22",
        "frame": "000050", "objects": [blanket],
    }
    backend = AGQAOracleQueryBackend(
        video_id="v", graph=graph, id_to_text=ONTOLOGY,
        graph_sha256=stable_hash("graph2"), budget=AGQAOracleToolBudget(2),
    )
    plan = parse_temporal_localized_object_question(
        "Which object did they touch after tidying up a blanket?"
    )
    result = execute_temporal_object_query(plan, backend)
    assert result.prediction is None
    assert result.candidate_objects == ("blanket", "pillow")
    assert result.reason == "OBJECT_BINDING_NOT_UNIQUE"


def test_oracle_backend_enforces_matched_maximum_budget():
    backend = _backend(max_calls=1)
    backend.locate_action("tidying up a blanket")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        backend.query_relation("touching", frames=[40])


@pytest.mark.parametrize(
    ("localized", "generic", "actor", "expected", "route"),
    [
        ("pillow", "blanket", "door", "pillow", "LOCALIZED"),
        (None, "blanket", "door", "blanket", "GENERIC_FALLBACK"),
        (None, None, "door", "door", "ACTOR_FALLBACK"),
        (None, None, None, None, "ABSTAINED"),
    ],
)
def test_composed_policy_has_fixed_guarded_precedence(
    localized, generic, actor, expected, route,
):
    result = compose_localized_with_generic(localized, generic, actor)
    assert result.prediction == expected
    assert result.route == route
