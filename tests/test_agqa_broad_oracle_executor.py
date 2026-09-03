from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan
from motif_transfer.agqa_broad_oracle_executor import execute_broad_public_plan
from motif_transfer.agqa_oracle_query_mdp import AGQAOracleQueryBackend, AGQAOracleToolBudget
from motif_transfer.contracts import stable_hash


ONTOLOGY = {
    "o1": "person", "o4": "blanket", "o26": "pillow",
    "r1": "looking at", "r22": "touching",
    "c000": "holding some clothes", "c075": "tidying up a blanket",
    "c076": "holding a pillow",
}


def _backend():
    graph = {
        **{f"{frame:06d}": {"id": f"{frame:06d}", "type": "frame"}
           for frame in (10, 20, 30, 40, 50)},
        "c075/1": {"id": "c075/1", "type": "action", "phrase": "tidying up a blanket", "all_f": ["000010", "000020"]},
        "c076/1": {"id": "c076/1", "type": "action", "phrase": "holding a pillow", "all_f": ["000030", "000040", "000050"]},
        "o26/000040": {"id": "o26/000040", "type": "object", "class": "o26"},
        "r22/000040": {"id": "r22/000040", "type": "contact", "class": "r22", "frame": "000040", "objects": [{"id": "o26/000040"}]},
    }
    return AGQAOracleQueryBackend(
        video_id="v", graph=graph, id_to_text=ONTOLOGY,
        graph_sha256=stable_hash("broad"), budget=AGQAOracleToolBudget(3),
    )


def test_broad_executor_orders_two_public_events():
    plan = parse_public_question_plan(
        "Did the person tidy up a blanket before or after they held a pillow?"
    )
    assert plan is not None
    result = execute_broad_public_plan(plan, _backend())
    assert result.prediction == "before"


def test_broad_executor_compares_duration():
    plan = parse_public_question_plan(
        "Which did they do for longer, tidied up a blanket or held a pillow?"
    )
    assert plan is not None
    result = execute_broad_public_plan(plan, _backend())
    assert result.prediction == "holding a pillow"


def test_broad_executor_queries_relation_object():
    plan = parse_public_question_plan("Which object were they touching?")
    assert plan is not None
    result = execute_broad_public_plan(plan, _backend())
    assert result.prediction == "pillow"


def test_broad_executor_closed_world_exists():
    plan = parse_public_question_plan("Was the person holding a pillow?")
    assert plan is not None
    assert execute_broad_public_plan(plan, _backend()).prediction == "yes"
