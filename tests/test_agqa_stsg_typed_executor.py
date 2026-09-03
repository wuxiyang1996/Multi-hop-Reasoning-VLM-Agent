from motif_transfer.agqa_stsg_typed_executor import AGQATypedSTSGExecutor
from motif_transfer.contracts import stable_hash


GRAPH = {
    "000001": {"type": "frame"}, "000002": {"type": "frame"},
    "000003": {"type": "frame"},
    "c1": {"type": "action", "phrase": "opening door", "all_f": [1, 2]},
    "c2": {"type": "action", "phrase": "holding cup", "all_f": [3]},
    "o2/000003": {"type": "object", "class": "o2"},
    "r1/000003": {"type": "relation", "class": "r1",
                   "objects": [{"id": "o2/000003"}]},
}
ONTOLOGY = {"o2": "cup", "r1": "holding"}
OPS = ("INTERVAL_OF", "TEMPORAL_SELECT", "FILTER_EQ", "UNIQUE", "PROJECT",
       "EXISTS", "FIRST", "LAST", "XOR", "AND", "CHOOSE", "COMPARE", "ARGMAX")


def executor(ops=OPS):
    return AGQATypedSTSGExecutor(
        graph=GRAPH, id_to_text=ONTOLOGY, graph_sha256=stable_hash(GRAPH),
        authorized_operators=ops,
    )


def test_temporal_relation_query_executes_through_source_gate():
    program = (
        "Query(class, OnlyItem(Iterate(Localize(after, opening door), "
        "Filter(frame, [relations, holding, objects]))))"
    )
    receipt = executor().execute(program)
    assert receipt.status == "COMMITTED"
    assert receipt.prediction == "cup"
    assert {"INTERVAL_OF", "TEMPORAL_SELECT", "FILTER_EQ", "UNIQUE", "PROJECT"} <= set(receipt.executed_operators)


def test_missing_source_capability_abstains_before_answer():
    program = "Exists(cup, Iterate(video, Filter(frame, [objects])))"
    receipt = executor(tuple(op for op in OPS if op != "EXISTS")).execute(program)
    assert receipt.status == "ABSTAINED"
    assert receipt.prediction is None
    assert "SOURCE_PROGRAM_NOT_ADMITTED" in receipt.reason
    assert "EXISTS" in receipt.reason


def test_duration_superlative_uses_source_argmax():
    program = (
        "Superlative(max, [Filter(video, [actions, opening door]), "
        "Filter(video, [actions, holding cup])], "
        "Subtract(Query(end, action), Query(start, action)))"
    )
    receipt = executor().execute(program)
    assert receipt.status == "COMMITTED"
    assert receipt.prediction == "opening door"
    assert "ARGMAX" in receipt.executed_operators


def test_first_action_uses_has_item_existence_capability():
    program = (
        "Query(class, OnlyItem(IterateUntil(forward, video, "
        "HasItem(Filter(frame, [actions])), Filter(frame, [actions]))))"
    )
    receipt = executor().execute(program)
    assert receipt.status == "COMMITTED"
    assert receipt.prediction == "opening door"
    assert {"FIRST", "EXISTS", "UNIQUE", "PROJECT"} <= set(receipt.executed_operators)
