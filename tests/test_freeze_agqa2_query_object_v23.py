from scripts.freeze_agqa2_query_object_v23_reserve import (
    _query_object_program_answer_space_matches,
)


def test_query_object_program_requires_query_answer_space():
    assert _query_object_program_answer_space_matches(
        "Query(class, OnlyItem(Iterate(video, Filter(frame, [relations, holding, objects]))))"
    )
    assert not _query_object_program_answer_space_matches(
        "Exists(Iterate(video, Filter(frame, [relations, holding, objects])))"
    )
