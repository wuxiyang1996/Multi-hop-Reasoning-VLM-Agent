from motif_transfer.agqa_open_vocabulary_grounder import Detection, PhraseDetection
from motif_transfer.agqa_relation_phrase_binder import (
    bind_phrase_regions_to_tracks, relation_query_phrases,
    slowfast_relation_frame_indices,
)


def test_relation_query_phrases_preserve_direction() -> None:
    assert relation_query_phrases("beneath")[0] == "object above a person"
    assert relation_query_phrases("carrying")[0] == "object a person is carrying"


def test_phrase_regions_bind_to_same_frame_ontology_box() -> None:
    phrases = (
        PhraseDetection(1, "object a person is carrying", .9, (10, 10, 30, 30)),
        PhraseDetection(2, "object a person is carrying", .8, (11, 10, 31, 30)),
    )
    ontology = (
        Detection(1, "person", .99, (0, 0, 100, 100)),
        Detection(1, "book", .8, (10, 10, 30, 30)),
        Detection(1, "chair", .9, (70, 70, 95, 95)),
        Detection(2, "book", .75, (11, 10, 31, 30)),
    )
    binding = bind_phrase_regions_to_tracks(phrases, ontology)
    assert binding is not None
    assert binding.label == "book"
    assert binding.evidence_frames == (1, 2)
    assert binding.score > binding.runner_up_score


def test_slowfast_scores_define_directional_relation_window() -> None:
    row = {
        "native_frame_index_views": [list(range(0, 32)), list(range(8, 40)), list(range(16, 48))],
        "obligations": [{"mapping_status": "EXACT_PUBLIC_ACTION_CLASS",
                         "window_scores": [0.0, 0.0, 1.0]}],
    }
    before = slowfast_relation_frame_indices(row, temporal_operator="BEFORE")
    after = slowfast_relation_frame_indices(row, temporal_operator="AFTER")
    assert before[0] == 0 and before[-1] < 47
    assert after[0] > 0 and after[-1] == 47


def test_slowfast_unmapped_anchor_fails_open_to_full_visual_search() -> None:
    indices = slowfast_relation_frame_indices(
        {"native_frame_index_views": [], "obligations": []}, temporal_operator="WHILE")
    assert indices[0] == 0 and indices[-1] == 47
