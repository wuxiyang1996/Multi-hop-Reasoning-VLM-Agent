from motif_transfer.agqa_layer_b_authority import cohort_crossed_authority


def test_new_public_projection_schema_is_accepted_by_contract_not_name():
    cohort = {
        "schema_version": "new-public-projection-v99",
        "answers_projected": False,
        "functional_programs_projected": False,
        "scene_graph_grounding_projected": False,
    }
    assert not cohort_crossed_authority(cohort)


def test_projected_answer_is_rejected():
    cohort = {
        "answers_projected": True,
        "functional_programs_projected": False,
        "scene_graph_grounding_projected": False,
    }
    assert cohort_crossed_authority(cohort)


def test_explicit_official_scene_graph_alias_is_accepted():
    cohort = {
        "answers_projected": False,
        "functional_programs_projected": False,
        "official_scene_graph_grounding_projected": False,
    }
    assert not cohort_crossed_authority(cohort)


def test_projected_official_scene_graph_alias_is_rejected():
    cohort = {
        "answers_projected": False,
        "functional_programs_projected": False,
        "official_scene_graph_grounding_projected": True,
    }
    assert cohort_crossed_authority(cohort)


def test_legacy_explicit_runtime_authority_is_preserved():
    cohort = {
        "answers_read": False,
        "scene_graphs_read": False,
        "functional_program_visible_at_runtime": False,
    }
    assert not cohort_crossed_authority(cohort)
