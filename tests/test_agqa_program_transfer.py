from motif_transfer.agqa_program_transfer import (
    COMPOSITE_ROUTE,
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    UNSUPPORTED_ROUTE,
    profile_program,
    select_program,
)
from motif_transfer.structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
)


def _source(
    *, program_sha256: str, ir_kind: str, operation: str,
    predicate_family: str, arity: int, value_kind: str,
    recurrent: bool, terminal: tuple[str, ...] = (),
) -> SourceIRContract:
    return SourceIRContract.create(
        program_sha256=program_sha256,
        ir_kind=ir_kind,
        operator_sequence=(OperatorSignature(
            operation=operation,
            predicate_family=predicate_family,
            arity=arity,
            value_kind=value_kind,
        ),),
        recurrent=recurrent,
        terminal_predicate_families=terminal,
        source_intervention_qualified=True,
        source_confirmation_sha256=f"confirmation-{program_sha256}",
    )


RELATION_SOURCE = _source(
    program_sha256="relation-source",
    ir_kind="RECURRENT_GOAL_RELATION_PROGRAM",
    operation="UPDATE",
    predicate_family="ENTITY_GOAL_RELATION",
    arity=2,
    value_kind="RELATION_COVERAGE",
    recurrent=True,
    terminal=("ENTITY_GOAL_RELATION",),
)
TEMPORAL_PAIR_SOURCE = _source(
    program_sha256="temporal-pair-source",
    ir_kind="SPARSE_TEMPORAL_EFFECT_FUNCTION",
    operation="SCORE",
    predicate_family="TEMPORAL_EFFECT_VECTOR",
    arity=2,
    value_kind="NORMALIZED_PROBABILITY",
    recurrent=True,
)
TEMPORAL_SINGLE_SOURCE = _source(
    program_sha256="temporal-single-source",
    ir_kind="SPARSE_TEMPORAL_EFFECT_FUNCTION",
    operation="SCORE",
    predicate_family="TEMPORAL_EFFECT_VECTOR",
    arity=1,
    value_kind="NORMALIZED_PROBABILITY",
    recurrent=False,
)
SOURCES = (RELATION_SOURCE, TEMPORAL_PAIR_SOURCE, TEMPORAL_SINGLE_SOURCE)


def _selected(profile):
    return select_program(
        SOURCES, profile, target_grounder_sha256="program-compiler-v1",
    )


def test_relation_program_selects_exact_recurrent_relation_contract():
    profile = profile_program(
        task_id="YSKX3-524",
        program=(
            "Exists(Query(class, OnlyItem(Iterate(video, Filter(frame, "
            "[relations, in front of, objects])))), Iterate(video, "
            "Filter(frame, [relations, tidying, objects])))"
        ),
    )
    receipt = _selected(profile)
    assert profile.route_kind == RELATION_ROUTE
    assert receipt["selected_program_sha256"] == "relation-source"
    assert receipt["source_identity_used_as_feature"] is False
    assert receipt["target_outcome_read"] is False
    assert receipt["target_action_emitted"] is False


def test_temporal_pair_program_selects_exact_recurrent_pair_contract():
    profile = profile_program(
        task_id="temporal-pair",
        program=(
            "Compare([before, after], Exists(blanket, Iterate(Localize("
            "temporal tag, holding a pillow), Filter(frame, [objects]))))"
        ),
    )
    receipt = _selected(profile)
    assert profile.route_kind == TEMPORAL_PAIR_ROUTE
    assert receipt["selected_program_sha256"] == "temporal-pair-source"


def test_duration_program_selects_exact_nonrecurrent_single_contract():
    profile = profile_program(
        task_id="duration",
        program=(
            "Superlative(max, [Filter(video, [actions, holding a vacuum]), "
            "Filter(video, [actions, smiling at something])], "
            "Subtract(Query(end, action), Query(start, action)))"
        ),
    )
    receipt = _selected(profile)
    assert profile.route_kind == TEMPORAL_SINGLE_ROUTE
    assert receipt["selected_program_sha256"] == "temporal-single-source"


def test_binary_duration_wrapper_keeps_same_inner_score_contract():
    profile = profile_program(
        task_id="binary-duration",
        program=(
            "Equals(holding a vacuum, Superlative(max, [Filter(video, "
            "[actions, holding a vacuum]), Filter(video, [actions, "
            "smiling at something])], Subtract(Query(end, action), "
            "Query(start, action))))"
        ),
    )
    receipt = _selected(profile)
    assert profile.route_kind == TEMPORAL_SINGLE_ROUTE
    assert receipt["selected_program_sha256"] == "temporal-single-source"


def test_composite_and_unsupported_programs_fail_closed():
    composite = profile_program(
        task_id="composite",
        program=(
            "Exists(Query(class, OnlyItem(IterateUntil(forward, video, "
            "Exists(touching, Filter(frame, [relations])), Filter(frame, "
            "[relations, touching, objects])))), Iterate(video, "
            "Filter(frame, [relations, holding, objects])))"
        ),
    )
    unsupported = profile_program(
        task_id="unsupported",
        program="Query(class, OnlyItem(Filter(video, [objects])))",
    )
    assert composite.route_kind == COMPOSITE_ROUTE
    assert unsupported.route_kind == UNSUPPORTED_ROUTE
    assert _selected(composite)["selected_program_sha256"] is None
    assert _selected(unsupported)["selected_program_sha256"] is None


def test_wrong_source_permutation_is_rejected_by_anonymous_type():
    profiles = (
        profile_program(
            task_id="relation",
            program=(
                "Exists(Query(class, OnlyItem(Iterate(video, Filter(frame, "
                "[relations, touching, objects])))), Iterate(video, "
                "Filter(frame, [relations, holding, objects])))"
            ),
        ),
        profile_program(
            task_id="pair",
            program=(
                "Choose([before, after], Iterate(Localize(temporal tag, "
                "holding a pillow), Filter(frame, [objects])))"
            ),
        ),
        profile_program(
            task_id="single",
            program=(
                "Superlative(max, [Filter(video, [actions, smiling])], "
                "Subtract(Query(end, action), Query(start, action)))"
            ),
        ),
    )
    wrong_sources = (
        TEMPORAL_PAIR_SOURCE,
        TEMPORAL_SINGLE_SOURCE,
        RELATION_SOURCE,
    )
    for profile, wrong_source in zip(profiles, wrong_sources):
        receipt = select_program(
            (wrong_source,), profile,
            target_grounder_sha256="program-compiler-v1",
        )
        assert receipt["status"] == "SOURCE_CONTRACT_SELECTION_ABSTAINED"
        assert receipt["selected_program_sha256"] is None


def test_profile_interface_has_no_target_outcome_or_source_identity_fields():
    profile = profile_program(
        task_id="blindness",
        program="Exists(person, Iterate(video, Filter(frame, [objects])))",
    )
    assert profile.answer_read is False
    assert profile.scene_graph_grounding_read is False
    assert profile.source_identity_read is False
