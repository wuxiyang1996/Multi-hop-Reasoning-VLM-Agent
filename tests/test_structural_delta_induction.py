from motif_transfer.structural_delta_induction import (
    StructuralPath,
    abstract_effect_sequence,
    induce_structural_program,
    structural_atom_descriptors,
    structural_delta_descriptor,
    structural_state_features,
    validate_structural_program,
)


def _state(
    *, carrying=None, objects=(), relations=(), agent_position=(1, 1),
):
    return {
        "agent_position": list(agent_position),
        "carrying": carrying,
        "objects": list(objects),
        "relations": list(relations),
    }


def _entity(name="key", color="red", *, position=(2, 2), opened=None, locked=None):
    return {
        "type": name,
        "color": color,
        "position": list(position),
        "is_open": opened,
        "is_locked": locked,
    }


def _step(before, after):
    return {
        "before_features": structural_state_features(before),
        "delta": structural_delta_descriptor(before, after),
    }


def test_delta_types_are_alpha_renaming_invariant_and_structurally_distinct():
    key = _entity()
    ball = _entity("ball", "blue")
    bind_key = structural_delta_descriptor(
        _state(objects=[key]),
        _state(carrying={"type": "key", "color": "red"}),
    )
    bind_ball = structural_delta_descriptor(
        _state(objects=[ball]),
        _state(carrying={"type": "ball", "color": "blue"}),
    )
    assert bind_key["delta_type_id"] == bind_ball["delta_type_id"]

    door_closed = _entity("door", "yellow", opened=False, locked=True)
    door_open = _entity("door", "yellow", opened=True, locked=False)
    mutate = structural_delta_descriptor(
        _state(objects=[door_closed]), _state(objects=[door_open]),
    )
    relate = structural_delta_descriptor(
        _state(carrying={"type": "key", "color": "red"}),
        _state(
            relations=[[["key", "red"], ["door", "yellow"]]],
        ),
    )
    assert len({
        bind_key["delta_type_id"], mutate["delta_type_id"],
        relate["delta_type_id"],
    }) == 3
    assert {row["arity"] for row in relate["atoms"]} == {1, 2}


def test_abstract_sequence_strips_navigation_and_collapses_repeats():
    base = _state(objects=[_entity()])
    moved = _state(objects=[_entity()], agent_position=(1, 2))
    bound = _state(carrying={"type": "key", "color": "red"})
    steps = [
        _step(base, moved),
        _step(base, bound),
        _step(base, bound),
    ]
    sequence = abstract_effect_sequence(steps)
    assert sequence == (
        structural_atom_descriptors(steps[1]["delta"])[0]["operator_type_id"],
    )


def test_induction_learns_sequence_guards_and_abstention_contract():
    start = _state(objects=[_entity()])
    bound = _state(carrying={"type": "key", "color": "red"})
    related = _state(
        relations=[[["key", "red"], ["door", "yellow"]]],
    )
    bind = _step(start, bound)
    relate = _step(bound, related)
    navigation = _step(start, _state(objects=[_entity()], agent_position=(2, 1)))
    success_steps = (navigation, bind, relate)
    paths = (
        StructuralPath("development", True, success_steps),
        StructuralPath("development", True, (bind, relate)),
        StructuralPath("development", False, (navigation,)),
        StructuralPath("qualification", True, success_steps),
        StructuralPath("qualification", False, (bind,)),
    )
    program = induce_structural_program(
        paths, source_receipts_sha256="source-receipts",
    )
    validate_structural_program(program)
    assert program["status"] == "SOURCE_STRUCTURAL_PROGRAM_QUALIFIED"
    assert program["induced_sequence"] == [
        structural_atom_descriptors(bind["delta"])[0]["operator_type_id"],
        *[
            row["operator_type_id"]
            for row in structural_atom_descriptors(relate["delta"])
        ],
    ]
    assert len(program["operators"]) == 3
    assert program["operators"][0]["learned_guards"]
    assert program["qualification_metrics"]["success_support"] == 1.0
    assert program["qualification_metrics"]["control_support"] == 0.0


def test_induction_abstains_without_replicated_success_paths():
    start = _state(objects=[_entity()])
    moved = _state(objects=[_entity()], agent_position=(1, 2))
    program = induce_structural_program(
        (
            StructuralPath("development", True, (_step(start, moved),)),
            StructuralPath("qualification", True, (_step(start, moved),)),
        ),
        source_receipts_sha256="insufficient",
    )
    validate_structural_program(program)
    assert program["status"] == "SOURCE_STRUCTURAL_PROGRAM_ABSTAINING"
    assert program["operators"] == []
