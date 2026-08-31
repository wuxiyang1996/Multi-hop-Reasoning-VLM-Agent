from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_source_function_induction import (
    ABSTAINING,
    QUALIFIED,
    function_trial_order,
    function_weights,
    induce_source_function_program,
    recalibrate_source_function_program,
    validate_source_function_program,
)
from motif_transfer.phase3_typed_effect_induction import (
    TYPED_EFFECTS,
    TypedCandidate,
    TypedInterventionSet,
)


def _candidate(rank, values, long_value):
    return TypedCandidate(
        candidate_rank=rank,
        effect_values=tuple(zip(TYPED_EFFECTS, values)),
        long_horizon_value=long_value,
        transition_receipt_sha256=stable_hash({"rank": rank, "values": values}),
    )


def _examples():
    rows = []
    # H1 alone and H8 alone each fail half the sets; their convex combination
    # ranks the verified candidate in all sets.  A second candidate recovers
    # some discovery misses under competing grid functions.
    patterns = (
        ((1.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.9, 0.0)),
        ((0.5, 0.0, 1.0, 0.0), (0.9, 0.0, 0.0, 0.0)),
    )
    for split in ("discovery", "qualification"):
        for index in range(8):
            left, right = patterns[index % 2]
            rows.append(TypedInterventionSet(
                snapshot_sha256=stable_hash({"split": split, "index": index}),
                source_split=split,
                candidates=(
                    _candidate(0, left, 1.0),
                    _candidate(1, right, 0.0),
                ),
                verified_candidate_rank=0,
            ))
    return rows


def test_induces_content_specific_function_and_executes_without_identity():
    program = induce_source_function_program(
        _examples(), source_receipts_sha256="source-tuples",
        minimum_authentic_minus_shuffled=0.0,
    )
    validate_source_function_program(program)
    assert program["status"] == QUALIFIED
    assert sum(value > 0 for value in function_weights(program)) >= 2
    assert program["source_identity_used_as_feature"] is False
    assert "source_game" not in program
    effects = [
        dict(zip(TYPED_EFFECTS, (1.0, 0.0, 1.0, 0.0))),
        dict(zip(TYPED_EFFECTS, (0.0, 0.0, 0.0, 0.0))),
    ]
    order, reason = function_trial_order(program, effects)
    assert reason is None
    assert order == (0, 1)


def test_cross_batch_calibration_can_only_remove_frozen_function():
    program = induce_source_function_program(
        _examples(), source_receipts_sha256="source-tuples",
        minimum_authentic_minus_shuffled=0.0,
    )
    rejected = recalibrate_source_function_program(
        program,
        calibration_metrics={
            "authentic": {"accuracy": 0.25, "varying_effect_fraction": 1.0},
            "shuffled_effect_binding": {"accuracy": 0.25},
        },
        calibration_receipt_sha256="reserve",
    )
    validate_source_function_program(rejected)
    assert rejected["status"] == ABSTAINING
    assert rejected["source_function"] == program["source_function"]
    effects = [{effect: value for effect in TYPED_EFFECTS} for value in (0.2, 0.8)]
    assert function_trial_order(rejected, effects) == (
        (), "SOURCE_DOMAIN_FUNCTION_NOT_QUALIFIED",
    )
