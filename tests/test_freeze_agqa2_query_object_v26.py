from scripts.freeze_agqa2_query_object_v26_reserve import (
    PER_GROUP,
    TOTAL_ROWS,
    _at_least_five_win_probability,
)


def test_v26_power_and_fixed_sample_size_are_frozen():
    assert PER_GROUP == 40
    assert TOTAL_ROWS == 120
    assert _at_least_five_win_probability() > 0.90
