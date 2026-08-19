from scripts.freeze_agqa2_query_object_v28_reserve import (
    PER_GROUP,
    TOTAL_ROWS,
    _power_probability,
)


def test_v28_uses_the_same_powered_fixed_sample_design():
    assert PER_GROUP == 40
    assert TOTAL_ROWS == 120
    assert _power_probability() > 0.90
