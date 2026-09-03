from scripts.calibrate_agqa_query_grounder_v3_threshold import (
    select_threshold,
    wilson_lower,
)


def test_wilson_lower_is_conservative_and_monotone_for_fixed_total():
    assert 0 < wilson_lower(30, 40) < .75
    assert wilson_lower(31, 40) > wilson_lower(30, 40)


def test_selection_maximizes_coverage_then_uses_higher_threshold_tie_break():
    constraints = {
        "unique_supported_count_minimum": 40,
        "unique_supported_precision_wilson_95_lower_bound_minimum": .6,
        "unique_supported_coverage_minimum": .2,
    }
    curve = [
        {"threshold": .6, "supported": 60, "coverage": .30, "precision_wilson_95_lower": .59},
        {"threshold": .625, "supported": 50, "coverage": .25, "precision_wilson_95_lower": .61},
        {"threshold": .65, "supported": 50, "coverage": .25, "precision_wilson_95_lower": .64},
        {"threshold": .675, "supported": 39, "coverage": .195, "precision_wilson_95_lower": .7},
    ]
    assert select_threshold(curve, constraints)["threshold"] == .65
