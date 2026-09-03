from scripts.calibrate_agqa_track_verified_threshold import (
    select_threshold,
    wilson_lower,
)


def test_selects_maximum_eligible_coverage() -> None:
    curve = [
        {"threshold": 0.7, "supported": 40, "coverage": 0.4,
         "precision_wilson_95_lower": 0.59},
        {"threshold": 0.75, "supported": 30, "coverage": 0.3,
         "precision_wilson_95_lower": 0.63},
        {"threshold": 0.8, "supported": 20, "coverage": 0.2,
         "precision_wilson_95_lower": 0.70},
    ]
    selected = select_threshold(curve, {
        "unique_supported_count_minimum": 20,
        "unique_supported_coverage_minimum": 0.2,
        "unique_supported_precision_wilson_95_lower_bound_minimum": 0.6,
    })
    assert selected["threshold"] == 0.75


def test_wilson_lower_is_conservative() -> None:
    assert 0 < wilson_lower(70, 100) < 0.70
