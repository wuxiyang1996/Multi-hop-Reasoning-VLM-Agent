import pytest

from motif_transfer.video_temporal_localization import (
    absolute_temporal_window,
    normalize_temporal_window,
    parse_temporal_localization,
)


def test_temporal_window_clamps_about_proposed_centre():
    assert normalize_temporal_window(
        .4, .45, minimum_width=.2, maximum_width=.6,
    ) == pytest.approx((.325, .525))
    assert normalize_temporal_window(
        .05, .95, minimum_width=.2, maximum_width=.6,
    ) == pytest.approx((.2, .8))


def test_full_context_and_absolute_mapping():
    parsed = parse_temporal_localization({
        "window_fraction": [.2, .3],
        "requires_full_context": True,
        "anchor_description": "compare the beginning and end",
        "sensor_reliability": .7,
    }, minimum_width=.2, maximum_width=.6)
    assert parsed["window_fraction"] == [0.0, 1.0]
    assert absolute_temporal_window(10, 30, [.25, .75]) == (15, 25)


def test_temporal_localization_rejects_invalid_schema():
    with pytest.raises(ValueError):
        parse_temporal_localization({
            "window_fraction": [0, 1],
            "requires_full_context": False,
            "anchor_description": "",
            "sensor_reliability": .8,
        }, minimum_width=.2, maximum_width=.6)
