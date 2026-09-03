import math

import pytest

from motif_transfer.agqa_track_verified_candidate import (
    track_persistence_score,
    track_verified_candidate_score,
)


def test_persistence_is_bounded_and_monotone() -> None:
    values = [track_persistence_score(count, 64) for count in (1, 2, 8, 64)]
    assert 0 < values[0] < values[1] < values[2] < values[3] == 1


def test_candidate_score_is_equal_weight_geometric_mean() -> None:
    value = track_verified_candidate_score(0.8, 0.9, 8, 64)
    expected_persistence = math.log1p(8) / math.log1p(64)
    assert value.track_persistence == pytest.approx(expected_persistence)
    assert value.score == pytest.approx((0.8 * 0.9 * expected_persistence) ** (1 / 3))


@pytest.mark.parametrize(
    "args",
    [(-0.1, 0.5, 1, 64), (1.1, 0.5, 1, 64),
     (0.5, -0.1, 1, 64), (0.5, 1.1, 1, 64),
     (0.5, 0.5, 0, 64), (0.5, 0.5, 65, 64),
     (0.5, 0.5, 1, 0)],
)
def test_invalid_evidence_fails_closed(args) -> None:
    with pytest.raises(ValueError):
        track_verified_candidate_score(*args)
