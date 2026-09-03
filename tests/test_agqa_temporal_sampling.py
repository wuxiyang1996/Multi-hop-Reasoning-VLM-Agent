import pytest

from motif_transfer.agqa_temporal_sampling import native_index_views


def test_uniform48_views_are_persisted_in_original_video_coordinates() -> None:
    native = tuple(round(index * 907 / 47) for index in range(48))
    proxy_views = (
        tuple(range(0, 32)),
        tuple(range(8, 40)),
        tuple(range(16, 48)),
    )

    views = native_index_views(native, proxy_views)

    assert views[0][0] == 0
    assert views[0][-1] == native[31]
    assert views[1][0] == native[8]
    assert views[1][-1] == native[39]
    assert views[2][0] == native[16]
    assert views[2][-1] == 907
    assert views != proxy_views


def test_native_views_allow_repeated_frames_for_short_videos() -> None:
    assert native_index_views((0, 0, 1, 1), ((0, 1), (2, 3))) == (
        (0, 0),
        (1, 1),
    )


@pytest.mark.parametrize(
    ("native", "views"),
    [
        ((), ((0,),)),
        ((1, 0), ((0,),)),
        ((0, 1), ()),
        ((0, 1), ((),)),
        ((0, 1), ((2,),)),
    ],
)
def test_native_view_mapping_rejects_invalid_coordinates(native, views) -> None:
    with pytest.raises(ValueError):
        native_index_views(native, views)
