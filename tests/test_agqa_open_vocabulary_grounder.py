from motif_transfer.agqa_open_vocabulary_grounder import Detection, _associate, _iou


def test_iou_and_track_association_preserve_physical_instances() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    detections = (
        Detection(0, "book", .8, (0, 0, 10, 10)),
        Detection(1, "book", .9, (1, 0, 11, 10)),
        Detection(1, "book", .7, (50, 50, 60, 60)),
        Detection(0, "chair", .6, (20, 20, 40, 40)),
    )
    tracks = _associate(detections, maximum_tracks=12)
    assert [(x.canonical_label, x.evidence_frames) for x in tracks] == [
        ("book", (0, 1)), ("chair", (0,)), ("book", (1,))
    ]
