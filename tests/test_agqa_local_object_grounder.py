from motif_transfer.agqa_active_frame_grounder import parse_operand_receipt
from motif_transfer.agqa_local_object_grounder import (
    AGQALocalDetection,
    AGQALocalObjectReceipt,
    inspection_indices,
    refine_query_object_receipt,
)


def _receipt(objects):
    return parse_operand_receipt({
        "operand_role": "A",
        "requested_operand": "a person putting down an unknown object",
        "observations": [
            {
                "occurrence_id": f"O{index}",
                "label": "putting down",
                "subject": "person",
                "predicate": "putting down",
                "object": name,
                "observability": "OBSERVED",
                "start_frame": start,
                "end_frame": end,
                "evidence_frames": [start, end],
                "confidence": confidence,
                "uncertainties": [],
            }
            for index, (name, start, end, confidence) in enumerate(objects)
        ],
        "coverage": "SUFFICIENT",
        "uncertainties": [],
    }, expected_role="A",
       expected_operand="a person putting down an unknown object",
       frame_count=48)


def _detector(*detections):
    return AGQALocalObjectReceipt(
        detector="test-coco-detector",
        model_sha256="model-hash",
        inspected_frame_indices=tuple(sorted({row.frame_index for row in detections})),
        detections=tuple(detections),
        question_read=False,
        answer_read=False,
        functional_program_read=False,
        answer_candidates_read=False,
        source_identity_read=False,
        receipt_sha256="receipt-hash",
    )


def test_pan_and_independent_bowl_canonicalize_to_broad_dish():
    receipt = _receipt([
        ("pan", 3, 6, 0.9),
        ("unknown object", 25, 28, 0.8),
    ])
    detector = _detector(AGQALocalDetection(
        frame_index=4,
        label="bowl",
        confidence=0.1,
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
    ))
    refined, markers = refine_query_object_receipt(receipt, detector)
    assert [row.object for row in refined.observations] == ["dish"]
    assert markers == ("O0:VLM_PAN_PLUS_COCO_BOWL_TO_DISH",)
    assert refined.canonicalizations == markers


def test_object_canonicalization_requires_same_interval_corroboration():
    receipt = _receipt([("pan", 3, 6, 0.9)])
    detector = _detector(AGQALocalDetection(
        frame_index=20,
        label="bowl",
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
    ))
    refined, markers = refine_query_object_receipt(receipt, detector)
    assert [row.object for row in refined.observations] == ["pan"]
    assert markers == ()


def test_generic_only_receipt_remains_unchanged_and_cannot_become_dish():
    receipt = _receipt([("unknown object", 3, 6, 0.9)])
    detector = _detector(AGQALocalDetection(
        frame_index=4,
        label="bowl",
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
    ))
    refined, markers = refine_query_object_receipt(receipt, detector)
    assert refined.receipt_sha256 == receipt.receipt_sha256
    assert markers == ()


def test_detector_receipt_declares_no_privileged_inputs():
    detector = _detector(AGQALocalDetection(
        frame_index=4,
        label="bowl",
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
    ))
    payload = detector.as_dict()
    assert not payload["question_read"]
    assert not payload["answer_read"]
    assert not payload["functional_program_read"]
    assert not payload["answer_candidates_read"]
    assert not payload["source_identity_read"]


def test_inspection_indices_are_limited_and_chronological():
    receipt = _receipt([("pan", 3, 20, 0.9)])
    indices = inspection_indices(receipt, maximum=5)
    assert indices == tuple(sorted(indices))
    assert len(indices) == 5
    assert indices[0] == 3
    assert indices[-1] == 20
