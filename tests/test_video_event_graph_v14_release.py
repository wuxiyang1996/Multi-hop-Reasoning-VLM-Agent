from __future__ import annotations

from scripts.audit_video_event_graph_v14 import audit


def test_portable_video_event_graph_formal_evidence() -> None:
    report = audit()
    assert report["status"] == "PORTABLE_CLEVRER_V14_FORMAL_EVIDENCE_VALIDATED"
    assert report["samples"] == 720
    assert report["success_delta"] == 22
    assert report["paired_wins"] == 27
    assert report["paired_losses"] == 5
    assert report["exact_two_sided_p"] < 0.001
    assert report["all_formal_gates_passed"]
    assert report["target_native_action_authority"]
