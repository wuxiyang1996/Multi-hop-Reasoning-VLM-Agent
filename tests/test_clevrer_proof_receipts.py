from __future__ import annotations

from types import SimpleNamespace

from motif_transfer.clevrer_proof_receipts import (
    PROOF_FEATURE_NAMES,
    execute_with_receipt,
    paired_proof_features,
)


class FakeExecutor:
    def __init__(self, collision_count: int) -> None:
        events = [
            {"type": "collision", "object": [0, 1], "frame": index * 5}
            for index in range(collision_count)
        ]
        self.existing_events = events
        self.unseens = []
        self.sim = SimpleNamespace(cf_events={})
        self.modules = {
            "events": {"nargs": 0, "func": lambda: events},
            "exist": {"nargs": 1, "func": lambda values: "yes" if values else "no"},
        }


def test_execute_with_receipt_matches_postfix_answer() -> None:
    receipt = execute_with_receipt(FakeExecutor(1), ["events", "exist"])
    assert receipt["answer"] == "yes"
    assert [step["module"] for step in receipt["steps"]] == ["events", "exist"]


def test_paired_proof_features_capture_event_divergence() -> None:
    features, receipts = paired_proof_features(
        FakeExecutor(1), FakeExecutor(0), ["exist"], [["events"]],
    )
    assert len(features) == len(PROOF_FEATURE_NAMES)
    assert features[0] > 0
    assert features[10] == 0.1
    assert features[11] == 0.0
    assert receipts[0]["explicit_answer"] == "yes"
    assert receipts[0]["trajectory_answer"] == "no"
