from __future__ import annotations

import json
from types import SimpleNamespace

from harness.reasoning_event_log import ReasoningEventKind, ReasoningEventRecorder
from harness.source_evidence_store import write_source_evidence_batch


def test_source_evidence_store_writes_four_batch_files_not_per_frame(tmp_path) -> None:
    recorder = ReasoningEventRecorder("e")
    for kind in (
        ReasoningEventKind.RESET,
        ReasoningEventKind.OBSERVATION,
        ReasoningEventKind.AGENT_PROPOSAL_SET,
        ReasoningEventKind.AGENT_RESPONSE,
        ReasoningEventKind.PARSED_DECISION,
        ReasoningEventKind.POLICY_TRANSFORM,
        ReasoningEventKind.NATIVE_ADMISSIBILITY,
        ReasoningEventKind.AGENT_DECISION,
        ReasoningEventKind.ENVIRONMENT_STEP,
        ReasoningEventKind.NATIVE_DELTA,
        ReasoningEventKind.OFFICIAL_STOP,
    ):
        recorder.append(kind, {"kind": kind.value})
    result = SimpleNamespace(
        episode_id="e", game="g", steps=1, total_reward=0.0,
        terminated=True, truncated=False,
        reasoning_event_log=recorder.to_dict(),
    )
    manifest = write_source_evidence_batch(
        tmp_path, [result], manifest_metadata={"model": "m"},
    )
    assert manifest["protocol_failures"] == {}
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "episodes.jsonl", "events.jsonl", "manifest.json",
    ]
    episode = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert episode["reasoning_log_sha256"] == recorder.to_dict()["log_sha256"]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 11
