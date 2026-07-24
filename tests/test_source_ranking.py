import json

import pytest

from motif_transfer.contracts import Observation, SourcePolicyStepRecord, SourceTransitionReceipt
from motif_transfer.source_ranking import (
    SourceSkillCandidate,
    SourceSkillRanker,
    build_native_segment_ranking_prompt,
    segment_native_policy,
)


def _record(step, skill_hash, skill_id="skill"):
    before = Observation({"observable_state": f"s{step}"}, ("left", "right"))
    after = Observation({"observable_state": f"s{step + 1}"}, ("left", "right"))
    transition = SourceTransitionReceipt.create(
        before,
        episode_id="episode",
        step=step,
        selected_skill_hash=skill_hash,
        action_response_hash=f"response-{step}",
        action="left",
        action_origin="AGENT",
        policy_adapter="action_taking",
        after=after,
        reward=0,
    )
    return SourcePolicyStepRecord(
        "episode", step, before, skill_id if skill_hash else None, skill_hash,
        "untrusted reasoning", f"response-{step}", "left", "AGENT",
        "action_taking", after, 0, transition,
    )


def test_segmentation_uses_exact_skill_id_not_dynamic_guidance_hash():
    records = (
        _record(0, "guidance-a0", "A"),
        _record(1, "guidance-a1", "A"),
        _record(2, "guidance-b0", "B"),
    )
    segmented = segment_native_policy(records)
    assert [tuple(row.step for row in span) for _, span in segmented] == [(0, 1), (2,)]
    assert all(receipt.validate() for receipt, _ in segmented)
    assert segmented[0][0].segmentation_rule == "MAXIMAL_RECORDED_SKILL_ID_RUN_V2"


class Backend:
    identity = {"adapter": "frozen-segment", "hash": "abc"}

    def __init__(self, response):
        self.response = response
        self.prompt = ""

    def complete_prompt(self, role, prompt):
        assert role == "segment"
        self.prompt = prompt
        return self.response


def test_native_ranker_requires_all_candidates_and_binds_receipt():
    records = (_record(0, "a"), _record(1, "a"))
    segment, span = segment_native_policy(records)[0]
    candidates = (
        SourceSkillCandidate("EXPLORE", "inspect state"),
        SourceSkillCandidate("COMMIT", "take scoring action"),
    )
    backend = Backend(json.dumps({
        "ranking": ["COMMIT", "EXPLORE"], "reasoning": "observed actions",
    }))
    receipt = SourceSkillRanker(backend).rank(segment, span, candidates)
    assert receipt.validate()
    assert receipt.ranking == ("COMMIT", "EXPLORE")
    assert "Rank ALL candidate skills" in backend.prompt
    assert backend.prompt == build_native_segment_ranking_prompt(segment, span, candidates)


def test_native_ranker_rejects_partial_or_unknown_ranking():
    records = (_record(0, None),)
    segment, span = segment_native_policy(records)[0]
    candidates = (SourceSkillCandidate("A", ""), SourceSkillCandidate("B", ""))
    backend = Backend('{"ranking":["A"],"reasoning":""}')
    with pytest.raises(ValueError, match="every candidate exactly once"):
        SourceSkillRanker(backend).rank(segment, span, candidates)
