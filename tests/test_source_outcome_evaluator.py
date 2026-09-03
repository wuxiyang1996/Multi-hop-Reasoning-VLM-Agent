from __future__ import annotations

import json

import pytest

from motif_transfer.cross_domain_memory_baselines import (
    MemoryBaseline,
    OutcomeAuthority,
    OutcomeLabel,
    canonical_source_episodes,
    source_projection,
)
from motif_transfer.source_outcome_evaluator import (
    FrozenJudgeEvaluator,
    OutcomeRuleError,
    benchmark_predicate_from_config,
    label_source_payload,
    load_outcome_config,
)

CONFIG_PATH = "configs/cross_domain_source_outcome_v1.json"


def _episode(episode_id, *, official=None, reward=0.0, terminated=False, truncated=True):
    return {
        "episode_id": episode_id,
        "source_domain": "tetris",
        "official_success": official,
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": reward,
        "steps": [{
            "receipt_id": f"{episode_id}-r0",
            "step": 0,
            "observation": "stack_h=4 holes=0 lines=0",
            "action": "L-up col0",
            "next_observation": "stack_h=6 holes=0 lines=0",
            "reward": reward,
            "terminal": terminated,
        }],
    }


class FakeJudgeBackend:
    def __init__(self, verdicts):
        self.verdicts = dict(verdicts)
        self.seen = []
        self.last_usage = {"total_tokens": 5}

    @property
    def identity(self):
        return {"model": "frozen-judge-test"}

    def complete(self, role, system, payload):
        assert role == "source_outcome_judge"
        # The judge must never be told about a target domain.
        assert "webshop" not in json.dumps(payload).lower()
        self.seen.append(payload)
        verdict = self.verdicts.get(payload["source_domain"], "UNKNOWN")
        return json.dumps({"verdict": verdict, "reason": "test"})


def test_shipped_config_declares_every_source_domain():
    config = load_outcome_config(CONFIG_PATH)
    predicate = benchmark_predicate_from_config(config)
    for domain in config["predicates"]:
        assert predicate({"source_domain": domain, "total_reward": 10.0}) is None


def test_undeclared_domain_is_an_error_not_a_silent_abstention():
    predicate = benchmark_predicate_from_config(
        {"predicates": {"tetris": {"kind": "ABSTAIN"}}}
    )
    with pytest.raises(OutcomeRuleError, match="no benchmark predicate declared"):
        predicate({"source_domain": "gymv_strider"})


def test_generic_reward_threshold_must_be_declared_explicitly():
    with pytest.raises(OutcomeRuleError, match="declared threshold"):
        benchmark_predicate_from_config(
            {"predicates": {"tetris": {"kind": "TOTAL_REWARD_AT_LEAST"}}}
        )(_episode("e0"))


def test_terminated_is_failure_rule_abstains_on_survival():
    predicate = benchmark_predicate_from_config(
        {"predicates": {"tetris": {"kind": "TERMINATED_IS_FAILURE"}}}
    )
    assert predicate(_episode("e0", terminated=True, truncated=False)) is False
    assert predicate(_episode("e1", terminated=False, truncated=True)) is None


def test_official_outcome_wins_and_judge_is_not_consulted():
    backend = FakeJudgeBackend({"tetris": "SUCCESS"})
    judge = FrozenJudgeEvaluator(backend)
    labelled = label_source_payload(
        {"episodes": [_episode("e0", official=False)]}, shared_evaluator=judge,
    )
    assert labelled["episodes"][0]["outcome"] == OutcomeLabel.FAILURE.value
    assert labelled["episodes"][0]["outcome_authority"] == OutcomeAuthority.OFFICIAL.value
    assert backend.seen == [], "the judge must not be paid for an already-official episode"


def test_judge_resolves_only_what_earlier_authorities_left_open():
    backend = FakeJudgeBackend({"tetris": "SUCCESS"})
    judge = FrozenJudgeEvaluator(backend)
    labelled = label_source_payload(
        {"episodes": [_episode("e0"), _episode("e1", official=True)]},
        benchmark_predicate=benchmark_predicate_from_config(
            {"predicates": {"tetris": {"kind": "ABSTAIN"}}}
        ),
        shared_evaluator=judge,
    )
    outcomes = {row["episode_id"]: row for row in labelled["episodes"]}
    assert outcomes["e0"]["outcome_authority"] == OutcomeAuthority.SHARED_EVALUATOR.value
    assert outcomes["e1"]["outcome_authority"] == OutcomeAuthority.OFFICIAL.value
    assert labelled["outcome_census"] == {"SUCCESS": 2, "FAILURE": 0, "UNKNOWN": 0}
    assert len(judge.receipts) == 1


def test_abstaining_judge_leaves_unknown_rather_than_failure():
    judge = FrozenJudgeEvaluator(FakeJudgeBackend({"tetris": "UNKNOWN"}))
    labelled = label_source_payload({"episodes": [_episode("e0")]}, shared_evaluator=judge)
    assert labelled["episodes"][0]["outcome"] == OutcomeLabel.UNKNOWN.value
    assert labelled["outcome_census"]["FAILURE"] == 0


def test_labelled_payload_is_hash_bound_and_refuses_relabelling():
    judge = FrozenJudgeEvaluator(FakeJudgeBackend({"tetris": "SUCCESS"}))
    labelled = label_source_payload({"episodes": [_episode("e0")]}, shared_evaluator=judge)
    assert labelled["source_labelled_sha256"]
    with pytest.raises(ValueError, match="already labelled"):
        label_source_payload(labelled, shared_evaluator=judge)


def test_labels_flow_into_identical_projections_for_every_method():
    judge = FrozenJudgeEvaluator(
        FakeJudgeBackend({"tetris": "SUCCESS", "gymv_strider": "FAILURE"})
    )
    failure = _episode("e1")
    failure["source_domain"] = "gymv_strider"
    labelled = label_source_payload(
        {"episodes": [_episode("e0"), failure]}, shared_evaluator=judge,
    )
    episodes = canonical_source_episodes(labelled)
    assert source_projection(MemoryBaseline.AWM, episodes)["eligible_episode_ids"] == ["e0"]
    for method in (MemoryBaseline.EXPEL, MemoryBaseline.REASONING_BANK):
        assert source_projection(method, episodes)["eligible_episode_ids"] == ["e0", "e1"]
