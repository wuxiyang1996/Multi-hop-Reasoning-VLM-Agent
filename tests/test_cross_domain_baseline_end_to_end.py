"""End-to-end: raw source episodes -> shared labels -> three artifacts -> target prompts.

This exercises the whole comparison in one place and asserts the fairness
properties the protocol claims, so that swapping in the real collected episodes
is a drop-in rather than a new integration.
"""

from __future__ import annotations

import json

import pytest

from motif_transfer.alfworld_llm_decision import decide_alfworld_action
from motif_transfer.cross_domain_memory_baselines import (
    MemoryBaseline,
    canonical_source_episodes,
    induce_memory_artifact,
    validate_memory_artifact,
)
from motif_transfer.cross_domain_memory_runtime import (
    MEMORY_PAYLOAD_KEY,
    MemoryAugmentedDecisionBackend,
)
from motif_transfer.source_outcome_evaluator import (
    FrozenJudgeEvaluator,
    benchmark_predicate_from_config,
    label_source_payload,
    load_outcome_config,
)


def _raw_source():
    """Six unlabelled arcade episodes, shaped like the real exporter's output."""
    episodes = []
    for index, domain in enumerate([
        "tetris", "tetris", "candy_crush",
        "gymv_strider", "gymv_columns", "gymv_thunder_force_iii",
    ]):
        episodes.append({
            "episode_id": f"{domain}:e{index}",
            "source_domain": domain,
            "official_success": None,
            "terminated": index % 2 == 0,
            "truncated": index % 2 == 1,
            "total_reward": float(index * 10),
            "steps": [
                {
                    "receipt_id": f"{domain}-e{index}-r{step}",
                    "step": step,
                    "observation": f"uncertain board state at {step}",
                    "action": "probe" if step == 0 else "commit",
                    "next_observation": f"resolved board state at {step + 1}",
                    "reward": float(step),
                    "terminal": step == 1,
                }
                for step in range(2)
            ],
        })
    return {"episodes": episodes}


class JudgeBackend:
    """Calls half the episodes successful, so every method has something to read."""

    identity = {"model": "e2e-judge"}
    last_usage = {"total_tokens": 4}

    def __init__(self):
        self.calls = 0

    def complete(self, role, system, payload):
        self.calls += 1
        verdict = "SUCCESS" if self.calls % 2 else "FAILURE"
        return json.dumps({"verdict": verdict, "reason": "e2e"})


class InductionBackend:
    identity = {"model": "e2e-inducer"}
    last_usage = {"total_tokens": 4}

    def __init__(self, method):
        self.method = method
        self.calls = 0
        self.seen_episode_ids = []

    def complete(self, role, system, payload):
        self.calls += 1
        episodes = payload["episodes"]
        self.seen_episode_ids.extend(row["episode_id"] for row in episodes)
        kind = {
            MemoryBaseline.EXPEL: "INSIGHT",
            MemoryBaseline.AWM: "WORKFLOW",
        }.get(self.method, "STRATEGY")
        return json.dumps({"items": [{
            "title": f"{self.method.value}-{self.calls}",
            "content": "Resolve the uncertain part of the state before committing.",
            "applicability": "When the observation is ambiguous.",
            "kind": kind,
            "source_episode_ids": [row["episode_id"] for row in episodes],
            "evidence_receipt_ids": [
                step["receipt_id"] for row in episodes for step in row["steps"]
            ],
        }]})


class Embedding:
    identity = {"model": "e2e-embed"}

    def embed(self, texts):
        return [[float(len(text) % 9 + 1), 2.0] for text in texts]


class DecisionBackend:
    identity = {"model": "e2e-decision"}
    last_usage = {"total_tokens": 6}

    def __init__(self):
        self.requests = []

    def complete(self, role, system, payload):
        self.requests.append(json.loads(json.dumps(payload)))
        return json.dumps({"action_index": 0, "reason": "e2e"})


@pytest.fixture(scope="module")
def labelled():
    config = load_outcome_config("configs/cross_domain_source_outcome_v1.json")
    return label_source_payload(
        _raw_source(),
        benchmark_predicate=benchmark_predicate_from_config(config),
        shared_evaluator=FrozenJudgeEvaluator(JudgeBackend()),
    )


@pytest.fixture(scope="module")
def artifacts(labelled):
    return {
        method: induce_memory_artifact(method, labelled, InductionBackend(method))
        for method in MemoryBaseline
    }


def test_shared_labels_leave_no_episode_silently_failed(labelled):
    assert labelled["outcome_census"]["UNKNOWN"] == 0
    assert labelled["outcome_census"]["SUCCESS"] == 3
    assert labelled["outcome_census"]["FAILURE"] == 3
    # Every label came from the one shared evaluator, not from per-method rules.
    assert labelled["outcome_authority_census"]["SHARED_EVALUATOR"] == 6


def test_every_method_binds_to_one_superset_and_reads_its_own_projection(artifacts):
    hashes = {row["source_superset_sha256"] for row in artifacts.values()}
    assert len(hashes) == 1

    eligible = {
        method: set(row["source_projection"]["eligible_episode_ids"])
        for method, row in artifacts.items()
    }
    assert len(eligible[MemoryBaseline.AWM]) == 3
    assert eligible[MemoryBaseline.EXPEL] == eligible[MemoryBaseline.REASONING_BANK]
    assert eligible[MemoryBaseline.AWM] < eligible[MemoryBaseline.EXPEL]

    for artifact in artifacts.values():
        validate_memory_artifact(artifact)
        assert artifact["online_memory_updates_allowed"] is False


@pytest.mark.parametrize("method", list(MemoryBaseline))
def test_memory_reaches_the_alfworld_decision_prompt_without_taking_authority(
    artifacts, method,
):
    inner = DecisionBackend()
    wrapped = MemoryAugmentedDecisionBackend(
        inner, artifact=artifacts[method], domain="alfworld",
        embedding_backend=Embedding(), top_k=2,
    )
    commands = ("go to cabinet 1", "take mug 1 from cabinet 1", "look")
    decision = decide_alfworld_action(
        wrapped,
        observation_text="You see an ambiguous countertop.",
        task_goal="put a clean mug in the coffeemachine",
        admissible_commands=commands,
    )
    # The environment still owns the action set.
    assert decision.action in commands
    sent = inner.requests[0]
    assert MEMORY_PAYLOAD_KEY in sent
    assert sent[MEMORY_PAYLOAD_KEY]["method"] == method.value
    assert sent["admissible_commands"] == list(commands)
    # Memory is advice, not an action: it must not name any admissible command.
    injected = json.dumps(sent[MEMORY_PAYLOAD_KEY]).casefold()
    assert not any(command.casefold() in injected for command in commands)
    assert wrapped.receipt()["decision_calls_augmented"] == 1


def test_target_only_arm_is_the_same_loop_unwrapped(artifacts):
    inner = DecisionBackend()
    decide_alfworld_action(
        inner,
        observation_text="You see an ambiguous countertop.",
        task_goal="put a clean mug in the coffeemachine",
        admissible_commands=("look", "go to cabinet 1"),
    )
    assert MEMORY_PAYLOAD_KEY not in inner.requests[0]


def test_no_source_episode_leaks_into_a_method_that_may_not_read_it(labelled, artifacts):
    episodes = {row.episode_id: row for row in canonical_source_episodes(labelled)}
    for method, artifact in artifacts.items():
        for episode_id in artifact["source_episode_ids"]:
            outcome = episodes[episode_id].outcome
            if method is MemoryBaseline.AWM:
                assert outcome == "SUCCESS"
            else:
                assert outcome in {"SUCCESS", "FAILURE"}
