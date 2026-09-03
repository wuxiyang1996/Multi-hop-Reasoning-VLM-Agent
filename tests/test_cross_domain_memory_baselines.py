from __future__ import annotations

import json

import pytest

from motif_transfer.contracts import AdvisoryVerdict
from motif_transfer.cross_domain_memory_baselines import (
    CrossDomainMemoryAdvisor,
    InsufficientEligibleSourceError,
    MemoryBaseline,
    OutcomeAuthority,
    OutcomeLabel,
    TargetDomain,
    adapt_target_context,
    bind_memory_artifact_to_target,
    canonical_source_episodes,
    induce_memory_artifact,
    resolve_source_outcome,
    retrieve_memory_items,
    source_projection,
    source_abstraction_audit,
    validate_memory_artifact,
)


SOURCE = {
    "episodes": [
        {
            "episode_id": "game-e0",
            "source_domain": "source_game",
            "official_success": True,
            "steps": [
                {
                    "receipt_id": "r0",
                    "step": 0,
                    "observation": "uncertain route",
                    "action": "probe",
                    "next_observation": "route identified",
                    "reward": 0,
                    "terminal": False,
                },
                {
                    "receipt_id": "r1",
                    "step": 1,
                    "observation": "route identified",
                    "action": "commit",
                    "next_observation": "complete",
                    "reward": 1,
                    "terminal": True,
                },
            ],
        },
        {
            "episode_id": "game-e1",
            "source_domain": "source_game",
            "official_success": False,
            "steps": [
                {
                    "receipt_id": "r2",
                    "step": 0,
                    "observation": "uncertain route",
                    "action": "commit",
                    "next_observation": "failure",
                    "reward": -1,
                    "terminal": True,
                }
            ],
        },
    ]
}


class FakeCompletion:
    def __init__(self, method: MemoryBaseline):
        self.method = method
        self.calls = 0
        self.last_usage = {"total_tokens": 10}

    @property
    def identity(self):
        return {"model": "open-weight-test", "method": self.method.value}

    def complete(self, role, system, payload):
        self.calls += 1
        episodes = payload["episodes"]
        episode_ids = [row["episode_id"] for row in episodes]
        receipts = [step["receipt_id"] for row in episodes for step in row["steps"]]
        if self.method == MemoryBaseline.EXPEL:
            kind = "INSIGHT"
        elif self.method == MemoryBaseline.AWM:
            kind = "WORKFLOW"
        else:
            kind = "STRATEGY" if episodes[0]["official_success"] else "PITFALL"
        return json.dumps({"items": [{
            "title": f"item-{self.calls}",
            "content": "Verify uncertain state before irreversible commitment.",
            "applicability": "Use only when live evidence shows uncertainty.",
            "kind": kind,
            "source_episode_ids": episode_ids,
            "evidence_receipt_ids": receipts,
        }]})


class FakeEmbedding:
    @property
    def identity(self):
        return {"model": "open-embedding-test"}

    def embed(self, texts):
        return [[float("uncertain" in text), float(len(text) % 7 + 1)] for text in texts]


class FakeBinder:
    identity = {"model": "open-weight-binder-test"}
    last_usage = {"total_tokens": 11}

    def complete(self, role, system, payload):
        if role == "memory_binding_verifier":
            return json.dumps({"decisions": [{
                "candidate_ref": row["candidate_ref"],
                "admit": True,
                "supporting_example_ids": [
                    payload["adaptation_examples"][0]["example_id"]
                ] if payload["adaptation_examples"] else [],
                "reason": "The target example exhibits an observable state check.",
            } for row in payload["candidates"]]})
        assert role == "memory_binder"
        return json.dumps({"items": [{
            "source_item_ref": row["source_item_ref"],
            "abstain": False,
            "title": "Verify state before commitment",
            "content": "Resolve uncertain task state before an irreversible commitment.",
            "applicability": "When the target observation leaves a required condition uncertain.",
            "information_to_check": "Whether the required condition is visibly satisfied.",
            "expected_observation": "The relevant condition becomes unambiguous.",
            "contradiction_condition": "The live state disproves the assumed condition.",
            "stop_condition": "Stop using this memory once the condition is resolved.",
        } for row in payload["source_items"]]})


class LeakyBinder(FakeBinder):
    def complete(self, role, system, payload):
        value = json.loads(super().complete(role, system, payload))
        value["items"][0]["content"] = "Watch the game score before committing."
        return json.dumps(value)


@pytest.mark.parametrize("method", list(MemoryBaseline))
def test_all_three_methods_build_frozen_source_only_artifacts(method):
    backend = FakeCompletion(method)
    artifact = induce_memory_artifact(method, SOURCE, backend)
    validate_memory_artifact(artifact)
    assert artifact["method"] == method.value
    assert artifact["source_domains"] == ["source_game"]
    assert artifact["online_memory_updates_allowed"] is False
    assert backend.calls == (2 if method == MemoryBaseline.REASONING_BANK else 1)


def test_source_schema_rejects_noncontiguous_and_duplicate_receipts():
    bad = json.loads(json.dumps(SOURCE))
    bad["episodes"][0]["steps"][1]["step"] = 7
    with pytest.raises(ValueError, match="not contiguous"):
        canonical_source_episodes(bad)
    bad = json.loads(json.dumps(SOURCE))
    bad["episodes"][1]["steps"][0]["receipt_id"] = "r0"
    with pytest.raises(ValueError, match="duplicated"):
        canonical_source_episodes(bad)


@pytest.mark.parametrize("domain", list(TargetDomain))
def test_four_target_adapters_are_outcome_blind(domain):
    context = adapt_target_context(
        domain,
        task="finish the task",
        observation={"observation": "current visible state", "ignored": "x"},
        native_actions=("native-a", "native-b"),
        history=({"action": "native-a", "next_observation": "changed"},),
        proposal={"action": "native-b"},
    )
    assert context["target_domain"] == domain.value
    assert context["native_actions"] == ["native-a", "native-b"]
    assert context["query_sha256"]


def test_target_adapter_rejects_gold_or_official_outcomes():
    with pytest.raises(ValueError, match="forbidden outcome"):
        adapt_target_context(
            "tirbench", task="q", observation={"prompt": "q", "answer": "secret"}
        )
    with pytest.raises(ValueError, match="forbidden outcome"):
        adapt_target_context(
            "alfworld", task="q", observation={"observation": "x"},
            history=({"official_success": True},),
        )


def test_retrieval_is_cross_domain_frozen_and_action_free():
    artifact = induce_memory_artifact(
        MemoryBaseline.REASONING_BANK, SOURCE, FakeCompletion(MemoryBaseline.REASONING_BANK)
    )
    target = adapt_target_context(
        "webshop", task="find product", observation={"observation": "uncertain options"},
        native_actions=("search", "click"), proposal={"action": "search"},
    )
    retrieval = retrieve_memory_items(artifact, target, FakeEmbedding(), top_k=2)
    assert retrieval["online_memory_updated"] is False
    advisory = CrossDomainMemoryAdvisor(retrieval).advisory()
    assert advisory.verdict == AdvisoryVerdict.ADMIT
    assert advisory.current_role == "CROSS_DOMAIN_REASONING_BANK"
    assert set(advisory.evidence_receipt_ids) <= {"r0", "r1", "r2"}
    assert "search" not in advisory.information_need


def test_retrieval_rejects_same_domain_memory():
    source = json.loads(json.dumps(SOURCE))
    for episode in source["episodes"]:
        episode["source_domain"] = "webshop"
    artifact = induce_memory_artifact(
        MemoryBaseline.EXPEL, source, FakeCompletion(MemoryBaseline.EXPEL)
    )
    target = adapt_target_context(
        "webshop", task="x", observation={"observation": "x"}
    )
    with pytest.raises(ValueError, match="cannot retrieve from the target domain"):
        retrieve_memory_items(artifact, target, FakeEmbedding())


def test_memory_artifact_tampering_is_detected():
    artifact = induce_memory_artifact(
        MemoryBaseline.AWM, SOURCE, FakeCompletion(MemoryBaseline.AWM)
    )
    artifact["items"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_memory_artifact(artifact)


def test_target_binding_is_frozen_traceable_and_domain_specific():
    source = induce_memory_artifact(
        MemoryBaseline.EXPEL, SOURCE, FakeCompletion(MemoryBaseline.EXPEL)
    )
    adaptation = {
        "target_domain": "alfworld",
        "split_role": "adaptation",
        "examples": [{
            "example_id": "alf-adapt-1",
            "task": "put an object in a receptacle",
            "observation": "A room description with an uncertain object location.",
            "native_actions": ["go to cabinet 1", "take mug 1 from cabinet 1"],
        }],
    }
    bound = bind_memory_artifact_to_target(
        source, "alfworld", adaptation, FakeBinder(), maximum_items_per_call=1,
    )
    validate_memory_artifact(bound)
    assert bound["artifact_kind"] == "FROZEN_TARGET_BOUND_CROSS_DOMAIN_MEMORY_BASELINE"
    assert bound["target_binding"]["source_artifact_sha256"] == source["artifact_sha256"]
    assert bound["target_binding"]["adaptation_example_ids"] == ["alf-adapt-1"]
    assert len(bound["items"]) == len(source["items"])


def test_target_binding_rejects_non_adaptation_data():
    source = induce_memory_artifact(
        MemoryBaseline.EXPEL, SOURCE, FakeCompletion(MemoryBaseline.EXPEL)
    )
    with pytest.raises(ValueError, match="adaptation split only"):
        bind_memory_artifact_to_target(
            source, "webshop",
            {"target_domain": "webshop", "split_role": "formal", "examples": []},
            FakeBinder(),
        )


def test_target_binding_rejects_source_vocabulary_and_outcome_proxies():
    source = induce_memory_artifact(
        MemoryBaseline.EXPEL, SOURCE, FakeCompletion(MemoryBaseline.EXPEL)
    )
    with pytest.raises(ValueError, match="forbidden source/outcome term"):
        bind_memory_artifact_to_target(
            source, "webshop",
            {"target_domain": "webshop", "split_role": "adaptation", "examples": [{
                "example_id": "adapt-1", "task": "find an item"
            }]},
            LeakyBinder(),
        )


def test_source_abstraction_audit_uses_provenance_not_keyword_deletion():
    source = json.loads(json.dumps(SOURCE))
    source["episodes"][1]["source_domain"] = "second_game"
    artifact = induce_memory_artifact(
        MemoryBaseline.EXPEL, source, FakeCompletion(MemoryBaseline.EXPEL)
    )
    audit = source_abstraction_audit(artifact)
    assert audit["cross_game_supported_items"] == 1
    assert audit["cross_game_supported_fraction"] == 1.0


def _unlabelled(**overrides):
    """One episode whose outcome the environment never reported."""
    episode = {
        "episode_id": "game-e2",
        "source_domain": "source_game",
        "official_success": None,
        "terminated": False,
        "truncated": True,
        "steps": [{
            "receipt_id": "r3",
            "step": 0,
            "observation": "uncertain route",
            "action": "probe",
            "next_observation": "still uncertain",
            "reward": 0,
            "terminal": False,
        }],
    }
    episode.update(overrides)
    return {"episodes": [episode]}


def test_absent_outcome_is_unknown_and_never_silently_a_failure():
    episode = canonical_source_episodes(_unlabelled())[0]
    assert episode.official_success is None
    assert episode.outcome == OutcomeLabel.UNKNOWN.value
    assert episode.outcome_authority == OutcomeAuthority.UNRESOLVED.value


def test_outcome_precedence_prefers_official_then_benchmark_then_evaluator():
    official = {"official_success": False}
    assert resolve_source_outcome(
        official,
        benchmark_predicate=lambda _: True,
        shared_evaluator=lambda _: True,
    ) == (OutcomeLabel.FAILURE, OutcomeAuthority.OFFICIAL)

    unlabelled = _unlabelled()["episodes"][0]
    assert resolve_source_outcome(
        unlabelled,
        benchmark_predicate=lambda _: True,
        shared_evaluator=lambda _: False,
    ) == (OutcomeLabel.SUCCESS, OutcomeAuthority.BENCHMARK_PREDICATE)

    # A predicate that abstains must fall through rather than decide.
    assert resolve_source_outcome(
        unlabelled,
        benchmark_predicate=lambda _: None,
        shared_evaluator=lambda _: False,
    ) == (OutcomeLabel.FAILURE, OutcomeAuthority.SHARED_EVALUATOR)


def test_unknown_episodes_are_withheld_from_every_method():
    superset = canonical_source_episodes(_unlabelled())
    for method in MemoryBaseline:
        projection = source_projection(method, superset)
        assert projection["eligible_episode_ids"] == []
        assert projection["withheld"] == [
            {"episode_id": "game-e2", "outcome": OutcomeLabel.UNKNOWN.value}
        ]


def test_awm_reads_only_successes_while_the_others_read_both():
    superset = canonical_source_episodes(SOURCE)
    assert source_projection(MemoryBaseline.AWM, superset)["eligible_episode_ids"] == ["game-e0"]
    for method in (MemoryBaseline.EXPEL, MemoryBaseline.REASONING_BANK):
        assert source_projection(method, superset)["eligible_episode_ids"] == [
            "game-e0", "game-e1",
        ]


def test_awm_refuses_to_build_an_empty_bank_from_unlabelled_source():
    with pytest.raises(InsufficientEligibleSourceError, match="UNKNOWN"):
        induce_memory_artifact(
            MemoryBaseline.AWM, _unlabelled(), FakeCompletion(MemoryBaseline.AWM)
        )


def test_all_methods_bind_to_one_shared_superset_hash():
    artifacts = {
        method: induce_memory_artifact(method, SOURCE, FakeCompletion(method))
        for method in MemoryBaseline
    }
    shared = {row["source_superset_sha256"] for row in artifacts.values()}
    assert len(shared) == 1, "methods must provably start from the same source pool"
    census = {"SUCCESS": 1, "FAILURE": 1, "UNKNOWN": 0}
    for method, artifact in artifacts.items():
        assert artifact["source_outcome_census"] == census
        assert artifact["source_projection"]["method"] == method.value
    # AWM reads a strict subset, so its own projection hash must differ.
    assert (
        artifacts[MemoryBaseline.AWM]["source_payload_sha256"]
        != artifacts[MemoryBaseline.EXPEL]["source_payload_sha256"]
    )


def test_shared_evaluator_labels_are_applied_identically_to_all_methods():
    payload = _unlabelled()
    payload["episodes"].append(json.loads(json.dumps(SOURCE["episodes"][0])))
    payload["episodes"][-1]["official_success"] = None
    payload["episodes"][-1]["episode_id"] = "game-e3"

    def evaluator(episode):
        return episode["episode_id"] == "game-e3"

    artifacts = {
        method: induce_memory_artifact(
            method, payload, FakeCompletion(method), shared_evaluator=evaluator,
        )
        for method in MemoryBaseline
    }
    for artifact in artifacts.values():
        assert artifact["source_outcome_census"] == {
            "SUCCESS": 1, "FAILURE": 1, "UNKNOWN": 0,
        }
    assert artifacts[MemoryBaseline.AWM]["source_projection"]["eligible_episode_ids"] == [
        "game-e3",
    ]
