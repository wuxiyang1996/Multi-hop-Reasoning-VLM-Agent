from __future__ import annotations

import json

import pytest

from motif_transfer.cross_domain_memory_baselines import (
    MemoryBaseline,
    induce_memory_artifact,
)
from motif_transfer.cross_domain_memory_runtime import (
    MEMORY_PAYLOAD_KEY,
    MemoryAugmentedDecisionBackend,
    TargetActionLeakError,
    advisory_prompt_block,
    assert_action_free,
    retrieve_target_advisory,
)

SOURCE = {
    "episodes": [{
        "episode_id": "game-e0",
        "source_domain": "source_game",
        "official_success": True,
        "steps": [{
            "receipt_id": "r0",
            "step": 0,
            "observation": "uncertain layout",
            "action": "probe",
            "next_observation": "layout identified",
            "reward": 1,
            "terminal": True,
        }],
    }]
}


class FakeInducer:
    identity = {"model": "test"}
    last_usage = {"total_tokens": 1}

    def complete(self, role, system, payload):
        return json.dumps({"items": [{
            "title": "check before committing",
            "content": "Confirm the uncertain part of the state before an irreversible step.",
            "applicability": "When the visible state is ambiguous.",
            "kind": "INSIGHT",
            "source_episode_ids": ["game-e0"],
            "evidence_receipt_ids": ["r0"],
        }]})


class FakeEmbedding:
    identity = {"model": "test-embed"}

    def embed(self, texts):
        return [[float(len(text) % 5 + 1), 1.0] for text in texts]


class RecordingBackend:
    identity = {"model": "decision-test"}
    last_usage = {"total_tokens": 7}

    def __init__(self):
        self.calls = []

    def complete(self, role, system, payload):
        self.calls.append((role, system, json.loads(json.dumps(payload))))
        return json.dumps({"candidates": [{"action": "click('42')"}]})


def _artifact():
    return induce_memory_artifact(MemoryBaseline.EXPEL, SOURCE, FakeInducer())


def _wrap(inner, **kwargs):
    return MemoryAugmentedDecisionBackend(
        inner, artifact=_artifact(), domain="webshop",
        embedding_backend=FakeEmbedding(), **kwargs,
    )


WEBSHOP_PAYLOAD = {
    "goal": "buy a blue mug under $20",
    "accessibility_tree": "[12] button 'Search'",
    "url": "http://webshop/index",
    "history": [{"action": "click('12')", "reward": 0.5}],
    "candidate_count": 5,
}


def test_memory_is_added_to_the_decision_request():
    inner = RecordingBackend()
    wrapped = _wrap(inner)
    wrapped.complete("decision", "system prompt", WEBSHOP_PAYLOAD)
    _, _, payload = inner.calls[0]
    assert MEMORY_PAYLOAD_KEY in payload
    assert "irreversible" in payload[MEMORY_PAYLOAD_KEY]["items"]
    # The target's own fields must survive untouched.
    assert payload["goal"] == WEBSHOP_PAYLOAD["goal"]
    assert payload["accessibility_tree"] == WEBSHOP_PAYLOAD["accessibility_tree"]


def test_non_decision_roles_are_passed_through_unchanged():
    inner = RecordingBackend()
    wrapped = _wrap(inner)
    wrapped.complete("grounder", "system", WEBSHOP_PAYLOAD)
    _, _, payload = inner.calls[0]
    assert MEMORY_PAYLOAD_KEY not in payload


def test_target_rewards_never_reach_the_memory_query():
    """history carries reward, which must be stripped before retrieval."""
    inner = RecordingBackend()
    wrapped = _wrap(inner)
    # adapt_target_context raises on forbidden outcome fields, so a leak here
    # would surface as an exception rather than a silently contaminated query.
    wrapped.complete("decision", "system", WEBSHOP_PAYLOAD)
    assert wrapped.retrieval_receipts[0]["retrieved_item_ids"]


def test_memory_quoting_a_target_action_fails_closed():
    with pytest.raises(TargetActionLeakError, match="target-native action"):
        assert_action_free(
            "Use click('42') to proceed.", ["click('42')", "go_back()"],
        )
    # Short tokens must not trigger false positives on ordinary prose.
    assert_action_free("Verify the state first.", ["a", "st"])
    # Generic one-word affordances are not copied executable command templates.
    assert_action_free("Look for contradictory evidence before committing.", ["look"])


def test_receipt_is_hash_bound_and_counts_augmented_calls():
    inner = RecordingBackend()
    wrapped = _wrap(inner)
    for _ in range(3):
        wrapped.complete("decision", "system", WEBSHOP_PAYLOAD)
    receipt = wrapped.receipt()
    assert receipt["decision_calls_augmented"] == 3
    assert receipt["receipt_sha256"]
    assert receipt["identity"]["cross_domain_memory"]["method"] == "expel"


def test_unwrapped_backend_reproduces_the_target_only_arm():
    inner = RecordingBackend()
    inner.complete("decision", "system", WEBSHOP_PAYLOAD)
    assert MEMORY_PAYLOAD_KEY not in inner.calls[0][2]


def test_empty_retrieval_is_byte_equivalent_to_target_only_request():
    inner = RecordingBackend()
    wrapped = _wrap(inner)
    wrapped._advisory_text = lambda payload: ("", {
        "method": "expel",
        "retrieval_sha256": "retrieval",
        "memory_artifact_sha256": wrapped.artifact["artifact_sha256"],
        "target_query_sha256": "query",
        "retrieved": [],
    })
    wrapped.complete("decision", "system prompt", WEBSHOP_PAYLOAD)
    assert inner.calls[0][2] == WEBSHOP_PAYLOAD
    receipt = wrapped.receipt()
    assert receipt["decision_calls"] == 1
    assert receipt["decision_calls_augmented"] == 0
    assert receipt["retrievals"][0]["request_augmented"] is False


TIRBENCH_PAYLOAD = {
    "prompt": "Which of the following moves reaches the goal?",
    "question": "solve the visual maze",
    "tool_trace": [],
    "available_tools": ["crop", "zoom"],
}


def test_direct_client_domains_get_the_same_advisory_without_a_backend():
    """TIRBench is multimodal and calls the OpenAI SDK directly.

    It must be able to inject identical memory text without being reshaped to
    fit a JSON-payload CompletionBackend.
    """
    artifact = _artifact()
    embedding = FakeEmbedding()
    text, retrieval = retrieve_target_advisory(
        artifact, "tirbench", TIRBENCH_PAYLOAD, embedding, top_k=3,
    )
    assert "irreversible" in text
    assert retrieval["online_memory_updated"] is False

    block = advisory_prompt_block(artifact["method"], text)
    assert "EXPEL CROSS-DOMAIN MEMORY" in block
    assert text in block

    # The very same helper is what the backend wrapper uses, so the two channels
    # cannot drift apart.
    inner = RecordingBackend()
    wrapped = _wrap(inner)
    wrapped.complete("decision", "system", WEBSHOP_PAYLOAD)
    assert inner.calls[0][2][MEMORY_PAYLOAD_KEY]["items"] == retrieve_target_advisory(
        artifact, "webshop", WEBSHOP_PAYLOAD, embedding, top_k=3,
    )[0]


def test_direct_client_path_still_fails_closed_on_action_leakage():
    leaky = dict(TIRBENCH_PAYLOAD, available_tools=["Confirm the uncertain part"])
    with pytest.raises(TargetActionLeakError):
        retrieve_target_advisory(_artifact(), "tirbench", leaky, FakeEmbedding())
