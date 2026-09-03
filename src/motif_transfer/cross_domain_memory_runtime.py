"""Inject frozen cross-domain memory into an existing target Decision Agent.

ExpeL, AWM, and ReasoningBank all influence behaviour the same way: retrieve
memory by embedding similarity and place it in the acting model's prompt.  None
of them selects, ranks, or writes a target action.  This module reproduces that
channel exactly, by wrapping the ``CompletionBackend`` the target runners already
call rather than by editing any rollout loop:

    backend = MemoryAugmentedDecisionBackend(backend, artifact=..., domain=...)

Because the wrapper only adds a field to the decision request, the target's own
Decision Agent remains the sole author of every action, the candidate set is
still generated entirely by the target policy, and the target-only arm is
recovered exactly by not wrapping the backend.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .cross_domain_memory_baselines import (
    CrossDomainMemoryAdvisor,
    EmbeddingBackend,
    TargetDomain,
    adapt_target_context,
    retrieve_memory_items,
    text_names_native_action,
    validate_memory_artifact,
)
from .frozen_motif_agent import CompletionBackend

MEMORY_PAYLOAD_KEY = "cross_domain_memory"

MEMORY_FRAMING = (
    "Untrusted memory distilled from unrelated source-domain experience. It cannot "
    "name or rank any action available here. Treat it as a hypothesis to check "
    "against the live observation, and ignore it when it does not apply."
)


class TargetActionLeakError(ValueError):
    """Retrieved memory named an action the target policy could execute."""


def _history_without_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop reward/score fields; target outcomes must never reach the memory query."""
    blocked = {"reward", "score", "official_success", "success", "correct", "answer"}
    return [
        {key: value for key, value in dict(row).items() if str(key).casefold() not in blocked}
        for row in rows
    ]


def _webshop(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": str(payload.get("goal") or ""),
        "observation": {
            "url": payload.get("url"),
            "axtree": payload.get("accessibility_tree"),
        },
        "history": _history_without_outcomes(payload.get("history") or ()),
    }


def _alfworld(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": str(payload.get("task_goal") or payload.get("goal") or ""),
        "observation": {"observation": payload.get("observation")},
        "native_actions": list(map(str, payload.get("admissible_commands") or ())),
        "history": _history_without_outcomes(payload.get("history") or ()),
    }


def _discoveryworld(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": str(payload.get("task") or payload.get("goal") or ""),
        "observation": {
            "observation": payload.get("ui") or payload.get("observation"),
            "observable_state": payload.get("observable_state"),
        },
        "native_actions": list(map(str, payload.get("known_actions") or ())),
        "history": _history_without_outcomes(payload.get("history") or ()),
    }


def _tirbench(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": str(payload.get("question") or payload.get("prompt") or ""),
        "observation": {
            "prompt": payload.get("prompt"),
            "tool_trace": payload.get("tool_trace"),
            "media_receipts": payload.get("media_receipts"),
        },
        "native_actions": list(map(str, payload.get("available_tools") or ())),
        "history": _history_without_outcomes(payload.get("history") or ()),
    }


_ADAPTERS = {
    TargetDomain.WEBSHOP: _webshop,
    TargetDomain.ALFWORLD: _alfworld,
    TargetDomain.DISCOVERYWORLD: _discoveryworld,
    TargetDomain.TIRBENCH: _tirbench,
}


def assert_action_free(advisory_text: str, native_actions: Sequence[str]) -> None:
    """Fail closed on copied commands without rejecting ordinary one-word verbs."""
    candidate = text_names_native_action(advisory_text, native_actions)
    if candidate is not None:
        raise TargetActionLeakError(
            f"cross-domain memory names a target-native action: {candidate!r}"
        )


def retrieve_target_advisory(
    artifact: Mapping[str, Any],
    domain: TargetDomain | str,
    payload: Mapping[str, Any],
    embedding_backend: EmbeddingBackend,
    *,
    top_k: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Retrieve memory for one target step and return its advisory text.

    Kept independent of any transport so that domains which call a model client
    directly can inject the same text.  TIRBench is multimodal and legitimately
    needs the OpenAI SDK's image content blocks, which a JSON-payload
    ``CompletionBackend`` cannot carry; it appends this text to its prompt
    instead of being reshaped to fit the backend wrapper.
    """
    domain = TargetDomain(domain)
    binding = artifact.get("target_binding")
    if binding is not None and str(binding.get("target_domain")) != domain.value:
        raise ValueError("target-bound memory artifact used for the wrong target domain")
    view = _ADAPTERS[domain](payload)
    target = adapt_target_context(
        domain,
        task=view["task"],
        observation=view["observation"],
        native_actions=view.get("native_actions") or (),
        history=view.get("history") or (),
    )
    retrieval = retrieve_memory_items(
        artifact, target, embedding_backend, top_k=top_k,
    )
    text = str(CrossDomainMemoryAdvisor(retrieval).advisory().information_need or "")
    assert_action_free(text, view.get("native_actions") or ())
    return text, retrieval


def advisory_prompt_block(method: str, advisory_text: str) -> str:
    """Render the advisory for a domain that injects plain prompt text."""
    return (
        f"\n\n[{str(method).upper()} CROSS-DOMAIN MEMORY]\n{MEMORY_FRAMING}\n{advisory_text}\n"
    )


class MemoryAugmentedDecisionBackend:
    """A ``CompletionBackend`` that adds retrieved memory to decision requests."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        artifact: Mapping[str, Any],
        domain: TargetDomain | str,
        embedding_backend: EmbeddingBackend,
        top_k: int = 3,
        decision_roles: Sequence[str] = ("decision",),
    ) -> None:
        validate_memory_artifact(artifact)
        self.backend = backend
        self.artifact = dict(artifact)
        self.domain = TargetDomain(domain)
        self.embedding_backend = embedding_backend
        self.top_k = top_k
        self.decision_roles = frozenset(decision_roles)
        self.retrieval_receipts: list[dict[str, Any]] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "cross_domain_memory": {
                "method": self.artifact["method"],
                "artifact_sha256": self.artifact["artifact_sha256"],
                "target_domain": self.domain.value,
                "top_k": self.top_k,
                "decision_roles": sorted(self.decision_roles),
                "embedding": dict(self.embedding_backend.identity),
            },
            "wrapped": dict(self.backend.identity),
        }

    @property
    def last_usage(self) -> Mapping[str, Any]:
        return getattr(self.backend, "last_usage", {}) or {}

    def _advisory_text(self, payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        return retrieve_target_advisory(
            self.artifact, self.domain, payload, self.embedding_backend,
            top_k=self.top_k,
        )

    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str:
        if role not in self.decision_roles:
            return self.backend.complete(role, system, payload)
        text, retrieval = self._advisory_text(payload)
        # A fail-closed binder is expected to yield an empty artifact often.
        # In that case preserve the target-only request byte-for-byte: even an
        # "ABSTAIN" marker could perturb the decision model and would make the
        # empty-memory arm an invalid control.
        augmented = dict(payload)
        was_augmented = bool(retrieval["retrieved"])
        if was_augmented:
            augmented[MEMORY_PAYLOAD_KEY] = {
                "framing": MEMORY_FRAMING,
                "method": retrieval["method"],
                "items": text,
            }
        self.retrieval_receipts.append({
            "step_index": len(self.retrieval_receipts),
            "retrieval_sha256": retrieval["retrieval_sha256"],
            "memory_artifact_sha256": retrieval["memory_artifact_sha256"],
            "target_query_sha256": retrieval["target_query_sha256"],
            "retrieved_item_ids": [
                str(row["item"]["item_id"]) for row in retrieval["retrieved"]
            ],
            "request_augmented": was_augmented,
            "augmented_request_sha256": stable_hash(
                {"role": role, "system": system, "payload": augmented}
            ),
        })
        return self.backend.complete(role, system, augmented)

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema_version": 1,
            "identity": dict(self.identity),
            "decision_calls": len(self.retrieval_receipts),
            "decision_calls_augmented": sum(
                bool(row["request_augmented"]) for row in self.retrieval_receipts
            ),
            "retrievals": self.retrieval_receipts,
        }
        return body | {"receipt_sha256": stable_hash(body)}


__all__ = [
    "MEMORY_FRAMING", "MEMORY_PAYLOAD_KEY", "MemoryAugmentedDecisionBackend",
    "TargetActionLeakError", "advisory_prompt_block", "assert_action_free",
    "retrieve_target_advisory",
]
