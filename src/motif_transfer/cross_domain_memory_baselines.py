"""Auditable open-weight baselines for cross-domain experience adaptation.

The module implements the common substrate needed to compare three published
memory strategies without importing their benchmark-specific agents:

* ExpeL-style global natural-language insights;
* Agent Workflow Memory (AWM)-style reusable workflows;
* ReasoningBank-style success/failure memory items with retrieval.

All three consume the same canonical *source* episodes and produce a frozen,
content-addressed artifact.  At target time they only provide advisory text;
they never construct or select a target-native action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import copy
import json
import math
import re
import hashlib
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import Advisory, AdvisoryVerdict, stable_hash
from .frozen_motif_agent import CompletionBackend


class MemoryBaseline(str, Enum):
    EXPEL = "expel"
    AWM = "awm"
    REASONING_BANK = "reasoning_bank"


class MemoryControl(str, Enum):
    EPISODIC_RAG = "episodic_rag"
    RANDOM_TRAJECTORY_ICL = "random_trajectory_icl"


OURS_MEMORY_METHOD = "ours"


def comparison_memory_methods(*, include_ours: bool = True) -> tuple[str, ...]:
    rows = tuple(row.value for row in MemoryBaseline) + tuple(row.value for row in MemoryControl)
    return rows + ((OURS_MEMORY_METHOD,) if include_ours else ())


class TargetDomain(str, Enum):
    WEBSHOP = "webshop"
    ALFWORLD = "alfworld"
    DISCOVERYWORLD = "discoveryworld"
    TIRBENCH = "tirbench"


class OutcomeLabel(str, Enum):
    """Resolved source-episode outcome.  ``UNKNOWN`` is never a silent failure."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"


class OutcomeAuthority(str, Enum):
    """Which rule resolved the outcome, in strict precedence order."""

    OFFICIAL = "OFFICIAL"
    BENCHMARK_PREDICATE = "BENCHMARK_PREDICATE"
    SHARED_EVALUATOR = "SHARED_EVALUATOR"
    UNRESOLVED = "UNRESOLVED"


class InsufficientEligibleSourceError(ValueError):
    """A method was left with too few labelled episodes to induce anything."""


class TargetBindingError(ValueError):
    """A target binder produced an invalid, leaky, or untraceable memory bank."""


@dataclass(frozen=True)
class CanonicalSourceStep:
    receipt_id: str
    step: int
    observation: str
    action: str
    next_observation: str
    reward: float
    terminal: bool


@dataclass(frozen=True)
class CanonicalSourceEpisode:
    episode_id: str
    source_domain: str
    official_success: bool | None
    outcome: str
    outcome_authority: str
    terminated: bool
    truncated: bool
    steps: tuple[CanonicalSourceStep, ...]


@dataclass(frozen=True)
class MemoryItem:
    item_id: str
    title: str
    content: str
    applicability: str
    kind: str
    source_episode_ids: tuple[str, ...]
    evidence_receipt_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        title: str,
        content: str,
        applicability: str,
        kind: str,
        source_episode_ids: Sequence[str],
        evidence_receipt_ids: Sequence[str],
    ) -> "MemoryItem":
        body = {
            "title": title.strip(),
            "content": content.strip(),
            "applicability": applicability.strip(),
            "kind": kind.strip().upper(),
            "source_episode_ids": tuple(dict.fromkeys(map(str, source_episode_ids))),
            "evidence_receipt_ids": tuple(dict.fromkeys(map(str, evidence_receipt_ids))),
        }
        if not all((body["title"], body["content"], body["applicability"], body["kind"])):
            raise ValueError("memory item contains an empty required field")
        if not body["source_episode_ids"] or not body["evidence_receipt_ids"]:
            raise ValueError("memory item must cite source episodes and receipts")
        return cls(item_id=stable_hash(body), **body)


class EmbeddingBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def _text(value: Any, *, limit: int = 4000) -> str:
    if isinstance(value, str):
        result = value
    else:
        result = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return result[:limit]


def _token_bounded_prefix(text: str, maximum_tokens: int) -> tuple[str, int]:
    matches = list(re.finditer(r"\w+|[^\w\s]", str(text), flags=re.UNICODE))
    used = min(max(0, int(maximum_tokens)), len(matches))
    return (str(text)[:matches[used - 1].end()] if used else "", used)


def resolve_source_outcome(
    raw_episode: Mapping[str, Any],
    *,
    benchmark_predicate: Callable[[Mapping[str, Any]], bool | None] | None = None,
    shared_evaluator: Callable[[Mapping[str, Any]], bool | None] | None = None,
) -> tuple[OutcomeLabel, OutcomeAuthority]:
    """Resolve an outcome under a fixed precedence.

    ``official predicate > benchmark-defined predicate > shared frozen evaluator >
    UNKNOWN``.  A missing or null ``official_success`` is *never* read as failure;
    it falls through to the next authority and ends at ``UNKNOWN`` if unresolved.
    The same ``shared_evaluator`` must be supplied for every method so that no
    single baseline receives outcome supervision the others were denied.
    """
    official = raw_episode.get("official_success")
    if isinstance(official, bool):
        return (
            OutcomeLabel.SUCCESS if official else OutcomeLabel.FAILURE,
            OutcomeAuthority.OFFICIAL,
        )
    for rule, authority in (
        (benchmark_predicate, OutcomeAuthority.BENCHMARK_PREDICATE),
        (shared_evaluator, OutcomeAuthority.SHARED_EVALUATOR),
    ):
        if rule is None:
            continue
        verdict = rule(raw_episode)
        if isinstance(verdict, bool):
            return (
                OutcomeLabel.SUCCESS if verdict else OutcomeLabel.FAILURE,
                authority,
            )
        if verdict is not None:
            raise ValueError("outcome rules must return True, False, or None")
    return OutcomeLabel.UNKNOWN, OutcomeAuthority.UNRESOLVED


def canonical_source_episodes(
    payload: Mapping[str, Any],
    *,
    benchmark_predicate: Callable[[Mapping[str, Any]], bool | None] | None = None,
    shared_evaluator: Callable[[Mapping[str, Any]], bool | None] | None = None,
) -> tuple[CanonicalSourceEpisode, ...]:
    """Validate the small interchange schema used by all three baselines."""
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("source payload needs a non-empty episodes list")
    episodes: list[CanonicalSourceEpisode] = []
    seen_episodes: set[str] = set()
    seen_receipts: set[str] = set()
    for raw_episode in raw_episodes:
        if not isinstance(raw_episode, Mapping):
            raise ValueError("each source episode must be an object")
        episode_id = str(raw_episode.get("episode_id") or "").strip()
        source_domain = str(raw_episode.get("source_domain") or "").strip()
        if not episode_id or not source_domain or episode_id in seen_episodes:
            raise ValueError("source episode identity is empty or duplicated")
        seen_episodes.add(episode_id)
        raw_steps = raw_episode.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"source episode {episode_id} has no steps")
        steps: list[CanonicalSourceStep] = []
        for expected_step, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise ValueError("each source step must be an object")
            step = int(raw_step.get("step", expected_step))
            if step != expected_step:
                raise ValueError(f"source episode {episode_id} steps are not contiguous from zero")
            receipt_id = str(raw_step.get("receipt_id") or "").strip()
            if not receipt_id or receipt_id in seen_receipts:
                raise ValueError("source receipt identity is empty or duplicated")
            seen_receipts.add(receipt_id)
            steps.append(CanonicalSourceStep(
                receipt_id=receipt_id,
                step=step,
                observation=_text(raw_step.get("observation", "")),
                action=_text(raw_step.get("action", ""), limit=1000),
                next_observation=_text(raw_step.get("next_observation", "")),
                reward=float(raw_step.get("reward", 0.0) or 0.0),
                terminal=bool(raw_step.get("terminal", False)),
            ))
        official = raw_episode.get("official_success")
        if official is not None and not isinstance(official, bool):
            raise ValueError(f"episode {episode_id} has a non-boolean official_success")
        declared = raw_episode.get("outcome")
        if declared is None:
            outcome, authority = resolve_source_outcome(
                raw_episode,
                benchmark_predicate=benchmark_predicate,
                shared_evaluator=shared_evaluator,
            )
        else:
            # A frozen evaluator pass already annotated this payload; keep its verdict
            # and its stated authority rather than silently re-deriving either.
            outcome = OutcomeLabel(str(declared))
            authority = OutcomeAuthority(str(raw_episode.get("outcome_authority") or ""))
            if (outcome is OutcomeLabel.UNKNOWN) != (authority is OutcomeAuthority.UNRESOLVED):
                raise ValueError(f"episode {episode_id} outcome and authority disagree")
        episodes.append(CanonicalSourceEpisode(
            episode_id=episode_id,
            source_domain=source_domain,
            official_success=official,
            outcome=outcome.value,
            outcome_authority=authority.value,
            terminated=bool(raw_episode.get("terminated", False)),
            truncated=bool(raw_episode.get("truncated", False)),
            steps=tuple(steps),
        ))
    return tuple(episodes)


def canonical_source_payload(episodes: Sequence[CanonicalSourceEpisode]) -> dict[str, Any]:
    return {
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "source_domain": episode.source_domain,
                "official_success": episode.official_success,
                "outcome": episode.outcome,
                "outcome_authority": episode.outcome_authority,
                "terminated": episode.terminated,
                "truncated": episode.truncated,
                "steps": [asdict(step) for step in episode.steps],
            }
            for episode in episodes
        ]
    }


_SYSTEMS = {
    MemoryBaseline.EXPEL: (
        "Implement the insight-extraction stage of ExpeL. Infer concise, domain-agnostic lessons "
        "from the supplied source trajectories. Use both successful and failed evidence. Do not "
        "name or propose any target action. Return exactly one JSON object {\"items\":[...]}; each "
        "item has title, content, applicability, kind, source_episode_ids, evidence_receipt_ids. "
        "kind must be INSIGHT. Every claim must cite real supplied IDs."
    ),
    MemoryBaseline.AWM: (
        "Implement offline Agent Workflow Memory induction. Infer reusable, domain-agnostic "
        "workflows from the supplied source trajectories. Express ordered verification or recovery "
        "steps in content, and applicability/stop conditions separately. Never emit a target action. "
        "Return exactly one JSON object {\"items\":[...]}; each item has title, content, "
        "applicability, kind, source_episode_ids, evidence_receipt_ids. kind must be WORKFLOW. "
        "Every workflow must cite real supplied IDs."
    ),
    MemoryBaseline.REASONING_BANK: (
        "Implement ReasoningBank memory extraction for one source trajectory. If successful, distill "
        "validated reasoning strategies; if failed, distill pitfalls and counterfactual guardrails. "
        "Keep every item domain-agnostic and never emit a target action. Return exactly one JSON "
        "object {\"items\":[...]}; each item has title, content, applicability, kind, "
        "source_episode_ids, evidence_receipt_ids. kind is STRATEGY or PITFALL. Every item must cite "
        "real supplied IDs."
    ),
}

_EXPEL_REFINER_SYSTEM = (
    "Implement ExpeL's iterative insight-refinement stage. Reconcile the current insight bank "
    "with the newly extracted candidate insights: merge duplicates, correct contradictions, "
    "retain useful failure lessons, and return the strongest bounded bank. Use only supplied "
    "source evidence and preserve real evidence receipt IDs. Return exactly one JSON object "
    "{\"items\":[...]}; every item has title, content, applicability, kind=INSIGHT, "
    "source_episode_ids, evidence_receipt_ids. Never mention or propose a target action."
)


# Which outcome labels each published method induces from.  These follow the
# original papers, not a house convention: ExpeL contrasts successes against
# failures, offline AWM abstracts workflows from successful/canonical trajectories
# only, and ReasoningBank distils strategies from successes and pitfalls from
# failures.  ``UNKNOWN`` is inducible by nobody, so an unlabelled episode is
# withheld from every method rather than counting as a failure for some of them.
_METHOD_ELIGIBLE_OUTCOMES: Mapping[MemoryBaseline, tuple[OutcomeLabel, ...]] = {
    MemoryBaseline.EXPEL: (OutcomeLabel.SUCCESS, OutcomeLabel.FAILURE),
    MemoryBaseline.AWM: (OutcomeLabel.SUCCESS,),
    MemoryBaseline.REASONING_BANK: (OutcomeLabel.SUCCESS, OutcomeLabel.FAILURE),
}


def _outcome_census(episodes: Sequence[CanonicalSourceEpisode]) -> dict[str, int]:
    census = {label.value: 0 for label in OutcomeLabel}
    for episode in episodes:
        census[OutcomeLabel(episode.outcome).value] += 1
    return census


def source_projection(
    method: MemoryBaseline | str,
    episodes: Sequence[CanonicalSourceEpisode],
) -> dict[str, Any]:
    """Split one shared superset into the episodes a given method may read."""
    method = MemoryBaseline(method)
    admitted = _METHOD_ELIGIBLE_OUTCOMES[method]
    eligible = tuple(
        episode for episode in episodes if OutcomeLabel(episode.outcome) in admitted
    )
    withheld = tuple(
        {"episode_id": episode.episode_id, "outcome": episode.outcome}
        for episode in episodes
        if OutcomeLabel(episode.outcome) not in admitted
    )
    return {
        "method": method.value,
        "eligible_outcomes": [label.value for label in admitted],
        "eligible_episode_ids": [episode.episode_id for episode in eligible],
        "withheld": list(withheld),
        "episodes": eligible,
    }


def _parse_items(
    raw: str,
    *,
    allowed_episode_ids: set[str],
    receipt_to_episode: Mapping[str, str],
    allowed_kinds: set[str],
    maximum_items: int,
) -> tuple[MemoryItem, ...]:
    value = json.loads(raw)
    rows = value.get("items") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or not rows or len(rows) > maximum_items:
        raise ValueError("memory extractor returned an invalid item count")
    items: list[MemoryItem] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("memory item is not an object")
        title = str(row.get("title") or "")
        receipts = tuple(row.get("evidence_receipt_ids") or ())
        # Diagnostics name the offending ids: these errors are fed back to the
        # extractor on retry, and "something disagrees" is not actionable.
        unknown_receipts = sorted(set(receipts) - set(receipt_to_episode))
        if unknown_receipts:
            raise ValueError(
                f"memory item {title!r} cites unknown source receipts "
                f"{unknown_receipts[:8]}"
            )
        cited_episodes = {receipt_to_episode[value] for value in receipts}
        outside = sorted(cited_episodes - allowed_episode_ids)
        if outside:
            raise ValueError(
                f"memory item {title!r} cites receipts from episodes {outside}, "
                f"which this call did not supply; it supplied "
                f"{sorted(allowed_episode_ids)}"
            )
        item = MemoryItem.create(
            title=title,
            content=str(row.get("content") or ""),
            applicability=str(row.get("applicability") or ""),
            kind=str(row.get("kind") or ""),
            # Derived from the receipts rather than declared separately.  A
            # cross-trajectory insight is provenanced by the receipts that
            # support it, and asking the extractor to keep a second list
            # consistent with the first only invents a way to disagree.
            source_episode_ids=tuple(sorted(cited_episodes)),
            evidence_receipt_ids=receipts,
        )
        if item.kind not in allowed_kinds:
            raise ValueError(f"invalid memory kind: {item.kind}")
        if item.item_id not in seen:
            items.append(item)
            seen.add(item.item_id)
    return tuple(items)


def induce_memory_artifact(
    method: MemoryBaseline | str,
    source_payload: Mapping[str, Any],
    backend: CompletionBackend,
    *,
    maximum_items_per_call: int = 8,
    maximum_episodes_per_call: int = 4,
    induction_retries: int = 3,
    minimum_eligible_episodes: int = 1,
    benchmark_predicate: Callable[[Mapping[str, Any]], bool | None] | None = None,
    shared_evaluator: Callable[[Mapping[str, Any]], bool | None] | None = None,
    expel_refinement_rounds: int = 0,
) -> dict[str, Any]:
    """Run frozen source-only induction and return a self-hashed artifact.

    Every method is handed the identical superset and binds to its hash; each then
    reads only the projection its published algorithm is defined over.
    """
    method = MemoryBaseline(method)
    if int(expel_refinement_rounds) < 0:
        raise ValueError("expel_refinement_rounds must be non-negative")
    superset = canonical_source_episodes(
        source_payload,
        benchmark_predicate=benchmark_predicate,
        shared_evaluator=shared_evaluator,
    )
    superset_payload = canonical_source_payload(superset)
    projection = source_projection(method, superset)
    episodes = projection.pop("episodes")
    if len(episodes) < minimum_eligible_episodes:
        raise InsufficientEligibleSourceError(
            f"{method.value} has {len(episodes)} eligible source episodes "
            f"(minimum {minimum_eligible_episodes}); admitted outcomes are "
            f"{projection['eligible_outcomes']} and the superset resolved to "
            + json.dumps(_outcome_census(superset), sort_keys=True)
        )
    canonical = canonical_source_payload(episodes)
    receipt_to_episode = {
        step.receipt_id: episode.episode_id
        for episode in episodes for step in episode.steps
    }
    if method == MemoryBaseline.REASONING_BANK:
        # ReasoningBank distils one memory item set per trajectory.
        calls = [[episode] for episode in episodes]
    else:
        # ExpeL iterates over trajectories, revising its insight list as it goes,
        # and AWM abstracts workflows from batches; neither requires a single
        # call holding every episode, which arcade observations would overflow.
        size = max(1, maximum_episodes_per_call)
        calls = [
            list(episodes[start:start + size])
            for start in range(0, len(episodes), size)
        ]
    allowed_kinds = {
        MemoryBaseline.EXPEL: {"INSIGHT"},
        MemoryBaseline.AWM: {"WORKFLOW"},
        MemoryBaseline.REASONING_BANK: {"STRATEGY", "PITFALL"},
    }[method]
    all_items: list[MemoryItem] = []
    call_receipts = []
    refinement_call_receipts = []
    cumulative_episode_ids: set[str] = set()
    for call_index, selected in enumerate(calls):
        payload = canonical_source_payload(selected)
        allowed_episode_ids = {episode.episode_id for episode in selected}
        # Long arcade episodes make the model unreliable at transcribing receipt
        # ids, so a rejected citation is fed back and retried.  The lineage check
        # itself is never relaxed: an item that cannot cite real receipts is not
        # admissible evidence.
        attempt_error = ""
        for attempt in range(1, induction_retries + 1):
            request = payload if not attempt_error else dict(payload) | {
                "previous_attempt_rejected": attempt_error,
                "receipt_id_to_episode_id": {
                    step.receipt_id: episode.episode_id
                    for episode in selected for step in episode.steps
                },
            }
            raw = backend.complete("memory_inducer", _SYSTEMS[method], request)
            try:
                items = _parse_items(
                    raw,
                    allowed_episode_ids=allowed_episode_ids,
                    receipt_to_episode=receipt_to_episode,
                    allowed_kinds=allowed_kinds,
                    maximum_items=maximum_items_per_call,
                )
                break
            except ValueError as error:
                attempt_error = str(error)[:300]
        else:
            raise ValueError(
                f"{method.value} induction call {call_index} failed the receipt "
                f"lineage check {induction_retries} times: {attempt_error}"
            )
        induction_usage = dict(getattr(backend, "last_usage", {}) or {})
        cumulative_episode_ids.update(allowed_episode_ids)
        if (
            method == MemoryBaseline.EXPEL
            and all_items
            and expel_refinement_rounds > 0
        ):
            candidate_bank = items
            for refinement_round in range(expel_refinement_rounds):
                refinement_request = {
                    "current_insights": [asdict(item) for item in all_items],
                    "new_candidate_insights": [asdict(item) for item in candidate_bank],
                    "maximum_items": maximum_items_per_call,
                }
                refinement_raw = backend.complete(
                    "memory_refiner", _EXPEL_REFINER_SYSTEM, refinement_request,
                )
                candidate_bank = _parse_items(
                    refinement_raw,
                    allowed_episode_ids=cumulative_episode_ids,
                    receipt_to_episode=receipt_to_episode,
                    allowed_kinds={"INSIGHT"},
                    maximum_items=maximum_items_per_call,
                )
                refinement_call_receipts.append({
                    "source_call_index": call_index,
                    "refinement_round": refinement_round,
                    "input_sha256": stable_hash(refinement_request),
                    "response_sha256": stable_hash(refinement_raw),
                    "item_ids": [item.item_id for item in candidate_bank],
                    "usage": dict(getattr(backend, "last_usage", {}) or {}),
                })
            all_items = list(candidate_bank)
        else:
            all_items.extend(items)
        call_receipts.append({
            "call_index": call_index,
            "input_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "attempts": attempt,
            "item_ids": [item.item_id for item in items],
            "usage": induction_usage,
        })
    body = {
        "schema_version": 2,
        "artifact_kind": "FROZEN_CROSS_DOMAIN_MEMORY_BASELINE",
        "method": method.value,
        "source_domains": sorted({episode.source_domain for episode in episodes}),
        "source_episode_ids": [episode.episode_id for episode in episodes],
        "source_episode_domains": {
            episode.episode_id: episode.source_domain for episode in episodes
        },
        # Shared across every method, so all four arms provably start from one pool.
        "source_superset_sha256": stable_hash(superset_payload),
        "source_superset_episode_ids": [episode.episode_id for episode in superset],
        "source_outcome_census": _outcome_census(superset),
        "source_projection": projection,
        "source_payload_sha256": stable_hash(canonical),
        "induction_calls": len(calls),
        "expel_refinement_rounds": (
            int(expel_refinement_rounds) if method == MemoryBaseline.EXPEL else 0
        ),
        "refinement_calls": len(refinement_call_receipts),
        "maximum_episodes_per_call": maximum_episodes_per_call,
        "source_receipt_ids": sorted(receipt_to_episode),
        "backend_identity": dict(backend.identity),
        "backend_identity_sha256": stable_hash(backend.identity),
        "items": [asdict(item) for item in all_items],
        "call_receipts": call_receipts,
        "refinement_call_receipts": refinement_call_receipts,
        "online_memory_updates_allowed": False,
        "target_actions_in_memory_allowed": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    validate_memory_artifact(artifact)
    return artifact


def build_trajectory_memory_artifact(
    method: MemoryControl | str,
    source_payload: Mapping[str, Any],
    *,
    random_seed: int = 0,
    maximum_trajectory_tokens: int = 2400,
    benchmark_predicate: Callable[[Mapping[str, Any]], bool | None] | None = None,
    shared_evaluator: Callable[[Mapping[str, Any]], bool | None] | None = None,
) -> dict[str, Any]:
    """Build a model-free one-item-per-episode control bank.

    ``episodic_rag`` and ``random_trajectory_icl`` contain identical trajectory
    text.  They differ only in the frozen retrieval policy, isolating semantic
    retrieval from the effect of adding source context.
    """
    method = MemoryControl(method)
    if maximum_trajectory_tokens < 1:
        raise ValueError("maximum_trajectory_tokens must be positive")
    episodes = canonical_source_episodes(
        source_payload,
        benchmark_predicate=benchmark_predicate,
        shared_evaluator=shared_evaluator,
    )
    # These controls do not induce outcome-conditioned lessons, so withholding
    # the middle tercile would give them less raw experience for no algorithmic
    # reason. UNKNOWN remains explicit inside each trajectory rather than being
    # silently converted to success or failure.
    eligible = episodes
    items = []
    for episode in eligible:
        trajectory = {
            "source_domain": episode.source_domain,
            "outcome": episode.outcome,
            "terminated": episode.terminated,
            "truncated": episode.truncated,
            "steps": [asdict(step) for step in episode.steps],
        }
        serialised, _ = _token_bounded_prefix(
            json.dumps(trajectory, ensure_ascii=False, sort_keys=True),
            maximum_trajectory_tokens,
        )
        item = MemoryItem.create(
            title=f"Source trajectory {episode.episode_id}",
            content=serialised,
            applicability="Use only if the target task and visible state make this experience relevant.",
            kind="TRAJECTORY",
            source_episode_ids=(episode.episode_id,),
            evidence_receipt_ids=tuple(step.receipt_id for step in episode.steps),
        )
        items.append(item)
    canonical = canonical_source_payload(episodes)
    body = {
        "schema_version": 2,
        "artifact_kind": "FROZEN_CROSS_DOMAIN_TRAJECTORY_CONTROL",
        "method": method.value,
        "retrieval_strategy": (
            "semantic" if method is MemoryControl.EPISODIC_RAG else "frozen_random"
        ),
        "random_seed": int(random_seed),
        "maximum_trajectory_tokens": int(maximum_trajectory_tokens),
        "trajectory_serialization": "canonical JSON token-bounded prefix without summarization",
        "source_domains": sorted({episode.source_domain for episode in eligible}),
        "source_episode_ids": [episode.episode_id for episode in eligible],
        "source_episode_domains": {
            episode.episode_id: episode.source_domain for episode in eligible
        },
        "source_superset_sha256": stable_hash(canonical),
        "source_superset_episode_ids": [episode.episode_id for episode in episodes],
        "source_outcome_census": _outcome_census(episodes),
        "source_projection": {
            "method": method.value,
            "eligible_outcomes": [label.value for label in OutcomeLabel],
            "eligible_episode_ids": [episode.episode_id for episode in eligible],
            "withheld": [],
        },
        "source_payload_sha256": stable_hash(canonical_source_payload(eligible)),
        "source_receipt_ids": sorted(
            step.receipt_id for episode in eligible for step in episode.steps
        ),
        "backend_identity": {"backend": "deterministic-trajectory-control"},
        "backend_identity_sha256": stable_hash({"backend": "deterministic-trajectory-control"}),
        "items": [asdict(item) for item in items],
        "call_receipts": [],
        "online_memory_updates_allowed": False,
        "target_actions_in_memory_allowed": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    validate_memory_artifact(artifact)
    return artifact


def build_external_memory_artifact(
    method: str,
    source_payload: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    producer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze an external method (notably Ours) into the shared runtime schema."""
    method = str(method)
    if method != OURS_MEMORY_METHOD:
        raise ValueError("external artifact builder currently accepts method='ours' only")
    episodes = canonical_source_episodes(source_payload)
    canonical = canonical_source_payload(episodes)
    known_episodes = {episode.episode_id for episode in episodes}
    receipt_to_episode = {
        step.receipt_id: episode.episode_id
        for episode in episodes for step in episode.steps
    }
    items = []
    for row in candidates:
        receipt_ids = tuple(map(str, row.get("evidence_receipt_ids") or ()))
        if not receipt_ids or set(receipt_ids) - set(receipt_to_episode):
            raise ValueError("ours candidate cites missing source receipts")
        derived_episodes = tuple(sorted({receipt_to_episode[value] for value in receipt_ids}))
        declared_episodes = tuple(sorted(map(str, row.get("source_episode_ids") or ())))
        if declared_episodes and declared_episodes != derived_episodes:
            raise ValueError("ours candidate episode lineage disagrees with its receipts")
        item = MemoryItem.create(
            title=str(row.get("title") or ""),
            content=str(row.get("content") or ""),
            applicability=str(row.get("applicability") or ""),
            kind=str(row.get("kind") or "SKILL"),
            source_episode_ids=derived_episodes,
            evidence_receipt_ids=receipt_ids,
        )
        items.append(item)
    if not items:
        raise ValueError("ours candidate bank is empty")
    body = {
        "schema_version": 2,
        "artifact_kind": "FROZEN_EXTERNAL_CROSS_DOMAIN_MEMORY",
        "method": method,
        "retrieval_strategy": "semantic",
        "source_domains": sorted({episode.source_domain for episode in episodes}),
        "source_episode_ids": [episode.episode_id for episode in episodes],
        "source_episode_domains": {
            episode.episode_id: episode.source_domain for episode in episodes
        },
        "source_superset_sha256": stable_hash(canonical),
        "source_superset_episode_ids": [episode.episode_id for episode in episodes],
        "source_outcome_census": _outcome_census(episodes),
        "source_projection": {
            "method": method,
            "eligible_outcomes": [label.value for label in OutcomeLabel],
            "eligible_episode_ids": sorted(known_episodes),
            "withheld": [],
        },
        "source_payload_sha256": stable_hash(canonical),
        "source_receipt_ids": sorted(receipt_to_episode),
        "backend_identity": dict(producer_identity),
        "backend_identity_sha256": stable_hash(producer_identity),
        "items": [asdict(item) for item in items],
        "call_receipts": [],
        "online_memory_updates_allowed": False,
        "target_actions_in_memory_allowed": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    validate_memory_artifact(artifact)
    return artifact


def validate_memory_artifact(artifact: Mapping[str, Any]) -> None:
    claimed = str(artifact.get("artifact_sha256") or "")
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("cross-domain memory artifact hash mismatch")
    method = str(artifact.get("method") or "")
    if method not in set(comparison_memory_methods()):
        raise ValueError(f"unknown cross-domain memory method: {method!r}")
    known_episodes = set(map(str, artifact.get("source_episode_ids") or ()))
    known_receipts = set(map(str, artifact.get("source_receipt_ids") or ()))
    if not known_episodes or not known_receipts:
        raise ValueError("memory artifact has empty source lineage")
    seen = set()
    for raw in artifact.get("items") or ():
        item = MemoryItem(**{
            **dict(raw),
            "source_episode_ids": tuple(raw["source_episode_ids"]),
            "evidence_receipt_ids": tuple(raw["evidence_receipt_ids"]),
        })
        rebuilt = MemoryItem.create(
            title=item.title, content=item.content, applicability=item.applicability,
            kind=item.kind, source_episode_ids=item.source_episode_ids,
            evidence_receipt_ids=item.evidence_receipt_ids,
        )
        if rebuilt != item or item.item_id in seen:
            raise ValueError("memory item is invalid or duplicated")
        if set(item.source_episode_ids) - known_episodes or set(item.evidence_receipt_ids) - known_receipts:
            raise ValueError("memory item cites lineage outside the frozen artifact")
        seen.add(item.item_id)
    binding = artifact.get("target_binding")
    if not seen and not (
        artifact.get("artifact_kind") == "FROZEN_TARGET_BOUND_CROSS_DOMAIN_MEMORY_BASELINE"
        and isinstance(binding, Mapping)
        and binding.get("binding_status") == "ALL_ITEMS_ABSTAINED"
    ):
        raise ValueError("memory artifact has no items")
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise ValueError("target binding metadata is not an object")
        target_domain = TargetDomain(str(binding.get("target_domain") or ""))
        if target_domain.value in set(map(str, artifact.get("source_domains") or ())):
            raise ValueError("target-bound memory includes its target among source domains")
        if not str(binding.get("source_artifact_sha256") or ""):
            raise ValueError("target binding is missing source artifact lineage")
        bindings = binding.get("item_bindings")
        if not isinstance(bindings, list):
            raise ValueError("target item bindings are not a list")
        if {str(row.get("bound_item_id")) for row in bindings} != seen:
            raise ValueError("target item bindings do not cover the frozen memory items")


_TARGET_BINDER_SYSTEM = (
    "You adapt source-domain memory to a named target domain using ONLY the supplied "
    "adaptation examples. The original memory may contain game-specific tactics. Preserve "
    "only a defensible abstract principle and bind it to observable target evidence; otherwise "
    "abstain. Never copy or recommend a target-native action, command, tool call, answer, or "
    "hidden outcome. Return exactly one JSON object {\"items\":[...]}. Emit one row per supplied "
    "source item with fields source_item_ref, abstain, title, content, applicability, "
    "information_to_check, expected_observation, contradiction_condition, stop_condition. "
    "When abstain is true the text fields may be empty. Do not mention a source game, game "
    "entity, controller button, or game-native command in a non-abstaining row."
)

_TARGET_BINDING_VERIFIER_SYSTEM = (
    "Verify whether each proposed cross-domain memory is directly supported by at least one "
    "supplied target adaptation example. Similar wording, generic plausibility, or metaphor is "
    "not evidence. Admit only when an adaptation trajectory visibly exhibits the proposed "
    "applicability condition and expected observation. Reject source-game mechanics that were "
    "merely renamed. Return exactly {\"decisions\":[...]} with one row per candidate_ref and "
    "fields candidate_ref, admit, supporting_example_ids, reason. An admitted row must cite at "
    "least one real supplied example_id."
)

_TARGET_GATE_ONLY_SYSTEM = (
    "You are a representation-neutral admission gate for cross-domain memory. "
    "For every supplied candidate, decide only ADMIT or REJECT from the supplied "
    "target adaptation examples. Never rewrite, summarize, repair, or translate a "
    "candidate. Admit only when at least one example directly demonstrates that the "
    "unchanged candidate is useful in this target domain; generic plausibility is not "
    "evidence. Return exactly {\"decisions\":[...]} with one row per candidate_ref "
    "and fields candidate_ref, admit, supporting_example_ids, reason. Every admitted "
    "row must cite at least one real example_id; rejected rows cite no examples."
)

_FORBIDDEN_BOUND_MEMORY_TERMS = (
    "tetris", "tetromino", "candy crush", "thunder force", "streets of rage",
    "game state", "game over", "line clear", "clearing lines", "hole-free base",
    "score", "lives", "reward signal", "reward value", "official_success",
    "gold answer", "correct answer",
)


def _normalise_command(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def text_names_native_action(text: str, native_actions: Sequence[str]) -> str | None:
    """Return the copied high-specificity command, while ignoring generic one-word verbs."""
    normalised_text = f" {_normalise_command(text)} "
    for raw_action in native_actions:
        raw = str(raw_action).strip()
        command = _normalise_command(raw)
        tokens = command.split()
        high_specificity = len(tokens) >= 2 or len(command) >= 12 or any(
            marker in raw for marker in ("(", ")", "[", "]", "{")
        )
        if high_specificity and command and f" {command} " in normalised_text:
            return raw
    return None


def _forbidden_bound_memory_term(text: str) -> str | None:
    lowered = str(text).casefold()
    for term in _FORBIDDEN_BOUND_MEMORY_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered):
            return term
    return None


def source_abstraction_audit(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Measure cross-game provenance without treating vocabulary deletion as abstraction."""
    validate_memory_artifact(artifact)
    episode_domains = {
        str(key): str(value)
        for key, value in (artifact.get("source_episode_domains") or {}).items()
    }
    rows = []
    for item in artifact["items"]:
        domains = sorted({
            episode_domains.get(
                str(episode_id),
                str(episode_id).split(":", 1)[0] if ":" in str(episode_id) else "UNKNOWN",
            )
            for episode_id in item["source_episode_ids"]
        })
        rows.append({
            "item_id": item["item_id"],
            "supporting_source_domains": domains,
            "cross_game_supported": len(set(domains) - {"UNKNOWN"}) >= 2,
        })
    supported = sum(bool(row["cross_game_supported"]) for row in rows)
    return {
        "items": rows,
        "item_count": len(rows),
        "cross_game_supported_items": supported,
        "cross_game_supported_fraction": supported / len(rows) if rows else 0.0,
    }


def _validated_adaptation_payload(
    payload: Mapping[str, Any], target_domain: TargetDomain,
) -> dict[str, Any]:
    if str(payload.get("split_role") or "") != "adaptation":
        raise TargetBindingError("target binding accepts adaptation split only")
    declared = TargetDomain(str(payload.get("target_domain") or ""))
    if declared is not target_domain:
        raise TargetBindingError("adaptation payload target domain mismatch")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise TargetBindingError("adaptation payload needs an examples list")
    seen: set[str] = set()
    for example in examples:
        if not isinstance(example, Mapping):
            raise TargetBindingError("adaptation example is not an object")
        example_id = str(example.get("example_id") or "").strip()
        if not example_id or example_id in seen:
            raise TargetBindingError("adaptation example identity is empty or duplicated")
        seen.add(example_id)
        split = str(example.get("split_role") or "adaptation")
        if split != "adaptation":
            raise TargetBindingError("qualification/formal example reached target binding")
    return json.loads(json.dumps(payload, ensure_ascii=False))


def gate_candidates_to_target(
    candidates: Sequence[Mapping[str, Any]],
    target_domain: TargetDomain | str,
    adaptation_payload: Mapping[str, Any],
    backend: CompletionBackend,
    *,
    maximum_items_per_call: int = 8,
    gate_retries: int = 3,
) -> dict[str, Any]:
    """Apply the same evidence-only, no-rewrite gate to any memory representation."""
    target_domain = TargetDomain(target_domain)
    adaptation = _validated_adaptation_payload(adaptation_payload, target_domain)
    native_actions = tuple(dict.fromkeys(
        str(action)
        for example in adaptation["examples"]
        for action in (example.get("native_actions") or ())
    ))
    known_examples = {str(example["example_id"]) for example in adaptation["examples"]}
    normalised: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw in candidates:
        candidate_id = str(raw.get("candidate_id") or "").strip()
        title = str(raw.get("title") or "").strip()
        content = str(raw.get("content") or "").strip()
        applicability = str(raw.get("applicability") or "").strip()
        if not candidate_id or candidate_id in seen_ids:
            raise TargetBindingError("candidate identity is empty or duplicated")
        if not title or not content or not applicability:
            raise TargetBindingError(f"candidate {candidate_id!r} has an empty rendered field")
        seen_ids.add(candidate_id)
        normalised.append({
            "candidate_id": candidate_id, "title": title,
            "content": content, "applicability": applicability,
        })

    decisions: list[dict[str, Any]] = []
    call_receipts: list[dict[str, Any]] = []
    size = max(1, int(maximum_items_per_call))
    for call_index, start in enumerate(range(0, len(normalised), size)):
        selected = normalised[start:start + size]
        eligible: list[dict[str, str]] = []
        for candidate in selected:
            copied = text_names_native_action(
                " ".join((candidate["title"], candidate["content"], candidate["applicability"])),
                native_actions,
            )
            if copied is not None:
                decisions.append({
                    "candidate_id": candidate["candidate_id"], "admit": False,
                    "supporting_example_ids": [],
                    "reason": f"mechanical target-action leakage: {copied}",
                    "decision_authority": "MECHANICAL_TARGET_ACTION_GATE",
                })
            else:
                eligible.append(candidate)
        if not eligible:
            continue
        reference_to_id = {
            f"G{start + offset:05d}": candidate["candidate_id"]
            for offset, candidate in enumerate(eligible)
        }
        candidate_by_id = {candidate["candidate_id"]: candidate for candidate in eligible}
        request = {
            "target_domain": target_domain.value,
            "adaptation_examples": adaptation["examples"],
            "candidates": [
                dict(candidate_by_id[candidate_id], candidate_ref=reference)
                for reference, candidate_id in reference_to_id.items()
            ],
        }
        error_text = ""
        for attempt in range(1, gate_retries + 1):
            call_request = request if not error_text else request | {
                "previous_attempt_rejected": error_text,
            }
            raw = backend.complete("memory_gate_verifier", _TARGET_GATE_ONLY_SYSTEM, call_request)
            usage = dict(getattr(backend, "last_usage", {}) or {})
            try:
                parsed = json.loads(raw)
                rows = parsed.get("decisions") if isinstance(parsed, Mapping) else None
                if not isinstance(rows, list):
                    raise TargetBindingError("memory gate returned no decisions list")
                returned = [str(row.get("candidate_ref") or "") for row in rows]
                if (len(rows) != len(reference_to_id) or set(returned) != set(reference_to_id)
                        or len(set(returned)) != len(returned)):
                    raise TargetBindingError("memory gate must decide every candidate exactly once")
                parsed_decisions: list[dict[str, Any]] = []
                for row in rows:
                    admit = bool(row.get("admit"))
                    supporting = list(map(str, row.get("supporting_example_ids") or ()))
                    if admit and (not supporting or set(supporting) - known_examples):
                        raise TargetBindingError("memory gate cited invalid adaptation evidence")
                    if not admit and supporting:
                        raise TargetBindingError("rejected memory gate row must not cite evidence")
                    parsed_decisions.append({
                        "candidate_id": reference_to_id[str(row["candidate_ref"])],
                        "admit": admit, "supporting_example_ids": supporting,
                        "reason": str(row.get("reason") or "")[:1000],
                        "decision_authority": "SHARED_EVIDENCE_GATE",
                    })
                decisions.extend(parsed_decisions)
                break
            except (ValueError, KeyError, TypeError) as error:
                error_text = str(error)[:400]
        else:
            raise TargetBindingError(
                f"memory gate call {call_index} failed {gate_retries} times: {error_text}"
            )
        call_receipts.append({
            "call_index": call_index, "input_sha256": stable_hash(request),
            "response_sha256": stable_hash(raw), "attempts": attempt, "usage": usage,
        })

    decision_by_id = {row["candidate_id"]: row for row in decisions}
    if set(decision_by_id) != seen_ids or len(decisions) != len(seen_ids):
        raise TargetBindingError("shared gate did not produce exactly one decision per candidate")
    ordered = [decision_by_id[candidate["candidate_id"]] for candidate in normalised]
    return {
        "schema_version": 1, "gate_policy": "SHARED_GATE_ONLY_NO_REWRITE",
        "target_domain": target_domain.value,
        "adaptation_payload_sha256": stable_hash(adaptation),
        "adaptation_example_ids": sorted(known_examples),
        "candidate_payload_sha256": stable_hash(normalised),
        "admitted_candidate_ids": [row["candidate_id"] for row in ordered if row["admit"]],
        "decisions": ordered, "call_receipts": call_receipts,
        "backend_identity": dict(backend.identity),
        "backend_identity_sha256": stable_hash(backend.identity),
    }


def gate_memory_artifact_to_target(
    source_artifact: Mapping[str, Any],
    target_domain: TargetDomain | str,
    adaptation_payload: Mapping[str, Any],
    backend: CompletionBackend,
    *,
    maximum_items_per_call: int = 8,
    gate_retries: int = 3,
) -> dict[str, Any]:
    """Freeze the admitted subset of a source bank without changing any item bytes."""
    validate_memory_artifact(source_artifact)
    target_domain = TargetDomain(target_domain)
    if target_domain.value in set(map(str, source_artifact["source_domains"])):
        raise TargetBindingError("cannot gate memory back to a source domain")
    source_items = list(source_artifact["items"])
    receipt = gate_candidates_to_target(
        [{"candidate_id": str(item["item_id"]), "title": str(item["title"]),
          "content": str(item["content"]), "applicability": str(item["applicability"])}
         for item in source_items],
        target_domain, adaptation_payload, backend,
        maximum_items_per_call=maximum_items_per_call, gate_retries=gate_retries,
    )
    admitted = set(receipt["admitted_candidate_ids"])
    decision_by_id = {row["candidate_id"]: row for row in receipt["decisions"]}
    bound_items = [copy.deepcopy(item)
                   for item in source_items if str(item["item_id"]) in admitted]
    item_bindings = [{
        "source_item_id": str(item["item_id"]), "bound_item_id": str(item["item_id"]),
        "verified_supporting_example_ids": decision_by_id[str(item["item_id"])]["supporting_example_ids"],
        "verification_reason": decision_by_id[str(item["item_id"])]["reason"],
    } for item in bound_items]
    body = {
        key: json.loads(json.dumps(value, ensure_ascii=False))
        for key, value in source_artifact.items()
        if key not in {"artifact_sha256", "items", "call_receipts", "backend_identity",
                       "backend_identity_sha256", "target_binding"}
    }
    body.update({
        "schema_version": 4,
        "artifact_kind": "FROZEN_TARGET_BOUND_CROSS_DOMAIN_MEMORY_BASELINE",
        "items": bound_items, "backend_identity": receipt["backend_identity"],
        "backend_identity_sha256": receipt["backend_identity_sha256"],
        "call_receipts": receipt["call_receipts"],
        "target_binding": {
            "target_domain": target_domain.value,
            "binding_mode": "SHARED_GATE_ONLY_NO_REWRITE",
            "binding_status": "BOUND_ITEMS_AVAILABLE" if bound_items else "ALL_ITEMS_ABSTAINED",
            "source_artifact_sha256": source_artifact["artifact_sha256"],
            "adaptation_payload_sha256": receipt["adaptation_payload_sha256"],
            "adaptation_example_ids": receipt["adaptation_example_ids"],
            "item_bindings": item_bindings, "gate_decisions": receipt["decisions"],
            "candidate_payload_sha256": receipt["candidate_payload_sha256"],
            "source_abstraction_audit": source_abstraction_audit(source_artifact),
        },
        "online_memory_updates_allowed": False, "target_actions_in_memory_allowed": False,
    })
    artifact = body | {"artifact_sha256": stable_hash(body)}
    validate_memory_artifact(artifact)
    return artifact


def bind_memory_artifact_to_target(
    source_artifact: Mapping[str, Any],
    target_domain: TargetDomain | str,
    adaptation_payload: Mapping[str, Any],
    backend: CompletionBackend,
    *,
    maximum_items_per_call: int = 8,
    binding_retries: int = 3,
) -> dict[str, Any]:
    """Translate a frozen source bank using a target adaptation split, then freeze it."""
    validate_memory_artifact(source_artifact)
    target_domain = TargetDomain(target_domain)
    if target_domain.value in set(map(str, source_artifact["source_domains"])):
        raise TargetBindingError("cannot bind memory back to a source domain")
    adaptation = _validated_adaptation_payload(adaptation_payload, target_domain)
    native_actions = tuple(dict.fromkeys(
        str(action)
        for example in adaptation["examples"]
        for action in (example.get("native_actions") or ())
    ))
    source_items = list(source_artifact["items"])
    size = max(1, int(maximum_items_per_call))
    bound_items: list[MemoryItem] = []
    item_bindings: list[dict[str, Any]] = []
    call_receipts: list[dict[str, Any]] = []
    source_by_id = {str(item["item_id"]): item for item in source_items}
    for call_index, start in enumerate(range(0, len(source_items), size)):
        selected = source_items[start:start + size]
        reference_to_id = {
            f"M{start + offset:05d}": str(item["item_id"])
            for offset, item in enumerate(selected)
        }
        allowed = set(reference_to_id)
        request = {
            "target_domain": target_domain.value,
            "forbidden_output_terms": list(_FORBIDDEN_BOUND_MEMORY_TERMS),
            "adaptation_examples": adaptation["examples"],
            "source_items": [
                {
                    "source_item_ref": reference,
                    "kind": item["kind"],
                    "title": item["title"],
                    "content": item["content"],
                    "applicability": item["applicability"],
                    "supporting_source_domains": sorted({
                        (source_artifact.get("source_episode_domains") or {}).get(
                            episode_id, "UNKNOWN"
                        )
                        for episode_id in item["source_episode_ids"]
                    }),
                }
                for reference, item in zip(reference_to_id, selected)
            ],
        }
        error_text = ""
        for attempt in range(1, binding_retries + 1):
            call_request = request if not error_text else request | {
                "previous_attempt_rejected": error_text
            }
            raw = backend.complete("memory_binder", _TARGET_BINDER_SYSTEM, call_request)
            try:
                parsed = json.loads(raw)
                rows = parsed.get("items") if isinstance(parsed, Mapping) else None
                if not isinstance(rows, list):
                    raise TargetBindingError("binder response has no items list")
                returned = [str(row.get("source_item_ref") or "") for row in rows]
                if len(rows) != len(selected) or set(returned) != allowed or len(set(returned)) != len(returned):
                    raise TargetBindingError("binder must return every supplied source item exactly once")
                pending_items: list[MemoryItem] = []
                pending_bindings: list[dict[str, Any]] = []
                for row in rows:
                    source_id = reference_to_id[str(row["source_item_ref"])]
                    if bool(row.get("abstain")):
                        continue
                    required = [
                        str(row.get(key) or "").strip()
                        for key in (
                            "title", "content", "applicability", "information_to_check",
                            "expected_observation", "contradiction_condition", "stop_condition",
                        )
                    ]
                    if not all(required):
                        raise TargetBindingError("non-abstaining binding has empty required fields")
                    rendered = (
                        f"Principle: {required[1]} Information to check: {required[3]} "
                        f"Expected observation: {required[4]} Contradiction: {required[5]} "
                        f"Stop condition: {required[6]}"
                    )
                    copied = text_names_native_action(
                        " ".join((required[0], rendered, required[2])), native_actions,
                    )
                    if copied is not None:
                        raise TargetBindingError(
                            f"target-bound memory copies native action {copied!r}"
                        )
                    forbidden = _forbidden_bound_memory_term(
                        " ".join((required[0], rendered, required[2]))
                    )
                    if forbidden is not None:
                        raise TargetBindingError(
                            f"target-bound memory retains forbidden source/outcome term {forbidden!r}"
                        )
                    source = source_by_id[source_id]
                    item = MemoryItem.create(
                        title=required[0], content=rendered, applicability=required[2],
                        kind=source["kind"],
                        source_episode_ids=source["source_episode_ids"],
                        evidence_receipt_ids=source["evidence_receipt_ids"],
                    )
                    pending_items.append(item)
                    pending_bindings.append({
                        "source_item_id": source_id,
                        "bound_item_id": item.item_id,
                        "adaptation_example_ids": [
                            str(example["example_id"]) for example in adaptation["examples"]
                        ],
                    })
                verifier_response = ""
                verifier_usage: dict[str, Any] = {}
                if pending_items:
                    candidate_refs = {
                        f"B{index:03d}": item.item_id
                        for index, item in enumerate(pending_items)
                    }
                    item_by_id = {item.item_id: item for item in pending_items}
                    binding_by_id = {
                        row["bound_item_id"]: row for row in pending_bindings
                    }
                    verifier_request = {
                        "target_domain": target_domain.value,
                        "adaptation_examples": adaptation["examples"],
                        "candidates": [
                            {
                                "candidate_ref": reference,
                                "title": item_by_id[item_id].title,
                                "content": item_by_id[item_id].content,
                                "applicability": item_by_id[item_id].applicability,
                            }
                            for reference, item_id in candidate_refs.items()
                        ],
                    }
                    verifier_response = backend.complete(
                        "memory_binding_verifier",
                        _TARGET_BINDING_VERIFIER_SYSTEM,
                        verifier_request,
                    )
                    verifier_usage = dict(getattr(backend, "last_usage", {}) or {})
                    verified = json.loads(verifier_response)
                    decisions = verified.get("decisions") if isinstance(verified, Mapping) else None
                    if not isinstance(decisions, list):
                        raise TargetBindingError("binding verifier returned no decisions list")
                    returned_refs = [str(row.get("candidate_ref") or "") for row in decisions]
                    if set(returned_refs) != set(candidate_refs) or len(returned_refs) != len(set(returned_refs)):
                        raise TargetBindingError("binding verifier must decide every candidate exactly once")
                    known_examples = {
                        str(example["example_id"]) for example in adaptation["examples"]
                    }
                    admitted_ids: set[str] = set()
                    for decision in decisions:
                        if not bool(decision.get("admit")):
                            continue
                        supporting = list(map(str, decision.get("supporting_example_ids") or ()))
                        if not supporting or set(supporting) - known_examples:
                            raise TargetBindingError("binding verifier cited invalid adaptation evidence")
                        item_id = candidate_refs[str(decision["candidate_ref"])]
                        admitted_ids.add(item_id)
                        binding_by_id[item_id]["verified_supporting_example_ids"] = supporting
                        binding_by_id[item_id]["verification_reason"] = str(
                            decision.get("reason") or ""
                        )[:1000]
                    pending_items = [item for item in pending_items if item.item_id in admitted_ids]
                    pending_bindings = [
                        row for row in pending_bindings if row["bound_item_id"] in admitted_ids
                    ]
                break
            except (ValueError, KeyError, TypeError) as error:
                error_text = str(error)[:400]
        else:
            raise TargetBindingError(
                f"target binding call {call_index} failed {binding_retries} times: {error_text}"
            )
        bound_items.extend(pending_items)
        item_bindings.extend(pending_bindings)
        call_receipts.append({
            "call_index": call_index,
            "input_sha256": stable_hash(request),
            "response_sha256": stable_hash(raw),
            "attempts": attempt,
            "admitted": len(pending_items),
            "abstained": len(selected) - len(pending_items),
            "usage": dict(getattr(backend, "last_usage", {}) or {}),
            "verification_response_sha256": (
                stable_hash(verifier_response) if verifier_response else None
            ),
            "verification_usage": verifier_usage,
        })
    body = {
        key: json.loads(json.dumps(value, ensure_ascii=False))
        for key, value in source_artifact.items()
        if key not in {"artifact_sha256", "items", "call_receipts", "backend_identity", "backend_identity_sha256"}
    }
    body.update({
        "schema_version": 3,
        "artifact_kind": "FROZEN_TARGET_BOUND_CROSS_DOMAIN_MEMORY_BASELINE",
        "items": [asdict(item) for item in bound_items],
        "backend_identity": dict(backend.identity),
        "backend_identity_sha256": stable_hash(backend.identity),
        "call_receipts": call_receipts,
        "target_binding": {
            "target_domain": target_domain.value,
            "binding_status": (
                "BOUND_ITEMS_AVAILABLE" if bound_items else "ALL_ITEMS_ABSTAINED"
            ),
            "source_artifact_sha256": source_artifact["artifact_sha256"],
            "adaptation_payload_sha256": stable_hash(adaptation),
            "adaptation_example_ids": [
                str(example["example_id"]) for example in adaptation["examples"]
            ],
            "item_bindings": item_bindings,
            "source_abstraction_audit": source_abstraction_audit(source_artifact),
        },
        "online_memory_updates_allowed": False,
        "target_actions_in_memory_allowed": False,
    })
    artifact = body | {"artifact_sha256": stable_hash(body)}
    validate_memory_artifact(artifact)
    return artifact


_SENSITIVE_TARGET_KEYS = {
    "answer", "gold", "gold_answer", "ground_truth", "label", "official_success",
    "reward", "score", "correct",
}


def _reject_target_outcomes(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _SENSITIVE_TARGET_KEYS:
                raise ValueError(f"target context exposes forbidden outcome field: {key}")
            _reject_target_outcomes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_target_outcomes(child)


def adapt_target_context(
    domain: TargetDomain | str,
    *,
    task: str,
    observation: Mapping[str, Any],
    native_actions: Sequence[str] = (),
    history: Sequence[Mapping[str, Any]] = (),
    proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the same outcome-blind query format for the first four targets."""
    domain = TargetDomain(domain)
    raw = {
        "task": task,
        "observation": dict(observation),
        "native_actions": list(map(str, native_actions)),
        "history": [dict(row) for row in history],
        "proposal": dict(proposal) if proposal is not None else None,
    }
    _reject_target_outcomes(raw)
    observation_keys = {
        TargetDomain.WEBSHOP: ("url", "axtree", "axtree_object", "observation"),
        TargetDomain.ALFWORLD: ("observation", "structured_state"),
        TargetDomain.DISCOVERYWORLD: ("observation", "observable_state", "structured_state"),
        TargetDomain.TIRBENCH: ("prompt", "task", "tool_trace", "media_receipts", "observation"),
    }[domain]
    compact_observation = {
        key: observation[key] for key in observation_keys if key in observation
    }
    if not compact_observation:
        compact_observation = dict(observation)
    result = {
        "schema_version": 1,
        "target_domain": domain.value,
        "task": task,
        "observation": compact_observation,
        "native_actions": list(map(str, native_actions)),
        "history": [dict(row) for row in history],
        "proposal": dict(proposal) if proposal is not None else None,
    }
    result["query_sha256"] = stable_hash(result)
    return result


def _item_text(item: Mapping[str, Any]) -> str:
    return "\n".join((str(item["title"]), str(item["content"]), str(item["applicability"])))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions are empty or mismatched")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else -1.0


def retrieve_memory_items(
    artifact: Mapping[str, Any],
    target_context: Mapping[str, Any],
    embedding_backend: EmbeddingBackend,
    *,
    top_k: int = 3,
    maximum_memory_tokens: int = 1200,
) -> dict[str, Any]:
    validate_memory_artifact(artifact)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    target_domain = str(target_context.get("target_domain") or "")
    if target_domain in set(map(str, artifact["source_domains"])):
        raise ValueError("cross-domain baseline cannot retrieve from the target domain")
    if maximum_memory_tokens < 1:
        raise ValueError("maximum_memory_tokens must be positive")
    query = json.dumps(target_context, ensure_ascii=False, sort_keys=True)
    items = list(artifact["items"])
    strategy = str(artifact.get("retrieval_strategy") or "semantic")
    if strategy == "semantic":
        vectors = list(embedding_backend.embed([query, *[_item_text(item) for item in items]]))
        if len(vectors) != len(items) + 1:
            raise ValueError("embedding backend returned the wrong vector count")
        ranked = sorted(
            [{"rank_score": _cosine(vectors[0], vectors[index + 1]), "item": item}
             for index, item in enumerate(items)],
            key=lambda row: (-row["rank_score"], str(row["item"]["item_id"])),
        )[:top_k]
    elif strategy == "frozen_random":
        seed = int(artifact.get("random_seed") or 0)
        ranked = sorted(
            [{
                "rank_score": None,
                "random_order_sha256": stable_hash({
                    "seed": seed,
                    "source_superset_sha256": artifact["source_superset_sha256"],
                    "target_query_sha256": target_context.get("query_sha256") or stable_hash(target_context),
                    "item_id": item["item_id"],
                }),
                "item": item,
            } for item in items],
            key=lambda row: (row["random_order_sha256"], str(row["item"]["item_id"])),
        )[:top_k]
    else:
        raise ValueError(f"unknown memory retrieval strategy: {strategy!r}")
    rendered_tokens = 0
    per_item = max(1, int(maximum_memory_tokens) // max(1, len(ranked)))
    remaining = int(maximum_memory_tokens)
    for index, row in enumerate(ranked):
        text = _item_text(row["item"])
        allowance = remaining if index == len(ranked) - 1 else min(per_item, remaining)
        row["rendered_text"], used = _token_bounded_prefix(text, allowance)
        row["rendered_token_count"] = used
        rendered_tokens += used
        remaining -= used
    body = {
        "schema_version": 1,
        "method": artifact["method"],
        "memory_artifact_sha256": artifact["artifact_sha256"],
        "target_domain": target_domain,
        "target_query_sha256": target_context.get("query_sha256") or stable_hash(target_context),
        "embedding_identity": dict(embedding_backend.identity),
        "embedding_identity_sha256": stable_hash(embedding_backend.identity),
        "retrieval_strategy": strategy,
        "top_k": top_k,
        "maximum_memory_tokens": maximum_memory_tokens,
        "memory_token_counter": "unicode_lexical_v1",
        "rendered_memory_tokens": rendered_tokens,
        "retrieved": ranked,
        "online_memory_updated": False,
    }
    return body | {"retrieval_sha256": stable_hash(body)}


class CrossDomainMemoryAdvisor:
    """Turn a frozen retrieval receipt into an action-free Decision advisory."""

    def __init__(self, retrieval: Mapping[str, Any]) -> None:
        claimed = retrieval.get("retrieval_sha256")
        body = {key: value for key, value in retrieval.items() if key != "retrieval_sha256"}
        if not claimed or stable_hash(body) != claimed:
            raise ValueError("memory retrieval receipt hash mismatch")
        self.retrieval = dict(retrieval)

    def advisory(self) -> Advisory:
        items = [row["item"] for row in self.retrieval.get("retrieved") or ()]
        if not items:
            return Advisory(AdvisoryVerdict.ABSTAIN, "no cross-domain memory was retrieved")
        evidence = tuple(dict.fromkeys(
            str(receipt_id)
            for item in items for receipt_id in item["evidence_receipt_ids"]
        ))
        memory_text = "\n\n".join(
            str(row.get("rendered_text") or "")
            for row in self.retrieval.get("retrieved") or ()
            if row.get("rendered_text")
        )
        return Advisory(
            AdvisoryVerdict.ADMIT,
            "Untrusted cross-domain memory; apply only after checking live target evidence.",
            evidence,
            current_role=f"CROSS_DOMAIN_{str(self.retrieval['method']).upper()}",
            information_need=memory_text,
            expected_transition="Use live target receipts to check whether the retrieved memory applies.",
            failure_route="Ignore the memory and continue target-only when contradicted or irrelevant.",
            termination_test="Use only the target environment's native completion signal.",
        )


class CrossDomainMemoryDecisionAgent:
    """Inject retrieved memory while leaving action authority with a base agent.

    This wrapper implements the normal DecisionAgent interface.  It is therefore
    usable with ``TwoAgentRuntime`` and every Environment adapter already in the
    repository.  The wrapped agent remains the only component that emits an
    action; the baseline contributes an ``Advisory`` only.
    """

    def __init__(
        self,
        decision_agent: Any,
        *,
        artifact: Mapping[str, Any],
        domain: TargetDomain | str,
        embedding_backend: EmbeddingBackend,
        top_k: int = 3,
        maximum_memory_tokens: int = 1200,
    ) -> None:
        validate_memory_artifact(artifact)
        self.decision_agent = decision_agent
        self.artifact = dict(artifact)
        self.domain = TargetDomain(domain)
        binding = self.artifact.get("target_binding")
        if binding is not None and str(binding.get("target_domain")) != self.domain.value:
            raise ValueError("target-bound memory artifact used for the wrong target domain")
        self.embedding_backend = embedding_backend
        self.top_k = top_k
        self.maximum_memory_tokens = maximum_memory_tokens
        self.retrieval_receipts: list[Mapping[str, Any]] = []

    @staticmethod
    def _history_view(history: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {
                "receipt_id": str(row.receipt_id),
                "action": str(row.action),
                "before_hash": str(row.before_hash),
                "after_hash": str(row.after_hash),
                "terminal": bool(row.done),
            }
            for row in history
        ]

    def propose_set(self, observation, goal, history, advisory):
        del advisory  # no second source channel is allowed in this baseline arm
        target = adapt_target_context(
            self.domain,
            task=goal,
            observation=observation.state,
            native_actions=observation.native_actions,
            history=self._history_view(history),
        )
        retrieval = retrieve_memory_items(
            self.artifact,
            target,
            self.embedding_backend,
            top_k=self.top_k,
            maximum_memory_tokens=self.maximum_memory_tokens,
        )
        self.retrieval_receipts.append(retrieval)
        memory_advisory = (
            CrossDomainMemoryAdvisor(retrieval).advisory()
            if retrieval["retrieved"] else None
        )
        copied = text_names_native_action(
            str(memory_advisory.information_need or "") if memory_advisory else "",
            observation.native_actions,
        )
        if copied is not None:
            raise ValueError(f"cross-domain memory names a target-native action: {copied!r}")
        return self.decision_agent.propose_set(observation, goal, history, memory_advisory)

    def assess_transition(self, before, proposal_set, after, reward, history):
        return self.decision_agent.assess_transition(
            before, proposal_set, after, reward, history,
        )


class LocalSentenceTransformerEmbeddingBackend:
    """Lazy open-weight embedding backend (default model: Qwen3-Embedding-0.6B)."""

    def __init__(self, model: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        self.model_name = model
        self._model = None

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"backend": "sentence-transformers", "model": self.model_name}

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors]


class LocalHashingEmbeddingBackend:
    """Dependency-light deterministic retriever for CPU smoke tests, not formal runs."""

    def __init__(self, dimensions: int = 4096) -> None:
        self.dimensions = int(dimensions)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "sklearn-hashing-vectorizer",
            "dimensions": self.dimensions,
            "pilot_only": True,
        }

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        rows = []
        for text in texts:
            tokens = re.findall(r"[a-z0-9]+", str(text).casefold())
            features = tokens + [
                f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
            ]
            vector = [0.0] * self.dimensions
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:8], "big") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            rows.append([value / norm for value in vector] if norm else vector)
        return rows


def method_code_reference(method: MemoryBaseline | str) -> str:
    return {
        MemoryBaseline.EXPEL: "https://github.com/LeapLabTHU/ExpeL",
        MemoryBaseline.AWM: "https://github.com/zorazrw/agent-workflow-memory",
        MemoryBaseline.REASONING_BANK: "https://github.com/google-research/reasoning-bank",
    }[MemoryBaseline(method)]


__all__ = [
    "CanonicalSourceEpisode", "CanonicalSourceStep", "CrossDomainMemoryAdvisor",
    "CrossDomainMemoryDecisionAgent",
    "EmbeddingBackend", "InsufficientEligibleSourceError",
    "LocalHashingEmbeddingBackend", "LocalSentenceTransformerEmbeddingBackend", "MemoryBaseline",
    "MemoryControl", "OURS_MEMORY_METHOD", "build_external_memory_artifact",
    "build_trajectory_memory_artifact",
    "comparison_memory_methods",
    "MemoryItem", "OutcomeAuthority", "OutcomeLabel", "TargetBindingError", "TargetDomain",
    "adapt_target_context", "canonical_source_episodes",
    "canonical_source_payload", "bind_memory_artifact_to_target",
    "gate_candidates_to_target", "gate_memory_artifact_to_target",
    "induce_memory_artifact", "method_code_reference", "source_abstraction_audit",
    "resolve_source_outcome", "retrieve_memory_items", "source_projection",
    "text_names_native_action", "validate_memory_artifact",
]
