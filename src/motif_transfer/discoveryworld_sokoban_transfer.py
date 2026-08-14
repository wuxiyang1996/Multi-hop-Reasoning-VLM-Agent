"""Bind the source-qualified Sokoban effect program to DiscoveryWorld candidates.

Only the intervention-supported POSITION/COMMIT control relation transfers.
Candidate actions, object UUIDs, spatial relations, and effect probabilities are
DiscoveryWorld-native neural groundings.  The selector is deterministic and has
no access to the official task scorecard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .discoveryworld_env import DiscoveryWorldObservation, stable_hash
from .discoveryworld_policy import (
    native_action_from_decision,
    parse_json_object,
    target_native_facts,
)


SOURCE_PROGRAM_SHA256 = "6b02dc1d7271bbd435e90539cedd7d56d04fcc1ad03798dd6dd06146d67f1fcd"
SOURCE_CONFIRMATION_SHA256 = "d64606c916ce6e812ae1b920771d5175cb48983a2f141b2dab6f43d491a6c1ed"
ROLES = frozenset({"POSITION", "COMMIT"})
CONDITIONS = frozenset({
    "target_native_myopic",
    "authentic_sokoban_effect_plus_target",
    "commit_availability_control_plus_target",
    "inverted_effect_control_plus_target",
    "position_prior_control_plus_target",
})


TARGET_GROUNDER_SYSTEM_PROMPT = """You are a target-native DiscoveryWorld candidate grounder.
You receive only the current policy observation, exact target-native facts, and recent
target history. You never receive source-game content or the hidden evaluator scorecard.

The payload contains a separately inferred target_binding. Treat its exact UUID/name,
required relation, and commit_action as the current target hypothesis; do not replace them
with a nearer object or a different final action.
Propose 2-4 compact, diverse native actions that are valid now. Use exact integer UUIDs and exact
listed teleport names. POSITION means reversible navigation, preparation, information
gathering, measurement, or hypothesis testing. COMMIT means an action that could
irreversibly or finally satisfy a task requirement. This is descriptive grounding; do
not decide which role should be preferred.

For every candidate estimate prerequisite_probability, positive_effect_probability,
and information_gain_probability in [0,1]. Ground prerequisites only in supplied facts.
Evidence entries must be one of:
  {"kind":"inventory","uuid":INT,"name":"exact supplied name"}
  {"kind":"accessible","uuid":INT,"name":"exact supplied name"}
  {"kind":"relative_object","uuid":INT,"name":"exact supplied name",
   "relation_from_agent":"...","distance":INT}
For a spatial COMMIT, include the exact target-object relation required by the task; an
inventory fact alone is insufficient. If that relation is absent, give a low prerequisite
probability and propose a POSITION action that can establish or inspect it. Never invent
object coordinates. Facts override memory. Never call an animal UUID a statue UUID (or
vice versa); bind the exact supplied object name. If your reason says any prerequisite is
not yet satisfied, prerequisite_probability must be below 0.5. Information gain includes
making an uncertain goal relation directly observable. TELEPORT_TO_LOCATION and
TELEPORT_TO_OBJECT have different effects. When the task requires a spatial relation to
an exact target object and that relation is not currently satisfied, always include a
TELEPORT_TO_OBJECT candidate using that exact object's UUID; the similarly named
TELEPORT_TO_LOCATION is not a substitute because it need not land beside the object.

Return one JSON object:
{"memory":"...","running_hypotheses":["..."],"candidates":[
 {"action":"...","arg1":...,"arg2":...,"target_role":"POSITION|COMMIT",
  "prerequisite_probability":0.0,"positive_effect_probability":0.0,
  "information_gain_probability":0.0,"expected_effect":"...",
  "evidence":[{...}],"reason":"..."}]}
Include only native arguments required by each action. Keep memory under 300 characters,
at most four hypotheses, and each expected_effect/reason under 40 words. Never narrate or
infer coordinates; use only supplied relations. No prose or Markdown."""


TARGET_BINDER_SYSTEM_PROMPT = """You are a target-native DiscoveryWorld entity/relation binder.
Use the task text, current scientific hypotheses, exact object facts, and known_actions.
Bind the object
that the current hypothesis says should receive the final task action; do not choose the
nearest same-type object and never invent a UUID or coordinate. Scientific memory may be
used to identify the hypothesis object (for example which sample is contaminated), but its
coordinates are untrusted and must never determine the goal relation.

If the task requires the commit subject (for example the held flag) to be in a spatial
relation to the target, copy that TASK GOAL relation without changing viewpoint. This is the
required goal relation, never the current observed relation. Example: "flag one square west
of target" means commit_subject_relation_to_target=west and target_distance=1. A symbolic
relation algebra will invert the viewpoint later. For "put X in Y", use inside with null
distance; containment is not same_location. Otherwise use null.
Return exactly:
{"target_uuid":INT,"target_name":"exact supplied name","commit_subject_relation_to_target":
 "north|east|south|west|same_location|inside|null","target_distance":INT_OR_NULL,
 "commit_action":{"action":"NATIVE_ACTION","arg1":...},
 "confidence":0.0,"hypothesis_used":"...","reason":"..."}
commit_action is the final native action named by the task. Its action string must be copied
exactly from known_actions, with only the arguments that known_actions requires and exact
currently supplied object UUID arguments. It is a symbolic binding, not a recommendation to
execute now. For PUT, bind target_uuid/name to the receiving container/device (arg2), and
bind the hypothesis object as arg1. Use PUT only when the task explicitly requires putting
an object inside a container/device. To place a held object on the ground beside a target,
bind DROP with only the held object's UUID; spatial positioning is a separate action.
Keep reason under 80 words.
No source-game content, scorecard, prose, or Markdown."""


@dataclass(frozen=True)
class DiscoveryWorldGroundedCandidate:
    action: Mapping[str, Any]
    target_role: str
    prerequisite_probability: float
    positive_effect_probability: float
    information_gain_probability: float
    expected_effect: str
    evidence: tuple[Mapping[str, Any], ...]
    reason: str

    @property
    def candidate_sha256(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class DiscoveryWorldTargetBinding:
    target_uuid: int
    target_name: str
    commit_subject_relation_to_target: str | None
    target_relation_from_agent: str | None
    target_distance: int | None
    commit_action: Mapping[str, Any]
    confidence: float
    hypothesis_used: str
    reason: str

    @property
    def binding_sha256(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class DiscoveryWorldSelectionReceipt:
    schema_version: str
    condition: str
    source_program_sha256: str | None
    source_confirmation_sha256: str | None
    candidate_bundle_sha256: str
    selected_candidate_sha256: str
    selected_action: Mapping[str, Any]
    selected_role: str
    evidence_supported: bool
    target_bound_position: bool
    commit_available: bool
    positive_commit_effect_witnessed: bool
    positive_commit_effect_kind: str | None
    selection_reason: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        condition: str,
        candidates: Sequence[DiscoveryWorldGroundedCandidate],
        selected: DiscoveryWorldGroundedCandidate,
        evidence_supported: bool,
        target_bound_position: bool,
        commit_available: bool,
        positive_commit_effect_witnessed: bool,
        positive_commit_effect_kind: str | None,
        selection_reason: str,
    ) -> "DiscoveryWorldSelectionReceipt":
        uses_source = condition not in {"target_native_myopic"}
        body = {
            "schema_version": "discoveryworld-sokoban-selection-v1",
            "condition": condition,
            "source_program_sha256": SOURCE_PROGRAM_SHA256 if uses_source else None,
            "source_confirmation_sha256": SOURCE_CONFIRMATION_SHA256 if uses_source else None,
            "candidate_bundle_sha256": stable_hash([asdict(row) for row in candidates]),
            "selected_candidate_sha256": selected.candidate_sha256,
            "selected_action": dict(selected.action),
            "selected_role": selected.target_role,
            "evidence_supported": bool(evidence_supported),
            "target_bound_position": bool(target_bound_position),
            "commit_available": bool(commit_available),
            "positive_commit_effect_witnessed": bool(
                positive_commit_effect_witnessed
            ),
            "positive_commit_effect_kind": positive_commit_effect_kind,
            "selection_reason": str(selection_reason),
        }
        return cls(receipt_sha256=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        expected = body.pop("receipt_sha256")
        return expected == stable_hash(body)


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    output = float(value)
    if not 0.0 <= output <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return output


def parse_grounded_candidates(
    raw: str,
    observation: DiscoveryWorldObservation,
) -> tuple[dict[str, Any], tuple[DiscoveryWorldGroundedCandidate, ...]]:
    bundle = parse_json_object(raw)
    rows = bundle.get("candidates")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 6:
        raise ValueError("candidate grounder must return 2-6 candidates")
    candidates = []
    rejections = []
    seen_actions = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            rejections.append({"index": index, "error": "candidate is not an object"})
            continue
        try:
            action = native_action_from_decision(row, observation)
            action_hash = stable_hash(action)
            if action_hash in seen_actions:
                raise ValueError("duplicate native action")
            role = str(row.get("target_role") or "")
            if role not in ROLES:
                raise ValueError("invalid target_role")
            evidence = row.get("evidence") or []
            if not isinstance(evidence, list) or not all(
                isinstance(item, Mapping) for item in evidence
            ):
                raise ValueError("evidence must be a list of objects")
            candidate = DiscoveryWorldGroundedCandidate(
                action=action,
                target_role=role,
                prerequisite_probability=_probability(
                    row.get("prerequisite_probability"), "prerequisite_probability",
                ),
                positive_effect_probability=_probability(
                    row.get("positive_effect_probability"), "positive_effect_probability",
                ),
                information_gain_probability=_probability(
                    row.get("information_gain_probability"), "information_gain_probability",
                ),
                expected_effect=str(row.get("expected_effect") or "")[:2000],
                evidence=tuple(dict(item) for item in evidence),
                reason=str(row.get("reason") or "")[:2000],
            )
        except (KeyError, TypeError, ValueError) as exc:
            rejections.append({
                "index": index,
                "error": f"{type(exc).__name__}: {exc}",
                "candidate_sha256": stable_hash(dict(row)),
            })
            continue
        seen_actions.add(action_hash)
        candidates.append(candidate)
    if len(candidates) < 2:
        raise ValueError(
            f"fewer than two distinct valid candidates; rejections={rejections}"
        )
    bundle = dict(bundle)
    bundle["candidate_parse_rejections"] = rejections
    return bundle, tuple(candidates)


def parse_target_binding(
    raw: str, observation: DiscoveryWorldObservation,
) -> DiscoveryWorldTargetBinding:
    row = parse_json_object(raw)
    uuid = row.get("target_uuid")
    name = row.get("target_name")
    if not isinstance(uuid, int) or isinstance(uuid, bool):
        raise ValueError("target_uuid must be an integer")
    if not isinstance(name, str) or not name:
        raise ValueError("target_name must be nonempty")
    facts = target_native_facts(observation)
    catalog = {
        (item.get("uuid"), item.get("name"))
        for key in ("inventory", "accessible_objects", "salient_relative_objects")
        for item in facts[key]
    }
    if (uuid, name) not in catalog:
        raise ValueError("target binding UUID/name is absent from exact target facts")
    subject_relation = row.get("commit_subject_relation_to_target")
    if subject_relation in {"null", "none", ""}:
        subject_relation = None
    if subject_relation not in {
        None, "north", "east", "south", "west", "same_location", "inside",
    }:
        raise ValueError("invalid commit_subject_relation_to_target")
    inverse_relation = {
        None: None,
        "north": "south",
        "east": "west",
        "south": "north",
        "west": "east",
        "same_location": "same_location",
        "inside": None,
    }
    relation = inverse_relation[subject_relation]
    distance = row.get("target_distance")
    if distance is not None and (
        not isinstance(distance, int) or isinstance(distance, bool) or distance < 0
    ):
        raise ValueError("target_distance must be a nonnegative integer or null")
    if subject_relation in {None, "inside"} and distance is not None:
        raise ValueError("target_distance requires a target relation")
    if subject_relation not in {None, "inside"} and distance is None:
        raise ValueError("target relation requires target_distance")
    commit_row = row.get("commit_action")
    if not isinstance(commit_row, Mapping):
        raise ValueError("commit_action must be a native action object")
    commit_action = native_action_from_decision(commit_row, observation)
    return DiscoveryWorldTargetBinding(
        target_uuid=uuid,
        target_name=name,
        commit_subject_relation_to_target=subject_relation,
        target_relation_from_agent=relation,
        target_distance=distance,
        commit_action=commit_action,
        confidence=_probability(row.get("confidence"), "confidence"),
        hypothesis_used=str(row.get("hypothesis_used") or "")[:2000],
        reason=str(row.get("reason") or "")[:2000],
    )


def evidence_supported(
    candidate: DiscoveryWorldGroundedCandidate,
    observation: DiscoveryWorldObservation,
    target_binding: DiscoveryWorldTargetBinding | None = None,
) -> bool:
    if not candidate.evidence:
        return candidate.target_role == "POSITION"
    facts = target_native_facts(observation)
    inventory = {
        (int(row["uuid"]), str(row.get("name") or ""))
        for row in facts["inventory"] if isinstance(row.get("uuid"), int)
    }
    accessible = {
        (int(row["uuid"]), str(row.get("name") or ""))
        for row in facts["accessible_objects"]
        if isinstance(row.get("uuid"), int)
    }
    relative = {
        (row.get("uuid"), row.get("name"), row.get("relation_from_agent"), row.get("distance"))
        for row in facts["salient_relative_objects"]
    }
    for row in candidate.evidence:
        kind = str(row.get("kind") or "")
        uuid = row.get("uuid")
        name = row.get("name")
        if not isinstance(uuid, int) or isinstance(uuid, bool):
            return False
        if not isinstance(name, str) or not name:
            return False
        if kind == "inventory" and (uuid, name) not in inventory:
            return False
        if kind == "accessible" and (uuid, name) not in accessible:
            return False
        if kind == "relative_object" and (
            uuid, name, row.get("relation_from_agent"), row.get("distance")
        ) not in relative:
            return False
        if kind not in {"inventory", "accessible", "relative_object"}:
            return False
    if (
        candidate.target_role == "COMMIT"
        and target_binding is not None
        and target_binding.target_relation_from_agent is not None
    ):
        required = {
            "kind": "relative_object",
            "uuid": target_binding.target_uuid,
            "name": target_binding.target_name,
            "relation_from_agent": target_binding.target_relation_from_agent,
            "distance": target_binding.target_distance,
        }
        if not any(all(row.get(key) == value for key, value in required.items()) for row in candidate.evidence):
            return False
    return True


def commit_available(
    candidate: DiscoveryWorldGroundedCandidate,
    target_binding: DiscoveryWorldTargetBinding | None,
) -> bool:
    """Whether a valid target-native candidate is the bound final action.

    Candidate parsing has already checked the native action's immediate API
    preconditions.  This predicate deliberately excludes its goal effect.
    """

    return bool(
        candidate.target_role == "COMMIT"
        and target_binding is not None
        and dict(candidate.action) == dict(target_binding.commit_action)
    )


def target_bound_position(
    candidate: DiscoveryWorldGroundedCandidate,
    target_binding: DiscoveryWorldTargetBinding | None,
) -> bool:
    """Recognize the target-native affordance that positions beside a bound object."""

    return bool(
        candidate.target_role == "POSITION"
        and target_binding is not None
        and candidate.action.get("action") == "TELEPORT_TO_OBJECT"
        and candidate.action.get("arg1") == target_binding.target_uuid
    )


def positive_commit_effect_witnessed(
    candidate: DiscoveryWorldGroundedCandidate,
    observation: DiscoveryWorldObservation,
    target_binding: DiscoveryWorldTargetBinding | None,
) -> bool:
    return positive_commit_effect_kind(candidate, observation, target_binding) is not None


def positive_commit_effect_kind(
    candidate: DiscoveryWorldGroundedCandidate,
    observation: DiscoveryWorldObservation,
    target_binding: DiscoveryWorldTargetBinding | None,
) -> str | None:
    """Join neural bindings with exact target symbols to witness direct progress.

    The neural modules bind the final target, relation, and native commit action.
    The truth of the spatial predicate is computed from the current observation;
    neural probabilities or narrated coordinates cannot override it.
    """

    if not commit_available(candidate, target_binding) or target_binding is None:
        return None
    action = candidate.action
    if (
        action.get("action") == "PUT"
        and action.get("arg2") == target_binding.target_uuid
        and target_binding.commit_subject_relation_to_target == "inside"
    ):
        # Native candidate parsing has already established arg1 is held and
        # arg2 is accessible. The neural binding supplies which hypothesis
        # object and receiver instantiate the task-native assignment.
        return "ASSIGNMENT_IMPROVEMENT_AVAILABLE"
    relation = target_binding.target_relation_from_agent
    distance = target_binding.target_distance
    # This version only claims a symbolic witness for explicitly bound spatial
    # effects. Other effect types require their own target-native predicates.
    if relation is None or distance is None:
        return None
    facts = target_native_facts(observation)
    expected = (
        target_binding.target_uuid,
        target_binding.target_name,
        relation,
        distance,
    )
    observed = {
        (row.get("uuid"), row.get("name"), row.get("relation_from_agent"), row.get("distance"))
        for row in facts["salient_relative_objects"]
    }
    return "DIRECT_PROGRESS_AVAILABLE" if expected in observed else None


def select_candidate(
    condition: str,
    candidates: Sequence[DiscoveryWorldGroundedCandidate],
    observation: DiscoveryWorldObservation,
    *,
    target_binding: DiscoveryWorldTargetBinding | None = None,
    prerequisite_threshold: float = 0.75,
    positive_effect_threshold: float = 0.65,
) -> tuple[DiscoveryWorldGroundedCandidate, DiscoveryWorldSelectionReceipt]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown DiscoveryWorld transfer condition: {condition}")
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    supported = {
        row.candidate_sha256: evidence_supported(row, observation, target_binding)
        for row in candidates
    }
    available = {
        row.candidate_sha256: commit_available(row, target_binding)
        for row in candidates
    }
    bound_positions = {
        row.candidate_sha256: target_bound_position(row, target_binding)
        for row in candidates
    }
    positive_witness = {
        row.candidate_sha256: positive_commit_effect_witnessed(
            row, observation, target_binding,
        )
        for row in candidates
    }
    positive_witness_kind = {
        row.candidate_sha256: positive_commit_effect_kind(
            row, observation, target_binding,
        )
        for row in candidates
    }
    positions = [
        row for row in candidates
        if row.target_role == "POSITION"
        and (supported[row.candidate_sha256] or bound_positions[row.candidate_sha256])
    ]
    commits = [row for row in candidates if row.target_role == "COMMIT"]

    if condition == "target_native_myopic":
        pool = list(candidates)
        key = lambda row: (
            row.positive_effect_probability,
            row.prerequisite_probability,
            row.information_gain_probability,
            row.candidate_sha256,
        )
        reason = "TARGET_NATIVE_MAX_PREDICTED_IMMEDIATE_EFFECT"
    elif condition == "authentic_sokoban_effect_plus_target":
        admissible_commits = [
            row for row in commits
            if positive_witness[row.candidate_sha256]
        ]
        pool = admissible_commits or positions
        if not pool:
            pool = list(candidates)
        key = lambda row: (
            row.positive_effect_probability,
            row.information_gain_probability,
            row.prerequisite_probability,
            row.candidate_sha256,
        )
        reason = (
            "SOURCE_POSITIVE_COMMIT_EFFECT_THEN_COMMIT_AND_VERIFY"
            if admissible_commits else "SOURCE_NO_POSITIVE_COMMIT_EFFECT_THEN_POSITION"
        )
    elif condition == "commit_availability_control_plus_target":
        available_commits = [
            row for row in commits
            if available[row.candidate_sha256]
        ]
        pool = available_commits or positions or list(candidates)
        key = lambda row: (
            row.prerequisite_probability,
            row.positive_effect_probability,
            row.candidate_sha256,
        )
        reason = "CONTROL_COMMIT_AVAILABILITY_WITHOUT_POSITIVE_EFFECT_GUARD"
    elif condition == "inverted_effect_control_plus_target":
        inverted = [
            row for row in commits
            if available[row.candidate_sha256]
            and not positive_witness[row.candidate_sha256]
        ]
        pool = inverted or positions or list(candidates)
        key = lambda row: (
            -row.positive_effect_probability if row.target_role == "COMMIT"
            else row.information_gain_probability,
            row.prerequisite_probability,
            row.candidate_sha256,
        )
        reason = "CONTROL_INVERTED_POSITIVE_EFFECT_GUARD"
    else:
        pool = positions or list(candidates)
        key = lambda row: (
            row.information_gain_probability,
            row.positive_effect_probability,
            row.prerequisite_probability,
            row.candidate_sha256,
        )
        reason = "CONTROL_ALWAYS_POSITION_WHEN_AVAILABLE"

    selected = max(pool, key=key)
    receipt = DiscoveryWorldSelectionReceipt.create(
        condition=condition,
        candidates=candidates,
        selected=selected,
        evidence_supported=supported[selected.candidate_sha256],
        target_bound_position=bound_positions[selected.candidate_sha256],
        commit_available=available[selected.candidate_sha256],
        positive_commit_effect_witnessed=positive_witness[selected.candidate_sha256],
        positive_commit_effect_kind=positive_witness_kind[selected.candidate_sha256],
        selection_reason=reason,
    )
    return selected, receipt


def grounder_prompt_payload(
    observation: DiscoveryWorldObservation,
    *,
    memory: str,
    hypotheses: Sequence[str],
    recent: Sequence[Mapping[str, Any]],
    target_binding: DiscoveryWorldTargetBinding,
    schema_error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "target_interface": {
            "scenario": observation.scenario,
            "difficulty": observation.difficulty,
            "seed": observation.seed,
            "episode_step": observation.episode_step,
            "known_actions": observation.known_actions,
            "last_action_result": observation.last_action_result,
            "in_dialog": observation.in_dialog,
        },
        "target_native_facts": target_native_facts(observation),
        "target_binding": asdict(target_binding),
        "persistent_memory_supplied": False,
        "running_hypotheses": [str(value) for value in hypotheses][-24:],
        "recent_target_history": [dict(row) for row in recent[-3:]],
        "source_content_visible_to_grounder": False,
    }
    if schema_error:
        payload["previous_response_rejected"] = str(schema_error)[:1200]
    return payload


def binder_prompt_payload(
    observation: DiscoveryWorldObservation,
    *,
    memory: str,
    hypotheses: Sequence[str],
    schema_error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "target_interface": {
            "scenario": observation.scenario,
            "known_actions": observation.known_actions,
        },
        "target_native_facts": target_native_facts(observation),
        "running_hypotheses": [str(value) for value in hypotheses][-24:],
        "untrusted_scientific_memory": str(memory)[-3000:],
        "persistent_memory_supplied": False,
        "source_content_visible_to_binder": False,
    }
    if schema_error:
        payload["previous_response_rejected"] = str(schema_error)[:1200]
    return payload


__all__ = [
    "CONDITIONS",
    "DiscoveryWorldGroundedCandidate",
    "DiscoveryWorldSelectionReceipt",
    "DiscoveryWorldTargetBinding",
    "SOURCE_CONFIRMATION_SHA256",
    "SOURCE_PROGRAM_SHA256",
    "TARGET_GROUNDER_SYSTEM_PROMPT",
    "TARGET_BINDER_SYSTEM_PROMPT",
    "binder_prompt_payload",
    "commit_available",
    "evidence_supported",
    "grounder_prompt_payload",
    "parse_grounded_candidates",
    "parse_target_binding",
    "positive_commit_effect_witnessed",
    "positive_commit_effect_kind",
    "select_candidate",
    "target_bound_position",
]
