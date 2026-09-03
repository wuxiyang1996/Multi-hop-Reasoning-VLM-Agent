"""Target-native structural induction for ALFWorld multiplicity tasks.

The exported source program remains an anonymous graph-edit sequence.  This
module learns the target side from consumed ALFWorld development trajectories:

* textual transition outcomes supervise ADD/REMOVE/UPDATE operator grounding;
* demonstrated object and receptacle bindings supervise two neural binding
  heads without exporting either binding into the source program; and
* successful target paths induce a counted target sequence.  A source program
  is applicable only when repetitions of its anonymous sequence explain that
  target sequence.

At inference, every neural head consumes only the current goal, observation,
candidate action, and action history.  Transition outcomes and official task
success are not inputs.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .alfworld_hierarchical_grounder import tokens
from .alfworld_multiplicity_grounder import workflow_status
from .contracts import stable_hash
from .structural_delta_induction import common_effect_subsequence
from .target_structural_induction import (
    ADD_ENTITY_SLOT,
    REMOVE_ENTITY_SLOT,
    anonymous_operator_descriptor,
)


UPDATE_ENTITY_ATTRIBUTE = anonymous_operator_descriptor(
    "UPDATE", "ENTITY_ATTRIBUTE", 1, "BOOLEAN_VECTOR",
)
OPERATOR_TYPES = (
    ADD_ENTITY_SLOT,
    REMOVE_ENTITY_SLOT,
    UPDATE_ENTITY_ATTRIBUTE,
)
OPERATOR_IDS = tuple(row["operator_type_id"] for row in OPERATOR_TYPES)
ADD_ID = str(ADD_ENTITY_SLOT["operator_type_id"])
REMOVE_ID = str(REMOVE_ENTITY_SLOT["operator_type_id"])
UPDATE_ID = str(UPDATE_ENTITY_ATTRIBUTE["operator_type_id"])

_SUCCESS_PATTERNS = {
    ADD_ID: (
        re.compile(r"\byou (?:pick up|take) the\b", re.IGNORECASE),
    ),
    REMOVE_ID: (
        re.compile(r"\byou (?:move|put) the\b", re.IGNORECASE),
    ),
    UPDATE_ID: (
        re.compile(
            r"\byou (?:open|close|heat|cool|clean|slice|turn on|turn off) the\b",
            re.IGNORECASE,
        ),
    ),
}


def observed_transition_operator_ids(
    transition: Mapping[str, Any],
) -> tuple[str, ...]:
    """Label an executed transition from the target-native response text.

    The action string alone is deliberately insufficient: a syntactically
    valid command that failed to change state must not become a positive
    structural example.
    """

    after = str(transition.get("after_observation") or "")
    output = []
    for type_id, patterns in _SUCCESS_PATTERNS.items():
        if any(pattern.search(after) for pattern in patterns):
            output.append(type_id)
    return tuple(output)


def episode_structural_sequence(
    episode: Mapping[str, Any], *, core_only: bool = True,
) -> tuple[str, ...]:
    output: list[str] = []
    for transition in episode.get("transitions") or ():
        for type_id in observed_transition_operator_ids(transition):
            if core_only and type_id not in {ADD_ID, REMOVE_ID}:
                continue
            output.append(type_id)
    return tuple(output)


def _hash_bin(value: str, bins: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % bins


def target_candidate_features(
    *, goal: str, observation: str, action: str, step: int,
    action_history: Sequence[str], feature_bins: int,
) -> np.ndarray:
    """Outcome-blind lexical/state features for all target neural heads."""

    action_tokens = tokens(action)
    goal_tokens = set(tokens(goal))
    observation_tokens = set(tokens(observation))
    action_set = set(action_tokens)
    lexical = np.zeros(int(feature_bins), dtype=np.float64)
    for index, token in enumerate(action_tokens):
        # Preserve action grammar while masking entity identities.  Earlier
        # artifacts hashed every token and consequently learned that an object
        # frequently used as a target during development was a target in every
        # future goal.  Here only the verb and closed-class relation markers
        # retain lexical identity; entity/receptacle tokens are represented by
        # their structural position and cross-text relations.
        position = min(index, 6)
        if index == 0 or token in {"to", "from", "in", "on", "at", "with"}:
            lexical[_hash_bin(f"grammar:{position}:{token}", feature_bins)] += 1.0
        lexical[_hash_bin(
            f"token-role:{position}:digit={int(token.isdigit())}", feature_bins,
        )] += 1.0
        if token in goal_tokens:
            lexical[_hash_bin(f"goal-overlap-position:{position}", feature_bins)] += 1.0
        if token in observation_tokens:
            lexical[_hash_bin(
                f"observation-overlap-position:{position}", feature_bins,
            )] += 1.0
    norm = float(np.linalg.norm(lexical))
    if norm:
        lexical /= norm
    status = workflow_status(goal, action_history)
    repeat = sum(str(previous) == str(action) for previous in action_history)
    overlap_goal = len(action_set & goal_tokens)
    overlap_observation = len(action_set & observation_tokens)
    first_argument_goal_overlap = float(
        len(action_tokens) > 1 and action_tokens[1] in goal_tokens
    )
    after_to_goal_overlap = float(any(
        action_tokens[index + 1] in goal_tokens
        for index, value in enumerate(action_tokens[:-1]) if value == "to"
    ))
    after_from_goal_overlap = float(any(
        action_tokens[index + 1] in goal_tokens
        for index, value in enumerate(action_tokens[:-1]) if value == "from"
    ))
    scalars = np.asarray((
        min(len(action_tokens), 12) / 12.0,
        min(overlap_goal, 4) / 4.0,
        overlap_goal / max(1, len(action_set)),
        min(overlap_observation, 6) / 6.0,
        overlap_observation / max(1, len(action_set)),
        min(repeat, 8) / 8.0,
        min(int(step), 180) / 180.0,
        float(status.held),
        min(status.placed_count, status.required_count) / max(1, status.required_count),
        status.remaining_count / max(1, status.required_count),
        float(status.required_count == 2),
        first_argument_goal_overlap,
        after_to_goal_overlap,
        after_from_goal_overlap,
    ), dtype=np.float64)
    return np.concatenate((scalars, lexical))


def infer_demonstrated_bindings(episode: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Induce episode-local entity/receptacle tokens from observed graph edits."""

    goal_tokens = set(tokens(str(episode.get("transitions", [{}])[0].get("goal") or "")))
    entity_votes: Counter[str] = Counter()
    destination_votes: Counter[str] = Counter()
    for transition in episode.get("transitions") or ():
        action_values = tuple(tokens(str(transition.get("expert_action") or "")))
        overlap = [value for value in action_values if value in goal_tokens and not value.isdigit()]
        observed = observed_transition_operator_ids(transition)
        if ADD_ID in observed:
            entity_votes.update(overlap)
        if REMOVE_ID in observed:
            destination_votes.update(overlap)
            entity_votes.update(overlap)
    # The target entity occurs in both ADD and REMOVE actions.  The destination
    # occurs only in REMOVE actions.  Counts, not a hand-written goal grammar,
    # establish their roles.
    entity = tuple(sorted(
        value for value, count in entity_votes.items()
        if count >= 2 and count > destination_votes[value]
    ))
    if not entity and entity_votes:
        maximum = max(entity_votes.values())
        entity = tuple(sorted(value for value, count in entity_votes.items() if count == maximum))
    destination = tuple(sorted(
        value for value, count in destination_votes.items()
        if value not in entity and count >= 1
    ))
    return {"entity": entity, "destination": destination}


def binding_labels(
    action: str, bindings: Mapping[str, Sequence[str]],
) -> tuple[int, int]:
    values = set(tokens(action))
    entity = int(bool(values & set(bindings.get("entity") or ())))
    destination = int(bool(values & set(bindings.get("destination") or ())))
    return entity, destination


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def export_binary_mlp(model: Any, *, label: str) -> dict[str, Any]:
    body = {
        "schema_version": "alfworld-target-native-binary-mlp-v1",
        "label": str(label),
        "activation": "RELU",
        "output_activation": "SIGMOID",
        "coefs": [np.asarray(value).tolist() for value in model.coefs_],
        "intercepts": [np.asarray(value).tolist() for value in model.intercepts_],
    }
    return body | {"head_sha256": stable_hash(body)}


def binary_mlp_probability(head: Mapping[str, Any], features: np.ndarray) -> float:
    body = dict(head)
    claimed = body.pop("head_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("ALFWorld structural neural head hash mismatch")
    hidden = np.asarray(features, dtype=np.float64)[None, :]
    coefs = [np.asarray(row, dtype=np.float64) for row in head["coefs"]]
    intercepts = [np.asarray(row, dtype=np.float64) for row in head["intercepts"]]
    for index, (weights, bias) in enumerate(zip(coefs, intercepts)):
        hidden = hidden @ weights + bias
        hidden = _sigmoid(hidden) if index + 1 == len(coefs) else np.maximum(hidden, 0.0)
    return float(hidden.ravel()[0])


def _binary_mlp_probabilities_validated(
    head: Mapping[str, Any], features: np.ndarray,
) -> np.ndarray:
    hidden = np.asarray(features, dtype=np.float64)
    for index, (weights, bias) in enumerate(zip(head["coefs"], head["intercepts"])):
        hidden = hidden @ np.asarray(weights, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
        hidden = _sigmoid(hidden) if index + 1 == len(head["coefs"]) else np.maximum(hidden, 0.0)
    return hidden.reshape(-1)


def validate_grounder(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = body.pop("grounder_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("ALFWorld structural grounder hash mismatch")
    if artifact.get("schema_version") != "alfworld-target-native-structural-grounder-v1":
        raise ValueError("unsupported ALFWorld structural grounder")
    if artifact.get("formal_target_outcome_read") is not False:
        raise ValueError("formal target outcome leaked into ALFWorld grounder")
    if artifact.get("outcome_fields_used_at_inference") is not False:
        raise ValueError("target outcome fields enabled at inference")
    if artifact.get("entity_and_receptacle_identity_tokens_masked") is not True:
        raise ValueError("target binding grounder permits entity-token memorization")
    expected = {*OPERATOR_IDS, "ENTITY_BINDING", "DESTINATION_BINDING", "BEHAVIOR"}
    if set(artifact.get("heads") or ()) != expected:
        raise ValueError("ALFWorld structural neural heads are incomplete")
    for name, head in artifact["heads"].items():
        if head.get("label") != name:
            raise ValueError("ALFWorld neural head label mismatch")
        binary_mlp_probability(
            head,
            np.zeros(
                int(artifact["feature_bins"]) + int(artifact["scalar_feature_count"]),
                dtype=np.float64,
            ),
        )


def _score_candidate_validated(
    artifact: Mapping[str, Any], *, goal: str, observation: str, action: str,
    step: int, action_history: Sequence[str],
) -> dict[str, Any]:
    features = target_candidate_features(
        goal=goal,
        observation=observation,
        action=action,
        step=step,
        action_history=action_history,
        feature_bins=int(artifact["feature_bins"]),
    )
    probabilities = {
        name: binary_mlp_probability(head, features)
        for name, head in artifact["heads"].items()
    }
    return {
        "operator_probabilities": {
            type_id: probabilities[type_id] for type_id in OPERATOR_IDS
        },
        "entity_binding_probability": probabilities["ENTITY_BINDING"],
        "destination_binding_probability": probabilities["DESTINATION_BINDING"],
        "behavior_probability": probabilities["BEHAVIOR"],
        "action_sha256": stable_hash({"target_native_action": str(action)}),
    }


def score_candidate(
    artifact: Mapping[str, Any], *, goal: str, observation: str, action: str,
    step: int, action_history: Sequence[str],
) -> dict[str, Any]:
    validate_grounder(artifact)
    return _score_candidate_validated(
        artifact, goal=goal, observation=observation, action=action,
        step=step, action_history=action_history,
    )


def score_candidates(
    artifact: Mapping[str, Any], *, goal: str, observation: str,
    actions: Sequence[str], step: int, action_history: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Validate once, then score a matched native-action batch."""

    validate_grounder(artifact)
    return {
        str(action): _score_candidate_validated(
            artifact, goal=goal, observation=observation, action=str(action),
            step=step, action_history=action_history,
        )
        for action in actions
    }


class ALFWorldStructuralGrounder:
    """Hash-validated immutable grounder for repeated online batch scoring."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        validate_grounder(artifact)
        self.artifact = artifact

    def score_candidates(
        self, *, goal: str, observation: str, actions: Sequence[str], step: int,
        action_history: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        normalized = tuple(map(str, actions))
        feature_matrix = np.asarray([
            target_candidate_features(
                goal=goal, observation=observation, action=action, step=step,
                action_history=action_history,
                feature_bins=int(self.artifact["feature_bins"]),
            )
            for action in normalized
        ], dtype=np.float64)
        probabilities = {
            name: _binary_mlp_probabilities_validated(head, feature_matrix)
            for name, head in self.artifact["heads"].items()
        }
        return {
            action: {
                "operator_probabilities": {
                    type_id: float(probabilities[type_id][index])
                    for type_id in OPERATOR_IDS
                },
                "entity_binding_probability": float(
                    probabilities["ENTITY_BINDING"][index]
                ),
                "destination_binding_probability": float(
                    probabilities["DESTINATION_BINDING"][index]
                ),
                "behavior_probability": float(probabilities["BEHAVIOR"][index]),
                "action_sha256": stable_hash({"target_native_action": action}),
            }
            for index, action in enumerate(normalized)
        }


def induce_target_sequence_program(
    paths: Sequence[Sequence[str]], *, development_receipts_sha256: str,
) -> dict[str, Any]:
    normalized = [tuple(map(str, path)) for path in paths if path]
    if not normalized:
        raise ValueError("ALFWorld target sequence induction requires paths")
    sequence = common_effect_subsequence(normalized)
    counts = Counter(sequence)
    body = {
        "schema_version": "target-induced-counted-sequence-program-v1",
        "development_receipts_sha256": str(development_receipts_sha256),
        "induced_sequence": list(sequence),
        "operator_requirements": [
            {"operator_type_id": key, "minimum_count": int(value)}
            for key, value in sorted(counts.items())
        ],
        "terminal_rule": "OBSERVED_TARGET_SEQUENCE_PREFIX_REACHES_INDUCED_LENGTH",
        "abstention_rule": {
            "missing_neural_binding": "ABSTAIN_TO_TARGET_POLICY",
            "observed_transition_mismatch": "DO_NOT_ADVANCE_SYMBOLIC_CURSOR",
        },
        "source_program_copied_as_target_body": False,
        "formal_target_data_read": False,
    }
    return body | {"program_sha256": stable_hash(body)}


def validate_target_sequence_program(program: Mapping[str, Any]) -> None:
    body = dict(program)
    claimed = body.pop("program_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("ALFWorld target sequence program hash mismatch")
    if program.get("source_program_copied_as_target_body") is not False:
        raise ValueError("source program copied into ALFWorld target body")
    if program.get("formal_target_data_read") is not False:
        raise ValueError("formal target data leaked into ALFWorld target program")


def repeated_source_support(
    source_sequence: Sequence[str], target_sequence: Sequence[str],
) -> dict[str, Any]:
    source = tuple(map(str, source_sequence))
    target = tuple(map(str, target_sequence))
    repeat_count = len(target) // len(source) if source and len(target) % len(source) == 0 else 0
    exact = bool(repeat_count and source * repeat_count == target)
    return {
        "applicable": exact,
        "repeat_count": repeat_count if exact else 0,
        "explained_operator_count": len(target) if exact else 0,
        "target_operator_count": len(target),
    }


def observed_action_history_sequence(
    goal: str, action_history: Sequence[str],
) -> tuple[str, ...]:
    """Target ledger events used only to verify executed neural bindings."""

    output = []
    prefix: list[str] = []
    before = workflow_status(goal, prefix)
    for action in action_history:
        prefix.append(str(action))
        after = workflow_status(goal, prefix)
        if not before.held and after.held:
            output.append(ADD_ID)
        if before.held and not after.held and after.placed_count > before.placed_count:
            output.append(REMOVE_ID)
        before = after
    return tuple(output)


__all__ = [
    "ADD_ID", "ALFWorldStructuralGrounder", "OPERATOR_IDS", "OPERATOR_TYPES",
    "REMOVE_ID", "UPDATE_ID",
    "UPDATE_ENTITY_ATTRIBUTE", "binary_mlp_probability", "binding_labels",
    "episode_structural_sequence", "export_binary_mlp",
    "induce_target_sequence_program", "infer_demonstrated_bindings",
    "observed_action_history_sequence", "observed_transition_operator_ids",
    "repeated_source_support", "score_candidate", "score_candidates",
    "target_candidate_features",
    "validate_grounder", "validate_target_sequence_program",
]
