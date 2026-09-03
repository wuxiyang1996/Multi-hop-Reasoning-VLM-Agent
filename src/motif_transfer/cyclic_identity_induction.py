"""Induce an anonymous cyclic identity-recovery constraint from source forks.

The learner never receives a rotation action name or a pre-written inverse
formula.  Its input is a collection of source-only intervention forks in an
anonymous cyclic state space.  Each fork records the observed effect of a
probe, the observed effect of a candidate recovery, and whether the composed
path returned to its initial state.  The learner selects a relation only when
one hypothesis exactly separates successful and unsuccessful forks and then
fails closed on zero or multiple target-native bindings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from .contracts import stable_hash


DATASET_VERSION = "source-cyclic-intervention-forks-v1"
PROGRAM_VERSION = "source-induced-cyclic-identity-program-v1"


def _compose_to_identity(probe: int, recovery: int, order: int) -> bool:
    return (probe + recovery) % order == 0


def _copy_probe_effect(probe: int, recovery: int, order: int) -> bool:
    return recovery % order == probe % order


def _recovery_is_identity(probe: int, recovery: int, order: int) -> bool:
    del probe
    return recovery % order == 0


def _recovery_is_generator(probe: int, recovery: int, order: int) -> bool:
    del probe
    return recovery % order == 1 % order


def _recovery_is_predecessor(probe: int, recovery: int, order: int) -> bool:
    del probe
    return recovery % order == (-1) % order


_HYPOTHESES: dict[str, Callable[[int, int, int], bool]] = {
    "COMPOSE_PROBE_RECOVERY_TO_IDENTITY": _compose_to_identity,
    "COPY_PROBE_EFFECT": _copy_probe_effect,
    "RECOVERY_IS_IDENTITY": _recovery_is_identity,
    "RECOVERY_IS_GENERATOR": _recovery_is_generator,
    "RECOVERY_IS_PREDECESSOR": _recovery_is_predecessor,
}


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def validate_cyclic_dataset(dataset: Mapping[str, Any]) -> None:
    """Validate hashes and the source-only export contract."""

    _self_hash(dataset, "dataset_sha256")
    if dataset.get("schema_version") != DATASET_VERSION:
        raise ValueError("unsupported cyclic source dataset")
    if dataset.get("target_data_read") is not False:
        raise ValueError("target data leaked into cyclic source dataset")
    if dataset.get("raw_source_action_tokens_exported") is not False:
        raise ValueError("raw source action token leaked")
    for episode in dataset.get("episodes") or ():
        _self_hash(episode, "episode_sha256")
        order = int(episode["group_order"])
        if order < 2:
            raise ValueError("cyclic episode must have nontrivial order")
        candidates = list(episode.get("candidates") or ())
        if len(candidates) < 3:
            raise ValueError("cyclic episode has too few candidate forks")
        for candidate in candidates:
            _self_hash(candidate, "candidate_sha256")
            probe = int(candidate["probe_effect_element"])
            recovery = int(candidate["recovery_effect_element"])
            if not 0 <= probe < order or not 0 <= recovery < order:
                raise ValueError("cyclic effect element is outside group")
            steps = list(candidate.get("primitive_transitions") or ())
            if not steps:
                raise ValueError("cyclic candidate omitted primitive transitions")
            for row in steps:
                _self_hash(row, "transition_sha256")
                if row.get("raw_action_exported") is not False:
                    raise ValueError("cyclic transition exported a source action")


def _rows(dataset: Mapping[str, Any]) -> list[tuple[int, int, int, bool]]:
    return [
        (
            int(candidate["probe_effect_element"]),
            int(candidate["recovery_effect_element"]),
            int(episode["group_order"]),
            bool(candidate["returned_to_identity"]),
        )
        for episode in dataset["episodes"]
        for candidate in episode["candidates"]
    ]


def hypothesis_diagnostics(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact classification diagnostics for the fixed relation class."""

    validate_cyclic_dataset(dataset)
    rows = _rows(dataset)
    diagnostics = {}
    for name, hypothesis in _HYPOTHESES.items():
        predictions = [hypothesis(probe, recovery, order) for probe, recovery, order, _ in rows]
        labels = [label for _, _, _, label in rows]
        true_positive = sum(prediction and label for prediction, label in zip(predictions, labels))
        true_negative = sum(not prediction and not label for prediction, label in zip(predictions, labels))
        diagnostics[name] = {
            "exactly_separates_forks": predictions == labels,
            "correct": sum(prediction == label for prediction, label in zip(predictions, labels)),
            "total": len(rows),
            "true_positive": true_positive,
            "true_negative": true_negative,
        }
    return diagnostics


def induce_cyclic_identity_program(
    dataset: Mapping[str, Any], *, minimum_episodes: int = 2,
) -> dict[str, Any]:
    """Select one receipt-grounded algebraic relation or abstain."""

    validate_cyclic_dataset(dataset)
    episodes = list(dataset["episodes"])
    diagnostics = hypothesis_diagnostics(dataset)
    exact = sorted(
        name
        for name, row in diagnostics.items()
        if row["exactly_separates_forks"]
    )
    non_self_inverse = sum(
        bool(candidate["returned_to_identity"])
        and int(candidate["probe_effect_element"])
        != int(candidate["recovery_effect_element"])
        for episode in episodes
        for candidate in episode["candidates"]
    )
    qualified = (
        len(episodes) >= int(minimum_episodes)
        and non_self_inverse > 0
        and exact == ["COMPOSE_PROBE_RECOVERY_TO_IDENTITY"]
    )
    body: dict[str, Any] = {
        "schema_version": PROGRAM_VERSION,
        "status": (
            "SOURCE_CYCLIC_IDENTITY_PROGRAM_INDUCED"
            if qualified
            else "ABSTAIN_AMBIGUOUS_OR_INSUFFICIENT_CYCLIC_EVIDENCE"
        ),
        "source_dataset_sha256": str(dataset["dataset_sha256"]),
        "selected_relation": (
            "COMPOSE_PROBE_RECOVERY_TO_IDENTITY" if qualified else None
        ),
        "typed_signature": {
            "state": "CYCLIC_GROUP_ELEMENT",
            "probe_effect": "CYCLIC_GROUP_ELEMENT",
            "recovery_effect": "CYCLIC_GROUP_ELEMENT",
            "terminal": "IDENTITY_EQUALITY",
        },
        "transition_constraint": (
            "COMPOSE(PROBE_EFFECT, RECOVERY_EFFECT) == IDENTITY"
            if qualified
            else None
        ),
        "abstention_rule": {
            "zero_grounded_recovery_candidates": "ABSTAIN",
            "multiple_grounded_recovery_candidates": "ABSTAIN",
            "unseen_group_contract": "ABSTAIN",
            "ambiguous_source_hypothesis": "ABSTAIN",
        },
        "diagnostics": {
            "episodes": len(episodes),
            "candidate_forks": sum(len(row["candidates"]) for row in episodes),
            "primitive_transitions": sum(
                len(candidate["primitive_transitions"])
                for row in episodes
                for candidate in row["candidates"]
            ),
            "non_self_inverse_successes": non_self_inverse,
            "exact_hypotheses": exact,
            "hypotheses": diagnostics,
        },
        "raw_source_action_tokens_exported": False,
        "target_data_read": False,
    }
    return body | {"program_sha256": stable_hash(body)}


def program_predicts_identity(
    program: Mapping[str, Any], *, probe_effect: int,
    recovery_effect: int, group_order: int,
) -> bool:
    _self_hash(program, "program_sha256")
    if program.get("status") != "SOURCE_CYCLIC_IDENTITY_PROGRAM_INDUCED":
        return False
    if program.get("selected_relation") != "COMPOSE_PROBE_RECOVERY_TO_IDENTITY":
        return False
    return _compose_to_identity(
        int(probe_effect), int(recovery_effect), int(group_order),
    )


def evaluate_cyclic_program(
    program: Mapping[str, Any], dataset: Mapping[str, Any],
) -> dict[str, Any]:
    validate_cyclic_dataset(dataset)
    rows = _rows(dataset)
    predictions = [
        program_predicts_identity(
            program, probe_effect=probe, recovery_effect=recovery,
            group_order=order,
        )
        for probe, recovery, order, _ in rows
    ]
    labels = [label for _, _, _, label in rows]
    return {
        "correct": sum(prediction == label for prediction, label in zip(predictions, labels)),
        "total": len(rows),
        "all_forks_classified": predictions == labels,
        "positive_support": sum(prediction and label for prediction, label in zip(predictions, labels)),
        "false_positive_support": sum(prediction and not label for prediction, label in zip(predictions, labels)),
    }


def permute_recovery_effect_bindings(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Rotate recovery effects within each episode while preserving labels."""

    validate_cyclic_dataset(dataset)
    value = deepcopy(dict(dataset))
    value.pop("dataset_sha256", None)
    for episode in value["episodes"]:
        episode.pop("episode_sha256", None)
        candidates = episode["candidates"]
        effects = [int(row["recovery_effect_element"]) for row in candidates]
        effects = effects[1:] + effects[:1]
        for candidate, effect in zip(candidates, effects):
            candidate.pop("candidate_sha256", None)
            candidate["recovery_effect_element"] = effect
            candidate["candidate_sha256"] = stable_hash(candidate)
        episode["episode_sha256"] = stable_hash(episode)
    value["control"] = "RECOVERY_EFFECT_BINDINGS_ROTATED"
    value["dataset_sha256"] = stable_hash(value)
    return value


def permute_terminal_labels(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Rotate source success labels within each episode."""

    validate_cyclic_dataset(dataset)
    value = deepcopy(dict(dataset))
    value.pop("dataset_sha256", None)
    for episode in value["episodes"]:
        episode.pop("episode_sha256", None)
        candidates = episode["candidates"]
        labels = [bool(row["returned_to_identity"]) for row in candidates]
        labels = labels[1:] + labels[:1]
        for candidate, label in zip(candidates, labels):
            candidate.pop("candidate_sha256", None)
            candidate["returned_to_identity"] = label
            candidate["candidate_sha256"] = stable_hash(candidate)
        episode["episode_sha256"] = stable_hash(episode)
    value["control"] = "TERMINAL_LABELS_ROTATED"
    value["dataset_sha256"] = stable_hash(value)
    return value


def subset_cyclic_dataset(
    dataset: Mapping[str, Any], episode_ids: Sequence[str],
) -> dict[str, Any]:
    validate_cyclic_dataset(dataset)
    selected = set(map(str, episode_ids))
    value = deepcopy(dict(dataset))
    value.pop("dataset_sha256", None)
    value["episodes"] = [
        row for row in value["episodes"]
        if str(row["episode_id"]) in selected
    ]
    value["dataset_sha256"] = stable_hash(value)
    return value


__all__ = [
    "DATASET_VERSION",
    "PROGRAM_VERSION",
    "evaluate_cyclic_program",
    "hypothesis_diagnostics",
    "induce_cyclic_identity_program",
    "permute_recovery_effect_bindings",
    "permute_terminal_labels",
    "program_predicts_identity",
    "subset_cyclic_dataset",
    "validate_cyclic_dataset",
]
