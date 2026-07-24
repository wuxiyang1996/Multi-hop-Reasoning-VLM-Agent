from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import SourcePolicyStepRecord, stable_hash
from .instrumented_import import ImportedSourceEpisode


OBJECTIVES = (
    "NEXT_TRANSITION",
    "OUTCOME_PREDICTION",
    "TRANSITION_MEMBERSHIP",
    "RECORDED_ADJACENCY",
    "MISSING_EVIDENCE_ABSTENTION",
)


def _reward_sign(value: float) -> str:
    return "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "ZERO"


@dataclass(frozen=True)
class HarnessTrainingExample:
    example_id: str
    game: str
    episode_id: str
    split: str
    objective: str
    input_payload: Mapping[str, Any]
    target_payload: Mapping[str, Any]
    evidence_receipt_ids: tuple[str, ...]
    authority: str = "MECHANICALLY_DERIVED_FROM_SOURCE_RECEIPTS"

    def validate(self) -> bool:
        body = asdict(self)
        example_id = body.pop("example_id")
        return (
            example_id == stable_hash(body)
            and self.objective in OBJECTIVES
            and bool(self.evidence_receipt_ids)
        )


def _split_by_episode(
    episodes: Sequence[ImportedSourceEpisode],
) -> dict[str, str]:
    labels = ("train", "validation", "source_held_out")
    episode_ids = sorted({episode.episode_id for episode in episodes})
    return {
        episode_id: labels[index % len(labels)]
        for index, episode_id in enumerate(episode_ids)
    }


def _make_example(
    *,
    game: str,
    episode_id: str,
    split: str,
    objective: str,
    input_payload: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    evidence_receipt_ids: Sequence[str],
) -> HarnessTrainingExample:
    body = {
        "game": game,
        "episode_id": episode_id,
        "split": split,
        "objective": objective,
        "input_payload": dict(input_payload),
        "target_payload": dict(target_payload),
        "evidence_receipt_ids": tuple(evidence_receipt_ids),
        "authority": "MECHANICALLY_DERIVED_FROM_SOURCE_RECEIPTS",
    }
    return HarnessTrainingExample(stable_hash(body), **body)


def _transition_examples(
    game: str,
    episode_id: str,
    split: str,
    record: SourcePolicyStepRecord,
) -> tuple[HarnessTrainingExample, ...]:
    shared_input = {
        "before": dict(record.before.state),
        "native_actions": record.before.native_actions,
        "selected_action": record.action,
        "action_origin": record.action_origin,
    }
    next_transition = _make_example(
        game=game,
        episode_id=episode_id,
        split=split,
        objective="NEXT_TRANSITION",
        input_payload=shared_input,
        target_payload={
            "after": dict(record.after.state),
            "reward_sign": _reward_sign(record.reward),
            "terminal": record.after.terminal,
            "official_success": record.after.official_success,
            "verdict": "OBSERVED",
        },
        evidence_receipt_ids=(record.transition.receipt_id,),
    )
    outcome = _make_example(
        game=game,
        episode_id=episode_id,
        split=split,
        objective="OUTCOME_PREDICTION",
        input_payload=shared_input,
        target_payload={
            "reward_sign": _reward_sign(record.reward),
            "terminal": record.after.terminal,
            "official_success": record.after.official_success,
        },
        evidence_receipt_ids=(record.transition.receipt_id,),
    )
    abstention = _make_example(
        game=game,
        episode_id=episode_id,
        split=split,
        objective="MISSING_EVIDENCE_ABSTENTION",
        input_payload={
            **shared_input,
            "after": None,
            "evidence_status": "WITHHELD_BY_TRAINING_CONSTRUCTION",
        },
        target_payload={
            "verdict": "INCONCLUSIVE",
            "continuation": "ABSTAIN",
        },
        evidence_receipt_ids=(record.transition.receipt_id,),
    )
    membership = _make_example(
        game=game,
        episode_id=episode_id,
        split=split,
        objective="TRANSITION_MEMBERSHIP",
        input_payload={
            **shared_input,
            "candidate_after": dict(record.after.state),
            "candidate_reward_sign": _reward_sign(record.reward),
            "candidate_terminal": record.after.terminal,
        },
        target_payload={"relation": "OBSERVED_FOR_THIS_RECEIPT"},
        evidence_receipt_ids=(record.transition.receipt_id,),
    )
    return next_transition, outcome, membership, abstention


def build_harness_training_examples(
    episodes: Sequence[ImportedSourceEpisode],
) -> tuple[HarnessTrainingExample, ...]:
    """Build source-only supervision without semantic predicates or Agent labels."""

    split_by_episode = _split_by_episode(episodes)
    examples = []
    for episode in sorted(episodes, key=lambda row: row.episode_id):
        split = split_by_episode[episode.episode_id]
        records = tuple(sorted(episode.records, key=lambda row: row.step))
        for record in records:
            examples.extend(
                _transition_examples(
                    episode.game, episode.episode_id, split, record,
                )
            )
        if len(records) >= 2:
            for index, record in enumerate(records):
                donor = records[(index + 1) % len(records)]
                if donor.transition.after_hash == record.transition.after_hash:
                    continue
                examples.append(_make_example(
                    game=episode.game,
                    episode_id=episode.episode_id,
                    split=split,
                    objective="TRANSITION_MEMBERSHIP",
                    input_payload={
                        "before": dict(record.before.state),
                        "native_actions": record.before.native_actions,
                        "selected_action": record.action,
                        "action_origin": record.action_origin,
                        "candidate_after": dict(donor.after.state),
                        "candidate_reward_sign": _reward_sign(donor.reward),
                        "candidate_terminal": donor.after.terminal,
                    },
                    target_payload={
                        "relation": "NOT_OBSERVED_FOR_THIS_RECEIPT",
                    },
                    evidence_receipt_ids=(
                        record.transition.receipt_id,
                        donor.transition.receipt_id,
                    ),
                ))
        for index, (left, right) in enumerate(zip(records, records[1:])):
            if right.step != left.step + 1:
                continue
            examples.append(_make_example(
                game=episode.game,
                episode_id=episode.episode_id,
                split=split,
                objective="RECORDED_ADJACENCY",
                input_payload={
                    "left_transition_id": left.transition.receipt_id,
                    "left_after_hash": left.transition.after_hash,
                    "right_transition_id": right.transition.receipt_id,
                    "right_before_hash": right.transition.before_hash,
                },
                target_payload={
                    "relation": "RECORDED_CONTIGUOUS",
                    "offset": index,
                },
                evidence_receipt_ids=(
                    left.transition.receipt_id,
                    right.transition.receipt_id,
                ),
            ))
        # A non-adjacent pair is not called impossible. It is labeled only as
        # absent from this recorded lineage, which is mechanically knowable.
        if len(records) >= 3:
            left, right = records[0], records[-1]
            examples.append(_make_example(
                game=episode.game,
                episode_id=episode.episode_id,
                split=split,
                objective="RECORDED_ADJACENCY",
                input_payload={
                    "left_transition_id": left.transition.receipt_id,
                    "left_after_hash": left.transition.after_hash,
                    "right_transition_id": right.transition.receipt_id,
                    "right_before_hash": right.transition.before_hash,
                },
                target_payload={"relation": "NOT_RECORDED_CONTIGUOUS"},
                evidence_receipt_ids=(
                    left.transition.receipt_id,
                    right.transition.receipt_id,
                ),
            ))
    if not all(example.validate() for example in examples):
        raise ValueError("constructed Harness training example failed validation")
    return tuple(examples)


def summarize_harness_training_examples(
    examples: Sequence[HarnessTrainingExample],
) -> dict[str, Any]:
    return {
        "examples": len(examples),
        "games": sorted({row.game for row in examples}),
        "episodes": len({row.episode_id for row in examples}),
        "objective_counts": dict(sorted(Counter(
            row.objective for row in examples
        ).items())),
        "split_counts": dict(sorted(Counter(
            row.split for row in examples
        ).items())),
        "all_valid": all(row.validate() for row in examples),
        "target_data_used": False,
        "agent_verdicts_used_as_labels": False,
        "human_predicates_used": False,
    }


__all__ = [
    "OBJECTIVES", "HarnessTrainingExample",
    "build_harness_training_examples", "summarize_harness_training_examples",
]
