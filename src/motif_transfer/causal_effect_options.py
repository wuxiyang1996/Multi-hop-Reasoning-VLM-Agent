"""Induce anonymous causal-effect options from matched visual forks.

The discovery procedure uses exact counterfactual frame equivalence.  It does
not assign semantic game labels (for example "shoot" or "move") and it does
not inspect future reward.  Source-native action names remain in a separate
grounding table; the transferable object is the class/transition structure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .visual_intervention_receipts import file_sha256, validate_plan


ARTIFACT_VERSION = "CAUSAL_VISUAL_EFFECT_OPTIONS_V1"
CLASS_NULL = "PERSISTENT_NULL_EFFECT"
CLASS_STABLE = "STABLE_CAUSAL_EFFECT"
CLASS_CONTEXTUAL = "CONTEXTUAL_CAUSAL_EFFECT"
EFFECT_CLASSES = (CLASS_NULL, CLASS_STABLE, CLASS_CONTEXTUAL)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in {path}")
                yield row


def _validate_hashed_mapping(row: Mapping[str, Any], hash_field: str) -> None:
    body = dict(row)
    claimed = str(body.pop(hash_field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"{hash_field} mismatch")


def _deranged_classes(
    action_classes: Mapping[str, str], *, seed: str,
) -> dict[str, str]:
    """Preserve class cardinalities while moving every action to a new class."""

    actions = sorted(action_classes, key=lambda action: stable_hash((seed, action)))
    remaining = Counter(action_classes.values())
    candidate: dict[str, str] = {}

    def assign(index: int) -> bool:
        if index == len(actions):
            return True
        action = actions[index]
        labels = sorted(
            (
                label for label, count in remaining.items()
                if count and label != action_classes[action]
            ),
            key=lambda label: stable_hash((seed, action, label)),
        )
        for label in labels:
            remaining[label] -= 1
            candidate[action] = label
            if assign(index + 1):
                return True
            del candidate[action]
            remaining[label] += 1
        return False

    if assign(0):
        return dict(sorted(candidate.items()))
    raise ValueError("effect classes cannot be fully deranged")


def _pairwise_equivalence(
    frames_by_snapshot: Mapping[str, Mapping[str, str]],
    actions: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1:]:
            comparisons = [
                frames[left] == frames[right]
                for frames in frames_by_snapshot.values()
            ]
            rows.append({
                "left": left,
                "right": right,
                "equal_count": sum(comparisons),
                "snapshot_count": len(comparisons),
                "equivalence_rate": sum(comparisons) / len(comparisons),
            })
    return rows


def build_causal_effect_option_artifact(
    plan: Mapping[str, Any],
    discovery_dir: Path,
    *,
    stable_effect_min_rate: float = 0.75,
    null_effect_max_rate: float = 0.0,
    minimum_snapshots: int = 6,
) -> dict[str, Any]:
    if not 0 <= null_effect_max_rate < stable_effect_min_rate <= 1:
        raise ValueError("invalid effect-rate thresholds")
    snapshots = {row.snapshot_id: row for row in validate_plan(plan)}
    discovery_dir = discovery_dir.resolve()
    manifest = json.loads((discovery_dir / "manifest.json").read_text())
    _validate_hashed_mapping(manifest, "manifest_sha256")
    if manifest.get("split") != "discovery":
        raise ValueError("option discovery may read only the discovery split")
    if not manifest.get("selection_complete"):
        raise ValueError("truncated mechanics smoke cannot induce an option")
    if not manifest.get("all_interventions_observed"):
        raise ValueError("discovery contains failed interventions")
    if not manifest.get("before_frame_consistent_per_snapshot"):
        raise ValueError("counterfactual forks do not share their before frame")
    if manifest.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("discovery manifest references a different plan")

    receipts = list(_read_jsonl(discovery_dir / str(manifest["receipts_file"])))
    if file_sha256(discovery_dir / str(manifest["receipts_file"])) != manifest.get(
        "receipts_sha256"
    ):
        raise ValueError("discovery receipt file hash mismatch")
    frames_by_snapshot: dict[str, dict[str, str]] = defaultdict(dict)
    rewards_by_action: dict[str, list[float]] = defaultdict(list)
    source_action_counts: Counter[str] = Counter()
    before_by_snapshot: dict[str, set[str]] = defaultdict(set)
    for receipt in receipts:
        _validate_hashed_mapping(receipt, "receipt_sha256")
        if receipt.get("status") != "INTERVENTION_OBSERVED":
            raise ValueError("failed intervention entered discovery")
        if receipt.get("split") != "discovery":
            raise ValueError("non-discovery receipt entered option induction")
        snapshot_id = str(receipt["snapshot_id"])
        snapshot = snapshots.get(snapshot_id)
        if snapshot is None or snapshot.split != "discovery":
            raise ValueError("receipt references an invalid discovery snapshot")
        action = str(receipt["intervention_action"])
        if action in frames_by_snapshot[snapshot_id]:
            raise ValueError("duplicate action fork at a snapshot")
        frames_by_snapshot[snapshot_id][action] = str(
            receipt["after_frame"]["png_sha256"]
        )
        before_by_snapshot[snapshot_id].add(str(
            receipt["before_frame"]["png_sha256"]
        ))
        rewards_by_action[action].append(float(receipt["reward"]))

    if len(frames_by_snapshot) < minimum_snapshots:
        raise ValueError("insufficient discovery snapshots")
    if any(len(values) != 1 for values in before_by_snapshot.values()):
        raise ValueError("before frame differs within a matched snapshot")
    action_sets = {tuple(sorted(frames)) for frames in frames_by_snapshot.values()}
    if len(action_sets) != 1:
        raise ValueError("native action coverage differs across snapshots")
    actions = tuple(next(iter(action_sets)))
    for snapshot_id, frames in frames_by_snapshot.items():
        snapshot = snapshots[snapshot_id]
        if set(frames) != set(snapshot.native_actions):
            raise ValueError("receipt action coverage differs from frozen plan")
        source_action_counts[snapshot.source_action] += 1

    modal_by_snapshot: dict[str, str] = {}
    modal_size_by_snapshot: dict[str, int] = {}
    effect_observations: dict[str, list[bool]] = defaultdict(list)
    for snapshot_id, frames in sorted(frames_by_snapshot.items()):
        frame_counts = Counter(frames.values())
        modal_count = max(frame_counts.values())
        modal_frame = min(
            frame for frame, count in frame_counts.items() if count == modal_count
        )
        modal_by_snapshot[snapshot_id] = modal_frame
        modal_size_by_snapshot[snapshot_id] = modal_count
        for action in actions:
            effect_observations[action].append(frames[action] != modal_frame)

    effect_rates = {
        action: sum(values) / len(values)
        for action, values in sorted(effect_observations.items())
    }
    action_classes: dict[str, str] = {}
    for action in actions:
        rate = effect_rates[action]
        if rate <= null_effect_max_rate:
            action_classes[action] = CLASS_NULL
        elif rate >= stable_effect_min_rate:
            action_classes[action] = CLASS_STABLE
        else:
            action_classes[action] = CLASS_CONTEXTUAL
    class_members = {
        effect_class: sorted(
            action for action, assigned in action_classes.items()
            if assigned == effect_class
        )
        for effect_class in EFFECT_CLASSES
    }
    if any(not members for members in class_members.values()):
        raise ValueError("discovery did not identify all required effect classes")

    program = {
        "predicates": [
            "ACTION_IN_PERSISTENT_VISUAL_NULLSPACE",
            "ACTION_HAS_STABLE_CAUSAL_EFFECT",
            "ACTION_HAS_CONTEXTUAL_CAUSAL_EFFECT",
            "PREDICTED_EFFECT_OBSERVED",
        ],
        "options": [
            {
                "option": "SELECT_EFFECT_BASIS",
                "requires": ["ACTION_HAS_STABLE_CAUSAL_EFFECT"],
                "expected_add": ["PREDICTED_EFFECT_OBSERVED"],
                "on_refutation": "PROBE_CONTEXTUAL_EFFECT",
            },
            {
                "option": "PROBE_CONTEXTUAL_EFFECT",
                "requires": ["ACTION_HAS_CONTEXTUAL_CAUSAL_EFFECT"],
                "expected_add": ["PREDICTED_EFFECT_OBSERVED"],
                "on_refutation": "FILTER_NULL_KERNEL",
            },
            {
                "option": "FILTER_NULL_KERNEL",
                "forbids": ["ACTION_IN_PERSISTENT_VISUAL_NULLSPACE"],
                "expected_add": [],
                "on_refutation": "ABSTAIN",
            },
        ],
        "selection_rule": (
            "PREFER_STABLE_THEN_CONTEXTUAL; NEVER_SELECT_PERSISTENT_NULL"
        ),
        "verification_rule": (
            "REQUIRE_TARGET_NATIVE_PREDICTED_EFFECT_AFTER_EACH_REALIZATION"
        ),
    }
    body: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "lifecycle": "DISCOVERY_CANDIDATE_NOT_SOURCE_QUALIFIED",
        "claim_boundary": (
            "INTERVENTION_GROUNDED_EFFECT_STRUCTURE_ONLY; SOURCE_ACTION_LABELS_"
            "ARE_NOT_TRANSFERRED"
        ),
        "source_domain": str(plan["game"]),
        "plan_sha256": str(plan["plan_sha256"]),
        "discovery_manifest_sha256": str(manifest["manifest_sha256"]),
        "thresholds": {
            "stable_effect_min_rate": stable_effect_min_rate,
            "null_effect_max_rate": null_effect_max_rate,
            "minimum_snapshots": minimum_snapshots,
        },
        "snapshot_count": len(frames_by_snapshot),
        "source_grounding": {
            "action_classes": dict(sorted(action_classes.items())),
            "class_members": class_members,
            "effect_rates": effect_rates,
            "modal_equivalence_class_size": modal_size_by_snapshot,
            "pairwise_visual_equivalence": _pairwise_equivalence(
                frames_by_snapshot, actions
            ),
            "one_step_reward": {
                action: {
                    "mean": sum(values) / len(values),
                    "positive_count": sum(value > 0 for value in values),
                    "count": len(values),
                }
                for action, values in sorted(rewards_by_action.items())
            },
            "source_policy_action_counts": dict(sorted(source_action_counts.items())),
        },
        "shuffled_control": {
            "action_classes": _deranged_classes(
                action_classes, seed=str(plan["plan_sha256"])
            ),
            "preserves_class_cardinality": True,
            "fixed_points": 0,
        },
        "transferable_symbolic_program": program,
        "source_lineage": [str(row["receipt_sha256"]) for row in receipts],
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_causal_effect_option_artifact(
    artifact: Mapping[str, Any], *, require_source_qualified: bool = False,
) -> None:
    _validate_hashed_mapping(artifact, "artifact_sha256")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported causal-effect artifact")
    if require_source_qualified and artifact.get("lifecycle") != "SOURCE_QUALIFIED":
        raise ValueError("artifact is not source-qualified")
    grounding = artifact.get("source_grounding") or {}
    classes = grounding.get("action_classes") or {}
    if set(classes.values()) != set(EFFECT_CLASSES):
        raise ValueError("artifact lacks the complete effect vocabulary")
    shuffled = (artifact.get("shuffled_control") or {}).get("action_classes") or {}
    if Counter(classes.values()) != Counter(shuffled.values()):
        raise ValueError("shuffled control changed class cardinalities")
    if any(classes[action] == shuffled.get(action) for action in classes):
        raise ValueError("shuffled control is not a derangement")
