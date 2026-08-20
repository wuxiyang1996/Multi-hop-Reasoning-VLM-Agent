"""Prospective cyclic-identity transfer to MiniGrid orientation recovery.

The target-native neural grounder observes rendered MiniGrid panels and emits
only compass-direction bindings.  The source program receives anonymous
cyclic-group effects, never pixels, MiniGrid action names, goal state, or
source identity.  A target task succeeds only when the chosen recovery macro
restores the pre-intervention heading closely enough for a precomputed native
navigation suffix to reach the environment goal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from .cyclic_identity_induction import program_predicts_identity


GROUP_ORDER = 4
TOKENS = ("A", "B", "C", "D")
DIRECTION_NAMES = ("right", "down", "left", "up")
CONDITIONS = (
    "source_induced",
    "alpha_renamed_source",
    "target_written_isomorphic",
    "neural_only_direct",
    "copy_effect_control",
    "fixed_token_control",
    "shuffled_binding_control",
)


def _digest(namespace: str, seed: int, field: str) -> bytes:
    return hashlib.sha256(
        f"{namespace}\0{seed}\0{field}".encode("utf-8")
    ).digest()


def _permutation(namespace: str, seed: int) -> tuple[int, ...]:
    values = list(range(GROUP_ORDER))
    digest = _digest(namespace, seed, "candidate-permutation")
    for index in range(len(values) - 1, 0, -1):
        swap = digest[index] % (index + 1)
        values[index], values[swap] = values[swap], values[index]
    return tuple(values)


def _probe_effects(namespace: str, seed: int) -> tuple[int, ...]:
    digest = _digest(namespace, seed, "probe-sequence")
    length = 7 + digest[0] % 7
    effects = [1 if digest[index + 1] % 2 else 3 for index in range(length)]
    if sum(effects) % GROUP_ORDER == 0:
        effects[-1] = 1 if effects[-1] == 3 else 3
        if sum(effects) % GROUP_ORDER == 0:
            effects.append(1)
    return tuple(effects)


@dataclass(frozen=True)
class OrientationTaskSpec:
    seed: int
    namespace: str
    probe_effects: tuple[int, ...]
    token_effects: tuple[int, ...]

    @property
    def probe_effect(self) -> int:
        return sum(self.probe_effects) % GROUP_ORDER

    @property
    def token_to_effect(self) -> dict[str, int]:
        return dict(zip(TOKENS, self.token_effects, strict=True))


def task_spec(seed: int, namespace: str) -> OrientationTaskSpec:
    spec = OrientationTaskSpec(
        seed=int(seed),
        namespace=str(namespace),
        probe_effects=_probe_effects(str(namespace), int(seed)),
        token_effects=_permutation(str(namespace), int(seed)),
    )
    if spec.probe_effect == 0:
        raise AssertionError("orientation task probe must be non-identity")
    if set(spec.token_effects) != set(range(GROUP_ORDER)):
        raise AssertionError("orientation task action effects are not bijective")
    return spec


def normalize_direction(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", " ")
    aliases = {
        "right": "right", "east": "right", "e": "right",
        "down": "down", "south": "down", "s": "down",
        "left": "left", "west": "left", "w": "left",
        "up": "up", "north": "up", "n": "up",
    }
    return aliases.get(text)


def direction_element(value: Any) -> int | None:
    direction = normalize_direction(value)
    return None if direction is None else DIRECTION_NAMES.index(direction)


def parse_neural_binding(
    payload: Mapping[str, Any], *, minimum_confidence: float,
) -> dict[str, Any]:
    """Parse a fail-closed seven-panel orientation grounding response."""

    raw_directions = payload.get("directions")
    raw_confidences = payload.get("confidences")
    if not isinstance(raw_directions, Mapping):
        raw_directions = {}
    if not isinstance(raw_confidences, Mapping):
        raw_confidences = {}
    required = ("I", "P", "C0", *TOKENS)
    directions: dict[str, str] = {}
    confidences: dict[str, float] = {}
    errors = []
    for label in required:
        direction = normalize_direction(raw_directions.get(label))
        try:
            confidence = float(raw_confidences.get(label))
        except (TypeError, ValueError):
            confidence = -1.0
        if direction is None:
            errors.append(f"{label}:invalid_direction")
        else:
            directions[label] = direction
        if not 0.0 <= confidence <= 1.0:
            errors.append(f"{label}:invalid_confidence")
        else:
            confidences[label] = confidence
        if 0.0 <= confidence < float(minimum_confidence):
            errors.append(f"{label}:below_threshold")
    direct = str(payload.get("direct_recovery") or "").strip().upper()
    if direct not in TOKENS:
        direct = "ABSTAIN"
    qualified = not errors and len(directions) == len(required)
    if qualified:
        initial = direction_element(directions["I"])
        post = direction_element(directions["P"])
        calibration = direction_element(directions["C0"])
        assert initial is not None and post is not None and calibration is not None
        probe_effect = (post - initial) % GROUP_ORDER
        token_effects = {
            token: (
                int(direction_element(directions[token])) - calibration
            ) % GROUP_ORDER
            for token in TOKENS
        }
    else:
        probe_effect = None
        token_effects = {}
        direct = "ABSTAIN"
    return {
        "qualified": qualified,
        "directions": directions,
        "confidences": confidences,
        "minimum_confidence": float(minimum_confidence),
        "errors": errors,
        "probe_effect": probe_effect,
        "token_effects": token_effects,
        "direct_recovery": direct,
    }


def select_recovery(
    program: Mapping[str, Any], binding: Mapping[str, Any], *, condition: str,
    shuffled_token_effects: Mapping[str, int] | None = None,
) -> str:
    """Choose one anonymous target macro, or fail closed."""

    if condition == "neural_only_direct":
        direct = str(binding.get("direct_recovery") or "ABSTAIN")
        return direct if direct in TOKENS else "ABSTAIN"
    if not bool(binding.get("qualified")):
        return "ABSTAIN"
    probe = int(binding["probe_effect"])
    effects = {
        str(token): int(effect)
        for token, effect in dict(binding["token_effects"]).items()
    }
    if condition == "shuffled_binding_control":
        if shuffled_token_effects is None:
            return "ABSTAIN"
        effects = {
            str(token): int(effect)
            for token, effect in shuffled_token_effects.items()
        }
    if set(effects) != set(TOKENS):
        return "ABSTAIN"
    if condition in {
        "source_induced", "alpha_renamed_source", "shuffled_binding_control",
    }:
        matches = [
            token for token in TOKENS
            if program_predicts_identity(
                program, probe_effect=probe, recovery_effect=effects[token],
                group_order=GROUP_ORDER,
            )
        ]
    elif condition == "target_written_isomorphic":
        # This ceiling is deliberately independent of the source artifact.  It
        # demonstrates that execution depends on program content, not origin.
        matches = [
            token for token in TOKENS
            if (probe + effects[token]) % GROUP_ORDER == 0
        ]
    elif condition == "copy_effect_control":
        matches = [token for token in TOKENS if effects[token] == probe]
    elif condition == "fixed_token_control":
        return TOKENS[0]
    else:
        raise ValueError(f"unknown orientation recovery condition: {condition}")
    return matches[0] if len(matches) == 1 else "ABSTAIN"


def ground_truth_binding(spec: OrientationTaskSpec, initial_direction: int) -> dict[str, Any]:
    """Build an evaluator-only perfect binding for unit and simulator checks."""

    post = (int(initial_direction) + spec.probe_effect) % GROUP_ORDER
    calibration = (int(initial_direction) + 2) % GROUP_ORDER
    directions = {
        "I": DIRECTION_NAMES[int(initial_direction)],
        "P": DIRECTION_NAMES[post],
        "C0": DIRECTION_NAMES[calibration],
        **{
            token: DIRECTION_NAMES[(calibration + effect) % GROUP_ORDER]
            for token, effect in spec.token_to_effect.items()
        },
    }
    return parse_neural_binding(
        {
            "directions": directions,
            "confidences": {label: 1.0 for label in directions},
            "direct_recovery": next(
                token for token, effect in spec.token_to_effect.items()
                if (spec.probe_effect + effect) % GROUP_ORDER == 0
            ),
        },
        minimum_confidence=0.0,
    )


def rotated_donor_bindings(
    rows: Sequence[Mapping[str, Any]], *, namespace: str,
) -> dict[int, dict[str, int]]:
    """Create a deterministic cross-task effect-binding negative control."""

    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{namespace}\0{int(row['seed'])}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) < 2:
        return {}
    return {
        int(row["seed"]): {
            str(token): int(effect)
            for token, effect in dict(
                ordered[(index + 1) % len(ordered)]["binding"]["token_effects"]
            ).items()
        }
        for index, row in enumerate(ordered)
    }


__all__ = [
    "CONDITIONS", "DIRECTION_NAMES", "GROUP_ORDER", "TOKENS",
    "OrientationTaskSpec", "direction_element", "ground_truth_binding",
    "normalize_direction", "parse_neural_binding", "rotated_donor_bindings",
    "select_recovery", "task_spec",
]
