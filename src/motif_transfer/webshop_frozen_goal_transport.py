"""Content-addressed transport for frozen synthetic WebShop goals."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Mapping

from .contracts import stable_hash


def _identity_without_sampled_price(goal: Mapping[str, Any]) -> dict[str, Any]:
    """Return native semantic identity while ignoring sampled price wording."""

    return {
        key: deepcopy(value)
        for key, value in goal.items()
        if key not in {"instruction_text", "price_upper"}
    }


def install_frozen_goal_overrides(
    app_module: Any, manifest: Mapping[str, Any],
) -> None:
    """Replay frozen selected goals after verifying their native identity.

    Upstream WebShop samples a price threshold while synthesizing goals.  Its
    random stream is not stable across every process/runtime combination.  A
    frozen experimental task must nevertheless denote identical semantics in
    every matched arm.  This adapter replaces only manifest-selected goal
    indices with their pre-outcome snapshots and rejects any native drift in
    ASIN, options, attributes, query, title, category, or weight.

    The adapter never reads a source artifact, condition, action, reward, or
    formal outcome.  Search and reward computation remain upstream-native.
    """

    if getattr(app_module, "_STRUCTURAL_FROZEN_GOALS_INSTALLED", False):
        return
    body = dict(manifest)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("frozen WebShop manifest hash mismatch")
    rows = [
        row for role_rows in (manifest.get("roles") or {}).values()
        for row in role_rows
    ]
    overrides: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["server_goal_index"])
        goal = deepcopy(dict(row["goal"]))
        if stable_hash(goal) != str(row["goal_sha256"]):
            raise ValueError(f"frozen goal hash mismatch at index {index}")
        if index in overrides:
            raise ValueError(f"duplicate frozen goal index: {index}")
        overrides[index] = goal

    original_get_goals = app_module.get_goals

    def get_goals(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        goals = list(original_get_goals(*args, **kwargs))
        # The upstream app resets GOAL_SEED and shuffles immediately after
        # get_goals() returns.  Compute that exact permutation locally so a
        # manifest's post-shuffle server index maps to the corresponding
        # pre-shuffle list slot.  Do not mutate the process-global RNG here.
        permutation = list(range(len(goals)))
        random.Random(int(app_module.GOAL_SEED)).shuffle(permutation)
        for server_index, frozen in overrides.items():
            if not 0 <= server_index < len(goals):
                raise RuntimeError(
                    f"frozen goal index outside native pool: {server_index}"
                )
            native_index = permutation[server_index]
            native = goals[native_index]
            if _identity_without_sampled_price(native) != (
                _identity_without_sampled_price(frozen)
            ):
                raise RuntimeError(
                    "native WebShop semantic identity drift at post-shuffle "
                    f"index {server_index}"
                )
            goals[native_index] = deepcopy(frozen)
        app_module._STRUCTURAL_FROZEN_GOALS_APPLIED = len(overrides)
        return goals

    app_module.get_goals = get_goals
    app_module._STRUCTURAL_FROZEN_GOALS_INSTALLED = True


__all__ = ["install_frozen_goal_overrides"]
