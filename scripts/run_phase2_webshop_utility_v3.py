#!/usr/bin/env python3
"""Run Phase-2 WebShop with a task-agnostic safe candidate fallback."""

from __future__ import annotations

from typing import Any, Mapping

from motif_transfer.webshop_candidate_failclosed_v3 import (
    failclosed_decision_candidates,
)
from motif_transfer.webshop_constraint_coverage_v14 import (
    augment_with_constraint_labels,
    augment_with_product_backtrack,
)
import scripts.run_phase2_webshop_utility_v1 as base_runner
from scripts.run_webshop_search_automaton_v16 import (
    ORIGINAL_DECISION_CANDIDATES,
)


def _candidate_augmenter(goal_options: Mapping[str, Any]):
    def decide(**kwargs: Any):
        candidates, raw, attempts = failclosed_decision_candidates(
            ORIGINAL_DECISION_CANDIDATES, **kwargs,
        )
        augmented = augment_with_constraint_labels(
            candidates,
            axtree=kwargs["axtree"],
            goal=str(kwargs["payload"]["goal"]),
            goal_options=goal_options,
        )
        augmented = augment_with_product_backtrack(
            augmented, url=str(kwargs["payload"].get("url") or ""),
        )
        return augmented, raw, attempts

    return decide


def main() -> int:
    base_runner._candidate_augmenter = _candidate_augmenter
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
