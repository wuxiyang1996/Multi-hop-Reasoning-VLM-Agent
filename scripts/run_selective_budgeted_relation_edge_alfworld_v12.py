#!/usr/bin/env python3
"""Run V12 using the shared executable-source-graph rollout engine."""

from __future__ import annotations

import run_executable_source_graph_alfworld_v9 as shared_runner

from motif_transfer.slot_aware_alfworld_harness_v12 import (
    CONDITIONS,
    choose_slot_aware_action,
    condition_required_property,
)


def main() -> int:
    shared_runner.CONDITIONS = CONDITIONS
    shared_runner.choose_slot_aware_action = choose_slot_aware_action
    shared_runner.condition_required_property = condition_required_property
    return shared_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
