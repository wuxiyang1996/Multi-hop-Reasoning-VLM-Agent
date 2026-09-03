#!/usr/bin/env python3
"""Freeze disjoint candidates, then apply a preregistered validity filter."""

from __future__ import annotations

from pathlib import Path

import prepare_alfworld_target_acquisition_train_itt_v20 as protocol


REPO = Path(__file__).resolve().parents[1]
# Reuse V20's ranking namespace and take the next rank slice, which makes the
# V21 candidates disjoint from every V20 reset attempt by construction.
protocol.NAMESPACE = "alfworld-target-acquisition-fresh-v20-train-itt"
protocol.STATUS = (
    "FROZEN_V21_CANDIDATES_AND_VALIDITY_RULE_BEFORE_COMPILATION_OR_POLICY_RESET"
)
protocol.TASK_OFFSET = 16
protocol.TASK_COUNT = 32
protocol.SOLVABLE_ELIGIBILITY_FILTER = True
protocol.MINIMUM_FORMAL_TASKS = 8
protocol.OUTPUT_DIR = (
    REPO / "configs/alfworld_target_acquisition_compiler_valid_v21"
)
protocol.SELECTION_PATH = protocol.OUTPUT_DIR / "selection.json"
protocol.COMPILER_AUDIT_PATH = protocol.OUTPUT_DIR / "compiler_audit.json"
protocol.CONFIG_PATH = protocol.OUTPUT_DIR / "formal.json"
protocol.RETRY_CONFIG_PATH = protocol.OUTPUT_DIR / "formal_retry.json"
protocol.RETRY2_CONFIG_PATH = protocol.OUTPUT_DIR / "formal_retry2.json"
protocol.GENERATED_DATA = (
    REPO / "runs/alfworld_target_acquisition_compiler_valid_v21/alfworld_data"
)
protocol.GENERATED_SPLIT = protocol.GENERATED_DATA / "json_2.1.1/train"
protocol.PREPARER_RELATIVE = (
    "scripts/prepare_alfworld_target_acquisition_compiler_valid_v21.py"
)
protocol.REPORT_OUTPUT = (
    "runs/alfworld_target_acquisition_compiler_valid_v21/report.json"
)


if __name__ == "__main__":
    raise SystemExit(protocol.main())
