#!/usr/bin/env python3
"""Freeze a disjoint compiler-valid reserve with a Python 3.11 contract."""

from __future__ import annotations

from pathlib import Path

import prepare_alfworld_target_acquisition_train_itt_v20 as protocol


REPO = Path(__file__).resolve().parents[1]
# Same ranking namespace, next disjoint slice after V20 [0,16) and V21 [16,48).
protocol.NAMESPACE = "alfworld-target-acquisition-fresh-v20-train-itt"
protocol.STATUS = (
    "FROZEN_V23_CANDIDATES_VALIDITY_AND_PY311_BEFORE_COMPILATION_OR_RESET"
)
protocol.TASK_OFFSET = 48
protocol.TASK_COUNT = 32
protocol.SOLVABLE_ELIGIBILITY_FILTER = True
protocol.MINIMUM_FORMAL_TASKS = 8
protocol.OUTPUT_DIR = REPO / "configs/alfworld_target_acquisition_py311_v23"
protocol.SELECTION_PATH = protocol.OUTPUT_DIR / "selection.json"
protocol.COMPILER_AUDIT_PATH = protocol.OUTPUT_DIR / "compiler_audit.json"
protocol.CONFIG_PATH = protocol.OUTPUT_DIR / "formal.json"
protocol.RETRY_CONFIG_PATH = protocol.OUTPUT_DIR / "formal_retry.json"
protocol.RETRY2_CONFIG_PATH = protocol.OUTPUT_DIR / "formal_retry2.json"
protocol.GENERATED_DATA = (
    REPO / "runs/alfworld_target_acquisition_py311_v23/alfworld_data"
)
protocol.GENERATED_SPLIT = protocol.GENERATED_DATA / "json_2.1.1/train"
protocol.PREPARER_RELATIVE = (
    "scripts/prepare_alfworld_target_acquisition_py311_v23.py"
)
protocol.REPORT_OUTPUT = (
    "runs/alfworld_target_acquisition_py311_v23/report.json"
)


if __name__ == "__main__":
    raise SystemExit(protocol.main())
