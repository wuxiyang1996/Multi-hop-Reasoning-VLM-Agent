#!/usr/bin/env python3
"""Run the V24 Normal matched fork without changing the frozen Easy adapter."""

from __future__ import annotations

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from motif_transfer.discoveryworld_normal_binding import parse_normal_target_binding
from scripts import run_discoveryworld_commit_recovery_v1 as frozen_runner


def main() -> None:
    # The generic runner imported the frozen Easy parser into its module scope.
    # Override only that target-native boundary for the diagnosed Normal route.
    frozen_runner.parse_target_binding = parse_normal_target_binding
    frozen_runner.main()


if __name__ == "__main__":
    main()
