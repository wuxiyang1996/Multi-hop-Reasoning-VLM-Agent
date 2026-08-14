#!/usr/bin/env python3
"""Run the frozen ALFWorld controller with identity-aware multiplicity state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer import alfworld_multiplicity_grounder as multiplicity  # noqa: E402
import scripts.run_multisource_alfworld_v2_qualification as base  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _choose_without_reversing_completed_bindings(**kwargs):
    grounded = kwargs["grounded"]
    admissible = {
        action: row for action, row in grounded.items()
        if not bool(row.get("reverses_completed_binding"))
    }
    if admissible:
        kwargs["grounded"] = admissible
    return _BASE_CHOOSE(**kwargs)


_BASE_CHOOSE = base._choose_action


def main() -> int:
    if "--config" not in sys.argv:
        raise SystemExit("--config is required")
    config_path = Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = config.get("multiplicity_extension", {})
    if config.get("status") == "FINAL_FROZEN_HELDOUT_EVALUATION":
        if contract.get("grounder_sha256") != _sha256(
            Path(multiplicity.__file__).resolve()
        ):
            raise SystemExit("frozen multiplicity grounder changed")
        if contract.get("runner_sha256") != _sha256(Path(__file__).resolve()):
            raise SystemExit("frozen multiplicity runner changed")
    base.workflow_status = multiplicity.workflow_status
    base.score_actions = multiplicity.score_actions
    base._choose_action = _choose_without_reversing_completed_bindings
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
