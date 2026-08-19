#!/usr/bin/env python3
"""V35 development collector with frozen syntax/transport normalization."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_operand_normalization import (  # noqa: E402
    parse_normalized_operand_receipt,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_active_grounding_v3 as base  # noqa: E402
from scripts.collect_agqa2_robust_temporal_v34_formal import (  # noqa: E402
    evaluate,
)


def _json_transport_retry(original, *, attempts: int):
    def invoke(*args, **kwargs):
        last_error = None
        for _ in range(attempts):
            try:
                return original(*args, **kwargs)
            except json.JSONDecodeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error
    return invoke


def collect_development(**kwargs):
    """Replay cached calls; repair syntax only before strict parsing."""

    original_parse = base.parse_operand_receipt
    original_provider = base._provider_json_call
    base.parse_operand_receipt = parse_normalized_operand_receipt
    base._provider_json_call = _json_transport_retry(
        original_provider, attempts=3,
    )
    try:
        return base.collect(**kwargs)
    finally:
        base.parse_operand_receipt = original_parse
        base._provider_json_call = original_provider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_robust_temporal_v35_development.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=(
            REPO_ROOT
            / "runs/agqa2_robust_temporal_v35_development/base_report.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            REPO_ROOT / "runs/agqa2_robust_temporal_v35_development/report.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    collect_development(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.base_report.resolve(), workers=args.workers,
        limit=None,
    )
    result = evaluate(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(),
    )
    body = deepcopy(result)
    body.pop("report_sha256", None)
    qualified = all(body["qualification_gates"].values())
    body.update({
        "schema_version": "agqa2-robust-temporal-v35-development-report-v1",
        "status": (
            "AGQA2_ROBUST_TEMPORAL_V35_DEVELOPMENT_QUALIFIED"
            if qualified else
            "AGQA2_ROBUST_TEMPORAL_V35_DEVELOPMENT_NOT_QUALIFIED"
        ),
        "split": "consumed_development",
        "confirmatory_claim": False,
        "claim_boundary": (
            "V34_OUTCOME_UNREAD_RUNTIME_POOL_CONSUMED_AS_V35_DEVELOPMENT;"
            "DETERMINISTIC_INTERVAL_EVIDENCE_ENVELOPE_NORMALIZATION;"
            "IDENTICAL_REQUEST_JSON_TRANSPORT_RETRY;NOT_CONFIRMATORY"
        ),
        "v34_formal_reclassified_as_success": False,
    })
    final = body | {"report_sha256": stable_hash(body)}
    args.output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: final[key]
        for key in (
            "status", "rows", "source_executor_authorizations",
            "source_vs_target_native", "qualification_gates",
            "provider_calls", "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
