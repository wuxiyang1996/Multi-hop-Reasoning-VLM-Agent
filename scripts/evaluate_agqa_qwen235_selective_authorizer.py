#!/usr/bin/env python3
"""Evaluate a frozen selective authorizer after runtime receipts are complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motif_transfer.agqa_qwen235_selective_authorizer import (  # noqa: E402
    authorize_source_override,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


def _normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    for prefix in ("the answer is ", "it is ", "they were ", "they are "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return {"true": "yes", "false": "no"}.get(text, text)


def _matches(prediction: object, gold: object) -> bool:
    predicted, expected = _normalize(prediction), _normalize(gold)
    if expected in {"yes", "no", "before", "after"}:
        return bool(predicted) and predicted.split(maxsplit=1)[0] == expected
    return predicted == expected


def evaluate(input_path: Path) -> dict:
    source = json.loads(input_path.read_text())
    rows = []
    for runtime in source["rows"]:
        authorization = authorize_source_override(runtime)
        gold = runtime["gold_answer_evaluator_only"]
        direct_correct = _matches(runtime["direct_response"], gold)
        source_correct = _matches(authorization["prediction"], gold)
        rows.append({
            "task_id": runtime["task_id"],
            "authorization": authorization,
            "direct_correct": direct_correct,
            "source_correct": source_correct,
            "win": source_correct and not direct_correct,
            "loss": direct_correct and not source_correct,
        })
    direct = sum(row["direct_correct"] for row in rows)
    source_correct = sum(row["source_correct"] for row in rows)
    wins = sum(row["win"] for row in rows)
    losses = sum(row["loss"] for row in rows)
    authorized = sum(row["authorization"]["authorized"] for row in rows)
    body = {
        "schema_version": "agqa-qwen235-selective-authorizer-report-v1",
        "status": "DEVELOPMENT_CANDIDATE_PASSED" if wins >= 1 and losses == 0 else "DEVELOPMENT_CANDIDATE_NOT_PASSED",
        "claim_boundary": "EVALUATOR-OPENED_CONSUMED_DEVELOPMENT;NOT_A_FORMAL_TRANSFER_RESULT",
        "input_report_sha256": source["report_sha256"],
        "sample_count": len(rows),
        "direct_correct": direct,
        "source_correct": source_correct,
        "authorized": authorized,
        "wins": wins,
        "losses": losses,
        "net_gain": source_correct - direct,
        "rows": rows,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "sample_count", "direct_correct", "source_correct", "authorized", "wins", "losses", "net_gain", "report_sha256")}, indent=2))


if __name__ == "__main__":
    main()
