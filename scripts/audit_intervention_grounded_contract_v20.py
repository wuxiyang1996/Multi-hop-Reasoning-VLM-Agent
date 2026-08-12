#!/usr/bin/env python3
"""Audit the failed V18/V19 transfer contract without reading new outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.neurosymbolic_transfer_contract import (  # noqa: E402
    transfer_contract_audit,
)
from train_intervention_grounded_option_advantage_v17 import (  # noqa: E402
    _grounded_splits,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_stable_hash(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = body.pop(field)
    if stable_hash(body) != claimed:
        raise ValueError(f"stable hash mismatch for {field}")


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty audit sample")
    return {
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _source_support(config: Mapping[str, Any]) -> dict[str, Any]:
    splits = _grounded_splits(config)
    output: dict[str, Any] = {}
    for split, rows in splits.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row.state_id)].append(float(row.features[11]))
        high_counts = [sum(value > 0.1 for value in values) for values in grouped.values()]
        pair_sums = [
            left + right
            for values in grouped.values()
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        ]
        output[split] = {
            "states": len(grouped),
            "option_rows": len(rows),
            "effect_score_quantiles": _quantiles([
                value for values in grouped.values() for value in values
            ]),
            "options_above_0p1_per_state_counts": {
                str(count): high_counts.count(count)
                for count in sorted(set(high_counts))
            },
            "pair_effect_sum_quantiles": _quantiles(pair_sums),
        }
    return output


def _target_shadow_support(shadow: Mapping[str, Any]) -> dict[str, Any]:
    scores: list[float] = []
    pair_sums: list[float] = []
    high_counts: list[int] = []
    selected_deltas: list[float] = []
    by_pair: dict[str, list[tuple[float, float]]] = defaultdict(list)
    contrasts = 0
    for task in shadow["tasks"]:
        for row in task["contrasts"]:
            contrasts += 1
            option_scores = {
                str(option): float(value)
                for option, value in row["neural_effect_by_option"].items()
            }
            scores.extend(option_scores.values())
            high_counts.append(sum(value > 0.1 for value in option_scores.values()))
            fallback = str(row["fallback_option"])
            selected = str(row["authentic_option"])
            fallback_score = option_scores[fallback]
            selected_score = option_scores[selected]
            selected_deltas.append(selected_score - fallback_score)
            pair_sums.append(selected_score + fallback_score)
            by_pair[f"{fallback}->{selected}"].append(
                (selected_score, fallback_score)
            )
    return {
        "states_with_authentic_contrast": contrasts,
        "legacy_score_semantics": "expert_action_imitation_score",
        "legacy_option_score_quantiles": _quantiles(scores),
        "options_above_0p1_per_state_counts": {
            str(count): high_counts.count(count)
            for count in sorted(set(high_counts))
        },
        "selected_plus_fallback_score_quantiles": _quantiles(pair_sums),
        "selected_minus_fallback_score_quantiles": _quantiles(selected_deltas),
        "by_option_contrast": {
            pair: {
                "count": len(values),
                "mean_selected_score": float(np.mean([row[0] for row in values])),
                "mean_fallback_score": float(np.mean([row[1] for row in values])),
                "mean_selected_minus_fallback": float(np.mean([
                    row[0] - row[1] for row in values
                ])),
            }
            for pair, values in sorted(by_pair.items())
        },
    }


def _longest_run(actions: Iterable[str]) -> tuple[int, str]:
    best_length = 0
    best_action = ""
    current_length = 0
    current_action = ""
    for action in map(str, actions):
        if action == current_action:
            current_length += 1
        else:
            current_action = action
            current_length = 1
        if current_length > best_length:
            best_length = current_length
            best_action = action
    return best_length, best_action


def _outcome_loops(outcomes: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition, episodes in outcomes["episodes"].items():
        adjacent_rates: list[float] = []
        longest_runs: list[int] = []
        examples: list[dict[str, Any]] = []
        for episode in episodes:
            actions = [str(row["action"]) for row in episode["records"]]
            adjacent_rates.append(
                sum(left == right for left, right in zip(actions, actions[1:]))
                / max(1, len(actions) - 1)
            )
            run_length, run_action = _longest_run(actions)
            longest_runs.append(run_length)
            examples.append({
                "task_id": str(episode["task_id"]),
                "longest_identical_action_run": run_length,
                "action": run_action,
                "source_interventions": int(episode["source_interventions"]),
            })
        result[condition] = {
            "tasks": len(episodes),
            "successes": sum(bool(row["official_success"]) for row in episodes),
            "mean_adjacent_exact_action_repeat_rate": float(np.mean(adjacent_rates)),
            "median_longest_identical_action_run": float(np.median(longest_runs)),
            "maximum_longest_identical_action_run": max(longest_runs),
            "worst_run_examples": sorted(
                [
                    row for row in examples
                    if condition == "authentic_intervention_effect"
                    and row["source_interventions"] > 0
                ],
                key=lambda row: (-row["longest_identical_action_run"], row["task_id"]),
            )[:5],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=REPO / "configs/intervention_grounded_option_advantage_source_v17.json",
    )
    parser.add_argument(
        "--target-grounder",
        type=Path,
        default=REPO / "configs/oracle_free_target_policy_candidate_v18.json",
    )
    parser.add_argument(
        "--shadow-report",
        type=Path,
        default=REPO / "runs/intervention_grounded_transfer_shadow_v18/report.json",
    )
    parser.add_argument(
        "--outcome-report",
        type=Path,
        default=REPO / "runs/intervention_grounded_outcome_forks_v19/report.json",
    )
    args = parser.parse_args()
    source_config = _read(args.source_config)
    target = _read(args.target_grounder)
    shadow = _read(args.shadow_report)
    outcomes = _read(args.outcome_report)
    _validate_stable_hash(target, "artifact_sha256")
    _validate_stable_hash(shadow, "report_sha256")
    _validate_stable_hash(outcomes, "report_sha256")
    if shadow.get("confirmation_or_valid_unseen_read_or_run") is not False:
        raise ValueError("V18 crossed the held-out boundary")
    if outcomes.get("confirmation_or_valid_unseen_read_or_run") is not False:
        raise ValueError("V19 crossed the held-out boundary")

    body = {
        "schema_version": "intervention-grounded-contract-audit-v20",
        "status": "V18_CONTRACT_INVALIDATED_V19_MECHANISM_REJECTED",
        "claim_boundary": (
            "Post-hoc implementation and representation audit of already-consumed "
            "V18/V19 development artifacts only; no new target outcomes and no "
            "confirmation or valid_unseen access."
        ),
        "lineage": {
            "source_config_sha256": _sha256(args.source_config),
            "target_grounder_sha256": _sha256(args.target_grounder),
            "target_grounder_artifact_sha256": target["artifact_sha256"],
            "shadow_report_sha256": shadow["report_sha256"],
            "outcome_report_sha256": outcomes["report_sha256"],
        },
        "contract_audit": transfer_contract_audit(
            target_grounder=target,
            source_target_support_receipt=None,
        ),
        "source_effect_support": _source_support(source_config),
        "target_shadow_score_support": _target_shadow_support(shadow),
        "matched_outcome_loop_audit": _outcome_loops(outcomes),
        "disposition": {
            "v18": (
                "POSTHOC_INVALIDATED: the outcome-blind contrast demonstrates that "
                "a source model can alter options under this numerical wiring, not "
                "that a causally grounded neural-symbolic skill transferred."
            ),
            "v19": (
                "MECHANISM_REJECTED: 0/64 successes in every condition and materially "
                "longer action loops under authentic overrides."
            ),
            "code_fix": (
                "Fail closed unless target scores are calibrated counterfactual "
                "successor-event probabilities and a joint source-target support "
                "receipt certifies conformal applicability."
            ),
        },
        "confirmation_or_valid_unseen_read_or_run": False,
    }
    report = body | {"audit_sha256": stable_hash(body)}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
