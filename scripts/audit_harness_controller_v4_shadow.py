#!/usr/bin/env python3
"""Audit exact-shadow predictions before a neural Harness may enter live mode."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.neural_harness_controller import (  # noqa: E402
    structural_controller_output_valid,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _strict_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _controller_input(prompt: str) -> dict[str, Any]:
    try:
        payload = prompt.split("\nCONTROLLER_INPUT=", 1)[1].rsplit(
            "\nOUTPUT_JSON=", 1,
        )[0]
        value = json.loads(payload)
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("malformed frozen controller prompt") from error
    if not isinstance(value, dict):
        raise ValueError("controller input is not a JSON object")
    return value


def _rates(counter: Counter) -> dict[str, Any]:
    rows = int(counter["rows"])
    return {
        "rows": rows,
        **{
            f"{key}_rate": counter[key] / rows
            for key in (
                "valid_json", "structural_valid", "decision_correct", "exact",
                "source_operator_authorized", "false_positive_execute",
                "false_negative_execute",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    output_root = REPO / "runs/harness_controller_qwen35_9b_v4/cardinality_sft_v1"
    parser.add_argument(
        "--reserve", type=Path,
        default=REPO / "runs/harness_controller_v4_fresh_target_reserve/reserve.jsonl",
    )
    parser.add_argument(
        "--activation", type=Path,
        default=output_root / "fresh_target_activation.json",
    )
    parser.add_argument(
        "--report", type=Path, default=output_root / "fresh_target_report.json",
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=output_root / "fresh_target_report.predictions.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=output_root / "fresh_target_shadow_audit.json",
    )
    args = parser.parse_args()

    activation = json.loads(args.activation.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if (
        report.get("status") != "TARGET_IR_ZERO_SHOT_CONTROLLER_GATE_PASSED"
        or not all(report.get("gates", {}).values())
    ):
        raise SystemExit("target controller gate did not pass; live shadow audit is closed")
    if (
        activation.get("status") != "FROZEN_TARGET_IR_ZERO_SHOT_EVALUATION_READY"
        or not all(activation.get("gates", {}).values())
    ):
        raise SystemExit("fresh target activation is not gate-clean")
    if (
        _sha256(args.reserve) != activation["evaluation_file"]["sha256"]
        or report.get("dataset_sha256") != _sha256(args.reserve)
        or report.get("adapter_model_sha256")
        != activation["frozen_model"]["adapter_model_sha256"]
    ):
        raise SystemExit("shadow inputs do not match the frozen activation")

    rows = _read_jsonl(args.reserve)
    lora_predictions = [
        row for row in _read_jsonl(args.predictions)
        if row.get("regime") == "CONTROLLER_LORA"
    ]
    predictions = {str(row["example_id"]): row for row in lora_predictions}
    if len(predictions) != len(lora_predictions):
        raise SystemExit("duplicate LoRA prediction example IDs")

    overall = Counter()
    by_group: dict[str, Counter] = defaultdict(Counter)
    by_control: dict[str, Counter] = defaultdict(Counter)
    by_cardinality: dict[str, Counter] = defaultdict(Counter)
    consistency = True
    prompt_forbidden = False
    for row in rows:
        example_id = str(row["example_id"])
        prediction = predictions.get(example_id)
        if prediction is None:
            consistency = False
            continue
        parsed = _strict_json(str(prediction["generated_text"]))
        target = json.loads(row["completion"])
        controller_input = _controller_input(str(row["prompt"]))
        structural = structural_controller_output_valid(
            anonymous_program=controller_input["program"],
            state=controller_input["symbolic_state"],
            candidates=controller_input["candidate_effects"],
            observed_effect=controller_input["observed_effect"],
            output=parsed,
        )
        valid = parsed is not None
        exact = parsed == target
        decision_correct = bool(
            valid and parsed.get("decision") == target["decision"]
        )
        authorized = bool(
            structural and parsed is not None
            and parsed.get("decision") == "EXECUTE_OPERATOR"
        )
        target_execute = target["decision"] == "EXECUTE_OPERATOR"
        values = {
            "rows": 1,
            "valid_json": int(valid),
            "structural_valid": int(structural),
            "decision_correct": int(decision_correct),
            "exact": int(exact),
            "source_operator_authorized": int(authorized),
            "false_positive_execute": int(authorized and not target_execute),
            "false_negative_execute": int(target_execute and not authorized),
            "exact_but_structurally_invalid": int(exact and not structural),
        }
        overall.update(values)
        by_group[str(row["target_eval_group_audit_only"])].update(values)
        by_control[str(row["control_variant_audit_only"])].update(values)
        by_cardinality[str(len(controller_input["candidate_effects"]))].update(values)
        consistency = consistency and (
            prediction.get("parsed") == parsed
            and bool(prediction.get("exact_json")) == exact
            and prediction.get("target") == target
        )
        prompt_forbidden = prompt_forbidden or any(
            token in str(row["prompt"])
            for token in (
                "official_success", "official_reward", "native_actions",
                "selected_action", "target_domain",
            )
        )

    group_rates = {key: _rates(value) for key, value in sorted(by_group.items())}
    control_rates = {key: _rates(value) for key, value in sorted(by_control.items())}
    cardinality_rates = {
        key: _rates(value) for key, value in sorted(
            by_cardinality.items(), key=lambda item: int(item[0]),
        )
    }
    gates = {
        "target_exact_gate_passed": all(report["gates"].values()),
        "one_lora_prediction_per_reserve_row": (
            len(rows) == len(lora_predictions) == len(predictions)
        ),
        "prediction_example_set_exact": (
            {str(row["example_id"]) for row in rows} == set(predictions)
        ),
        "stored_prediction_fields_consistent": consistency,
        "all_exact_outputs_are_structurally_valid": (
            overall["exact_but_structurally_invalid"] == 0
        ),
        "no_false_positive_source_operator_authorization": (
            overall["false_positive_execute"] == 0
        ),
        "every_group_structural_valid_rate_at_least_0p98": (
            bool(group_rates)
            and min(row["structural_valid_rate"] for row in group_rates.values()) >= 0.98
        ),
        "controller_prompt_has_no_target_outcome_or_native_action": not prompt_forbidden,
    }
    payload = {
        "schema_version": "harness-controller-v4-live-shadow-audit-v1",
        "status": (
            "NEURAL_HARNESS_STRUCTURAL_LIVE_QUALIFIED"
            if all(gates.values()) else "NEURAL_HARNESS_STRUCTURAL_LIVE_BLOCKED"
        ),
        "authority": (
            "POST_TARGET_GATE_EXACT_SHADOW;NO_TARGET_ACTION;"
            "STRUCTURAL_LIVE_MODE_DOES_NOT_RECOMPUTE_CANDIDATE_ARGMAX"
        ),
        "inputs": {
            "reserve": {"path": str(args.reserve.resolve()), "sha256": _sha256(args.reserve)},
            "activation": {
                "path": str(args.activation.resolve()), "sha256": _sha256(args.activation),
            },
            "report": {"path": str(args.report.resolve()), "sha256": _sha256(args.report)},
            "predictions": {
                "path": str(args.predictions.resolve()), "sha256": _sha256(args.predictions),
            },
        },
        "overall": _rates(overall),
        "by_target_group": group_rates,
        "by_control": control_rates,
        "by_cardinality": cardinality_rates,
        "gates": gates,
        "next_legal_step": (
            "FREEZE_LIVE_WRAPPER_AND_RUN_NON_ACTING_DOMAIN_RECEIPT_REPLAY"
            if all(gates.values()) else "DO_NOT_ENABLE_NEURAL_HARNESS_LIVE_AUTHORITY"
        ),
        "claim_boundary": (
            "This audit may qualify the structural-only wrapper for non-acting domain "
            "receipt replay. It is not an environment success result. Exact target "
            "completions are used only by this shadow audit and are unavailable to "
            "the structural-only live controller."
        ),
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"], "overall": payload["overall"],
        "gates": gates, "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
