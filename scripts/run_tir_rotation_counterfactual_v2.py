#!/usr/bin/env python3
"""Run fresh Tetris-to-TIR rotation transfer with panel interventions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.tetris_counterfactual_rotation import (  # noqa: E402
    execute_counterfactual_inverse,
)
from motif_transfer.tetris_rotation_transfer import (  # noqa: E402
    exact_sign_p,
    parse_rotation_options,
)
from scripts.develop_tir_rotation_counterfactual_v2 import (  # noqa: E402
    _contact_sheet,
    _data_url,
)


CONDITIONS = (
    "raw_target_only",
    "authentic_tetris_inverse",
    "alpha_renamed_authentic",
    "target_written_isomorphic",
    "opposite_group_control",
    "binding_rotation_control",
    "half_turn_marginal_control",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_call(
    client: OpenAI, *, model: str, system: str, text: str, image_url: str,
    maximum_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model, temperature=0, max_tokens=maximum_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    usage = response.usage
    return payload, {
        "model": str(response.model),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "response_sha256": stable_hash(payload),
    }


def _collect(
    sample_id: str, *, row: Mapping[str, Any], dataset_root: Path,
    config: Mapping[str, Any], api_key: str, contract_sha256: str,
) -> dict[str, Any]:
    options = parse_rotation_options(str(row["prompt"]))
    image_path = dataset_root / str(row["image_1"])
    with Image.open(image_path) as image:
        sheet, mapping = _contact_sheet(image, sample_id=sample_id, options=options)
        sheet_url = _data_url(sheet)
        original_url = _data_url(image.convert("RGB"))
    model = config["model"]
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    grounded, grounding_usage = _json_call(
        client, model=str(model["id"]),
        system=(
            "You are a visual orientation verifier. You cannot see numeric "
            "rotation angles, answer choices, or a gold label. Return strict JSON."
        ),
        text=(
            "Each panel is the same scene after a different anonymous rotation "
            "intervention. Which ONE panel restores the scene to its physically "
            "original upright orientation? Use gravity, walls, floors, text, "
            "people, furniture, and architecture. Return "
            "{\"panel_id\":\"P0-P5\",\"confidence\":0-1," 
            "\"visual_cues\":[string]}."
        ),
        image_url=sheet_url,
        maximum_tokens=int(model["maximum_output_tokens"]),
    )
    panel = str(grounded.get("panel_id") or "").strip().upper()
    confidence = float(grounded.get("confidence", 0.0))
    if panel not in mapping or not 0 <= confidence <= 1:
        raise ValueError("invalid anonymous counterfactual binding")
    baseline, baseline_usage = _json_call(
        client, model=str(model["id"]),
        system=(
            "Solve the visual rotation multiple-choice task. Return strict JSON "
            "{\"answer\":\"A-F\",\"reason\":\"brief\"}."
        ),
        text=str(row["prompt"]), image_url=original_url,
        maximum_tokens=int(model["maximum_output_tokens"]),
    )
    baseline_answer = str(baseline.get("answer") or "").strip().upper()[:1]
    if baseline_answer not in options:
        raise ValueError("raw target baseline returned an invalid slot")
    body: dict[str, Any] = {
        "schema_version": "tir-rotation-counterfactual-receipt-v2",
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample_id,
        "image_sha256": _sha256(image_path),
        "prompt_sha256": stable_hash(str(row["prompt"])),
        "options": options,
        "panel_to_slot": mapping,
        "selected_identity_panel": panel,
        "binding_confidence": confidence,
        "binding_visual_cues": [str(x) for x in grounded.get("visual_cues", [])][:8],
        "numeric_angles_slots_or_gold_seen_by_grounder": False,
        "grounding_usage": grounding_usage,
        "baseline_answer": baseline_answer,
        "baseline_usage": baseline_usage,
        "gold_answer_evaluator_only": str(row["answer"]),
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _paired(rows: list[dict[str, Any]], right: str) -> dict[str, Any]:
    left = "authentic_tetris_inverse"
    wins = sum(row["correct"][left] and not row["correct"][right] for row in rows)
    losses = sum(row["correct"][right] and not row["correct"][left] for row in rows)
    return {
        "wins": wins, "losses": losses, "ties": len(rows) - wins - losses,
        "exact_two_sided_p": exact_sign_p(wins, losses),
    }


def _evaluate(
    receipts: list[dict[str, Any]], *, split: str, config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for receipt in sorted(receipts, key=lambda value: str(value["sample_id"])):
        answers = {"raw_target_only": str(receipt["baseline_answer"])}
        for condition in CONDITIONS[1:]:
            answers[condition] = execute_counterfactual_inverse(
                panel_to_slot=receipt["panel_to_slot"],
                selected_identity_panel=str(receipt["selected_identity_panel"]),
                options=receipt["options"], condition=condition,
            )
        gold = str(receipt["gold_answer_evaluator_only"])
        rows.append({
            "sample_id": str(receipt["sample_id"]),
            "answers": answers,
            "correct": {name: answer == gold for name, answer in answers.items()},
            "gold_answer_evaluator_only": gold,
            "selected_identity_panel": receipt["selected_identity_panel"],
            "binding_confidence": receipt["binding_confidence"],
        })
    metrics = {
        condition: {
            "tasks": len(rows),
            "correct": sum(row["correct"][condition] for row in rows),
            "accuracy": sum(row["correct"][condition] for row in rows) / len(rows),
            "changes_vs_raw": sum(
                row["answers"][condition] != row["answers"]["raw_target_only"]
                for row in rows
            ),
        }
        for condition in CONDITIONS
    }
    paired = {
        condition: _paired(rows, condition)
        for condition in CONDITIONS if condition != "authentic_tetris_inverse"
    }
    authentic = metrics["authentic_tetris_inverse"]
    destructive = (
        "opposite_group_control", "binding_rotation_control",
        "half_turn_marginal_control",
    )
    expected = len(config["splits"][split])
    gates = {
        "exact_frozen_task_count": len(rows) == expected,
        "all_bindings_valid": all(0 <= row["binding_confidence"] <= 1 for row in rows),
        "minimum_authentic_accuracy": authentic["accuracy"] >= float(config["formal_gates"]["minimum_authentic_accuracy"]),
        "nontrivial_action_changes": authentic["changes_vs_raw"] >= max(2, expected // 5),
        "alpha_rename_invariance": all(row["answers"]["authentic_tetris_inverse"] == row["answers"]["alpha_renamed_authentic"] for row in rows),
        "target_isomorphic_equivalence": all(row["answers"]["authentic_tetris_inverse"] == row["answers"]["target_written_isomorphic"] for row in rows),
        "authentic_strictly_above_raw": authentic["correct"] > metrics["raw_target_only"]["correct"],
        "paired_wins_above_losses": paired["raw_target_only"]["wins"] > paired["raw_target_only"]["losses"],
        "authentic_strictly_above_destructive_controls": all(authentic["correct"] > metrics[name]["correct"] for name in destructive),
    }
    passed = all(gates.values())
    stage = "FRESH_QUALIFICATION" if split == "qualification" else "FRESH_FORMAL"
    body: dict[str, Any] = {
        "schema_version": "tir-rotation-counterfactual-report-v2",
        "status": f"{stage}_COUNTERFACTUAL_ROTATION_GATE_{'PASSED' if passed else 'FAILED'}",
        "split": split, "metrics": metrics, "paired_authentic": paired,
        "gates": gates, "rows": rows, "claim_boundary": config["claim_boundary"],
        "source_artifact_sha256": config["source"]["artifact_content_sha256"],
    }
    body["report_sha256"] = stable_hash(body)
    return body


def _validate_report(path: Path, expected_status: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    claimed = body.pop("report_sha256", None)
    if claimed != stable_hash(body) or value["status"] != expected_status:
        raise SystemExit(f"authority report does not authorize target calls: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("qualification", "heldout"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_FRESH_COUNTERFACTUAL_CALLS":
        raise SystemExit("counterfactual protocol is not frozen")
    for relative, expected in config["integrity"]["file_sha256"].items():
        if _sha256(REPO / relative) != expected:
            raise SystemExit(f"frozen dependency changed: {relative}")
    _validate_report(
        REPO / config["authority"]["development_report"],
        "CONSUMED_DEVELOPMENT_COUNTERFACTUAL_GATE_PASSED",
    )
    if args.split == "heldout":
        _validate_report(
            REPO / config["authority"]["qualification_report"],
            "FRESH_QUALIFICATION_COUNTERFACTUAL_ROTATION_GATE_PASSED",
        )
    dataset_path = args.dataset_root / "TIR-Bench.json"
    if _sha256(dataset_path) != config["dataset"]["file_sha256"]:
        raise SystemExit("TIR dataset drift")
    source_path = REPO / config["source"]["artifact"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if _sha256(source_path) != config["source"]["artifact_file_sha256"]:
        raise SystemExit("Tetris source receipt drift")
    if source.get("artifact_sha256") != config["source"]["artifact_content_sha256"]:
        raise SystemExit("Tetris source receipt self-hash drift")
    index = {str(row["id"]): row for row in json.loads(dataset_path.read_text())}
    sample_ids = [str(value) for value in config["splits"][args.split]]
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    contract = stable_hash({
        "config_sha256": _sha256(args.config), "split": args.split,
        "source_artifact_sha256": source["artifact_sha256"],
    })
    receipts_dir = args.output_dir / f"{args.split}_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, dict[str, Any]] = {}
    pending = []
    for sample_id in sample_ids:
        path = receipts_dir / f"{sample_id}.json"
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("collection_contract_sha256") != contract:
                raise SystemExit(f"receipt contract drift: {sample_id}")
            receipts[sample_id] = value
        else:
            pending.append(sample_id)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _collect, sample_id, row=index[sample_id], dataset_root=args.dataset_root,
                config=config, api_key=str(key), contract_sha256=contract,
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            value = future.result()
            (receipts_dir / f"{sample_id}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            receipts[sample_id] = value
            print(json.dumps({"sample_id": sample_id, "panel": value["selected_identity_panel"], "baseline": value["baseline_answer"]}), flush=True)
    report = _evaluate(
        [receipts[sample_id] for sample_id in sample_ids], split=args.split,
        config=config,
    )
    output = args.output_dir / f"{args.split}_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"], "paired": report["paired_authentic"]["raw_target_only"], "gates": report["gates"], "output": str(output)}, indent=2, sort_keys=True))
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
