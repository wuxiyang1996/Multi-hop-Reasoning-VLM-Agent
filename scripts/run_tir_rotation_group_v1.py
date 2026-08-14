#!/usr/bin/env python3
"""Run Tetris rotation-group transfer on TIR rotation_game."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.tetris_rotation_transfer import (  # noqa: E402
    exact_sign_p,
    parse_rotation_options,
    select_rotation_action,
)


CONDITIONS = (
    "raw_target_only",
    "authentic_tetris_inverse",
    "alpha_renamed_authentic",
    "target_written_isomorphic",
    "no_inverse_control",
    "shuffled_binding_control",
    "half_turn_marginal_control",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_data(image: Image.Image, *, max_side: int, quality: int) -> str:
    rendered = image.convert("RGB").copy()
    rendered.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rendered.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _json_call(
    client: OpenAI, *, model: str, system: str, text: str, image_url: str,
    maximum_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=maximum_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("empty target neural response")
    payload = json.loads(raw)
    usage = response.usage
    return payload, {
        "model": str(response.model),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "cost": float(getattr(usage, "cost", 0.0) or 0.0),
        "response_sha256": stable_hash(payload),
    }


def _bind(
    client: OpenAI, *, model: str, image_url: str, maximum_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Estimate image orientation only. You cannot see answer choices or "
            "a gold label. Return strict JSON."
        ),
        text=(
            "Treat the natural scene's physically upright orientation as the "
            "identity. Estimate how many degrees COUNTERCLOCKWISE the current "
            "image has been displaced from upright, in [0,360). Do not return a "
            "multiple-choice slot and do not convert it into a clockwise repair. "
            "Return {\"observed_ccw_degrees\":number,\"confidence\":number," 
            "\"visual_cues\":[string]}."
        ),
        image_url=image_url,
        maximum_tokens=maximum_tokens,
    )
    angle = float(payload["observed_ccw_degrees"]) % 360.0
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("rotation confidence is outside [0,1]")
    return {
        "role": "TARGET_NATIVE_NEURAL_ORIENTATION_BINDING",
        "answer_choices_seen": False,
        "gold_seen": False,
        "observed_ccw_degrees": angle,
        "confidence": confidence,
        "visual_cues": [str(value) for value in payload.get("visual_cues", [])][:8],
    }, usage


def _baseline(
    client: OpenAI, *, model: str, prompt: str, image_url: str,
    maximum_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Solve the visual rotation multiple-choice task. Return strict JSON "
            "{\"answer\":\"A-F\",\"reason\":\"brief\"}."
        ),
        text=prompt,
        image_url=image_url,
        maximum_tokens=maximum_tokens,
    )
    answer = str(payload.get("answer") or "").strip().upper()[:1]
    if answer not in "ABCDEF":
        raise ValueError("rotation baseline did not return A-F")
    return answer, usage


def _collect(
    sample_id: str, *, row: Mapping[str, Any], dataset_root: Path,
    config: Mapping[str, Any], api_key: str, contract_sha256: str,
) -> dict[str, Any]:
    if row.get("task") != "rotation_game" or row.get("image_2"):
        raise ValueError("rotation runner accepts single-image rotation_game only")
    image_path = dataset_root / str(row["image_1"])
    with Image.open(image_path) as handle:
        image_url = _image_data(
            handle,
            max_side=int(config["media"]["max_side"]),
            quality=int(config["media"]["jpeg_quality"]),
        )
    model_config = config["model"]
    client = OpenAI(
        api_key=api_key,
        base_url=str(model_config["base_url"]),
        timeout=float(model_config["timeout_seconds"]),
        max_retries=int(model_config["max_retries"]),
    )
    binding, binding_usage = _bind(
        client,
        model=str(model_config["id"]),
        image_url=image_url,
        maximum_tokens=int(model_config["maximum_output_tokens"]),
    )
    baseline, baseline_usage = _baseline(
        client,
        model=str(model_config["id"]),
        prompt=str(row["prompt"]),
        image_url=image_url,
        maximum_tokens=int(model_config["maximum_output_tokens"]),
    )
    body = {
        "schema_version": "tir-rotation-group-receipt-v1",
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample_id,
        "family": "rotation_game",
        "image_sha256": _sha256(image_path),
        "prompt_sha256": stable_hash(str(row["prompt"])),
        "options": parse_rotation_options(str(row["prompt"])),
        "neural_binding": binding,
        "binding_usage": binding_usage,
        "baseline_answer": baseline,
        "baseline_usage": baseline_usage,
    }
    body["gold_answer_evaluator_only"] = str(row["answer"])
    return body | {"receipt_sha256": stable_hash(body)}


def _paired(rows: list[dict[str, Any]], right: str) -> dict[str, Any]:
    left = "authentic_tetris_inverse"
    wins = sum(row["correct"][left] and not row["correct"][right] for row in rows)
    losses = sum(row["correct"][right] and not row["correct"][left] for row in rows)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "net_wins": wins - losses,
        "exact_two_sided_p": exact_sign_p(wins, losses),
    }


def _evaluate(
    receipts: list[dict[str, Any]], *, split: str, config: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(receipts, key=lambda row: str(row["sample_id"]))
    donor_order = sorted(
        ordered,
        key=lambda row: hashlib.sha256(
            f"tir-rotation-shuffled-v1\0{row['sample_id']}".encode()
        ).hexdigest(),
    )
    donor = {
        str(row["sample_id"]): donor_order[(index + 1) % len(donor_order)]
        for index, row in enumerate(donor_order)
    }
    rows = []
    for receipt in ordered:
        sample_id = str(receipt["sample_id"])
        angle = float(receipt["neural_binding"]["observed_ccw_degrees"])
        options = receipt["options"]
        answers = {"raw_target_only": str(receipt["baseline_answer"])}
        for condition in CONDITIONS[1:]:
            answers[condition] = select_rotation_action(
                options,
                angle,
                condition=condition,
                donor_ccw_degrees=float(
                    donor[sample_id]["neural_binding"]["observed_ccw_degrees"]
                ) if condition == "shuffled_binding_control" else None,
            )
        gold = str(receipt["gold_answer_evaluator_only"])
        rows.append({
            "sample_id": sample_id,
            "answers": answers,
            "correct": {name: value == gold for name, value in answers.items()},
            "gold_answer_evaluator_only": gold,
            "observed_ccw_degrees": angle,
            "binding_confidence": float(receipt["neural_binding"]["confidence"]),
            "shuffled_donor": str(donor[sample_id]["sample_id"]),
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
    paired = {condition: _paired(rows, condition) for condition in CONDITIONS if condition != "authentic_tetris_inverse"}
    destructive = (
        "no_inverse_control", "shuffled_binding_control", "half_turn_marginal_control",
    )
    expected = len(config["splits"][split])
    primary = paired["raw_target_only"]
    gates = {
        "exact_frozen_task_count": len(rows) == expected,
        "source_gate_passed": all(source_artifact["gates"].values()),
        "all_bindings_valid": all(0 <= row["binding_confidence"] <= 1 for row in rows),
        "nontrivial_action_changes": metrics["authentic_tetris_inverse"]["changes_vs_raw"] >= max(2, expected // 6),
        "alpha_rename_invariance": all(
            row["answers"]["authentic_tetris_inverse"] == row["answers"]["alpha_renamed_authentic"]
            for row in rows
        ),
        "target_isomorphic_equivalence": all(
            row["answers"]["authentic_tetris_inverse"] == row["answers"]["target_written_isomorphic"]
            for row in rows
        ),
        "authentic_strictly_above_raw": metrics["authentic_tetris_inverse"]["correct"] > metrics["raw_target_only"]["correct"],
        "paired_wins_above_losses": primary["wins"] > primary["losses"],
        "authentic_strictly_above_destructive_controls": all(
            metrics["authentic_tetris_inverse"]["correct"] > metrics[name]["correct"]
            for name in destructive
        ),
    }
    passed = all(gates.values())
    stage = {
        "consumed_development": "CONSUMED_DEVELOPMENT",
        "qualification": "FRESH_QUALIFICATION",
        "heldout": "FRESH_FORMAL",
    }[split]
    body: dict[str, Any] = {
        "schema_version": "tir-rotation-group-report-v1",
        "status": f"{stage}_ROTATION_TRANSFER_GATE_{'PASSED' if passed else 'FAILED'}",
        "split": split,
        "metrics": metrics,
        "paired_authentic": paired,
        "gates": gates,
        "rows": rows,
        "claim_boundary": config["claim_boundary"],
        "source_artifact_sha256": source_artifact["artifact_sha256"],
    }
    body["report_sha256"] = stable_hash(body)
    return body


def _self_hash(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    claimed = body.pop("report_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("authority report self-hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("consumed_development", "qualification", "heldout"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_TARGET_ROTATION_CALLS":
        raise SystemExit("rotation config is not frozen")
    for relative, expected in config["integrity"]["file_sha256"].items():
        if _sha256(REPO / relative) != expected:
            raise SystemExit(f"rotation frozen dependency changed: {relative}")
    if args.split in {"qualification", "heldout"}:
        development = json.loads((REPO / config["authority"]["development_report"]).read_text())
        _self_hash(development)
        if development["status"] != "CONSUMED_DEVELOPMENT_ROTATION_TRANSFER_GATE_PASSED":
            raise SystemExit("rotation development did not authorize fresh calls")
    if args.split == "heldout":
        qualification = json.loads((REPO / config["authority"]["qualification_report"]).read_text())
        _self_hash(qualification)
        if qualification["status"] != "FRESH_QUALIFICATION_ROTATION_TRANSFER_GATE_PASSED":
            raise SystemExit("rotation qualification did not authorize formal calls")
    dataset_path = args.dataset_root / "TIR-Bench.json"
    if _sha256(dataset_path) != config["dataset"]["file_sha256"]:
        raise SystemExit("TIR dataset hash mismatch")
    index = {str(row["id"]): row for row in json.loads(dataset_path.read_text())}
    sample_ids = [str(value) for value in config["splits"][args.split]]
    source_path = REPO / config["source"]["artifact"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if _sha256(source_path) != config["source"]["artifact_file_sha256"]:
        raise SystemExit("Tetris source artifact file changed")
    if source["status"] != "SOURCE_ROTATION_GROUP_CONFIRMED":
        raise SystemExit("Tetris rotation source gate failed")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    contract_sha256 = stable_hash({
        "config_sha256": _sha256(args.config),
        "split": args.split,
        "source_artifact_sha256": source["artifact_sha256"],
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir = args.output_dir / f"{args.split}_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, dict[str, Any]] = {}
    pending = []
    for sample_id in sample_ids:
        path = receipts_dir / f"{sample_id}.json"
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("collection_contract_sha256") != contract_sha256:
                raise SystemExit(f"receipt contract changed: {sample_id}")
            receipts[sample_id] = value
        else:
            pending.append(sample_id)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _collect,
                sample_id,
                row=index[sample_id],
                dataset_root=args.dataset_root,
                config=config,
                api_key=str(key),
                contract_sha256=contract_sha256,
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            value = future.result()
            (receipts_dir / f"{sample_id}.json").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )
            receipts[sample_id] = value
            print(json.dumps({
                "sample_id": sample_id,
                "baseline": value["baseline_answer"],
                "observed_ccw": value["neural_binding"]["observed_ccw_degrees"],
            }), flush=True)
    ordered = [receipts[sample_id] for sample_id in sample_ids]
    report = _evaluate(ordered, split=args.split, config=config, source_artifact=source)
    output = args.output_dir / f"{args.split}_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "metrics": report["metrics"],
        "paired_primary": report["paired_authentic"]["raw_target_only"],
        "gates": report["gates"], "output": str(output),
    }, indent=2, sort_keys=True))
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
