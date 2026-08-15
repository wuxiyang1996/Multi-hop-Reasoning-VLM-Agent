#!/usr/bin/env python3
"""Collect target-native TIR bindings and execute six direct source cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    file_sha256,
    make_cell_execution_receipt,
    read_object,
    validate_manifest,
    validate_self_hash,
)
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)
from motif_transfer.tir_maze_topology import validate_neural_binding  # noqa: E402
from motif_transfer.tir_search_automaton_v16 import (  # noqa: E402
    AUTHENTIC,
    CONDITIONS,
    execute_tir_maze_search,
)
from scripts.run_tir_maze_topology_v2 import (  # noqa: E402
    _baseline,
    _bind,
    _image_data,
)


def _cell(manifest: dict, game: str) -> dict:
    rows = [
        row for row in manifest["cells"]
        if row["source_game"] == game and row["target_domain"] == "tirbench"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one TIRBench cell for {game}")
    return dict(rows[0])


def _target_grounding(
    *,
    output_dir: Path,
    row: dict,
    image: Image.Image,
    image_path: Path,
    model_config: dict,
    api_key: str,
    contract_sha256: str,
) -> dict:
    path = output_dir / "target_grounding.json"
    if path.is_file():
        existing = read_object(path)
        validate_self_hash(existing, "target_grounding_sha256")
        if existing.get("collection_contract_sha256") != contract_sha256:
            raise RuntimeError("TIR target-grounding resume contract changed")
        return existing
    image_url = _image_data(
        image,
        max_side=int(model_config["max_side"]),
        quality=int(model_config["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key,
        base_url=str(model_config["base_url"]),
        timeout=180,
        max_retries=2,
    )
    binding, binding_usage = _bind(
        client,
        model=str(model_config["id"]),
        prompt=str(row["prompt"]),
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
    validate_neural_binding(binding)
    body = {
        "schema_version": "phase1-direct-tir-target-grounding-v1",
        "collection_contract_sha256": contract_sha256,
        "sample_id": str(row["id"]),
        "image_path": str(image_path),
        "image_file_sha256": file_sha256(image_path),
        "prompt_sha256": stable_hash(str(row["prompt"])),
        "answer_or_gold_seen_by_binder": False,
        "neural_binding": binding,
        "neural_binding_valid": True,
        "binding_usage": binding_usage,
        "baseline_answer": baseline,
        "baseline_usage": baseline_usage,
        "gold_answer_present": False,
    }
    payload = body | {"target_grounding_sha256": stable_hash(body)}
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _run_cell(
    *, manifest: dict, game: str, output_root: Path, api_key: str
) -> dict:
    cell = _cell(manifest, game)
    output_dir = output_root / game
    output_path = output_dir / "report.json"
    if output_path.is_file():
        existing = read_object(output_path)
        receipt = existing.get("cell_execution_receipt")
        if receipt:
            validate_self_hash(receipt, "cell_receipt_sha256")
            if receipt.get("manifest_sha256") == manifest["manifest_sha256"]:
                return existing
        raise RuntimeError(f"refusing incompatible TIR resume: {output_path}")

    source_path = REPO / str(cell["source_artifact"])
    source = SourceSearchAutomaton(
        read_object(source_path),
        expected_sha256=str(cell["source_artifact_sha256"]),
    )
    target = dict(manifest["targets"]["tirbench"])
    dataset_path = Path(str(target["dataset"]))
    if file_sha256(dataset_path) != str(target["dataset_file_sha256"]):
        raise RuntimeError("TIR dataset changed after freeze")
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    matches = [row for row in rows if str(row["id"]) == str(cell["target_task_id"])]
    if len(matches) != 1:
        raise RuntimeError("frozen TIR sample identity is not unique")
    row = dict(matches[0])
    if row.get("task") != "maze" or row.get("image_2"):
        raise RuntimeError("frozen TIR target is not a single-image maze")
    image_path = dataset_path.parent / str(row["image_1"])
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    initial_state_hash = stable_hash({
        "sample_id": str(row["id"]),
        "prompt_sha256": stable_hash(str(row["prompt"])),
        "image_file_sha256": file_sha256(image_path),
    })
    contract = stable_hash({
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "source_artifact_sha256": source.artifact_sha256,
        "runner_file_sha256": file_sha256(Path(__file__)),
        "target_model": target["model"],
        "initial_state_sha256": initial_state_hash,
    })
    runtime_error = None
    conditions = {}
    grounding = None
    try:
        grounding = _target_grounding(
            output_dir=output_dir,
            row=row,
            image=image,
            image_path=image_path,
            model_config=dict(target["model"]),
            api_key=api_key,
            contract_sha256=contract,
        )
        for condition in CONDITIONS:
            conditions[condition] = execute_tir_maze_search(
                image=image,
                prompt=str(row["prompt"]),
                sample_id=str(row["id"]),
                baseline_answer=str(grounding["baseline_answer"]),
                neural_binding=grounding["neural_binding"],
                source=source,
                condition=condition,
            )
            print(json.dumps({
                "cell_id": cell["cell_id"],
                "condition": condition,
                "selected_answer": conditions[condition]["selected_answer"],
                "source_decisions": len(
                    conditions[condition].get("source_decisions") or ()
                ),
            }), flush=True)
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"

    authentic_trace = list(
        conditions.get(AUTHENTIC, {}).get("source_decisions") or ()
    )
    receipt = make_cell_execution_receipt(
        manifest_sha256=str(manifest["manifest_sha256"]),
        cell=cell,
        source_artifact_sha256=source.artifact_sha256,
        conditions_executed=[condition for condition in CONDITIONS if condition in conditions],
        expected_conditions=CONDITIONS,
        target_initial_state_hashes=[initial_state_hash] * len(conditions),
        authentic_source_decisions=authentic_trace,
        target_native_grounding_used=bool(
            grounding and grounding.get("neural_binding_valid")
        ),
        target_reset_or_sample_open_count=1,
        outcome_was_reused=False,
        runtime_error=runtime_error,
    )
    # Evaluator-only attachment happens after every policy condition returns.
    gold = str(row["answer"])
    condition_correct = {
        name: str(result["selected_answer"]) == gold
        for name, result in conditions.items()
    }
    body = {
        "schema_version": "phase1-direct-tirbench-cell-v1",
        "status": receipt["status"],
        "claim_boundary": manifest["claim_boundary"],
        "cell": cell,
        "collection_contract_sha256": contract,
        "source_artifact_file_sha256": file_sha256(source_path),
        "target_grounding_file_sha256": (
            file_sha256(output_dir / "target_grounding.json")
            if (output_dir / "target_grounding.json").is_file() else None
        ),
        "conditions": conditions,
        "gold_answer_evaluator_only": gold,
        "condition_correct_evaluator_only": condition_correct,
        "cell_execution_receipt": receipt,
    }
    report = body | {"report_sha256": stable_hash(body)}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase1_direct_prospective_v1/manifest.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--source-game", choices=SOURCE_GAMES, action="append")
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "runs/phase1_direct_prospective_v1/tirbench",
    )
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_manifest(manifest, repo=REPO)
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    os.environ["PHASE1_DIRECT_TIR_OPENROUTER_KEY"] = str(key)
    games = tuple(args.source_game or SOURCE_GAMES)
    reports = [
        _run_cell(
            manifest=manifest, game=game, output_root=args.output_root,
            api_key=str(key),
        )
        for game in games
    ]
    passed = sum(
        report["cell_execution_receipt"]["status"]
        == "DIRECT_PROSPECTIVE_CELL_PASSED"
        for report in reports
    )
    print(json.dumps({
        "domain": "tirbench", "passed": passed, "attempted": len(reports)
    }, indent=2))
    return 0 if passed == len(reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
