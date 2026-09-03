#!/usr/bin/env python3
"""Add an outcome-blind baseline-support verifier to consumed TIR receipts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_collector():
    path = REPO / "scripts/collect_phase3_tir_visual_search_v3.py"
    spec = importlib.util.spec_from_file_location("phase3_tir_v3_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load TIR V3 collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECTOR = _load_collector()


def _validate_receipt(row: dict) -> None:
    body = dict(row)
    claimed = str(body.pop("receipt_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"receipt hash mismatch: {row.get('sample_id')}")
    if row.get("formal_outcome_exposed_to_neural_calls") is not False:
        raise ValueError("receipt lacks outcome-isolation attestation")
    if row.get("source_program_or_identity_exposed_to_neural_calls") is not False:
        raise ValueError("receipt exposed source information")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stages", nargs="+", choices=(
            "development_train", "development_validation",
            "development_holdout",
        ), required=True,
    )
    parser.add_argument(
        "--verifier-model", default="google/gemini-3.7-flash",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    config_body = dict(config)
    claimed = str(config_body.pop("config_sha256", ""))
    if not claimed or stable_hash(config_body) != claimed:
        raise SystemExit("TIR manifest hash mismatch")
    if config.get("status") != "FROZEN_CONSUMED_DEVELOPMENT_DIAGNOSTIC_ONLY":
        raise SystemExit("augmentation is allowed only for consumed development")
    dataset_file = args.dataset_root / "TIR-Bench.json"
    if file_sha256(dataset_file) != config["dataset"]["sha256"]:
        raise SystemExit("TIR dataset drift")
    dataset = json.loads(dataset_file.read_text())
    index = {str(row["id"]): row for row in dataset}
    api_key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    runtime_config = json.loads(json.dumps(config))
    runtime_config["media"]["overview_max_side"] = COLLECTOR.OVERVIEW_MAX_SIDE
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stage in args.stages:
        input_path = args.input_dir / f"{stage}_receipts.json"
        output_path = args.output_dir / f"{stage}_receipts.json"
        source_rows = json.loads(input_path.read_text())
        expected = list(map(str, config["splits"][stage]))
        if [str(row["sample_id"]) for row in source_rows] != expected:
            raise SystemExit(f"{stage} input receipt order does not match manifest")
        for row in source_rows:
            _validate_receipt(row)
        existing = {}
        if output_path.is_file():
            for row in json.loads(output_path.read_text()):
                _validate_receipt(row)
                existing[str(row["sample_id"])] = row
        pending = [row for row in source_rows if str(row["sample_id"]) not in existing]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for row in pending:
                sample_id = str(row["sample_id"])
                target_input = {
                    key: value for key, value in index[sample_id].items()
                    if key != "answer"
                }
                future = executor.submit(
                    COLLECTOR.augment_with_baseline_verifier,
                    row, target_input=target_input,
                    dataset_root=args.dataset_root, config=runtime_config,
                    api_key=str(api_key), verifier_model=args.verifier_model,
                )
                futures[future] = sample_id
            for future in as_completed(futures):
                sample_id = futures[future]
                existing[sample_id] = future.result()
                ordered = [existing[value] for value in expected if value in existing]
                output_path.write_text(
                    json.dumps(ordered, ensure_ascii=False, indent=2) + "\n"
                )
                print(json.dumps({
                    "stage": stage, "completed": sample_id,
                    "progress": f"{len(ordered)}/{len(expected)}",
                }), flush=True)
        missing = [value for value in expected if value not in existing]
        if missing:
            raise SystemExit(f"incomplete verifier augmentation: {missing}")
        print(json.dumps({
            "stage": stage,
            "receipts": len(expected),
            "verifier_model": args.verifier_model,
            "output_file_sha256": file_sha256(output_path),
            "qualification_or_formal_id_opened": False,
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
