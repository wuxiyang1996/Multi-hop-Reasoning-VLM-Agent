#!/usr/bin/env python3
"""Collect independent cross-model proof receipts on consumed V15 development."""

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


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_focused_verify_v16 as focused  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _RuntimeSample:
    def __init__(self, row: Mapping[str, Any]) -> None:
        public = row["sample_public"]
        self.question = str(public["question"])
        self.options = dict(public["options"])
        self.answer_slots = tuple(public["answer_slots"])

    def format_question(self) -> str:
        options = "\n".join(f"{slot}. {self.options[slot]}" for slot in self.answer_slots)
        return f"{self.question}\nOptions:\n{options}\nReturn one option letter."


def _step(candidate: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    return next(step for step in candidate["proof_steps"] if step["kind"] == kind)


def _collect_one(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    panels = focused._proof_panels(row, config)
    proof, raw, usage = paired._proof_call(
        client, sample=_RuntimeSample(row), panels=panels, config=config,
    )
    primary_answer = str(row["primary"]["answer"])
    proof_answer = str(proof["answer"])
    by_slot = {str(candidate["slot"]): candidate for candidate in proof["candidates"]}
    primary_entailment = _step(by_slot[primary_answer], "ANSWER_ENTAILMENT")
    proof_entailment = _step(by_slot[proof_answer], "ANSWER_ENTAILMENT")
    authentic_recover = (
        proof_answer != primary_answer
        and primary_entailment["status"] == "REFUTED"
        and proof_entailment["status"] == "SUPPORTED"
    )
    authentic_answer = proof_answer if authentic_recover else primary_answer
    gold = str(row["gold_answer"])
    return {
        "schema_version": 17,
        "benchmark": row["benchmark"],
        "split": "development",
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "family": row["family"],
        "gold_answer": gold,
        "primary": row["primary"],
        "primary_correct": bool(row["primary_correct"]),
        "cross_model_proof": proof,
        "proof_correct": proof_answer == gold,
        "proof_uplift": int(proof_answer == gold) - int(bool(row["primary_correct"])),
        "primary_answer_entailment": primary_entailment,
        "proof_answer_entailment": proof_entailment,
        "authentic_recover": authentic_recover,
        "authentic_answer": authentic_answer,
        "authentic_correct": authentic_answer == gold,
        "authentic_uplift": int(authentic_answer == gold) - int(bool(row["primary_correct"])),
        "cross_model_proof_raw": raw,
        "usage": usage,
        "proof_panel_sha256": [hashlib.sha256(value).hexdigest() for value in panels],
        "input_v15_row_sha256": _content_hash(row),
        "collection_contract_sha256": contract_sha256,
        "runtime_saw_gold_or_official_structure": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "input_receipts_sha256": Path(config["input_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "focused_collector_sha256": REPO / "scripts/collect_natural_video_focused_verify_v16.py",
        "contract_module_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
    }
    for key, path in paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V17 frozen lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    input_rows = json.loads(Path(config["input_receipts"]).read_text(encoding="utf-8"))
    ordered_keys = [(str(row["benchmark"]), str(row["sample_id"])) for row in input_rows]
    if len(ordered_keys) != len(set(ordered_keys)):
        raise ValueError("V17 input rows contain duplicates")
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "input_receipts_sha256": _sha256(Path(config["input_receipts"])),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "ordered_keys": ordered_keys,
    })
    keys = runpy.run_path(str(args.keys))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured cross-model API key is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached cross-model receipt contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    row_by_key = {key: row for key, row in zip(ordered_keys, input_rows)}
    pending = [key for key in ordered_keys if key not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[key] for key in ordered_keys if key in existing],
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one, row_by_key[key], config=config,
                api_key=str(api_key), contract_sha256=contract_sha256,
            ): key for key in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                existing[key] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": list(key), "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": list(key), "progress": f"{len(existing)}/{len(ordered_keys)}",
            }), flush=True)
    missing = [key for key in ordered_keys if key not in existing]
    if missing:
        raise SystemExit(f"incomplete V17 cross-model proof; rerun: {missing}")
    rows = [existing[key] for key in ordered_keys]
    print(json.dumps({
        "status": "NATURAL_VIDEO_V17_CROSS_MODEL_DEVELOPMENT_COLLECTED",
        "samples": len(rows),
        "primary_correct": sum(row["primary_correct"] for row in rows),
        "proof_correct": sum(row["proof_correct"] for row in rows),
        "authentic_correct": sum(row["authentic_correct"] for row in rows),
        "authentic_recoveries": sum(row["authentic_recover"] for row in rows),
        "proof_uplift_counts": {
            str(value): sum(row["proof_uplift"] == value for row in rows)
            for value in (-1, 0, 1)
        },
        "authentic_uplift_counts": {
            str(value): sum(row["authentic_uplift"] == value for row in rows)
            for value in (-1, 0, 1)
        },
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
