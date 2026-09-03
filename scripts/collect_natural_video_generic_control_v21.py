#!/usr/bin/env python3
"""Collect a model/frame-matched generic-direct control for V19 typed proofs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_recovery_v15 as paired  # noqa: E402
import collect_natural_video_v19_formal as v19  # noqa: E402
from motif_transfer.natural_video_recovery import parse_primary_receipt  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generic_call(
    client: OpenAI,
    *,
    sample: Any,
    panels: Sequence[bytes],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    slots = tuple(sample.answer_slots)
    prompt = (
        "Answer this video question directly from the dense chronological frames. "
        "Use ordinary end-to-end visual reasoning. Do not execute a candidate-"
        "factorized proof, do not emit typed proof steps, and do not assume any "
        "earlier answer. Return probability mass for every option plus concise "
        "observed evidence and unresolved uncertainty. No annotations, official "
        "programs, graphs, or gold are available.\n" + sample.format_question()
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Dense chronological evidence panel {index + 1}/{len(panels)}:"},
            paired.media_helpers._image_content(panel),
        ])
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": "Schema error: " + last_error})
        payload, usage = v19._provider_json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {answer:string, probabilities:{slot:number},"
                "observed_evidence:[string],unresolved_uncertainties:[string],"
                "reason:string}. answer must be the unique probability argmax."
            ),
            content=attempt,
            max_tokens=int(model["max_tokens"]),
        )
        try:
            return parse_primary_receipt(payload, slots), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("generic-direct receipt schema retries exhausted: " + last_error)


def _collect_one(
    row: Mapping[str, Any],
    sample: Any,
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
    panel_config = {**config, "media": config["media"]}
    _primary_panel, proof_panels, metadata = paired._panels(sample, panel_config)
    panel_hashes = [hashlib.sha256(value).hexdigest() for value in proof_panels]
    if panel_hashes != list(row["proof_panel_sha256"]):
        raise ValueError("generic control did not reconstruct the exact V19 proof panels")
    generic, raw, usage = _generic_call(
        client, sample=sample, panels=proof_panels, config=config,
    )
    gold = str(row["gold_answer"])
    return {
        "schema_version": 21,
        "benchmark": str(row["benchmark"]),
        "split": "consumed_v19_development_control",
        "sample_id": str(row["sample_id"]),
        "video_id": str(row["video_id"]),
        "family": str(row["family"]),
        "gold_answer": gold,
        "primary_correct": bool(row["primary_correct"]),
        "typed_proof_correct": bool(row["proof_correct"]),
        "typed_proof_answer": str(row["proof"]["answer"]),
        "generic_direct": generic,
        "generic_direct_correct": str(generic["answer"]) == gold,
        "generic_minus_typed": (
            int(str(generic["answer"]) == gold) - int(bool(row["proof_correct"]))
        ),
        "generic_raw": raw,
        "usage": usage,
        "video_metadata": metadata,
        "proof_panel_sha256": panel_hashes,
        "input_v19_row_sha256": _content_hash(row),
        "collection_contract_sha256": contract_sha256,
        "runtime_saw_gold_or_official_structure": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lineage_paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "input_v19_receipts_sha256": Path(config["input_v19_receipts"]),
        "v19_config_sha256": Path(config["v19_config"]),
        "collector_sha256": Path(__file__).resolve(),
        "v19_collector_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "contract_module_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V21 generic-control lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    rows = json.loads(Path(config["input_v19_receipts"]).read_text(encoding="utf-8"))
    if len(rows) != int(config["expected_rows"]):
        raise ValueError("V21 generic control requires the complete V19 receipt set")
    ordered_pairs = [(str(row["benchmark"]), str(row["sample_id"])) for row in rows]
    if len(set(ordered_pairs)) != len(ordered_pairs):
        raise ValueError("duplicate V19 identities")
    samples = {
        benchmark: paired._load_samples(
            benchmark,
            [sample_id for name, sample_id in ordered_pairs if name == benchmark],
            config,
        )
        for benchmark in ("star", "nextqa")
    }
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "input_v19_receipts_sha256": _sha256(Path(config["input_v19_receipts"])),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "ordered_pairs": ordered_pairs,
    })
    key_values = runpy.run_path(str(args.keys))
    api_key = key_values.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured OpenRouter key is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V21 generic-control contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    row_by_pair = {pair: row for pair, row in zip(ordered_pairs, rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[pair] for pair in ordered_pairs if pair in existing],
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                row_by_pair[pair],
                samples[pair[0]][pair[1]],
                config=config,
                api_key=str(api_key),
                contract_sha256=contract_sha256,
            ): pair
            for pair in pending
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                existing[pair] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": list(pair),
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": list(pair),
                "progress": f"{len(existing)}/{len(ordered_pairs)}",
            }), flush=True)
    missing = [pair for pair in ordered_pairs if pair not in existing]
    if missing:
        raise SystemExit(f"incomplete V21 generic control; rerun: {missing}")
    output = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": "NATURAL_VIDEO_V21_MATCHED_GENERIC_CONTROL_COLLECTED",
        "rows": len(output),
        "typed_proof_correct": sum(row["typed_proof_correct"] for row in output),
        "generic_direct_correct": sum(row["generic_direct_correct"] for row in output),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
