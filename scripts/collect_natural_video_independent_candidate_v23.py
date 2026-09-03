#!/usr/bin/env python3
"""Collect isolated option proofs and execute the frozen symbolic verifier."""

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
from motif_transfer.independent_video_verifier import (  # noqa: E402
    execute_candidate_program,
    execute_source_guard,
    parse_independent_candidate,
)
from motif_transfer.natural_video_recovery import PROOF_KINDS  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _candidate_call(
    client: OpenAI,
    *,
    question: str,
    candidate_text: str,
    panels: Sequence[bytes],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    prompt = (
        "Independently verify one candidate answer claim for a video question. "
        "No alternative choices and no earlier model answer are available. Judge "
        "only whether this claim is established by the visible chronological "
        "frames. Execute exactly five target-native checks in this order: "
        + ", ".join(PROOF_KINDS)
        + ". SUPPORTED requires visible positive evidence, REFUTED requires visible "
        "contradiction, and UNKNOWN is mandatory when the needed fact is not "
        "observable. CAUSAL_LINK requires temporal evidence beyond co-occurrence. "
        "Return a calibrated probability that this candidate correctly answers "
        "the question. No annotations, official programs, graphs, gold, option "
        "position, or competing candidate is available.\n"
        f"Question: {question}\nCandidate answer claim: {candidate_text}"
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
                "Return JSON only: {support_probability:number,sensor_reliability:"
                "number,proof_steps:[{kind:string,status:SUPPORTED|REFUTED|UNKNOWN,"
                "confidence:number,visible_fact:string}],unresolved_uncertainties:"
                "[string],reason:string}. proof_steps must preserve "
                + ",".join(PROOF_KINDS) + "."
            ),
            content=attempt,
            max_tokens=int(model["max_tokens"]),
        )
        try:
            return parse_independent_candidate(payload), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("independent candidate schema retries exhausted: " + last_error)


def _collect_one(
    row: Mapping[str, Any],
    generic_row: Mapping[str, Any],
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
    _primary_panel, panels, metadata = paired._panels(sample, config)
    panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
    if panel_hashes != list(row["proof_panel_sha256"]):
        raise ValueError("independent verifier did not reconstruct exact V19 panels")
    candidates = []
    raw_receipts = {}
    usage = {}
    for slot in sample.answer_slots:
        receipt, raw, candidate_usage = _candidate_call(
            client,
            question=str(sample.question),
            candidate_text=str(sample.options[slot]),
            panels=panels,
            config=config,
        )
        candidates.append({"slot": str(slot), **receipt})
        raw_receipts[str(slot)] = raw
        usage[str(slot)] = candidate_usage
    family = str(row["family"])
    primary_answer = str(row["primary"]["answer"])
    executor = execute_candidate_program(candidates, family=family)
    source = execute_source_guard(primary_answer, candidates, family=family)
    binding = execute_source_guard(
        primary_answer, candidates, family=family, shuffled_binding=True,
    )
    topology = execute_source_guard(
        primary_answer, candidates, family=family, shuffled_topology=True,
    )
    gold = str(row["gold_answer"])
    return {
        "schema_version": 23,
        "benchmark": str(row["benchmark"]),
        "split": "consumed_v19_independent_candidate_pilot",
        "sample_id": str(row["sample_id"]),
        "video_id": str(row["video_id"]),
        "family": family,
        "gold_answer": gold,
        "primary_answer": primary_answer,
        "primary_correct": bool(row["primary_correct"]),
        "joint_typed_proof_answer": str(row["proof"]["answer"]),
        "joint_typed_proof_correct": bool(row["proof_correct"]),
        "generic_direct_answer": str(generic_row["generic_direct"]["answer"]),
        "generic_direct_correct": bool(generic_row["generic_direct_correct"]),
        "independent_candidates": candidates,
        "independent_executor": executor,
        "independent_executor_correct": str(executor["answer"]) == gold,
        "source_authentic": source,
        "source_authentic_correct": str(source["answer"]) == gold,
        "shuffled_binding_control": binding,
        "shuffled_binding_control_correct": str(binding["answer"]) == gold,
        "shuffled_topology_control": topology,
        "shuffled_topology_control_correct": str(topology["answer"]) == gold,
        "candidate_raw": raw_receipts,
        "usage": usage,
        "video_metadata": metadata,
        "proof_panel_sha256": panel_hashes,
        "input_v19_row_sha256": _content_hash(row),
        "input_generic_row_sha256": _content_hash(generic_row),
        "collection_contract_sha256": contract_sha256,
        "each_neural_call_saw_exactly_one_candidate": True,
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
        "pilot_manifest_sha256": Path(config["pilot_manifest"]),
        "input_v19_receipts_sha256": Path(config["input_v19_receipts"]),
        "input_generic_receipts_sha256": Path(config["input_generic_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "executor_module_sha256": REPO / "src/motif_transfer/independent_video_verifier.py",
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "transport_module_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V23 independent-candidate lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    manifest = json.loads(Path(config["pilot_manifest"]).read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_V23_INDEPENDENT_CANDIDATE_PILOT":
        raise ValueError("V23 pilot manifest is not sealed")
    ordered_pairs = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("star", "nextqa")
        for row in manifest["benchmarks"][benchmark]
    ]
    v19_rows = json.loads(Path(config["input_v19_receipts"]).read_text(encoding="utf-8"))
    generic_rows = json.loads(Path(config["input_generic_receipts"]).read_text(encoding="utf-8"))
    v19_by_pair = {(str(row["benchmark"]), str(row["sample_id"])): row for row in v19_rows}
    generic_by_pair = {
        (str(row["benchmark"]), str(row["sample_id"])): row for row in generic_rows
    }
    if any(pair not in v19_by_pair or pair not in generic_by_pair for pair in ordered_pairs):
        raise ValueError("V23 pilot identity is missing a matched baseline")
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
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "ordered_pairs": ordered_pairs,
    })
    key_values = runpy.run_path(str(args.keys))
    api_key = key_values.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured OpenRouter key is missing")
    existing = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V23 pilot contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
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
                v19_by_pair[pair],
                generic_by_pair[pair],
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
                    "failed": list(pair), "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": list(pair),
                "progress": f"{len(existing)}/{len(ordered_pairs)}",
            }), flush=True)
    missing = [pair for pair in ordered_pairs if pair not in existing]
    if missing:
        raise SystemExit(f"incomplete V23 pilot; rerun: {missing}")
    rows = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": "NATURAL_VIDEO_V23_INDEPENDENT_CANDIDATE_PILOT_COLLECTED",
        "rows": len(rows),
        "primary_correct": sum(row["primary_correct"] for row in rows),
        "generic_direct_correct": sum(row["generic_direct_correct"] for row in rows),
        "joint_typed_proof_correct": sum(row["joint_typed_proof_correct"] for row in rows),
        "independent_executor_correct": sum(row["independent_executor_correct"] for row in rows),
        "source_authentic_correct": sum(row["source_authentic_correct"] for row in rows),
        "binding_control_correct": sum(row["shuffled_binding_control_correct"] for row in rows),
        "topology_control_correct": sum(row["shuffled_topology_control_correct"] for row in rows),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
