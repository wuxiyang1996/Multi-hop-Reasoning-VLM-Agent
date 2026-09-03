#!/usr/bin/env python3
"""Prospective STAR/NExT-QA evaluation of source-compatible VERIFY transfer."""

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

import collect_natural_video_active_recovery_v18 as v18  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402
from motif_transfer.natural_video_active_recovery import source_compatible  # noqa: E402
from motif_transfer.natural_video_recovery import build_features  # noqa: E402
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


def _provider_json_call(
    client: OpenAI,
    *,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve V17 semantics; use a transport-only Gemini fallback if needed."""

    failures = []
    for attempt in range(1, 4):
        is_gemini = "gemini" in model.casefold()
        fallback = bool(is_gemini and attempt > 1)
        kwargs: dict[str, Any] = {}
        if fallback:
            kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=max(16_000, max_tokens) if fallback else max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        try:
            payload, wrapper = v18._decode_json_object(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            usage = response.usage
            failures.append(
                f"attempt {attempt}: {exc}; finish={choice.finish_reason}; "
                f"chars={len(raw)}; completion_tokens="
                f"{int(usage.completion_tokens if usage else 0)}"
            )
            continue
        usage = response.usage
        return payload, {
            "model": str(response.model),
            "finish_reason": str(choice.finish_reason),
            "prompt_tokens": int(usage.prompt_tokens if usage else 0),
            "completion_tokens": int(usage.completion_tokens if usage else 0),
            "cost": float(getattr(usage, "cost", 0.0) or 0.0),
            "response_sha256": _content_hash(payload),
            "transport_attempts": attempt,
            "transport_fallback": fallback,
            "transport_wrapper_removed": wrapper,
            "prior_transport_errors": failures,
        }
    raise ValueError("provider JSON transport retries exhausted: " + " | ".join(failures))


def _step(candidate: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    return next(step for step in candidate["proof_steps"] if step["kind"] == kind)


def _collect_one(
    sample: Any,
    *,
    benchmark: str,
    config: Mapping[str, Any],
    primary_api_key: str,
    proof_api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    primary_client = OpenAI(
        api_key=primary_api_key,
        base_url=str(config["primary_model"]["base_url"]),
        timeout=float(config["primary_model"]["timeout_seconds"]),
        max_retries=int(config["primary_model"]["max_retries"]),
    )
    proof_client = OpenAI(
        api_key=proof_api_key,
        base_url=str(config["proof_model"]["base_url"]),
        timeout=float(config["proof_model"]["timeout_seconds"]),
        max_retries=int(config["proof_model"]["max_retries"]),
    )
    primary_panel, proof_panels, metadata = paired._panels(sample, config)
    primary_config = {**config, "model": config["primary_model"]}
    proof_config = {**config, "model": config["proof_model"]}
    primary, primary_raw, primary_usage = paired._primary_call(
        primary_client, sample=sample, panel=primary_panel, config=primary_config,
    )
    proof, proof_raw, proof_usage = paired._proof_call(
        proof_client, sample=sample, panels=proof_panels, config=proof_config,
    )
    family = str(
        getattr(sample, "question_family", None)
        or getattr(sample, "question_type", "")
    )
    compatible = source_compatible(benchmark, family)
    primary_answer = str(primary["answer"])
    proof_answer = str(proof["answer"])
    by_slot = {str(candidate["slot"]): candidate for candidate in proof["candidates"]}
    primary_entailment = _step(by_slot[primary_answer], "ANSWER_ENTAILMENT")
    proof_entailment = _step(by_slot[proof_answer], "ANSWER_ENTAILMENT")
    proof_guard = bool(
        proof_answer != primary_answer
        and primary_entailment["status"] == "REFUTED"
        and proof_entailment["status"] == "SUPPORTED"
    )
    authentic_recover = bool(compatible and proof_guard)
    unrestricted_recover = proof_guard
    inverted_applicability_recover = bool((not compatible) and proof_guard)
    authentic_answer = proof_answer if authentic_recover else primary_answer
    unrestricted_answer = proof_answer if unrestricted_recover else primary_answer
    inverted_answer = proof_answer if inverted_applicability_recover else primary_answer
    # Gold is attached only after both blind neural branches and all symbolic
    # conditions have been frozen for this row.
    gold = str(sample.answer)
    features = build_features(
        benchmark=benchmark, family=family, primary=primary, proof=proof,
    )
    return {
        "schema_version": 19,
        "benchmark": benchmark,
        "split": "formal",
        "sample_id": str(sample.sample_id),
        "video_id": str(sample.video_id),
        "family": family,
        "gold_answer": gold,
        "source_compatible": compatible,
        "primary": primary,
        "proof": proof,
        "primary_answer_entailment": primary_entailment,
        "proof_answer_entailment": proof_entailment,
        "proof_guard": proof_guard,
        "authentic_recover": authentic_recover,
        "authentic_answer": authentic_answer,
        "unrestricted_recover": unrestricted_recover,
        "unrestricted_answer": unrestricted_answer,
        "inverted_applicability_recover": inverted_applicability_recover,
        "inverted_applicability_answer": inverted_answer,
        "primary_correct": primary_answer == gold,
        "proof_correct": proof_answer == gold,
        "authentic_correct": authentic_answer == gold,
        "unrestricted_correct": unrestricted_answer == gold,
        "inverted_applicability_correct": inverted_answer == gold,
        "authentic_uplift": int(authentic_answer == gold) - int(primary_answer == gold),
        "proof_uplift": int(proof_answer == gold) - int(primary_answer == gold),
        "unrestricted_uplift": int(unrestricted_answer == gold) - int(primary_answer == gold),
        "inverted_applicability_uplift": int(inverted_answer == gold) - int(primary_answer == gold),
        "features": list(map(float, features)),
        "primary_raw": primary_raw,
        "proof_raw": proof_raw,
        "usage": {"primary": primary_usage, "proof": proof_usage},
        "video_metadata": metadata,
        "video_sha256": _sha256(Path(sample.video_path)),
        "primary_panel_sha256": hashlib.sha256(primary_panel).hexdigest(),
        "proof_panel_sha256": [hashlib.sha256(value).hexdigest() for value in proof_panels],
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
        "formal_manifest_sha256": Path(config["formal_manifest"]),
        "collector_sha256": Path(__file__).resolve(),
        "v15_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "transport_module_sha256": REPO / "scripts/collect_natural_video_active_recovery_v18.py",
        "contract_module_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
        "applicability_module_sha256": REPO / "src/motif_transfer/natural_video_active_recovery.py",
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V19 formal lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    manifest = json.loads(Path(config["formal_manifest"]).read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_V19_EXPANDED_FORMAL_OUTCOMES":
        raise ValueError("V19 formal manifest is not sealed")
    ordered_pairs = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("star", "nextqa")
        for row in manifest["benchmarks"][benchmark]
    ]
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
        "formal_manifest_sha256": _sha256(Path(config["formal_manifest"])),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "ordered_pairs": ordered_pairs,
    })
    key_values = runpy.run_path(str(args.keys))
    primary_api_key = key_values.get(config["primary_model"]["api_key_name"])
    proof_api_key = key_values.get(config["proof_model"]["api_key_name"])
    if not primary_api_key or not proof_api_key:
        raise SystemExit("V19 primary/proof API key is missing")
    # Both imported V15 calls dispatch through this single, frozen transport.
    paired.media_helpers._json_call = _provider_json_call
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V19 formal receipt contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[pair] for pair in ordered_pairs if pair in existing],
            ensure_ascii=False,
            indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                samples[benchmark][sample_id],
                benchmark=benchmark,
                config=config,
                primary_api_key=str(primary_api_key),
                proof_api_key=str(proof_api_key),
                contract_sha256=contract_sha256,
            ): (benchmark, sample_id)
            for benchmark, sample_id in pending
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
        raise SystemExit(f"incomplete V19 formal collection; rerun: {missing}")
    rows = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": "NATURAL_VIDEO_V19_EXPANDED_FORMAL_COLLECTED",
        "samples": len(rows),
        "benchmark_counts": {
            benchmark: sum(row["benchmark"] == benchmark for row in rows)
            for benchmark in ("star", "nextqa")
        },
        "source_compatible": sum(row["source_compatible"] for row in rows),
        "primary_correct": sum(row["primary_correct"] for row in rows),
        "proof_correct": sum(row["proof_correct"] for row in rows),
        "authentic_correct": sum(row["authentic_correct"] for row in rows),
        "unrestricted_correct": sum(row["unrestricted_correct"] for row in rows),
        "authentic_recoveries": sum(row["authentic_recover"] for row in rows),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
