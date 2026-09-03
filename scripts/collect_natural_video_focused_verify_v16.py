#!/usr/bin/env python3
"""Collect focused VERIFY_EXPECTED_EFFECT receipts on consumed V15 development."""

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

import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402
from motif_transfer.natural_video_recovery import (  # noqa: E402
    PROOF_KINDS,
    parse_focused_verification,
)
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


def _proof_panels(row: Mapping[str, Any], config: Mapping[str, Any]) -> list[bytes]:
    public = row["sample_public"]
    media = config["media"]
    frames, metadata = structured._sample_clip(
        Path(public["video_path"]),
        start_sec=float(public["clip_start_seconds"]),
        end_sec=(
            float(public["clip_end_seconds"])
            if public.get("clip_end_seconds") is not None else None
        ),
        frame_count=int(media["proof_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    seconds = metadata["proxy_sample_seconds"]
    per_panel = int(media["proof_frames_per_panel"])
    output = []
    for start in range(0, len(frames), per_panel):
        indices = list(range(start, min(start + per_panel, len(frames))))
        output.append(media_helpers._panel_bytes(
            [frames[index] for index in indices],
            labels=[f"V{index} {seconds[index]:.2f}s" for index in indices],
            frame_width=int(media["proof_frame_width"]),
            quality=int(media["jpeg_quality"]),
        ))
    return output


def _focused_call(
    client: OpenAI,
    *,
    row: Mapping[str, Any],
    panels: Sequence[bytes],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    public = row["sample_public"]
    slots = tuple(public["answer_slots"])
    primary_answer = str(row["primary"]["answer"])
    primary_claim = str(public["options"][primary_answer])
    formatted_options = "\n".join(
        f"{slot}. {public['options'][slot]}" for slot in slots
    )
    prompt = (
        "A first-pass video agent has COMMITTED to an expected answer. Execute "
        "VERIFY_EXPECTED_EFFECT on that exact commitment. Actively seek both visible "
        "support and visible counterevidence; do not preserve the commitment merely "
        "because it was proposed. For the expected answer, execute exactly these "
        "typed proof steps in order: " + ", ".join(PROOF_KINDS) + ". "
        "ANSWER_ENTAILMENT is SUPPORTED only if the necessary visible facts jointly "
        "entail the expected option; REFUTED only if visible evidence contradicts a "
        "necessary fact or clearly supports an incompatible option; otherwise UNKNOWN. "
        "verification_status must be OBSERVED, REFUTED, or UNRESOLVED according to "
        "that final typed step. Also produce evidence probabilities for all options "
        "and a recovery_answer equal to their argmax. Do not use annotations, official "
        "programs, graphs, or gold.\n"
        f"Question: {public['question']}\nOptions:\n{formatted_options}\n"
        f"Committed expected answer: {primary_answer}. {primary_claim}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Chronological verification evidence {index + 1}/{len(panels)}:"},
            media_helpers._image_content(panel),
        ])
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": f"Schema error: {last_error}"})
        payload, usage = media_helpers._json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {verification_status:"
                "OBSERVED|REFUTED|UNRESOLVED,recovery_answer:string,probabilities:"
                "{slot:number},expected_answer_proof_steps:[{kind:string,status:"
                "SUPPORTED|REFUTED|UNKNOWN,confidence:number,visible_fact:string}],"
                "supporting_evidence:[string],counterevidence:[string],"
                "unresolved_uncertainties:[string],reason:string}. Preserve the five "
                "typed steps exactly; recovery_answer is the unique probability argmax."
            ),
            content=attempt,
            max_tokens=int(model["max_verify_tokens"]),
        )
        try:
            parsed = parse_focused_verification(payload, slots, primary_answer)
            return parsed, payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"focused VERIFY schema failed: {last_error}")


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
    panels = _proof_panels(row, config)
    verification, raw, usage = _focused_call(
        client, row=row, panels=panels, config=config,
    )
    gold = str(row["gold_answer"])
    recovery_correct = verification["recovery_answer"] == gold
    primary_correct = bool(row["primary_correct"])
    authentic_recover = (
        verification["verification_status"] == "REFUTED"
        and verification["recovery_answer"] != verification["expected_answer"]
    )
    authentic_answer = (
        verification["recovery_answer"] if authentic_recover
        else verification["expected_answer"]
    )
    return {
        "schema_version": 16,
        "benchmark": row["benchmark"],
        "split": "development",
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "family": row["family"],
        "gold_answer": gold,
        "primary": row["primary"],
        "primary_correct": primary_correct,
        "focused_verification": verification,
        "recovery_correct": recovery_correct,
        "recovery_uplift": int(recovery_correct) - int(primary_correct),
        "authentic_recover": authentic_recover,
        "authentic_answer": authentic_answer,
        "authentic_correct": authentic_answer == gold,
        "authentic_uplift": int(authentic_answer == gold) - int(primary_correct),
        "focused_verification_raw": raw,
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
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "input_receipts_sha256": Path(config["input_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "contract_module_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
    }
    for key, path in paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V16 frozen lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    input_rows = json.loads(Path(config["input_receipts"]).read_text(encoding="utf-8"))
    ordered_keys = [(str(row["benchmark"]), str(row["sample_id"])) for row in input_rows]
    if len(ordered_keys) != len(set(ordered_keys)):
        raise ValueError("V16 input rows contain duplicates")
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "input_receipts_sha256": _sha256(Path(config["input_receipts"])),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "ordered_keys": ordered_keys,
    })
    keys = runpy.run_path(str(args.keys))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured focused-verifier API key is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached focused-verification contract mismatch")
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
        raise SystemExit(f"incomplete V16 focused verification; rerun: {missing}")
    rows = [existing[key] for key in ordered_keys]
    print(json.dumps({
        "status": "NATURAL_VIDEO_V16_FOCUSED_DEVELOPMENT_COLLECTED",
        "samples": len(rows),
        "primary_correct": sum(row["primary_correct"] for row in rows),
        "recovery_correct": sum(row["recovery_correct"] for row in rows),
        "authentic_correct": sum(row["authentic_correct"] for row in rows),
        "authentic_recoveries": sum(row["authentic_recover"] for row in rows),
        "verification_status_counts": {
            status: sum(
                row["focused_verification"]["verification_status"] == status
                for row in rows
            ) for status in ("OBSERVED", "REFUTED", "UNRESOLVED")
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
