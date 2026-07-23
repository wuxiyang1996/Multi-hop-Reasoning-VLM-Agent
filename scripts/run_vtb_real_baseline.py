#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import runpy
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed real-image VisualToolBench smoke.")
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-id", required=True, help="Frozen row:<zero-based index>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()

    try:
        row_index = int(args.sample_id.removeprefix("row:"))
    except ValueError as exc:
        raise SystemExit(f"invalid frozen sample ID: {args.sample_id}") from exc
    import duckdb
    from PIL import Image

    # DuckDB is intentional: the official Parquet hash is valid but current
    # PyArrow releases in this workspace reject one repetition histogram.
    row = duckdb.sql(
        "SELECT * FROM read_parquet(?) LIMIT 1 OFFSET ?", params=[str(args.parquet), row_index]
    ).fetchone()
    columns = [item[0] for item in duckdb.sql(
        "DESCRIBE SELECT * FROM read_parquet(?)", params=[str(args.parquet)]
    ).fetchall()]
    if row is None:
        raise SystemExit(f"row does not exist: {args.sample_id}")
    record = dict(zip(columns, row))
    images = record.get("images") or []
    if not images or not images[0].get("bytes"):
        raise SystemExit(f"row has no inline image bytes: {args.sample_id}")
    image = Image.open(BytesIO(images[0]["bytes"]))
    image.load()

    sys.path.insert(0, str(args.legacy_repo))
    from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
        VisualToolBenchSample,
        parse_visual_toolbench_sample,
    )
    prompts = record.get("turn_prompts") or []
    answers = record.get("turn_golden_answers") or []
    if not prompts:
        raise SystemExit(f"row has no prompt: {args.sample_id}")
    sample = VisualToolBenchSample(
        sample_id=args.sample_id,
        turncase=str(record.get("turncase") or ""),
        prompt_category=record.get("prompt_category"),
        question=str(prompts[0]),
        gold_answer=str(answers[0]) if answers else None,
        image_cell=None,
        eval_focus=record.get("eval_focus"),
    )
    api_key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    result = parse_visual_toolbench_sample(
        sample, image=image, model=args.model, api_key=str(api_key),
        base_url=args.base_url, temperature=0, max_rounds=args.max_rounds,
        chain=["tool_loop"],
    )
    trace = result.get("tool_trace") or []
    if not trace:
        raise SystemExit("NOT_RUNNABLE: visual tool loop returned no concrete tool trace")
    payload = {
        "schema_version": 1,
        "cell": "visual_toolbench",
        "condition": "target_only",
        "executor_kind": "real_media_tool_loop",
        "evaluator_kind": "diagnostic_string_match_not_official_rubric",
        "sample_id": args.sample_id,
        "parquet_sha256": _sha256(args.parquet),
        "image_sha256": hashlib.sha256(images[0]["bytes"]).hexdigest(),
        "model": args.model,
        "max_rounds": args.max_rounds,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_id": args.sample_id,
        "answer": result.get("answer"),
        "ground_truth": result.get("ground_truth"),
        "diagnostic_correct": result.get("correct"),
        "tool_events": len(trace),
        "rounds": result.get("rounds"),
        "head_used": result.get("head_used"),
        "validation": result.get("validation"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
