#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed real-media TIR baseline smoke.")
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.legacy_repo))
    from PIL import Image
    from visual_reasoning_wrapper.benchmarks.tir_bench import TIRBenchSample, parse_tir_bench_sample

    values = runpy.run_path(str(args.keys))
    api_key = values.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    annotation = args.dataset_root / "TIR-Bench.json"
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    matches = [row for row in rows if str(row.get("id")) == args.sample_id]
    if len(matches) != 1:
        raise SystemExit(f"sample ID must resolve exactly once: {args.sample_id}")
    row = matches[0]
    image_path = args.dataset_root / row["image_1"]
    if not image_path.is_file():
        raise SystemExit(f"image is missing: {image_path}")
    image = Image.open(image_path)
    image.load()
    sample = TIRBenchSample(
        sample_id=str(row["id"]), task=str(row.get("task") or ""),
        prompt=str(row.get("prompt") or ""), answer=str(row.get("answer") or ""),
        image_1=str(image_path), image_2=row.get("image_2"), meta_data=row.get("meta_data") or {},
    )
    result = parse_tir_bench_sample(
        sample, image=image, model=args.model, api_key=str(api_key),
        base_url=args.base_url, temperature=0, max_rounds=args.max_rounds,
        chain=["tool_loop"],
    )
    tool_trace = result.get("tool_trace") or []
    # A schema-only answer is not an agentic TIR run. Do not accept the old
    # parser's nominal output unless at least one concrete tool event exists.
    if not tool_trace:
        raise SystemExit("NOT_RUNNABLE: tool_loop returned no concrete tool trace")
    payload = {
        "schema_version": 1,
        "cell": "tir_bench",
        "condition": "target_only",
        "executor_kind": "real_media_tool_loop",
        "official_evaluator": "exact_answer",
        "sample_id": args.sample_id,
        "annotation_sha256": _sha256(annotation),
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
        "model": args.model,
        "max_rounds": args.max_rounds,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_id": args.sample_id, "answer": result.get("answer"),
        "ground_truth": result.get("ground_truth"), "correct": result.get("correct"),
        "tool_events": len(tool_trace), "rounds": result.get("rounds"),
        "head_used": result.get("head_used"), "warnings": result.get("warnings"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
