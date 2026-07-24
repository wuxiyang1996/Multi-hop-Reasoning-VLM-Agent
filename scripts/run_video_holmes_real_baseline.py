#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Fail-closed real-video Video-Holmes baseline smoke.")
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-id", required=True, help="<video_id>.Q<question_id>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()

    sys.path.insert(0, str(args.legacy_repo))
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples,
        parse_video_holmes_sample,
    )

    values = runpy.run_path(str(args.keys))
    api_key = values.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    try:
        video_id, question = args.sample_id.rsplit(".Q", 1)
        question_id = int(question)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"invalid sample ID: {args.sample_id}") from exc

    matches = [
        sample for sample in iter_video_holmes_samples(
            split="test", video_holmes_root=args.dataset_root, video_ids=[video_id]
        )
        if sample.question_id == question_id
    ]
    if len(matches) != 1:
        raise SystemExit(f"sample ID must resolve exactly once: {args.sample_id}")
    sample = matches[0]
    if sample.video_path is None:
        native_path = args.dataset_root / "Benchmark/videos_cropped" / f"{video_id}.mp4"
        if native_path.is_file():
            sample.video_path = native_path
    if sample.video_path is None or not sample.video_path.is_file():
        raise SystemExit(f"video is missing for {args.sample_id}")

    result = parse_video_holmes_sample(
        sample,
        num_frames=args.num_frames,
        model=args.model,
        api_key=str(api_key),
        base_url=args.base_url,
        temperature=0,
        max_rounds=args.max_rounds,
    )
    trace = result.get("tool_trace") or []
    if not trace:
        raise SystemExit("NOT_RUNNABLE: video tool loop returned no concrete tool trace")
    payload = {
        "schema_version": 1,
        "cell": "video_holmes",
        "condition": "target_only",
        "executor_kind": "real_video_tool_loop",
        "official_evaluator": "exact_answer",
        "sample_id": args.sample_id,
        "annotation_sha256": _sha256(args.dataset_root / "Benchmark/test_Video-Holmes.json"),
        "video_path": str(sample.video_path),
        "video_sha256": _sha256(sample.video_path),
        "model": args.model,
        "num_frames": args.num_frames,
        "max_rounds": args.max_rounds,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_id": args.sample_id,
        "answer": result.get("answer"),
        "ground_truth": result.get("ground_truth"),
        "correct": result.get("correct"),
        "tool_events": len(trace),
        "rounds": result.get("rounds"),
        "head_used": result.get("head_used"),
        "validation": result.get("validation"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
