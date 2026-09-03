#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    LocalHashingEmbeddingBackend,
    LocalSentenceTransformerEmbeddingBackend,
    MemoryBaseline,
    validate_memory_artifact,
)
from motif_transfer.cross_domain_fairness import (  # noqa: E402
    require_formal_suite_audit,
    require_nonpilot_embedding,
)
from motif_transfer.cross_domain_memory_runtime import (  # noqa: E402
    advisory_prompt_block,
    retrieve_target_advisory,
)


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
    parser.add_argument(
        "--arm", default="target_only",
        choices=["target_only", *[row.value for row in MemoryBaseline]],
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--run-mode", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--fairness-audit", type=Path)
    args = parser.parse_args()
    if args.arm != "target_only" and args.artifact is None:
        raise SystemExit("--artifact is required for a memory arm")

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
    prompt = str(row.get("prompt") or "")
    memory_retrieval = None
    memory_artifact_sha256 = None
    artifact = None
    if args.arm != "target_only":
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        validate_memory_artifact(artifact)
        if artifact["method"] != args.arm:
            raise SystemExit("memory artifact method does not match --arm")
        embedding_backend = (
            LocalHashingEmbeddingBackend()
            if args.embedding_model == "hashing-pilot"
            else LocalSentenceTransformerEmbeddingBackend(args.embedding_model)
        )
        require_nonpilot_embedding(embedding_backend.identity, run_mode=args.run_mode)
        advisory, memory_retrieval = retrieve_target_advisory(
            artifact,
            "tirbench",
            {
                "prompt": prompt,
                "question": prompt,
                "tool_trace": [],
                "available_tools": ["crop", "zoom", "object_detection"],
            },
            embedding_backend,
            top_k=3,
        )
        memory_artifact_sha256 = artifact["artifact_sha256"]
        # Empty verified memory is a strict target-only no-op.
        if advisory:
            prompt += advisory_prompt_block(args.arm, advisory)
    require_formal_suite_audit(
        args.fairness_audit,
        run_mode=args.run_mode,
        target_domain="tirbench",
        method=None if args.arm == "target_only" else args.arm,
        artifact_sha256=artifact["artifact_sha256"] if artifact else None,
    )
    sample = TIRBenchSample(
        sample_id=str(row["id"]), task=str(row.get("task") or ""),
        prompt=prompt, answer=str(row.get("answer") or ""),
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
        "condition": args.arm,
        "run_mode": args.run_mode,
        "implementation_fidelity": "clean_room_style",
        "result_label": "target-only" if args.arm == "target_only" else f"{args.arm}-style",
        "executor_kind": "real_media_tool_loop",
        "official_evaluator": "exact_answer",
        "sample_id": args.sample_id,
        "annotation_sha256": _sha256(annotation),
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
        "model": args.model,
        "max_rounds": args.max_rounds,
        "memory_artifact_sha256": memory_artifact_sha256,
        "memory_retrieval": memory_retrieval,
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
