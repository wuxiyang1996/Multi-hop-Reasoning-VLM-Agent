#!/usr/bin/env python3
"""Resolve one shared outcome label per source episode for every memory method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.source_outcome_evaluator import (  # noqa: E402
    FrozenJudgeEvaluator,
    benchmark_predicate_from_config,
    label_source_payload,
    load_outcome_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Canonical source episode JSON")
    parser.add_argument("--config", type=Path, required=True, help="Frozen outcome config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--maximum-output-tokens", type=int, default=400)
    parser.add_argument("--maximum-steps-shown", type=int, default=24)
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Native-fidelity pass: resolve official and benchmark predicates only",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen labels: {args.output}")

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("source input must be one JSON object")
    config = load_outcome_config(args.config)

    judge = None
    if not args.no_judge:
        backend = MemoizedCompletionBackend(
            OpenAICompatibleBackend(
                args.base_url,
                {"source_outcome_judge": args.model},
                api_key_env=args.api_key_env,
                json_mode=True,
                temperature=0,
                request_overrides={"max_tokens": args.maximum_output_tokens},
            ),
            cache_path=args.cache,
        )
        judge = FrozenJudgeEvaluator(backend, maximum_steps_shown=args.maximum_steps_shown)

    predicate = benchmark_predicate_from_config(config, payload.get("episodes") or [])
    labelled = label_source_payload(
        payload,
        benchmark_predicate=predicate,
        shared_evaluator=judge,
        attribution={
            "outcome_config_sha256": stable_hash(config),
            "shared_evaluator_identity": dict(judge.identity) if judge else None,
            "native_fidelity_pass": bool(args.no_judge),
            "label_semantics": dict(config.get("label_semantics") or {}),
            "cohort_bounds": getattr(predicate, "cohort_bounds", {}),
        },
    )
    if judge is not None:
        labelled["shared_evaluator_receipts"] = judge.receipts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(labelled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "episodes": len(labelled["episodes"]),
        "outcome_census": labelled["outcome_census"],
        "outcome_authority_census": labelled["outcome_authority_census"],
        "judge_calls": len(judge.receipts) if judge is not None else 0,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
