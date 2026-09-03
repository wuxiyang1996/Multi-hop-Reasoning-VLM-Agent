#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.api_decision_agent import OpenAIJSONDecisionAgent  # noqa: E402
from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    CrossDomainMemoryDecisionAgent,
    LocalHashingEmbeddingBackend,
    MemoryBaseline,
    validate_memory_artifact,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.metrics import measure_episode  # noqa: E402
from motif_transfer.neutral_motif_agent import NeutralMotifAgent  # noqa: E402
from motif_transfer.runtime import TwoAgentRuntime  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receipt-rich target-only ALFWorld rollout collector"
    )
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-name", default="adaptation")
    parser.add_argument("--task-offset", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision-cache", required=True, type=Path)
    parser.add_argument("--alfworld-split", default="train")
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--decision-max-tokens", type=int, default=512)
    parser.add_argument("--decision-model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument(
        "--arm", default="target_only",
        choices=["target_only", *[row.value for row in MemoryBaseline]],
    )
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.arm != "target_only" and args.artifact is None:
        raise SystemExit("--artifact is required for a memory arm")

    keys = runpy.run_path(str(args.keys))
    value = keys.get("OPENROUTER_API_KEY")
    if value and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = str(value)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"]["alfworld_valid_unseen"]
    task_ids = list(cell["splits"][args.split_name])
    if not 0 <= args.task_offset < len(task_ids):
        raise SystemExit("task offset outside frozen split")
    task_id = str(task_ids[args.task_offset])
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            "https://openrouter.ai/api/v1",
            {"decision": args.decision_model},
            api_key_env="OPENROUTER_API_KEY",
            json_mode=True,
            request_overrides={
                "max_tokens": args.decision_max_tokens,
                "top_p": 1.0,
                "seed": args.seed + args.task_offset,
                "reasoning": {"effort": "none", "exclude": True},
            },
        ),
        cache_path=args.decision_cache,
    )
    base_decision = OpenAIJSONDecisionAgent(backend)
    memory_decision = None
    if args.arm != "target_only":
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        validate_memory_artifact(artifact)
        if artifact["method"] != args.arm:
            raise SystemExit("memory artifact method does not match --arm")
        memory_decision = CrossDomainMemoryDecisionAgent(
            base_decision, artifact=artifact, domain="alfworld",
            embedding_backend=LocalHashingEmbeddingBackend(), top_k=3,
        )
    decision = memory_decision or base_decision
    # Construct a one-game official batch instead of scanning the full split
    # and calling ``skip``.  AlfredTWEnv shuffles on reset, so skip(index) did
    # not reliably execute the manifest-resolved game and could silently make
    # paired arms face different goals.
    environment = ALFWorldTextBatchEnvironment(
        config_path=str(args.config),
        data_path=str(args.data),
        split=args.alfworld_split,
        seed=args.seed + args.task_offset,
        game_ids=[task_id],
        max_steps=args.max_steps,
    )
    started = time.monotonic()
    result = None
    error = None
    try:
        result = TwoAgentRuntime(decision, NeutralMotifAgent()).run(
            environment,
            "Follow the official ALFWorld task stated in the observation.",
            max_steps=args.max_steps,
        )
    except Exception as exc:
        result = getattr(exc, "partial_episode_result", None)
        error = f"{type(exc).__name__}:{exc}"
    finally:
        environment.close()
    terminal_failure_kind = None
    if error and error.startswith("ValueError:decision model "):
        terminal_failure_kind = "DECISION_AGENT_INVALID_OUTPUT"
    payload = {
        "schema_version": 1,
        "authority": "TARGET_ONLY_ADAPTATION_ROLLOUT",
        "target_domain": "alfworld_valid_unseen",
        "collection_split": args.split_name,
        "task_offset": args.task_offset,
        "task_id": task_id,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "alfworld_split": args.alfworld_split,
        "seed": args.seed + args.task_offset,
        "decision_model": args.decision_model,
        "decision_backend": backend.identity,
        "condition": "BASE_DECISION_TARGET_ONLY" if args.arm == "target_only" else "CROSS_DOMAIN_MEMORY",
        "arm": args.arm,
        "harness_used": False,
        "source_motif_used": False,
        "max_steps": args.max_steps,
        "decision_max_tokens": args.decision_max_tokens,
        "resolved_game_file": environment.resolved_game_file,
        "metrics": asdict(measure_episode(result)) if result else None,
        "records": [asdict(row) for row in result.records] if result else [],
        "transition_receipts": (
            [asdict(row) for row in result.receipts] if result else []
        ),
        "decision_call_receipts": base_decision.call_receipts,
        "memory_retrieval_receipts": (
            list(memory_decision.retrieval_receipts) if memory_decision else []
        ),
        "source_failures": list(result.source_failures) if result else [],
        "error": error,
        "terminal_failure_kind": terminal_failure_kind,
        "wall_time_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "task_offset": args.task_offset,
        "task_id": task_id,
        "steps": len(payload["records"]),
        "official_success": (
            payload["metrics"].get("official_success")
            if payload["metrics"] else None
        ),
        "error": error,
    }, ensure_ascii=False), flush=True)
    if error is not None and terminal_failure_kind is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
