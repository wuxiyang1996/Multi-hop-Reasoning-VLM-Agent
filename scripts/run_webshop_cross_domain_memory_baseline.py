#!/usr/bin/env python3
"""Run the cross-domain memory baseline arms on WebShop.

Each baseline arm is the *target-only* action selector plus retrieved memory in
the Decision Agent's prompt.  Nothing in the rollout loop changes: the target
policy still proposes every candidate and still picks the action, so the only
difference from the ``target_only`` arm is the extra prompt field.  That is the
channel ExpeL, AWM, and ReasoningBank actually use.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    LocalSentenceTransformerEmbeddingBackend,
    LocalHashingEmbeddingBackend,
    MemoryBaseline,
    validate_memory_artifact,
)
from motif_transfer.cross_domain_memory_runtime import (  # noqa: E402
    MemoryAugmentedDecisionBackend,
)
from motif_transfer.cross_domain_fairness import (  # noqa: E402
    require_formal_suite_audit,
    require_nonpilot_embedding,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)

TARGET_ONLY = "target_only"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected one JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True,
        choices=[TARGET_ONLY, *[row.value for row in MemoryBaseline]],
    )
    parser.add_argument(
        "--artifact", type=Path,
        help=f"Frozen memory artifact; required unless --arm {TARGET_ONLY}",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-grounder", type=Path, required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="V17_WEBSHOP_OPENROUTER_KEY")
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--run-mode", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--fairness-audit", type=Path)
    parser.add_argument("--maximum-output-tokens", type=int, default=1200)
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--run-id", default="webshop-xd-memory-v1")
    args = parser.parse_args()

    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    os.environ[args.api_key_env] = str(key)

    if args.arm != TARGET_ONLY and args.artifact is None:
        raise SystemExit(f"--artifact is required for arm {args.arm}")

    import run_webshop_neural_symbolic_v9 as v9_runner  # noqa: E402
    from motif_transfer.webshop_neural_symbolic_v9 import (  # noqa: E402
        TargetOutcomeMLP,
    )

    manifest = _load(args.manifest)
    # Support both the older flat phase-2 manifest and the current frozen
    # WebShop split, whose task rows live under roles.{development,formal}.
    manifest_rows = manifest.get("tasks")
    if manifest_rows is None:
        manifest_rows = [
            row
            for role_rows in (manifest.get("roles") or {}).values()
            for row in role_rows
        ]
    rows = {str(row["task_id"]): row for row in manifest_rows}
    tasks = list(args.task_ids or sorted(rows))
    unknown = [task for task in tasks if task not in rows]
    if unknown:
        raise SystemExit(f"unknown task ids: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # One cache per arm: the augmented request differs, so sharing a cache across
    # arms would silently serve a target-only completion to a memory arm.
    cache = args.output_dir / f"decision_cache.{args.arm}.json"
    raw_backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url, {"decision": args.model},
            api_key_env=args.api_key_env, json_mode=True,
            temperature=0, timeout_seconds=180,
            request_overrides={
                "max_tokens": args.maximum_output_tokens,
                "reasoning": {"effort": "none", "exclude": True},
            },
        ),
        cache_path=cache,
    )

    memory_backend = None
    artifact = None
    if args.arm != TARGET_ONLY:
        artifact = _load(args.artifact)
        validate_memory_artifact(artifact)
        if artifact["method"] != args.arm:
            raise SystemExit(
                f"artifact method {artifact['method']!r} does not match arm {args.arm!r}"
            )
        if "webshop" in set(map(str, artifact["source_domains"])):
            raise SystemExit("artifact was induced from the target domain")
        embedding_backend = (
            LocalHashingEmbeddingBackend()
            if args.embedding_model == "hashing-pilot"
            else LocalSentenceTransformerEmbeddingBackend(args.embedding_model)
        )
        require_nonpilot_embedding(embedding_backend.identity, run_mode=args.run_mode)
        memory_backend = MemoryAugmentedDecisionBackend(
            raw_backend,
            artifact=artifact,
            domain="webshop",
            embedding_backend=embedding_backend,
            top_k=args.top_k,
        )
    require_formal_suite_audit(
        args.fairness_audit,
        run_mode=args.run_mode,
        target_domain="webshop",
        method=None if args.arm == TARGET_ONLY else args.arm,
        artifact_sha256=artifact["artifact_sha256"] if artifact else None,
    )
    backend = memory_backend or raw_backend

    grounder_artifact = _load(args.target_grounder)
    if not (
        grounder_artifact.get("status") == "TARGET_NATIVE_LOW_SAMPLE_GROUNDER_QUALIFIED"
        or grounder_artifact.get("preflight_passed") is True
    ):
        raise SystemExit("low-sample target grounder is not qualified")
    grounder = TargetOutcomeMLP.from_dict(grounder_artifact["grounder"])

    receipts = []
    for task_id in tasks:
        goal_row = rows[task_id]
        receipt = v9_runner._run_condition(
            task_id=task_id,
            # The memory arms keep the target-only selector; memory reaches the
            # model through the prompt and never through action selection.
            condition="target_only",
            backend=backend,
            grounder=grounder,
            source_models={"artifact": {}},
            source_policy={"uncertainty_scale": 0.0, "decision_margin": 0.0},
            expected_goal=str(goal_row["instruction_text"]),
            wrapper_root=args.wrapper_root,
            session_namespace=f"{args.run_id}.{args.arm.replace('_', '-')}",
            number_of_goals=int(manifest["number_of_registered_tasks_required"]),
            maximum_steps=args.maximum_steps,
            candidate_count=args.candidate_count,
            schema_retries=args.schema_retries,
        )
        receipt["arm"] = args.arm
        receipts.append(receipt)
        (args.output_dir / f"{task_id}.{args.arm}.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    body = {
        "schema_version": 1,
        "run_id": args.run_id,
        "arm": args.arm,
        "run_mode": args.run_mode,
        "implementation_fidelity": "clean_room_style",
        "result_label": (
            "target-only" if args.arm == TARGET_ONLY else f"{args.arm}-style"
        ),
        "target_domain": "webshop",
        "tasks": tasks,
        "decision_model": args.model,
        "maximum_steps": args.maximum_steps,
        "candidate_count": args.candidate_count,
        "backend_identity": dict(backend.identity),
        "memory_receipt": memory_backend.receipt() if memory_backend else None,
        "strict_success": sum(1 for row in receipts if row.get("strict_success")),
        "pass_success": sum(1 for row in receipts if row.get("pass_success")),
        "episodes": len(receipts),
    }
    report = body | {"report_sha256": stable_hash(body)}
    (args.output_dir / f"report.{args.arm}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "arm": args.arm, "episodes": len(receipts),
        "strict_success": report["strict_success"],
        "pass_success": report["pass_success"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
