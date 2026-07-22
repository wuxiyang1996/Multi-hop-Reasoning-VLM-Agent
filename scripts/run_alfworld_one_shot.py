#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextEnvironment
from motif_transfer.api_decision_agent import OpenAIJSONDecisionAgent
from motif_transfer.artifact_io import adaptation_example_view, load_first_source_motif
from motif_transfer.contracts import Lifecycle, MotifNode, SourceStepSignature, stable_hash
from motif_transfer.controls import shuffled_topology
from motif_transfer.frozen_motif_agent import FrozenJSONMotifAgent, OpenAICompatibleBackend
from motif_transfer.metrics import measure_episode
from motif_transfer.neutral_motif_agent import NeutralMotifAgent
from motif_transfer.runtime import TwoAgentRuntime


CONDITIONS = ("target_only", "generic_protocol", "authentic", "shuffled_topology", "other_source")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _masked(candidate):
    signature = SourceStepSignature(False, "MASKED", "MASKED", False, None)
    nodes = tuple(MotifNode(
        node.node_id,
        node.transition_receipt_ids,
        tuple(signature for _ in node.decision_signatures),
    ) for node in candidate.nodes)
    return replace(
        candidate,
        motif_id=f"{candidate.motif_id}:generic-protocol",
        nodes=nodes,
        status=Lifecycle.GENERIC_ONLY,
        untrusted_description="content-free generic protocol control",
    )


def _load_keys(path: Path) -> None:
    values = runpy.run_path(str(path))
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        value = values.get(name)
        if value and not os.environ.get(name):
            os.environ[name] = str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adaptation-demo", type=Path, required=True)
    parser.add_argument("--authentic-motif", type=Path, required=True)
    parser.add_argument("--other-source-motif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--split", default="eval_out_of_distribution")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--decision-model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--harness-model", default="gpt-5-mini")
    args = parser.parse_args()

    _load_keys(args.keys)
    adaptation = adaptation_example_view(args.adaptation_demo)
    authentic = load_first_source_motif(args.authentic_motif, status=Lifecycle.GENERIC_ONLY)
    other = load_first_source_motif(args.other_source_motif, status=Lifecycle.GENERIC_ONLY)
    motifs = {
        "generic_protocol": _masked(authentic),
        "authentic": authentic,
        "shuffled_topology": shuffled_topology(authentic),
        "other_source": other,
    }
    rows = []
    for condition in args.conditions:
        decision_backend = OpenAICompatibleBackend(
            "https://openrouter.ai/api/v1",
            {"decision": args.decision_model},
            api_key_env="OPENROUTER_API_KEY",
            json_mode=True,
            request_overrides={
                "max_tokens": 256,
                "top_p": 1.0,
                "reasoning": {"effort": "none", "exclude": True},
            },
        )
        decision = OpenAIJSONDecisionAgent(decision_backend)
        binding = None
        harness_calls = []
        binding_error = None
        if condition == "target_only":
            motif_agent = NeutralMotifAgent()
        else:
            harness_backend = OpenAICompatibleBackend(
                "https://us.api.openai.com/v1",
                {"binding": args.harness_model, "review": args.harness_model},
                api_key_env="OPENAI_API_KEY",
                json_mode=True,
                temperature=None,
            )
            motif_agent = FrozenJSONMotifAgent(
                harness_backend,
                allowed_verifier_ids=("official_transition_and_outcome",),
            )
            try:
                binding = motif_agent.initialize_binding_from_example(motifs[condition], adaptation)
            except Exception as exc:
                binding_error = f"BINDING_ERROR:{type(exc).__name__}:{exc}"
            harness_calls.append({
                "phase": "binding_driver",
                "admitted_provisional_binding": binding is not None,
                "error": binding_error,
            })
        environment = ALFWorldTextEnvironment(
            config_path=str(args.config),
            data_path=str(args.data),
            split=args.split,
            seed=args.seed,
            game_index=args.game_index,
            max_steps=args.max_steps,
        )
        started = time.monotonic()
        error = None
        result = None
        try:
            if condition != "target_only" and binding is None:
                error = binding_error or "PROVISIONAL_BINDING_ABSTAINED"
            else:
                result = TwoAgentRuntime(decision, motif_agent).run(
                    environment,
                    "Follow the official ALFWorld task stated in the observation.",
                    binding=binding,
                    max_steps=args.max_steps,
                )
        except Exception as exc:
            result = getattr(exc, "partial_episode_result", None)
            error = f"{type(exc).__name__}:{exc}"
        finally:
            environment.close()
        if condition != "target_only":
            harness_calls.extend(motif_agent.call_receipts)
        metrics = asdict(measure_episode(result)) if result else None
        row = {
            "condition": condition,
            "source_status": motifs[condition].status.value if condition in motifs else None,
            "binding": asdict(binding) if binding else None,
            "adaptation_example_sha256": stable_hash(adaptation),
            "initial_state_hash": (
                result.records[0].transition.before_hash if result and result.records else None
            ),
            "target_task_goal": (
                result.records[0].before.state.get("task_goal")
                if result and result.records else None
            ),
            "metrics": metrics,
            "actions": [record.proposal_set.selected.action for record in result.records] if result else [],
            "transition_receipts": [asdict(row) for row in result.receipts] if result else [],
            "decision_call_receipts": decision.call_receipts,
            "harness_calls": harness_calls,
            "error": error,
            "wall_time_s": time.monotonic() - started,
        }
        rows.append(row)
        print(json.dumps({
            "condition": condition,
            "error": error,
            "metrics": metrics,
        }, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "schema_version": 1,
            "claim_limit": "MECHANISM_DIAGNOSTIC_GENERIC_ONLY_UNTIL_SOURCE_QUALIFICATION",
            "one_shot_semantics": "one example initializes a provisional binding; live target transitions test it",
            "decision_model": args.decision_model,
            "harness_model": args.harness_model,
            "decision_backend": decision_backend.identity,
            "source_artifacts": {
                "adaptation_demo": {
                    "path": str(args.adaptation_demo.resolve()),
                    "sha256": _file_sha256(args.adaptation_demo),
                },
                "authentic_motif": {
                    "path": str(args.authentic_motif.resolve()),
                    "sha256": _file_sha256(args.authentic_motif),
                },
                "other_source_motif": {
                    "path": str(args.other_source_motif.resolve()),
                    "sha256": _file_sha256(args.other_source_motif),
                },
            },
            "split": args.split,
            "seed": args.seed,
            "game_index": args.game_index,
            "max_steps": args.max_steps,
            "rows": rows,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
