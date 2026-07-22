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
from motif_transfer.binding import validate_structural_binding
from motif_transfer.api_decision_agent import OpenAIJSONDecisionAgent
from motif_transfer.artifact_io import (
    adaptation_example_view,
    load_first_source_motif,
    load_frozen_binding_artifact,
)
from motif_transfer.contracts import Lifecycle, MotifNode, SourceStepSignature, stable_hash
from motif_transfer.controls import shuffled_topology
from motif_transfer.frozen_motif_agent import (
    FrozenJSONMotifAgent,
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
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
    parser.add_argument("--skip-alpha-control", action="store_true")
    parser.add_argument("--disable-alpha-invariance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--binding-repetitions", type=int, default=2)
    parser.add_argument(
        "--binding-artifact-dir", type=Path,
        help="Directory containing <condition>.json artifacts frozen before evaluation.",
    )
    parser.add_argument(
        "--allow-inline-adaptation", action="store_true",
        help="Diagnostic only: permit binding generation in the evaluation process.",
    )
    args = parser.parse_args()
    skip_alpha_control = args.skip_alpha_control or args.disable_alpha_invariance

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
    shared_decision_backend = MemoizedCompletionBackend(OpenAICompatibleBackend(
        "https://openrouter.ai/api/v1",
        {"decision": args.decision_model},
        api_key_env="OPENROUTER_API_KEY",
        json_mode=True,
        request_overrides={
            "max_tokens": 256,
            "top_p": 1.0,
            "seed": args.seed,
            "reasoning": {"effort": "none", "exclude": True},
        },
    ))
    rows = []
    for condition in args.conditions:
        decision_backend = shared_decision_backend
        decision = OpenAIJSONDecisionAgent(decision_backend)
        bindings = ()
        binding_attributions = []
        binding_artifact_hash = None
        harness_calls = []
        binding_error = None
        if condition == "target_only":
            motif_agent = NeutralMotifAgent()
        else:
            harness_backend = OpenAICompatibleBackend(
                "https://us.api.openai.com/v1",
                {
                    "binding": args.harness_model,
                    "review": args.harness_model,
                    "verify": args.harness_model,
                },
                api_key_env="OPENAI_API_KEY",
                json_mode=True,
                temperature=None,
            )
            motif_agent = FrozenJSONMotifAgent(
                harness_backend,
                allowed_verifier_ids=("official_transition_and_outcome",),
            )
            motif_agent.register_motif(motifs[condition])
            try:
                if args.binding_artifact_dir is not None:
                    artifact = load_frozen_binding_artifact(
                        args.binding_artifact_dir / f"{condition}.json"
                    )
                    if artifact.motif_id != motifs[condition].motif_id:
                        raise ValueError("frozen binding motif does not match evaluation condition")
                    if artifact.adaptation_example_sha256 != stable_hash(adaptation):
                        raise ValueError("frozen binding adaptation example does not match")
                    for frozen in artifact.bindings:
                        hypothesis = frozen.hypothesis
                        signature = validate_structural_binding(
                            motifs[condition],
                            target_cycle_count=len(adaptation.get("transitions", [])),
                            node_alignment=hypothesis.node_alignment,
                            edge_alignment=hypothesis.edge_alignment,
                        )
                        if signature != hypothesis.invariance_signature:
                            raise ValueError("frozen binding structural signature mismatch")
                        if hypothesis.adaptation_receipt_ids != (stable_hash(adaptation),):
                            raise ValueError("frozen binding does not cite the adaptation example")
                    bindings = artifact.hypotheses
                    binding_attributions = [row.attribution.value for row in artifact.bindings]
                    binding_artifact_hash = artifact.artifact_hash
                elif args.allow_inline_adaptation:
                    artifact = motif_agent.build_binding_artifact(
                        motifs[condition], adaptation,
                        run_alpha_control=not skip_alpha_control,
                        induction_repetitions=args.binding_repetitions,
                    )
                    bindings = artifact.hypotheses
                    binding_attributions = [row.attribution.value for row in artifact.bindings]
                    binding_artifact_hash = artifact.artifact_hash
                else:
                    raise ValueError(
                        "non-target evaluation requires --binding-artifact-dir; "
                        "use --allow-inline-adaptation only for diagnostics"
                    )
            except Exception as exc:
                binding_error = f"BINDING_ERROR:{type(exc).__name__}:{exc}"
            harness_calls.append({
                "phase": "binding_driver",
                "admitted_provisional_bindings": len(bindings),
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
            result = TwoAgentRuntime(decision, motif_agent).run(
                environment,
                "Follow the official ALFWorld task stated in the observation.",
                bindings=bindings,
                max_steps=args.max_steps,
                max_source_replans=1,
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
            "bindings": [asdict(binding) for binding in bindings],
            "binding_attributions": binding_attributions,
            "binding_artifact_hash": binding_artifact_hash,
            "binding_frozen_before_evaluation": args.binding_artifact_dir is not None,
            "binding_status": (
                "TARGET_ONLY"
                if condition == "target_only"
                else "FROZEN_RAW_STABLE_PROVISIONAL"
                if bindings
                else "TARGET_ONLY_FALLBACK_NO_RAW_STABLE_BINDING"
            ),
            "binding_error": binding_error,
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
            "binding_evidence": [asdict(row) for row in result.binding_evidence] if result else [],
            "source_fallback_step": (
                result.source_fallback_step
                if result and result.source_fallback_step is not None
                else 0
                if condition != "target_only" and not bindings
                else None
            ),
            "source_replans": result.source_replans if result else 0,
            "source_failures": list(result.source_failures) if result else [],
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
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "alpha_control_run": not skip_alpha_control,
            "binding_repetitions": args.binding_repetitions,
            "rows": rows,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
