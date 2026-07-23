#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextEnvironment  # noqa: E402
from motif_transfer.api_decision_agent import OpenAIJSONDecisionAgent  # noqa: E402
from motif_transfer.contracts import (  # noqa: E402
    Advisory,
    AdvisoryVerdict,
    BindingEvidence,
    BindingHypothesis,
    EvidenceVerdict,
    TransferObjectKind,
    stable_hash,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.instrumented_import import import_native_source_batch  # noqa: E402
from motif_transfer.metrics import measure_episode  # noqa: E402
from motif_transfer.runtime import TwoAgentRuntime  # noqa: E402


CONDITIONS = (
    "target_only",
    "generic_reasoning",
    "source_receipts_only",
    "authentic_weak_control_prior",
    "shuffled_evidence_prior",
    "other_game_abstain",
)


def _pad_json_to_exact_tokens(payload: Mapping[str, Any], target_tokens: int) -> str:
    import tiktoken

    value = dict(payload)
    value["_matched_padding"] = ""
    encoding = tiktoken.get_encoding("o200k_base")
    for _ in range(target_tokens * 3):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        count = len(encoding.encode(text))
        if count == target_tokens:
            return text
        if count > target_tokens:
            raise ValueError(f"context exceeds matched token target: {count}>{target_tokens}")
        value["_matched_padding"] += " p"
    raise RuntimeError("could not construct exact-token source context")


def _load_keys(path: Path) -> None:
    values = runpy.run_path(str(path))
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        value = values.get(name)
        if value and not os.environ.get(name):
            os.environ[name] = str(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_records(evidence_dirs: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for game_index, evidence_dir in enumerate(evidence_dirs):
        episodes = import_native_source_batch(evidence_dir)
        for episode_index, episode in enumerate(episodes):
            if not episode.records:
                continue
            actions = sorted({
                action for row in episode.records for action in row.before.native_actions
            }, key=len, reverse=True)
            game_names = {
                str(row.before.state.get("structured_state", {}).get("display_name", ""))
                for row in episode.records[:1]
            } - {""}

            def redact(value: str) -> str:
                output = value
                for token in (*sorted(game_names, key=len, reverse=True), *actions):
                    output = re.sub(
                        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
                        "SOURCE_SYMBOL",
                        output,
                        flags=re.IGNORECASE,
                    )
                return output

            for row in episode.records:
                receipt = row.transition
                result[receipt.receipt_id] = {
                    "receipt_id": receipt.receipt_id,
                    "source_game_alias": f"G{game_index}",
                    "source_episode_alias": f"G{game_index}_E{episode_index}",
                    "step": row.step,
                    "before_observation": redact(
                        str(row.before.state.get("observable_state", ""))
                    )[:700],
                    "untrusted_source_reasoning": redact(row.action_reasoning)[:400],
                    "after_observation": redact(
                        str(row.after.state.get("observable_state", ""))
                    )[:700],
                    "reward": row.reward,
                    "terminal": row.after.terminal,
                }
    return result


def _contexts(
    authentic_artifact: Mapping[str, Any],
    other_artifact: Mapping[str, Any],
    source_registry: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, str]], int]:
    clauses = [dict(row) for row in authentic_artifact["knowledge"]["clauses"]]
    authentic_ids = list(dict.fromkeys(
        receipt_id
        for clause in clauses
        for receipt_id in clause["source_receipt_ids"]
        if receipt_id in source_registry
    ))
    evidence = [dict(source_registry[receipt_id]) for receipt_id in authentic_ids]
    rotated = [clause["source_receipt_ids"] for clause in clauses[1:]] + [
        clauses[0]["source_receipt_ids"]
    ]
    shuffled_clauses = [
        {**clause, "source_receipt_ids": list(rotated[index])}
        for index, clause in enumerate(clauses)
    ]
    generic_clauses = [{
        "clause_id": f"GENERIC_{index}",
        "role": clause["role"],
        "untrusted_hypothesis": (
            "Use only live target evidence to identify missing information, test the selected "
            "proposal, recover from contradiction, and stop when official evidence is sufficient."
        ),
        "source_receipt_ids": [],
    } for index, clause in enumerate(clauses)]
    other_clauses = [
        dict(row) for row in (other_artifact.get("knowledge", {}).get("clauses") or [])
    ]
    raw_contexts = {
        "target_only": {
            "mode": "TARGET_ONLY_NEUTRAL_TEACHER",
            "clauses": [],
            "source_receipts": [],
        },
        "generic_reasoning": {
            "mode": "MATCHED_GENERIC_REASONING",
            "clauses": generic_clauses,
            "source_receipts": [],
        },
        "source_receipts_only": {
            "mode": "RAW_SOURCE_RECEIPTS_WITHOUT_INDUCED_KNOWLEDGE",
            "clauses": [],
            "source_receipts": evidence,
        },
        "authentic_weak_control_prior": {
            "mode": "DISCOVERY_PROVISIONAL_AUTHENTIC_WEAK_KNOWLEDGE",
            "clauses": clauses,
            "source_receipts": evidence,
        },
        "shuffled_evidence_prior": {
            "mode": "SHUFFLED_CLAUSE_TO_EVIDENCE_CONTROL",
            "clauses": shuffled_clauses,
            "source_receipts": evidence,
        },
        "other_game_abstain": {
            "mode": "OTHER_GAME_TEACHER_ABSTAINED_NO_CLAUSE",
            "clauses": other_clauses,
            "source_receipts": [],
        },
    }
    receipt_aliases: dict[str, dict[str, str]] = {}
    for condition, context in raw_contexts.items():
        receipt_ids = list(dict.fromkeys(
            str(row["receipt_id"]) for row in context["source_receipts"]
        ))
        real_to_alias = {
            receipt_id: f"S{index}" for index, receipt_id in enumerate(receipt_ids)
        }
        receipt_aliases[condition] = {
            alias: receipt_id for receipt_id, alias in real_to_alias.items()
        }

        def replace_receipt_ids(value: Any) -> Any:
            if isinstance(value, str):
                return real_to_alias.get(value, value)
            if isinstance(value, list):
                return [replace_receipt_ids(row) for row in value]
            if isinstance(value, dict):
                return {
                    key: replace_receipt_ids(row) for key, row in value.items()
                }
            return value

        raw_contexts[condition] = replace_receipt_ids(context)

    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    lengths = {
        condition: len(encoding.encode(json.dumps(value, sort_keys=True)))
        for condition, value in raw_contexts.items()
    }
    target_tokens = max(lengths.values()) + 32
    padded = {
        condition: _pad_json_to_exact_tokens(value, target_tokens)
        for condition, value in raw_contexts.items()
    }
    return padded, receipt_aliases, target_tokens


def _adaptation_summary(input_dir: Path, count: int) -> tuple[dict[str, Any], tuple[str, ...]]:
    episodes = []
    receipt_ids = []
    candidates = []
    for path in input_dir.glob("task_*.json"):
        match = re.fullmatch(r"task_(\d+)\.json", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    selected_paths = [path for _, path in sorted(candidates)[:count]]
    if len(selected_paths) != count:
        raise ValueError(
            f"requested {count} canonical adaptation artifacts, found "
            f"{len(selected_paths)}"
        )
    artifacts = []
    for path in selected_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw.get("records") or []
        artifacts.append({
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "task_id": raw.get("task_id"),
            "record_count": len(records),
            "collection_error": raw.get("error"),
        })
        receipt_ids.extend(str(row["transition"]["receipt_id"]) for row in records)
        def selected_proposal(record: Mapping[str, Any]) -> Mapping[str, Any]:
            proposal_set = record["proposal_set"]
            selected_id = proposal_set["selected_proposal_id"]
            return next(
                row for row in proposal_set["proposals"]
                if row["proposal_id"] == selected_id
            )

        episodes.append({
            "episode_alias": f"A{len(episodes)}",
            "official_success": bool((raw.get("metrics") or {}).get("official_success")),
            "steps": len(records),
            "task_goal": (
                records[0]["before"]["state"].get("task_goal") if records else None
            ),
            "transitions": [{
                "step": index,
                "selected_action": selected_proposal(row)["action"],
                "prediction": selected_proposal(row).get("prediction", ""),
                "after_observation": row["after"]["state"].get("observation"),
                "reward": row["reward"],
                "terminal": row["after"]["terminal"],
                "official_success": row["after"]["official_success"],
                "receipt_id": row["transition"]["receipt_id"],
            } for index, row in enumerate(records)],
        })
    return {"artifacts": artifacts, "episodes": episodes}, tuple(receipt_ids)


class PilotTeacherHarness:
    def __init__(
        self,
        backend,
        *,
        condition: str,
        matched_context_json: str,
        source_receipt_aliases: Mapping[str, str],
        target_hypothesis: Mapping[str, Any],
    ) -> None:
        self.backend = backend
        self.condition = condition
        self.matched_context_json = matched_context_json
        self.source_receipt_aliases = dict(source_receipt_aliases)
        self.target_hypothesis = dict(target_hypothesis)
        self.call_receipts: list[dict[str, Any]] = []
        self.protocol_violations: list[str] = []

    def _complete(self, role: str, system: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw = json.loads(self.backend.complete(role, system, payload))
        if not isinstance(raw, dict):
            raise ValueError("teacher response is not a JSON object")
        self.call_receipts.append({
            "role": role,
            "payload_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "usage": dict(self.backend.last_usage),
        })
        return raw

    def review_bindings(self, proposal, observation, bindings, history):
        payload = {
            "condition": self.condition,
            "matched_source_context_json": self.matched_context_json,
            "target_hypothesis": self.target_hypothesis,
            "already_selected_target_action": asdict(proposal),
            "target_observation": dict(observation.state),
            "recent_live_receipts": [asdict(row) for row in history[-6:]],
        }
        system = (
            "You are a no-action-authority Harness teacher. The Decision Agent already selected the "
            "target-native action. You may only test the supplied target_hypothesis's abstract control "
            "precondition and predicted observable transition. Do not assess task semantics, invent target "
            "commands, recommend an alternative action, or use general ALFWorld knowledge. If the control "
            "precondition is not visibly SATISFIED, you must ADMIT. REPLAN or ABSTAIN is allowed only when "
            "the precondition is SATISFIED and the selected proposal violates the supplied abstract control "
            "constraint. Source evidence uses short S-number aliases; live target receipts are not source "
            "evidence. Return exact JSON with verdict, reason, precondition_status "
            "(SATISFIED, NOT_SATISFIED, UNKNOWN), and expected_observable_outcome. For target_only, emit "
            "ADMIT; its teacher call is compute matching only. If target_hypothesis.abstain is true, also "
            "ADMIT with UNKNOWN. Never put a target action in any output field."
        )
        raw = self._complete("teacher_review", system, payload)
        if set(raw) != {
            "verdict", "reason", "precondition_status",
            "expected_observable_outcome",
        }:
            self.protocol_violations.append("teacher_review_invalid_schema")
            raw = {
                "verdict": "ADMIT",
                "reason": "Teacher schema invalid; fail-closed to no intervention.",
                "precondition_status": "UNKNOWN",
                "expected_observable_outcome": "",
            }
        precondition_status = str(raw.get("precondition_status"))
        if precondition_status not in {"SATISFIED", "NOT_SATISFIED", "UNKNOWN"}:
            self.protocol_violations.append(
                f"teacher_review_invalid_precondition:{precondition_status}"
            )
            precondition_status = "UNKNOWN"
        try:
            verdict = AdvisoryVerdict(str(raw["verdict"]))
        except ValueError:
            self.protocol_violations.append(
                f"teacher_review_invalid_verdict:{raw.get('verdict')}"
            )
            verdict = AdvisoryVerdict.ADMIT
        if verdict != AdvisoryVerdict.ADMIT and precondition_status != "SATISFIED":
            verdict = AdvisoryVerdict.ADMIT
        if self.condition == "target_only":
            verdict = AdvisoryVerdict.ADMIT
        citations = tuple(sorted(self.source_receipt_aliases.values()))
        return Advisory(
            verdict,
            str(raw.get("reason") or ""),
            citations,
            "CONTROL_PRECONDITION_REVIEW",
            (str(self.target_hypothesis.get("target_claim") or ""),),
            str(self.target_hypothesis.get("information_need") or ""),
            str(raw.get("expected_observable_outcome") or ""),
            str(self.target_hypothesis.get("failure_route") or ""),
            str(self.target_hypothesis.get("termination_test") or ""),
        )

    def verify_bindings(self, bindings, before, proposal, after, transition, history):
        payload = {
            "condition": self.condition,
            "matched_source_context_json": self.matched_context_json,
            "target_hypothesis": self.target_hypothesis,
            "recent_live_receipts_before_transition": [
                asdict(row) for row in history[-6:]
            ],
            "before_observation": dict(before.state),
            "already_executed_target_action": asdict(proposal),
            "after_observation": dict(after.state),
            "live_transition_receipt": asdict(transition),
        }
        system = (
            "Verify the provisional test-time hypothesis against the live target receipt. Never output an "
            "action. First require the hypothesis's full stated precondition to be visibly satisfied by the "
            "recent receipts before this transition. If it is absent, unknown, or only partially satisfied, "
            "return INCONCLUSIVE. SUPPORTED requires both a satisfied precondition and its predicted live "
            "outcome. REFUTED requires a satisfied precondition and a contradictory outcome; task "
            "incompletion alone is not contradiction. Return exact JSON with verdict "
            "(SUPPORTED, REFUTED, INCONCLUSIVE) and reason. For target_only return INCONCLUSIVE because "
            "there is no source hypothesis."
        )
        raw = self._complete("teacher_verify", system, payload)
        if set(raw) != {"verdict", "reason"}:
            self.protocol_violations.append("teacher_verify_invalid_schema")
            raw = {
                "verdict": "INCONCLUSIVE",
                "reason": "Teacher schema invalid; no evidence credited.",
            }
        try:
            verdict = EvidenceVerdict(str(raw["verdict"]))
        except ValueError:
            self.protocol_violations.append(
                f"teacher_verify_invalid_verdict:{raw.get('verdict')}"
            )
            verdict = EvidenceVerdict.INCONCLUSIVE
        if self.condition == "target_only":
            verdict = EvidenceVerdict.INCONCLUSIVE
        return tuple(
            BindingEvidence(
                binding.binding_id,
                transition.receipt_id,
                binding.verifier_id,
                verdict,
            )
            for binding in bindings
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adaptation-dir", type=Path, required=True)
    parser.add_argument("--authentic-artifact", type=Path, required=True)
    parser.add_argument("--other-artifact", type=Path, required=True)
    parser.add_argument("--source-evidence-dir", type=Path, action="append", required=True)
    parser.add_argument("--task-offset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--adaptation-count", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--alfworld-split", default="eval_out_of_distribution")
    parser.add_argument("--decision-model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--decision-max-tokens", type=int, default=1024)
    parser.add_argument("--teacher-model", default="gpt-5-mini")
    parser.add_argument(
        "--run-seed",
        type=int,
        help="Decision sampling seed; defaults to 81000 + task offset",
    )
    parser.add_argument(
        "--environment-seed",
        type=int,
        help="Environment seed; defaults to 81000 + task offset",
    )
    parser.add_argument(
        "--frozen-hypotheses-from",
        type=Path,
        help="reuse condition hypotheses from a completed canonical artifact",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    _load_keys(args.keys)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task_ids = manifest["cells"]["alfworld_valid_unseen"]["splits"]["qualification"]
    task_id = str(task_ids[args.task_offset])
    authentic = json.loads(args.authentic_artifact.read_text(encoding="utf-8"))
    other = json.loads(args.other_artifact.read_text(encoding="utf-8"))
    registry = _source_records(args.source_evidence_dir)
    contexts, receipt_aliases, context_tokens = _contexts(authentic, other, registry)
    adaptation, adaptation_receipt_ids = _adaptation_summary(
        args.adaptation_dir, args.adaptation_count
    )
    run_seed = (
        args.run_seed if args.run_seed is not None else 81000 + args.task_offset
    )
    environment_seed = (
        args.environment_seed
        if args.environment_seed is not None
        else 81000 + args.task_offset
    )
    frozen_hypotheses: dict[str, Mapping[str, Any]] = {}
    frozen_hypotheses_sha256 = None
    if args.frozen_hypotheses_from:
        frozen = json.loads(
            args.frozen_hypotheses_from.read_text(encoding="utf-8")
        )
        if frozen.get("task_id") != task_id:
            raise ValueError("frozen hypothesis artifact targets a different task")
        frozen_adaptation_hashes = [
            row["sha256"] for row in frozen.get("adaptation_artifacts") or []
        ]
        current_adaptation_hashes = [
            row["sha256"] for row in adaptation["artifacts"]
        ]
        if frozen_adaptation_hashes != current_adaptation_hashes:
            raise ValueError("frozen hypothesis adaptation lineage changed")
        expected_source_hashes = {
            "authentic": _sha256(args.authentic_artifact),
            "other": _sha256(args.other_artifact),
        }
        observed_source_hashes = {
            key: value["sha256"]
            for key, value in (frozen.get("source_artifacts") or {}).items()
        }
        if observed_source_hashes != expected_source_hashes:
            raise ValueError("frozen hypothesis source lineage changed")
        frozen_hypotheses = {
            str(row["condition"]): row
            for row in frozen.get("rows") or []
        }
        if not set(args.conditions) <= set(frozen_hypotheses):
            raise ValueError("frozen artifact is missing a requested condition")
        frozen_hypotheses_sha256 = _sha256(args.frozen_hypotheses_from)
    decision_backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            "https://openrouter.ai/api/v1",
            {"decision": args.decision_model},
            api_key_env="OPENROUTER_API_KEY",
            json_mode=True,
            request_overrides={
                "max_tokens": args.decision_max_tokens,
                "top_p": 1.0,
                "seed": run_seed,
                "reasoning": {"effort": "none", "exclude": True},
            },
        ),
        cache_path=args.cache_dir / "decision.json",
    )
    rows = []
    for condition in args.conditions:
        teacher_backend = MemoizedCompletionBackend(
            OpenAICompatibleBackend(
                "https://us.api.openai.com/v1",
                {
                    "teacher_initialize": args.teacher_model,
                    "teacher_review": args.teacher_model,
                    "teacher_verify": args.teacher_model,
                },
                api_key_env="OPENAI_API_KEY",
                json_mode=True,
                temperature=None,
                request_overrides={
                    "max_completion_tokens": 1800,
                    "reasoning_effort": "low",
                },
            ),
            cache_path=args.cache_dir / f"{condition}_teacher.json",
        )
        init_payload = {
            "condition": condition,
            "matched_source_context_json": contexts[condition],
            "target_adaptation_examples": adaptation,
        }
        init_system = (
            "Instantiate at most one weak, falsifiable target-time reasoning hypothesis from the supplied "
            "adaptation receipts and condition context. Source receipts are identified only by short S-number "
            "aliases; never cite a target receipt hash as source support. The hypothesis may describe only an "
            "abstract relation among information state, proposal history, observable state change, reward, "
            "verification, recovery, and termination. Do not output or recommend a target action, use target "
            "task semantics to invent a rule, or align source graph nodes/edges. Express its target test using "
            "only observation/reward/history fields actually present in the target adaptation receipts. Do not "
            "retain source-interface concepts such as pixels, frames, lives, health, playfields, or game "
            "objects unless those fields literally occur in target receipts. If source receipts are present "
            "and used, cite the minimal supporting aliases. Return exact JSON with abstain, target_claim, "
            "testable_prediction, information_need, failure_route, termination_test, and "
            "supporting_source_receipt_aliases. For target_only return abstain=true and no aliases. For an "
            "empty source context, abstain unless the condition explicitly supplies a generic control."
        )
        aliases_for_condition = receipt_aliases[condition]
        if frozen_hypotheses:
            frozen_row = frozen_hypotheses[condition]
            target_hypothesis = dict(frozen_row["target_hypothesis"])
            cited_aliases = {
                str(value)
                for value in (
                    target_hypothesis.get("supporting_source_receipt_aliases") or ()
                )
            }
            if not cited_aliases <= set(aliases_for_condition):
                raise ValueError("frozen hypothesis cites an unknown source alias")
            expected_receipt_ids = [
                aliases_for_condition[alias] for alias in sorted(cited_aliases)
            ]
            if (
                target_hypothesis.get("supporting_source_receipt_ids")
                != expected_receipt_ids
            ):
                raise ValueError("frozen hypothesis receipt lineage changed")
            init_raw = {
                key: value for key, value in target_hypothesis.items()
                if key != "supporting_source_receipt_ids"
            }
            init_usage = {"cache_hit": True, "frozen_initialization": True}
            init_validation_error = frozen_row.get(
                "initialization_validation_error"
            )
        else:
            init_raw = json.loads(
                teacher_backend.complete(
                    "teacher_initialize", init_system, init_payload
                )
            )
            init_usage = dict(teacher_backend.last_usage)
            if set(init_raw) != {
                "abstain", "target_claim", "testable_prediction",
                "information_need", "failure_route", "termination_test",
                "supporting_source_receipt_aliases",
            }:
                raise ValueError("teacher initialization has an invalid schema")
            cited_aliases = {
                str(value)
                for value in init_raw.get("supporting_source_receipt_aliases") or ()
            }
            init_validation_error = None
            if condition in {
                "target_only", "generic_reasoning", "other_game_abstain"
            }:
                cited_aliases = set()
            elif not cited_aliases <= set(aliases_for_condition):
                init_validation_error = "teacher fabricated a source receipt alias"
                cited_aliases &= set(aliases_for_condition)
            if aliases_for_condition and not cited_aliases:
                init_validation_error = (
                    init_validation_error or
                    "source hypothesis has no receipt support"
                )
                init_raw["abstain"] = True
            if condition in {"target_only", "other_game_abstain"}:
                init_raw["abstain"] = True
            target_hypothesis = {
                **init_raw,
                "supporting_source_receipt_aliases": sorted(cited_aliases),
                "supporting_source_receipt_ids": [
                    aliases_for_condition[alias]
                    for alias in sorted(cited_aliases)
                ],
            }
        binding_body = {
            "condition": condition,
            "target_hypothesis": target_hypothesis,
            "adaptation_receipts": adaptation_receipt_ids,
        }
        binding = BindingHypothesis(
            stable_hash(binding_body),
            stable_hash({"condition": condition, "context": contexts[condition]}),
            str(target_hypothesis.get("target_claim") or "neutral teacher control"),
            str(target_hypothesis.get("testable_prediction") or "no source prediction"),
            adaptation_receipt_ids,
            "live_target_receipt",
            transfer_object_kind=TransferObjectKind.WEAK_CONTROL_PRIOR,
        )
        harness = PilotTeacherHarness(
            teacher_backend,
            condition=condition,
            matched_context_json=contexts[condition],
            source_receipt_aliases={
                alias: aliases_for_condition[alias] for alias in cited_aliases
            },
            target_hypothesis=target_hypothesis,
        )
        decision = OpenAIJSONDecisionAgent(decision_backend)
        environment = ALFWorldTextEnvironment(
            config_path=str(args.config),
            data_path=str(args.data),
            split=args.alfworld_split,
            seed=environment_seed,
            game_id=task_id,
            max_steps=args.max_steps,
        )
        started = time.monotonic()
        result = None
        error = None
        try:
            result = TwoAgentRuntime(decision, harness).run(
                environment,
                "Follow the official ALFWorld task stated in the observation.",
                binding=binding,
                max_steps=args.max_steps,
                max_source_replans=args.max_steps,
            )
        except Exception as exc:
            result = getattr(exc, "partial_episode_result", None)
            error = f"{type(exc).__name__}:{exc}"
        finally:
            environment.close()
        rows.append({
            "condition": condition,
            "target_hypothesis": target_hypothesis,
            "initialization_validation_error": init_validation_error,
            "initial_state_hash": (
                result.records[0].transition.before_hash if result and result.records else None
            ),
            "metrics": asdict(measure_episode(result)) if result else None,
            "steps": len(result.records) if result else 0,
            "actions": [
                row.proposal_set.selected.action for row in result.records
            ] if result else [],
            "source_fallback_step": result.source_fallback_step if result else None,
            "source_replans": result.source_replans if result else 0,
            "source_failures": list(result.source_failures) if result else [],
            "teacher_protocol_violations": list(harness.protocol_violations),
            "binding_evidence": [
                asdict(row) for row in result.binding_evidence
            ] if result else [],
            "decision_calls": decision.call_receipts,
            "teacher_calls": [{
                "role": "teacher_initialize",
                "usage": init_usage,
                "response_sha256": stable_hash(init_raw),
            }, *harness.call_receipts],
            "error": error,
            "wall_time_s": time.monotonic() - started,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "schema_version": 1,
            "authority": "GPT5MINI_WEAK_KNOWLEDGE_TEACHER_PILOT",
            "claim_limit": "PILOT_DISCOVERY_PROVISIONAL_NOT_SOURCE_SUPPORTED",
            "task_offset": args.task_offset,
            "task_id": task_id,
            "run_seed": run_seed,
            "environment_seed": environment_seed,
            "frozen_hypotheses": {
                "path": str(args.frozen_hypotheses_from.resolve()),
                "sha256": frozen_hypotheses_sha256,
            } if args.frozen_hypotheses_from else None,
            "conditions": list(args.conditions),
            "matched_source_context_tokens": context_tokens,
            "adaptation_count": args.adaptation_count,
            "adaptation_artifacts": adaptation["artifacts"],
            "max_steps": args.max_steps,
            "decision_model": args.decision_model,
            "decision_max_tokens": args.decision_max_tokens,
            "teacher_model": args.teacher_model,
            "source_artifacts": {
                "authentic": {
                    "path": str(args.authentic_artifact.resolve()),
                    "sha256": _sha256(args.authentic_artifact),
                },
                "other": {
                    "path": str(args.other_artifact.resolve()),
                    "sha256": _sha256(args.other_artifact),
                },
            },
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "task_offset": args.task_offset,
            "condition": condition,
            "steps": len(result.records) if result else 0,
            "success": (
                asdict(measure_episode(result)).get("official_success")
                if result else None
            ),
            "error": error,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
