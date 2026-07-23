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
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import Lifecycle  # noqa: E402
from motif_transfer.control_priors import (  # noqa: E402
    ControlKnowledgeRole,
    ReceiptGroundedClause,
    ReceiptGroundedKnowledge,
    audit_receipt_grounded_knowledge,
    knowledge_to_mapping,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.instrumented_import import import_native_source_batch  # noqa: E402
from motif_transfer.source_execution_motifs import build_execution_traces  # noqa: E402


SYSTEM = """You are a source-knowledge induction teacher with no environment action authority.
Use only the supplied DISCOVERY source receipts. Propose weak, falsifiable knowledge about
control regularities, failure signatures, verification routines, or applicability boundaries.
Do not propose a transferable skill, policy, source-to-target mapping, target action, predicate
ontology, or complete graph alignment. Natural-language statements are untrusted hypotheses.
Every clause must be independent of source game names, source action names, skill names, and the
literal signature vocabulary FIRST/SAME/CHANGED/ZERO/POSITIVE/NEGATIVE. A clause that merely
restates reward signs, execution lengths, or action-repeat patterns is invalid. Describe an
information-control relation instead: what observable evidence is missing, what future observation
would test a proposal, what contradiction should trigger recovery/source-off, or what evidence
permits termination. SOURCE_SYMBOL is a redaction marker and must not appear in a clause. Do not
prescribe an action.

Return JSON only:
{
  "abstain": false,
  "description": "...",
  "clauses": [{
    "role": "CONTROL_REGULARITY|FAILURE_SIGNATURE|VERIFICATION_ROUTINE|APPLICABILITY_BOUNDARY",
    "untrusted_hypothesis": "...",
    "source_receipt_ids": ["...", "..."]
  }]
}

Every clause must cite supplied transition receipt aliases from at least two different discovery episodes
in each of at least two different source games.
Prefer knowledge that makes a future observable prediction or specifies when evidence should
disable a hypothesis. Do not cite qualification or held-out data because none is supplied. If the
receipts do not support non-trivial knowledge, return abstain=true and an empty clause list."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_payload(
    evidence_dirs: list[Path],
    *,
    include_all_episodes: bool,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str], set[str]]:
    receipt_to_episode: dict[str, str] = {}
    receipt_to_game: dict[str, str] = {}
    source_action_tokens: set[str] = set()
    compact = []
    hint_flags = []
    for game_index, evidence_dir in enumerate(evidence_dirs):
        episodes = import_native_source_batch(evidence_dir)
        traces = tuple(
            row for row in build_execution_traces(episodes)
            if include_all_episodes or row.split == "discovery"
        )
        episode_by_id = {row.episode_id: row for row in episodes}
        game_alias = f"G{game_index}"
        game_action_tokens = sorted({
            str(action)
            for episode in episodes
            for row in episode.records
            for action in row.before.native_actions
        }, key=len, reverse=True)
        game_names = sorted({
            str(row.before.state.get("structured_state", {}).get("display_name", ""))
            for episode in episodes
            for row in episode.records[:1]
        } - {""}, key=len, reverse=True)

        def sanitize_reasoning(value: str) -> str:
            result = value
            for token in (*game_names, *game_action_tokens):
                result = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
                    "SOURCE_SYMBOL",
                    result,
                    flags=re.IGNORECASE,
                )
            return result

        for trace_index, trace in enumerate(traces):
            episode = episode_by_id[trace.episode_id]
            if not episode.records:
                continue
            record_by_receipt = {
                row.transition.receipt_id: row for row in episode.records
            }
            source_action_tokens.update(
                str(action).upper()
                for row in episode.records
                for action in row.before.native_actions
                if len(str(action)) > 1
            )
            executions = []
            for execution in trace.executions:
                for receipt_id in execution.transition_receipt_ids:
                    receipt_to_episode[receipt_id] = trace.episode_id
                    receipt_to_game[receipt_id] = game_alias
                reasoning = []
                for receipt_id in execution.transition_receipt_ids:
                    value = sanitize_reasoning(
                        record_by_receipt[receipt_id].action_reasoning.strip()
                    )
                    if value and value not in reasoning:
                        reasoning.append(value[:280])
                signature = asdict(execution.signature)
                signature["action_repeat_sequence"] = [
                    "FIRST" if value == "START" else value
                    for value in signature["action_repeat_sequence"]
                ]
                executions.append({
                    "execution_id": execution.execution_id,
                    "transition_receipt_ids": list(execution.transition_receipt_ids),
                    "signature": signature,
                    "untrusted_reasoning_samples": reasoning[:2],
                })
            compact.append({
                "game_alias": game_alias,
                "episode_alias": f"{game_alias}_E{trace_index}",
                "episode_receipt_count": len(episode.records),
                "official_return": sum(row.reward for row in episode.records),
                "terminal": bool(episode.records[-1].after.terminal),
                "executions": executions,
            })
        manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
        hint_flags.append(manifest["metadata"].get("human_policy_hints"))
    return {
        "authority": "DISCOVERY_ONLY_SOURCE_RECEIPTS",
        "game_count": len(evidence_dirs),
        "human_policy_hints": hint_flags,
        "episodes": compact,
    }, receipt_to_episode, receipt_to_game, source_action_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use GPT-5-mini to propose discovery-only receipt-grounded source knowledge."
    )
    parser.add_argument("--evidence-dir", type=Path, required=True, action="append")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--endpoint", default="https://us.api.openai.com/v1")
    parser.add_argument(
        "--include-all-source-episodes",
        action="store_true",
        help="Control-only diagnostic for legacy source data; never use for a source-support claim.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    keys = runpy.run_path(str(args.keys))
    key = keys.get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key:
        raise RuntimeError("missing OPENAI_API_KEY")
    os.environ["SOURCE_KNOWLEDGE_OPENAI_API_KEY"] = key

    payload, receipt_to_episode, receipt_to_game, source_action_tokens = _compact_payload(
        args.evidence_dir,
        include_all_episodes=args.include_all_source_episodes,
    )
    receipt_aliases = {
        receipt_id: f"R{index}"
        for index, receipt_id in enumerate(sorted(receipt_to_episode))
    }
    receipt_ids_by_alias = {alias: receipt_id for receipt_id, alias in receipt_aliases.items()}
    for episode in payload["episodes"]:
        for execution in episode["executions"]:
            execution["transition_receipt_ids"] = [
                receipt_aliases[receipt_id]
                for receipt_id in execution["transition_receipt_ids"]
            ]
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.endpoint,
            {"source_knowledge": args.model},
            api_key_env="SOURCE_KNOWLEDGE_OPENAI_API_KEY",
            json_mode=True,
            temperature=None,
            request_overrides={
                "max_completion_tokens": 8000,
                "reasoning_effort": "low",
            },
        ),
        cache_path=args.cache,
    )
    raw = json.loads(backend.complete("source_knowledge", SYSTEM, payload))
    grounded_response = {
        **raw,
        "clauses": [{
            **row,
            "source_receipt_ids": [
                receipt_ids_by_alias.get(str(value), str(value))
                for value in row.get("source_receipt_ids") or ()
            ],
        } for row in raw.get("clauses") or ()],
    }
    clauses = tuple(
        ReceiptGroundedClause.create(
            ControlKnowledgeRole(str(row["role"])),
            str(row.get("untrusted_hypothesis") or ""),
            tuple(str(value) for value in row.get("source_receipt_ids") or ()),
        )
        for row in grounded_response.get("clauses") or ()
    )
    knowledge = ReceiptGroundedKnowledge.create(
        tuple(sorted(set(receipt_to_episode.values()))),
        clauses,
        status=Lifecycle.CANDIDATE,
    )
    audit = audit_receipt_grounded_knowledge(
        knowledge,
        receipt_to_episode=receipt_to_episode,
        require_source_supported=False,
        minimum_episode_support=2,
    )
    forbidden_protocol_tokens = {
        "FIRST", "SAME", "CHANGED", "ZERO", "POSITIVE", "NEGATIVE", "SOURCE_SYMBOL",
    }
    semantic_failures = []
    game_support_by_clause = {}
    for clause in clauses:
        words = {
            word.strip(".,:;!?()[]{}'\"").upper()
            for word in clause.untrusted_hypothesis.split()
        }
        leaked_actions = sorted(words & source_action_tokens)
        leaked_protocol = sorted(words & forbidden_protocol_tokens)
        game_support = {
            receipt_to_game[receipt_id]
            for receipt_id in clause.source_receipt_ids
            if receipt_id in receipt_to_game
        }
        game_support_by_clause[clause.clause_id] = len(game_support)
        if leaked_actions:
            semantic_failures.append(
                f"SOURCE_ACTION_TOKEN_LEAKAGE:{clause.clause_id}:{','.join(leaked_actions)}"
            )
        if leaked_protocol:
            semantic_failures.append(
                f"SURFACE_SIGNATURE_RESTATEMENT:{clause.clause_id}:{','.join(leaked_protocol)}"
            )
        if len(game_support) < 2:
            semantic_failures.append(
                f"INSUFFICIENT_CROSS_GAME_SUPPORT:{clause.clause_id}"
            )
    result = {
        "schema_version": 1,
        "authority": "UNTRUSTED_GPT_SOURCE_KNOWLEDGE_PLUS_MECHANICAL_DISCOVERY_AUDIT",
        "claim_limit": "DISCOVERY_PROVISIONAL_NOT_SOURCE_SUPPORTED",
        "discovery_only": not args.include_all_source_episodes,
        "qualification_or_held_out_used": False,
        "evidence_dirs": [str(path.resolve()) for path in args.evidence_dir],
        "evidence_files_sha256": [{
            "directory": str(path.resolve()),
            "files": {
                name: _sha256(path / name)
                for name in ("manifest.json", "events.jsonl", "episodes.jsonl")
            },
        } for path in args.evidence_dir],
        "model_identity": dict(backend.identity),
        "agent_response": raw,
        "grounded_agent_response": grounded_response,
        "receipt_alias_registry_sha256": hashlib.sha256(
            json.dumps(receipt_ids_by_alias, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "knowledge": knowledge_to_mapping(knowledge),
        "audit": asdict(audit),
        "pilot_semantic_audit": {
            "accepted": audit.accepted and not semantic_failures,
            "failure_codes": semantic_failures,
            "game_support_by_clause": game_support_by_clause,
            "forbidden_source_action_tokens_sha256": hashlib.sha256(
                json.dumps(sorted(source_action_tokens)).encode("utf-8")
            ).hexdigest(),
        },
        "usage": dict(backend.last_usage),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "accepted_discovery_provenance": audit.accepted,
        "accepted_pilot_semantics": audit.accepted and not semantic_failures,
        "clauses": len(clauses),
        "failure_codes": list(audit.failure_codes) + semantic_failures,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
