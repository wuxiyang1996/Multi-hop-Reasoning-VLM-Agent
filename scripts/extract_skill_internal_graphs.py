#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import runpy

from motif_transfer.frozen_motif_agent import OpenAICompatibleBackend
from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.skill_internal import (
    SourcePromptCondition,
    SkillInternalGraphAgent,
    audit_internal_graph,
    build_execution_sets,
    load_skill_hypotheses,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose receipt-grounded graphs inside a historical Phase-1 skill"
    )
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument(
        "--condition", choices=[row.value for row in SourcePromptCondition],
        default=SourcePromptCondition.AUTHENTIC.value,
    )
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument(
        "--replay-report", type=Path,
        help="Re-audit the frozen agent response in an earlier report without an API call",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.replay_report is None and (not args.endpoint or not args.model):
        parser.error("--endpoint and --model are required unless --replay-report is used")
    if args.key_file is not None and args.replay_report is None:
        value = runpy.run_path(str(args.key_file)).get(args.api_key_env)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"missing {args.api_key_env} in key file")
        os.environ["SKILL_INTERNAL_API_KEY"] = value
        api_key_env = "SKILL_INTERNAL_API_KEY"
    else:
        api_key_env = args.api_key_env

    episodes = import_native_source_batch(args.evidence_dir)
    games = {episode.game for episode in episodes}
    if len(games) != 1:
        raise ValueError(f"expected one game, found {sorted(games)}")
    game = next(iter(games))
    hypotheses = load_skill_hypotheses(args.bank)
    execution_sets = {
        row.skill_id: row for row in build_execution_sets(game, episodes, hypotheses)
    }
    if args.skill_id not in execution_sets:
        raise ValueError(f"skill not executed in evidence: {args.skill_id}")
    execution_set = execution_sets[args.skill_id]
    records_by_receipt = {
        record.transition.receipt_id: record
        for episode in episodes for record in episode.records
    }
    if args.replay_report is not None:
        frozen = json.loads(args.replay_report.read_text(encoding="utf-8"))

        class RecordedBackend:
            identity = {
                "backend": "recorded-response-replay",
                "source_report": str(args.replay_report),
                "source_response_hash": frozen["agent_call"]["response_hash"],
            }
            last_usage = {"replayed": True}

            def complete(self, role, system, payload):
                return json.dumps(frozen["agent_call"]["response"])

        backend = RecordedBackend()
    else:
        backend = OpenAICompatibleBackend(
            args.endpoint,
            {"skill_internal_graph": args.model},
            api_key_env=api_key_env,
            json_mode=True,
            temperature=None,
        )
    agent = SkillInternalGraphAgent(backend)
    candidates = agent.propose(
        execution_set, records_by_receipt, hypotheses.get(args.skill_id),
        condition=SourcePromptCondition(args.condition),
    )
    audits = [
        audit_internal_graph(candidate, execution_set, records_by_receipt)
        for candidate in candidates
    ]
    report = {
        "schema_version": 1,
        "authority": "UNTRUSTED_AGENT_PROPOSAL_WITH_MECHANICAL_RECEIPT_AUDIT",
        "prompt_condition": args.condition,
        "model_identity": dict(backend.identity),
        "agent_call": dict(agent.last_call),
        "intervention_requests": [asdict(row) for row in agent.intervention_requests],
        "execution_set": asdict(execution_set),
        "candidates": [asdict(row) for row in candidates],
        "audits": [asdict(row) for row in audits],
        "warning": (
            "Accepted means discovery receipts support the proposed topology; "
            "it is not SOURCE_SUPPORTED until qualification, interventions, and held-out tests pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidates": len(candidates),
        "audit_accepted": sum(row.accepted for row in audits),
        "nontrivial": sum(row.nontrivial for row in audits),
        "backbone_eligible": sum(row.backbone_eligible for row in audits),
        "intervention_requests": len(agent.intervention_requests),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
