#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from motif_transfer.frozen_motif_agent import (
    FrozenJSONMotifAgent,
    OpenAICompatibleBackend,
    PromptCondition,
)
from motif_transfer.harness import DeterministicHarness
from motif_transfer.instrumented_import import import_instrumented_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen-model motif proposals from complete receipts")
    parser.add_argument("evidence_dir")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--segment-model", required=True)
    parser.add_argument("--condition", choices=[row.value for row in PromptCondition], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    backend = OpenAICompatibleBackend(
        args.endpoint,
        {"segment": args.segment_model, "binding": args.segment_model, "review": args.segment_model},
    )
    agent = FrozenJSONMotifAgent(backend, condition=PromptCondition(args.condition))
    harness = DeterministicHarness()
    episode_results = []
    for episode in import_instrumented_batch(args.evidence_dir):
        receipt_map = {
            row.transition.receipt_id: row.transition for row in episode.records
        } | {row.receipt_id: row for row in episode.replay_forks}
        try:
            candidates = agent.propose_motifs(episode.records, episode.replay_forks)
            audits = [harness.audit_motif(candidate, receipt_map) for candidate in candidates]
            error = None
        except Exception as exc:
            candidates = ()
            audits = []
            error = f"{type(exc).__name__}:{exc}"
        episode_results.append(
            {
                "episode_id": episode.episode_id,
                "game": episode.game,
                "records": len(episode.records),
                "replay_forks": len(episode.replay_forks),
                "import_gaps": list(episode.gaps),
                "candidates": [asdict(candidate) for candidate in candidates],
                "audits": [asdict(audit) for audit in audits],
                "model_error": error,
            }
        )
    report = {
        "schema_version": 1,
        "model_identity": dict(backend.identity),
        "prompt_condition": args.condition,
        "authority": "UNTRUSTED_MOTIF_PROPOSAL_ONLY",
        "episodes": episode_results,
        "totals": {
            "episodes": len(episode_results),
            "candidates": sum(len(row["candidates"]) for row in episode_results),
            "accepted": sum(
                audit["accepted"] for row in episode_results for audit in row["audits"]
            ),
            "model_errors": sum(row["model_error"] is not None for row in episode_results),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
