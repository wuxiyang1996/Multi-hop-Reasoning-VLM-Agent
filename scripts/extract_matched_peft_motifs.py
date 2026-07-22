#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.frozen_motif_agent import (
    FrozenJSONMotifAgent,
    PromptCondition,
    TransformersPeftBackend,
)
from motif_transfer.harness import DeterministicHarness
from motif_transfer.instrumented_import import import_native_source_batch


def evaluate(backend, episodes, condition: PromptCondition, use_adapter: bool):
    backend.use_adapter_by_role = {"segment": use_adapter}
    agent = FrozenJSONMotifAgent(backend, condition=condition)
    harness = DeterministicHarness()
    rows = []
    for episode in episodes:
        receipt_map = {
            row.transition.receipt_id: row.transition for row in episode.records
        } | {row.receipt_id: row for row in episode.replay_forks}
        try:
            candidates = agent.propose_source_motifs(episode.records, episode.replay_forks)
            audits = [harness.audit_motif(candidate, receipt_map) for candidate in candidates]
            error = None
        except Exception as exc:
            candidates, audits = (), []
            error = f"{type(exc).__name__}:{exc}"
        raw_response = backend.last_completion
        rows.append({
            "episode_id": episode.episode_id,
            "game": episode.game,
            "records": len(episode.records),
            "replay_forks": len(episode.replay_forks),
            "import_gaps": list(episode.gaps),
            "candidates": [asdict(row) for row in candidates],
            "audits": [asdict(row) for row in audits],
            "model_error": error,
            "model_response_sha256": hashlib.sha256(raw_response.encode()).hexdigest(),
            "model_response_preview": raw_response[:500],
        })
    return {
        "schema_version": 1,
        "model_identity": dict(backend.identity),
        "prompt_condition": condition.value,
        "authority": "UNTRUSTED_MOTIF_PROPOSAL_ONLY",
        "episodes": rows,
        "totals": {
            "episodes": len(rows),
            "candidates": sum(len(row["candidates"]) for row in rows),
            "accepted": sum(audit["accepted"] for row in rows for audit in row["audits"]),
            "model_errors": sum(row["model_error"] is not None for row in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched base/PEFT frozen motif extraction")
    parser.add_argument("evidence_dir")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--segment-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    episodes = import_native_source_batch(args.evidence_dir)
    backend = TransformersPeftBackend(
        args.base_model,
        args.segment_adapter,
        use_adapter_by_role={"segment": False},
        max_new_tokens=args.max_new_tokens,
    )
    treatments = (
        ("base_authentic", PromptCondition.AUTHENTIC, False),
        ("coevolved_authentic", PromptCondition.AUTHENTIC, True),
        ("coevolved_receipt_only", PromptCondition.RECEIPT_ONLY, True),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, condition, use_adapter in treatments:
        report = evaluate(backend, episodes, condition, use_adapter)
        (output_dir / f"{name}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        summary[name] = report["totals"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
