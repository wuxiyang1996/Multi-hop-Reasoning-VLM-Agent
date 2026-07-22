#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from motif_transfer.frozen_motif_agent import TransformersPeftBackend
from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.source_ranking import (
    SourceSkillRanker,
    load_source_skill_bank,
    segment_native_policy,
)


def evaluate(backend, episodes, candidates, *, use_adapter: bool):
    backend.use_adapter_by_role = {"segment": use_adapter}
    ranker = SourceSkillRanker(backend)
    rows = []
    for episode in episodes:
        for segment, records in segment_native_policy(episode.records):
            try:
                receipt = ranker.rank(segment, records, candidates)
                error = None
            except Exception as exc:
                receipt = None
                error = f"{type(exc).__name__}:{exc}"
            selected_ids = sorted({
                row.selected_skill_id for row in records if row.selected_skill_id
            })
            rows.append({
                "episode_id": episode.episode_id,
                "game": episode.game,
                "segment": asdict(segment),
                "selected_skill_ids_untrusted": selected_ids,
                "ranking_receipt": asdict(receipt) if receipt else None,
                "model_error": error,
                "model_response_preview": backend.last_completion[:500],
            })
    valid = [row for row in rows if row["ranking_receipt"] is not None]
    comparable = [
        row for row in valid if len(row["selected_skill_ids_untrusted"]) == 1
    ]
    top1_matches = sum(
        row["ranking_receipt"]["ranking"][0]
        == row["selected_skill_ids_untrusted"][0]
        for row in comparable
    )
    return {
        "schema_version": 1,
        "authority": "UNTRUSTED_NATIVE_SKILL_RANKING_ONLY",
        "model_identity": dict(backend.identity),
        "adapter_enabled": use_adapter,
        "rows": rows,
        "totals": {
            "segments": len(rows),
            "valid_rankings": len(valid),
            "model_errors": len(rows) - len(valid),
            "selected_skill_comparable": len(comparable),
            "selected_skill_top1_matches": top1_matches,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the old segment head on its native skill-ranking task"
    )
    parser.add_argument("evidence_dir")
    parser.add_argument("--skill-bank", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--segment-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    episodes = import_native_source_batch(args.evidence_dir)
    candidates = load_source_skill_bank(args.skill_bank)
    backend = TransformersPeftBackend(
        args.base_model,
        args.segment_adapter,
        use_adapter_by_role={"segment": False},
        max_new_tokens=args.max_new_tokens,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, use_adapter in (("base", False), ("coevolved_segment", True)):
        report = evaluate(
            backend, episodes, candidates, use_adapter=use_adapter,
        )
        (output_dir / f"{name}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary[name] = report["totals"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
