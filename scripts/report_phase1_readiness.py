#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
from collections import Counter
import json
from pathlib import Path

from motif_transfer.instrumented_import import import_instrumented_batch, import_native_source_batch
from motif_transfer.phase1_assets import audit_evidence_batch, discover_evidence_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile one fail-closed Phase-1 readiness report")
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    condition_games: dict[str, set[str]] = {}
    for batch in discover_evidence_batches(root):
        audit = audit_evidence_batch(batch)
        relative = batch.relative_to(root)
        condition = relative.parts[0] if len(relative.parts) >= 3 else "unknown"
        imported = (
            import_native_source_batch(batch)
            if audit.protocol_profile == "source_agent"
            else import_instrumented_batch(batch)
        )
        records = [record for episode in imported for record in episode.records]
        import_gap_count = sum(len(episode.gaps) for episode in imported)
        if audit.protocol_profile == "source_agent":
            profile_metrics = {
                "selected_skill_id_counts": dict(sorted(Counter(
                    record.selected_skill_id or "NONE" for record in records
                ).items())),
                "action_origin_counts": dict(sorted(Counter(
                    record.action_origin for record in records
                ).items())),
                "policy_adapter_counts": dict(sorted(Counter(
                    record.policy_adapter for record in records
                ).items())),
                "reward_sign_counts": dict(sorted(Counter(
                    "POSITIVE" if record.reward > 0 else "NEGATIVE" if record.reward < 0 else "ZERO"
                    for record in records
                ).items())),
            }
        else:
            profile_metrics = {
                "cycles_with_multiple_action_proposals": sum(
                    len(record.proposal_set.proposals) > 1 for record in records
                ),
                "proposal_cardinality_counts": dict(sorted(Counter(
                    len(record.proposal_set.proposals) for record in records
                ).items())),
                "selected_ordinal_counts": dict(sorted(Counter(
                    next(
                        index for index, proposal in enumerate(record.proposal_set.proposals)
                        if proposal.proposal_id == record.proposal_set.selected_proposal_id
                    )
                    for record in records
                ).items())),
                "unique_native_action_counts": dict(sorted(Counter(
                    len({proposal.action for proposal in record.proposal_set.proposals})
                    for record in records
                ).items())),
                "post_verdict_counts": dict(sorted(Counter(
                    record.assessment.verdict.value for record in records
                ).items())),
                "continuation_decision_counts": dict(sorted(Counter(
                    record.assessment.continuation.value for record in records
                ).items())),
            }
        condition_games.setdefault(condition, set()).add(audit.game)
        rows.append(
            {
                "condition": condition,
                "audit": asdict(audit),
                "treatment_integrity": (
                    (
                        audit.native_action_policy_steps == audit.steps
                        and import_gap_count == 0
                        and (
                            audit.selected_skill_receipts == audit.native_action_policy_steps
                            if condition == "authentic_skill_loaded"
                            else audit.selected_skill_receipts == 0
                        )
                    ) if audit.protocol_profile == "source_agent" else (
                        audit.action_adapter_grounded_proposal_sets == audit.action_proposal_sets
                        and (
                            audit.skill_conditioned_proposal_sets == audit.action_proposal_sets
                            if condition == "authentic_skill_loaded"
                            else audit.skill_conditioned_proposal_sets == 0
                        )
                    )
                ),
                "imported_records": len(records),
                "imported_replay_forks": sum(len(episode.replay_forks) for episode in imported),
                "import_gaps": import_gap_count,
                "episode_total_rewards": [episode.total_reward for episode in imported],
                "initial_state_hashes": sorted(
                    episode.records[0].transition.before_hash
                    for episode in imported if episode.records
                ),
                **profile_metrics,
            }
        )
    required_conditions = {"authentic_skill_loaded", "skill_disabled"}
    shared_games = (
        set.intersection(*(condition_games.get(name, set()) for name in required_conditions))
        if all(condition_games.get(name) for name in required_conditions)
        else set()
    )
    report = {
        "schema_version": 1,
        "root": str(root),
        "batches": rows,
        "conditions": {name: sorted(games) for name, games in sorted(condition_games.items())},
        "shared_skill_on_off_games": sorted(shared_games),
        "gates": {
            "paired_initial_states_match": all(
                sorted(
                    row["initial_state_hashes"]
                    for row in rows
                    if row["condition"] == condition and row["audit"]["game"] == game
                ) == sorted(
                    row["initial_state_hashes"]
                    for row in rows
                    if row["condition"] == other and row["audit"]["game"] == game
                )
                for game in shared_games
                for condition, other in [("authentic_skill_loaded", "skill_disabled")]
            ),
            "six_games_have_skill_on_off": len(shared_games) == 6,
            "all_batches_motif_ready": bool(rows) and all(row["audit"]["motif_ready"] for row in rows),
            "all_batches_import_nonempty": bool(rows) and all(row["imported_records"] > 0 for row in rows),
            "all_batches_treatment_integrity": bool(rows) and all(
                row["treatment_integrity"] for row in rows
            ),
            "source_qualification_can_start": (
                len(shared_games) == 6
                and bool(rows)
                and all(row["audit"]["motif_ready"] for row in rows)
                and all(row["imported_records"] > 0 for row in rows)
                and all(row["treatment_integrity"] for row in rows)
                and all(
                    next(
                        row["initial_state_hashes"] for row in rows
                        if row["condition"] == "authentic_skill_loaded"
                        and row["audit"]["game"] == game
                    ) == next(
                        row["initial_state_hashes"] for row in rows
                        if row["condition"] == "skill_disabled"
                        and row["audit"]["game"] == game
                    )
                    for game in shared_games
                )
            ),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
