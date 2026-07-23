#!/usr/bin/env python3
"""Mechanically audit whether source logs support a reasoning-backbone claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.reasoning_event_log import reasoning_event_log_from_dict, validate_reasoning_protocol


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, action="append", required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = []
    totals = Counter()
    for root in args.source_batch:
        manifest = json.loads((root / "manifest.json").read_text())
        for name, receipt in manifest["files"].items():
            if _file_hash(root / name) != receipt["sha256"]:
                raise ValueError(f"batch file hash mismatch: {root / name}")
        by_episode = defaultdict(list)
        for line in (root / "events.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                by_episode[str(row["episode_id"])].append(row)
        for episode_id, rows in sorted(by_episode.items()):
            rows.sort(key=lambda row: int(row["sequence"]))
            log = {"schema_version": 1, "episode_id": episode_id, "events": rows}
            log["log_sha256"] = _hash(log)
            events = reasoning_event_log_from_dict(log)
            failures = list(validate_reasoning_protocol(events, profile="source_agent"))
            parsed = [row for row in rows if row["kind"] == "PARSED_DECISION"]
            proposals = [row for row in rows if row["kind"] == "AGENT_PROPOSAL_SET"]
            action_proposals = [
                row for row in rows if row["kind"] == "AGENT_ACTION_PROPOSAL_SET"
            ]
            post_verdict_rows = [
                row for row in rows if row["kind"] == "AGENT_POST_TRANSITION_VERDICT"
            ]
            decisions = [row for row in rows if row["kind"] == "AGENT_DECISION"]
            stops = [row for row in rows if row["kind"] == "OFFICIAL_STOP"]
            legacy_action_proposal_steps = sum(
                row["payload"].get("claim_boundary") == "ACTION_PROPOSALS"
                for row in proposals
            )
            action_proposal_steps = legacy_action_proposal_steps + sum(
                bool(row["payload"].get("schema_valid")) for row in action_proposals
            )
            post_transition_verdicts = sum(
                bool(row["payload"].get("schema_valid")) for row in post_verdict_rows
            )
            legacy_replan_support = sum(
                bool(row["payload"].get("agent_protocol_supports_replan_abstain"))
                for row in parsed
            )
            explicit_replan_support = legacy_replan_support + post_transition_verdicts
            official_available = bool(stops) and bool(
                stops[-1]["payload"].get("official_success_evaluator_available")
            )
            native_outcome_available = bool(stops) and all(
                key in stops[-1]["payload"]
                for key in ("total_reward", "terminated", "truncated", "native_final_info")
            )
            row = {
                "batch": str(root), "episode_id": episode_id,
                "protocol_failures": failures, "n_steps": len(decisions),
                "n_agent_origin_steps": sum(
                    bool(item["payload"].get("can_support_agent_reasoning_induction"))
                    for item in decisions
                ),
                "n_action_proposal_steps": action_proposal_steps,
                "n_post_transition_agent_verdicts": post_transition_verdicts,
                "n_explicit_replan_abstain_steps": explicit_replan_support,
                "official_outcome_available": official_available,
                "native_environment_outcome_available": native_outcome_available,
            }
            episodes.append(row)
            totals.update({key: value for key, value in row.items() if key.startswith("n_")})
            totals["episodes"] += 1
            totals["episodes_with_official_outcome"] += int(official_available)
            totals["episodes_with_native_environment_outcome"] += int(
                native_outcome_available
            )
            totals["episodes_protocol_complete"] += int(not failures)
    source = json.loads(args.source_artifact.read_text())
    unsigned_source = dict(source)
    claimed_source_hash = unsigned_source.pop("artifact_sha256", None)
    if not claimed_source_hash or _hash(unsigned_source) != claimed_source_hash:
        raise ValueError("source artifact hash mismatch")
    traces = [row.get("source_reasoning_trace") or {} for row in source.get("programs") or ()]
    for trace in traces:
        unsigned_trace = dict(trace)
        claimed = unsigned_trace.pop("trace_sha256", None)
        if not claimed or _hash(unsigned_trace) != claimed:
            raise ValueError("source reasoning trace hash mismatch")
    gates = {
        "complete_event_protocol": totals["episodes_protocol_complete"] == totals["episodes"],
        "action_proposals_observed_every_step": totals["n_action_proposal_steps"] == totals["n_steps"],
        "post_transition_agent_verdict_observed_every_step": totals["n_post_transition_agent_verdicts"] == totals["n_steps"],
        "explicit_replan_abstain_supported_every_step": totals["n_explicit_replan_abstain_steps"] == totals["n_steps"],
        "native_environment_outcome_available_every_episode": (
            totals["episodes_with_native_environment_outcome"] == totals["episodes"]
        ),
        "reasoning_receipts_cover_compiled_program_steps": all(
            len(trace.get("steps") or ()) > 0 for trace in traces
        ),
    }
    output = {
        "schema_version": 1,
        "source_artifact_sha256": claimed_source_hash,
        "source_batches": [str(path) for path in args.source_batch],
        "totals": dict(totals), "gates": gates, "episodes": episodes,
        "reasoning_backbone_ready": all(gates.values()),
        "diagnostics": {
            "official_success_evaluator_available_every_episode": (
                totals["episodes_with_official_outcome"] == totals["episodes"]
            ),
        },
        "claim_limit": (
            "Protocol completeness proves logging integrity only. Agent reasoning text "
            "remains untrusted, and this audit does not establish transfer value."
        ),
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "totals": output["totals"], "gates": gates,
        "reasoning_backbone_ready": output["reasoning_backbone_ready"],
        "artifact_sha256": output["artifact_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
