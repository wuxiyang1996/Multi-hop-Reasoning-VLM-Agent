#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import (  # noqa: E402
    Advisory,
    AdvisoryVerdict,
    ContinuationDecision,
    DecisionCycleReceipt,
    DecisionCycleRecord,
    DecisionProposal,
    DecisionProposalSet,
    EvidenceVerdict,
    Observation,
    PostTransitionAssessment,
    TransitionReceipt,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.target_native_motifs import (  # noqa: E402
    TargetEpisodeView,
    audit_target_native_motif,
    target_motif_from_agent_response,
)


SYSTEM = """You are a Motif/Harness Agent proposing one target-native execution graph from
adaptation receipts only. You cannot choose actions and must not infer hidden state. Find a
non-trivial recurrent control structure such as attempt-observe-branch-recover/continue, not a
taxonomy of action names, task names, rewards, or span lengths.

Return JSON only with:
description;
spans: [{span_id, episode_id, start_offset, end_offset, intent}];
nodes: [{node_id, span_ids, role}];
edges: [{source, target}].

Every span must be contiguous and cite exact offsets from the supplied episode. Each node and each
edge must recur in at least two different episodes. Spans cannot overlap. The graph must have at
least two nodes and contain a branch or cycle. If the receipts do not support this, return empty
spans/nodes/edges and explain why in description. Natural-language roles are untrusted hypotheses;
the deterministic auditor decides acceptance. Respect each episode's record_count exactly. Assign
each span_id to exactly one node; if one interval contains multiple roles, split it into
non-overlapping spans instead of reusing the same span."""


def _observation(raw: dict[str, Any]) -> Observation:
    return Observation(
        dict(raw["state"]),
        tuple(map(str, raw["native_actions"])),
        bool(raw["terminal"]),
        bool(raw["official_success"]),
        float(raw["score"]),
    )


def _record(raw: dict[str, Any]) -> DecisionCycleRecord:
    proposal_set = DecisionProposalSet(
        str(raw["proposal_set"]["proposal_set_id"]),
        tuple(DecisionProposal(
            str(row["proposal_id"]),
            str(row["action"]),
            str(row.get("prediction", "")),
            str(row.get("rationale", "")),
            str(row.get("agent_id", "decision-agent")),
        ) for row in raw["proposal_set"]["proposals"]),
        str(raw["proposal_set"]["selected_proposal_id"]),
    )
    advisory = Advisory(
        AdvisoryVerdict(raw["advisory"]["verdict"]),
        str(raw["advisory"]["reason"]),
        tuple(map(str, raw["advisory"].get("evidence_receipt_ids", []))),
        str(raw["advisory"].get("current_role", "")),
        tuple(map(str, raw["advisory"].get("open_hypotheses", []))),
        str(raw["advisory"].get("information_need", "")),
        str(raw["advisory"].get("expected_transition", "")),
        str(raw["advisory"].get("failure_route", "")),
        str(raw["advisory"].get("termination_test", "")),
    )
    assessment = PostTransitionAssessment(
        EvidenceVerdict(raw["assessment"]["verdict"]),
        ContinuationDecision(raw["assessment"]["continuation"]),
        str(raw["assessment"].get("reason", "")),
    )
    record = DecisionCycleRecord(
        _observation(raw["before"]),
        proposal_set,
        advisory,
        _observation(raw["after"]),
        float(raw["reward"]),
        TransitionReceipt(**raw["transition"]),
        assessment,
        DecisionCycleReceipt(**raw["receipt"]),
    )
    if not record.validate():
        raise ValueError("target Decision cycle failed receipt validation")
    return record


def _payload(episodes: list[TargetEpisodeView]) -> dict[str, Any]:
    aliases = {
        episode.episode_id: f"E{index}"
        for index, episode in enumerate(episodes)
    }
    return {
        "authority": "ADAPTATION_RECEIPTS_ONLY",
        "episodes": [{
            "episode_id": aliases[episode.episode_id],
            "record_count": len(episode.records),
            "records": [{
                "offset": offset,
                "cycle_receipt_id": record.receipt.cycle_id,
                "before_observation": record.before.state.get("observation"),
                "selected_action": record.proposal_set.selected.action,
                "prediction": record.proposal_set.selected.prediction,
                "rationale": record.proposal_set.selected.rationale,
                "after_observation": record.after.state.get("observation"),
                "reward": record.reward,
                "terminal": record.after.terminal,
                "official_success": record.after.official_success,
            } for offset, record in enumerate(episode.records)],
        } for episode in episodes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT proposal plus mechanical target-native motif audit"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--endpoint", default="https://us.api.openai.com/v1")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    key = runpy.run_path(str(args.key_file)).get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key:
        raise RuntimeError("missing OPENAI_API_KEY")
    os.environ["TARGET_MOTIF_OPENAI_API_KEY"] = key

    episodes = []
    source_files = []
    for path in sorted(args.input_dir.glob("task_[0-7].json")):
        raw = json.loads(path.read_text())
        records = tuple(_record(row) for row in raw["records"])
        episodes.append(TargetEpisodeView(
            str(raw["task_id"]), "alfworld_valid_unseen", "adaptation", records,
        ))
        source_files.append({
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(records),
        })
    if len(episodes) != 8:
        raise ValueError("expected exactly eight frozen adaptation episodes")
    backend = MemoizedCompletionBackend(OpenAICompatibleBackend(
        args.endpoint,
        {"target_motif": args.model},
        api_key_env="TARGET_MOTIF_OPENAI_API_KEY",
        json_mode=True,
        temperature=None,
        request_overrides={
            "max_completion_tokens": 12000,
            "reasoning_effort": "low",
        },
    ), cache_path=args.cache)
    response_text = backend.complete(
        "target_motif", SYSTEM, _payload(episodes),
    )
    response = json.loads(response_text)
    aliases = {
        f"E{index}": episode.episode_id
        for index, episode in enumerate(episodes)
    }
    grounded_response = {
        **response,
        "spans": [{
            **span,
            "episode_id": aliases.get(
                str(span.get("episode_id")), str(span.get("episode_id")),
            ),
        } for span in response.get("spans", [])],
    }
    motif = target_motif_from_agent_response(
        "alfworld_valid_unseen", episodes, grounded_response,
    )
    audit = audit_target_native_motif(motif, episodes)
    payload = {
        "schema_version": 1,
        "authority": "UNTRUSTED_GPT_TARGET_MOTIF_PLUS_MECHANICAL_AUDIT",
        "adaptation_only": True,
        "qualification_or_held_out_used": False,
        "model_identity": dict(backend.identity),
        "source_files": source_files,
        "agent_response": response,
        "episode_aliases": aliases,
        "grounded_agent_response": grounded_response,
        "motif": asdict(motif),
        "audit": asdict(audit),
        "usage": dict(backend.last_usage),
        "claim_boundary": (
            "An accepted adaptation motif is only a candidate. Qualification "
            "and matched official value are still required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "accepted": audit.accepted,
        "failures": audit.failure_codes,
        "spans": audit.spans,
        "nodes": audit.nodes,
        "edges": audit.edges,
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
