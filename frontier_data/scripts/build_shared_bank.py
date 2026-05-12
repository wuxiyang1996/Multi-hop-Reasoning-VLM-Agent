#!/usr/bin/env python
"""Build the shared abstract bank from all 850 per-task skills.

Reads per_task_banks/<task>/skill_bank.jsonl and:
  1. Normalises skill IDs (strip phase/version/translation decorations)
  2. Clusters skills into abstract groups by normalised stem
  3. Lifts each cluster into a SharedAbstractSkill with:
     - template_signature inferred from contract/protocol steps
     - lineage entries pointing back to each (task, skill_id) origin
  4. Writes the TwoLayerSkillStore layout:
       shared_skill_bank/abstract.jsonl
       shared_skill_bank/by_task/<task>/bindings.jsonl

Output:
  frontier_data/output/shared_skill_bank/
    abstract.jsonl
    by_task/<task>/bindings.jsonl
    SUMMARY.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("build_shared_bank")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from skill_bank.shared_abstract_bank import (
    BoundConcreteSkill,
    LineageEntry,
    ProtocolStep,
    SharedAbstractSkill,
    SubEpisodeRef,
    TemplateStep,
    normalise_skill_id,
)

# ── Template step vocabulary ─────────────────────────────────────────
TEMPLATE_OPS = {
    "PERCEIVE", "RECALL", "COMPARE", "FILTER",
    "DECIDE", "COMMIT", "VERIFY", "RECOVER",
}
FALLBACK_SIGNATURES = {
    "ATTACK": "PERCEIVE → DECIDE → COMMIT",
    "DEFEND": "PERCEIVE → DECIDE → COMMIT → VERIFY",
    "MOVE": "PERCEIVE → DECIDE → COMMIT",
    "NAVIGATE": "PERCEIVE → RECALL → DECIDE → COMMIT",
    "EVADE": "PERCEIVE → FILTER → DECIDE → COMMIT",
    "COLLECT": "PERCEIVE → FILTER → DECIDE → COMMIT",
    "SHOOT": "PERCEIVE → DECIDE → COMMIT → VERIFY",
    "JUMP": "PERCEIVE → DECIDE → COMMIT",
    "CLEAR": "PERCEIVE → COMPARE → FILTER → DECIDE → COMMIT",
    "SETUP": "PERCEIVE → RECALL → DECIDE → COMMIT",
    "INSPECT": "PERCEIVE → COMPARE → DECIDE",
    "RECOVER": "PERCEIVE → RECALL → DECIDE → COMMIT → VERIFY",
    "EXPLORE": "PERCEIVE → RECALL → FILTER → DECIDE → COMMIT",
    "COMBO": "PERCEIVE → RECALL → COMPARE → DECIDE → COMMIT",
    "MANAGE": "PERCEIVE → COMPARE → FILTER → DECIDE → COMMIT",
    "SELECT": "PERCEIVE → COMPARE → FILTER → DECIDE",
    "PLAN": "PERCEIVE → RECALL → COMPARE → FILTER → DECIDE",
    "EXECUTE": "PERCEIVE → DECIDE → COMMIT → VERIFY",
    "OPTIMIZE": "PERCEIVE → COMPARE → FILTER → DECIDE → COMMIT → VERIFY",
}

COHORT_MAP = {
    "Temporal_": "gymv_game",
    "tetris": "env_wr_game", "super_mario": "env_wr_game",
    "candy_crush": "env_wr_game", "twenty_forty_eight": "env_wr_game",
    "browsergym": "web", "miniwob": "web", "webshop": "web",
    "osworld": "web",
    "siv_bench": "vr_video", "video_holmes": "vr_video",
    "tir_bench": "vr_image", "visual_toolbench": "vr_image",
}


def task_to_cohort(task: str) -> str:
    for prefix, cohort in COHORT_MAP.items():
        if task.startswith(prefix) or task == prefix:
            return cohort
    return "unknown"


def extract_ops_from_name(name: str) -> List[str]:
    """Extract verb tokens from a skill name like 'COMMIT/CLEAR'."""
    return [t.strip().upper() for t in re.split(r"[/\-_ ]", name) if t.strip()]


def infer_template_signature(skill: dict) -> str:
    """Infer PERCEIVE → DECIDE → COMMIT signature from a per-task skill record."""
    contract = skill.get("contract", {})
    if isinstance(contract, str):
        try:
            contract = json.loads(contract)
        except json.JSONDecodeError:
            contract = {}

    # Try explicit template_signature first
    ts = skill.get("template_signature", "")
    if ts and "→" in ts:
        return ts

    # Try protocol_steps or template_steps
    for key in ("protocol_steps", "template_steps", "protocol"):
        steps = skill.get(key, [])
        if steps and isinstance(steps, list):
            ops = []
            for s in steps:
                if isinstance(s, dict):
                    op = s.get("op", s.get("template_op", ""))
                elif isinstance(s, str):
                    op = s
                else:
                    continue
                op = op.upper().strip()
                if op in TEMPLATE_OPS:
                    ops.append(op)
            if ops:
                return " → ".join(ops)

    # Heuristic from skill name verbs
    s = skill.get("skill", skill)
    sid = s.get("skill_id", s.get("name", ""))
    stem = normalise_skill_id(sid)
    verbs = extract_ops_from_name(stem)
    for v in verbs:
        if v in FALLBACK_SIGNATURES:
            return FALLBACK_SIGNATURES[v]

    # Ultimate fallback
    return "PERCEIVE → DECIDE → COMMIT"


def build_template_steps(signature: str) -> List[dict]:
    """Convert 'PERCEIVE → DECIDE → COMMIT' into TemplateStep dicts."""
    predicates = {
        "PERCEIVE": "Observe and encode the current visual/textual state",
        "RECALL": "Retrieve relevant past observations or knowledge",
        "COMPARE": "Contrast current state against target or baseline",
        "FILTER": "Narrow candidates to the most relevant subset",
        "DECIDE": "Select the best action from filtered candidates",
        "COMMIT": "Execute the chosen action in the environment",
        "VERIFY": "Check whether the action achieved its intended effect",
        "RECOVER": "Handle failure and attempt corrective action",
    }
    ops = [op.strip() for op in signature.split("→")]
    return [{"op": op, "predicate": predicates.get(op, f"Execute {op.lower()} step")} for op in ops]


def build_protocol_steps(signature: str, skill: dict) -> List[dict]:
    """Build abstract protocol steps from signature + contract info."""
    ops = [op.strip() for op in signature.split("→")]
    steps = []
    for op in ops:
        step = {
            "op": op,
            "payload": {f"{op.lower()}_target": f"${{{op.lower()}_target}}"},
            "slot_types": {f"{op.lower()}_target": "tracked_entity"},
            "preconditions": [],
            "effects_add": [],
            "effects_del": [],
            "evidence_role": op.lower(),
            "notes": "",
        }
        steps.append(step)
    return steps


def load_per_task_skills(per_task_root: Path) -> Dict[str, List[dict]]:
    """Load all per-task skill_bank.jsonl files. Returns {task: [skill_dicts]}."""
    result: Dict[str, List[dict]] = {}
    for task_dir in sorted(per_task_root.iterdir()):
        if not task_dir.is_dir():
            continue
        sb = task_dir / "skill_bank.jsonl"
        if not sb.exists():
            continue
        skills = []
        with open(sb) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    skills.append(d)
                except json.JSONDecodeError:
                    continue
        result[task_dir.name] = skills
    return result


def extract_skill_core(d: dict) -> dict:
    """Unwrap the {skill:{...}, report:{...}} envelope if present."""
    if "skill" in d and isinstance(d["skill"], dict):
        return d["skill"]
    return d


def main() -> int:
    per_task_root = REPO_ROOT / "frontier_data" / "output" / "per_task_banks"
    out_root = REPO_ROOT / "frontier_data" / "output" / "shared_skill_bank"
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading per-task banks from %s", per_task_root)
    all_tasks = load_per_task_skills(per_task_root)
    total_skills = sum(len(v) for v in all_tasks.values())
    logger.info("Loaded %d skills across %d tasks", total_skills, len(all_tasks))

    # ── Phase 1: cluster by normalised stem ────────────────────────
    # key = normalised_skill_id_stem → list of (task, skill_dict)
    clusters: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
    for task, skills in all_tasks.items():
        for raw in skills:
            s = extract_skill_core(raw)
            sid = s.get("skill_id", s.get("name", ""))
            if not sid:
                continue
            stem = normalise_skill_id(sid)
            clusters[stem].append((task, raw))

    logger.info("Clustered into %d unique skill stems", len(clusters))

    # ── Phase 2: lift each cluster → SharedAbstractSkill ───────────
    abstracts: List[SharedAbstractSkill] = []
    all_bindings: Dict[str, List[BoundConcreteSkill]] = defaultdict(list)
    now_iso = datetime.now(timezone.utc).isoformat()

    for stem, members in sorted(clusters.items()):
        # Pick the richest record as the representative
        rep_task, rep_raw = max(members, key=lambda x: len(json.dumps(x[1])))
        rep = extract_skill_core(rep_raw)

        # Infer template signature
        sig = infer_template_signature(rep)

        # Build lineage from all members
        lineage_entries = []
        cohorts_seen: Set[str] = set()
        for task, raw in members:
            s = extract_skill_core(raw)
            sid = s.get("skill_id", s.get("name", ""))
            cohort = task_to_cohort(task)
            cohorts_seen.add(cohort)
            le = LineageEntry(
                task=task,
                concrete_skill_id=normalise_skill_id(sid),
                raw_skill_id=sid,
                cohort=cohort,
                discovered_via="mining",
                is_native=True,
                n_uses=0,
                n_success=0,
                n_translated_uses=0,
                contract_hash=hashlib.md5(
                    json.dumps(s.get("contract", {}), sort_keys=True).encode()
                ).hexdigest()[:8],
            )
            lineage_entries.append(le)

        # Human-readable name from the representative
        name = rep.get("name", rep.get("contract", {}).get("name", stem))
        if isinstance(name, dict):
            name = name.get("name", stem)

        abstract = SharedAbstractSkill(
            abstract_skill_id=stem,
            name=str(name),
            template_signature=sig,
            template_steps=[TemplateStep.from_dict(ts) for ts in build_template_steps(sig)],
            protocol_steps=[ProtocolStep.from_dict(ps) for ps in build_protocol_steps(sig, rep)],
            lineage=lineage_entries,
            cohorts_seen=sorted(cohorts_seen),
            discovered_via="mining",
            schema_version=1,
            created_at=now_iso,
            updated_at=now_iso,
        )
        abstracts.append(abstract)

        # Build bound concrete skills for each (task, skill)
        for task, raw in members:
            s = extract_skill_core(raw)
            sid = s.get("skill_id", s.get("name", ""))
            contract = s.get("contract", {})
            if isinstance(contract, str):
                try:
                    contract = json.loads(contract)
                except json.JSONDecodeError:
                    contract = {}

            bound = BoundConcreteSkill(
                concrete_skill_id=normalise_skill_id(sid),
                task=task,
                abstract_skill_id=stem,
                name=str(s.get("name", contract.get("name", stem))),
                protocol=[ProtocolStep.from_dict(ps) for ps in build_protocol_steps(sig, s)],
                contract=contract,
                sub_episodes=[],
                binding_status="VALIDATED",
                binding_source="mining",
                raw_skill_id=sid,
                schema_version=2,
                created_at=now_iso,
                updated_at=now_iso,
            )
            all_bindings[task].append(bound)

    logger.info("Created %d abstract mega-skills", len(abstracts))
    logger.info("Created bindings for %d tasks", len(all_bindings))

    # ── Phase 3: write the TwoLayerSkillStore ──────────────────────
    # abstract.jsonl
    abstract_path = out_root / "abstract.jsonl"
    with open(abstract_path, "w") as f:
        for a in abstracts:
            f.write(json.dumps(a.to_dict()) + "\n")
    logger.info("Wrote %d abstracts to %s", len(abstracts), abstract_path)

    # by_task/<task>/bindings.jsonl
    for task, bindings in sorted(all_bindings.items()):
        task_dir = out_root / "by_task" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        bind_path = task_dir / "bindings.jsonl"
        with open(bind_path, "w") as f:
            for b in bindings:
                f.write(json.dumps(b.to_dict()) + "\n")

    # Summary
    sig_counts = Counter(a.template_signature for a in abstracts)
    cohort_counts = Counter()
    for a in abstracts:
        for c in a.cohorts_seen:
            cohort_counts[c] += 1

    multi_task = [a for a in abstracts if a.n_bound_tasks >= 2]
    summary = {
        "generated_utc": now_iso,
        "n_abstracts": len(abstracts),
        "n_multi_task_abstracts": len(multi_task),
        "n_tasks_with_bindings": len(all_bindings),
        "total_bindings": sum(len(v) for v in all_bindings.values()),
        "top_signatures": sig_counts.most_common(15),
        "cohort_coverage": dict(cohort_counts),
        "multi_task_examples": [
            {
                "id": a.abstract_skill_id,
                "name": a.name,
                "signature": a.template_signature,
                "tasks": sorted({L.task for L in a.lineage}),
            }
            for a in sorted(multi_task, key=lambda x: -x.n_bound_tasks)[:20]
        ],
    }
    with open(out_root / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("═" * 60)
    logger.info("SHARED BANK BUILT SUCCESSFULLY")
    logger.info("  %d abstract mega-skills", len(abstracts))
    logger.info("  %d multi-task mega-skills (span ≥2 tasks)", len(multi_task))
    logger.info("  %d tasks with concrete bindings", len(all_bindings))
    logger.info("  %d total bindings", sum(len(v) for v in all_bindings.values()))
    logger.info("  Top 5 template signatures:")
    for sig, cnt in sig_counts.most_common(5):
        logger.info("    %3d  %s", cnt, sig)
    logger.info("  Cohort coverage: %s", dict(cohort_counts))
    logger.info("  Output: %s", out_root)
    logger.info("═" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
