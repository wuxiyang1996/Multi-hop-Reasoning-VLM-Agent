#!/usr/bin/env python
"""Build archetype skill banks for miniwob and webshop.

Reads the per-episode rollouts from Cold-start-out-browsergym/, runs
the sequence-episode lift, injects a cluster_key into provenance, then
aggregates into archetype banks matching the VR/video format.

Cluster strategies:
  miniwob  → task family (first word: click, email, drag, ...)
  webshop  → step-count bucket (short ≤5, medium 6-12, long 13+)

Output:
  frontier_data/output/per_task_banks/miniwob/skill_bank.jsonl   (archetype)
  frontier_data/output/per_task_banks/webshop/skill_bank.jsonl   (archetype)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("build_web_skill_banks")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DOWNLOAD_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/main_project"
)
BROWSERGYM_ROOT = DOWNLOAD_ROOT / "Cold-start-out-browsergym"


def _miniwob_family(game_name: str) -> str:
    """miniwob.click-button → click, miniwob.email-inbox → email."""
    stem = game_name.replace("miniwob.", "")
    family = stem.split("-")[0]
    return family


def _webshop_step_bucket(n_steps: int) -> str:
    if n_steps <= 5:
        return "short_interaction"
    elif n_steps <= 12:
        return "medium_interaction"
    else:
        return "long_interaction"


def _extract_prose_steps(experiences: List[Dict]) -> List[Tuple[str, str]]:
    """Extract (intent, action_prose) from each experience step."""
    out = []
    for exp in experiences:
        intentions = exp.get("intentions") or []
        intent_text = ""
        if isinstance(intentions, list) and intentions:
            if isinstance(intentions[0], dict):
                intent_text = intentions[0].get("text", intentions[0].get("goal", ""))
            elif isinstance(intentions[0], str):
                intent_text = intentions[0]
        if not intent_text:
            intent_text = exp.get("summary", "") or exp.get("summary_state", "") or ""

        action = exp.get("action", "")
        out.append((str(intent_text)[:200], str(action)[:200]))
    return out


def _build_protocol_hops(prose_steps: List[Tuple[str, str]]) -> List[Dict]:
    """Convert prose steps into typed protocol hops."""
    hops = []
    for intent, action in prose_steps:
        if intent:
            hops.append({
                "op": "PERCEIVE",
                "payload": {"description": intent},
                "slot_types": {},
                "preconditions": [],
                "effects_add": [],
                "effects_del": [],
                "evidence_role": "perceive",
                "notes": intent,
            })
        if action:
            verb = "COMMIT"
            verb_match = re.match(r"^\s*\(?([\w]+)", action)
            if verb_match:
                v = verb_match.group(1).lower()
                if v in ("click", "press", "submit", "select"):
                    verb = "COMMIT"
                elif v in ("type", "fill", "enter", "input"):
                    verb = "COMMIT"
                elif v in ("scroll", "navigate", "goto", "search"):
                    verb = "PERCEIVE"
                elif v in ("drag", "drop", "move"):
                    verb = "COMMIT"
            hops.append({
                "op": verb,
                "payload": {"action": action},
                "slot_types": {},
                "preconditions": [],
                "effects_add": [],
                "effects_del": [],
                "evidence_role": "commit",
                "notes": action,
            })
    return hops


def lift_episode(
    episode: Dict[str, Any],
    *,
    corpus: str,
    cluster_key: str,
    episode_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Lift one episode into a {report, skill} envelope with cluster_key."""
    experiences = episode.get("experiences") or []
    if not experiences:
        return None

    prose_steps = _extract_prose_steps(experiences)
    if not prose_steps:
        return None

    hops = _build_protocol_hops(prose_steps)
    if not hops:
        return None

    meta = episode.get("metadata", {})
    task_id = episode.get("game_name") or episode.get("episode_id") or "unknown"
    goal = episode.get("task", "")
    goal = re.sub(
        r"^Solve\s+the\s+BrowserGym\s+task\s+\S+\.\s*", "", goal, flags=re.I
    ).strip()

    success = episode.get("outcome")
    if success is None:
        reward = meta.get("total_reward", 0)
        success = reward is not None and float(reward) > 0

    n_steps = len(experiences)
    reward_total = meta.get("total_reward", 0)

    now = datetime.now(timezone.utc).isoformat()
    skill_id = f"{corpus}.{task_id}"

    name_stem = goal[:60].lower().replace(" ", "_") if goal else task_id
    name = f"episode/{name_stem}"

    skill = {
        "skill_id": skill_id,
        "name": name,
        "strategic_description": goal[:300],
        "applicable_domains": ["browser"],
        "feasible_tasks": [task_id, cluster_key],
        "feasible_domains": ["browser"],
        "verified_domains": ["browser"] if success else [],
        "verified_tasks": [task_id] if success else [],
        "evidence_role": "COMMIT",
        "execution_hint": None,
        "expected_tag_pattern": None,
        "protocol": hops,
        "protocol_history": [],
        "protocol_raw": {"steps": [i for i, _ in prose_steps], "actions": [a for _, a in prose_steps]},
        "contract": {
            "preconditions": [],
            "success_criteria": [f"task {'succeeded' if success else 'failed'} ({task_id})"],
            "abort_criteria": [],
            "effects_add": [{"type": "task_completed", "args": {"task": task_id}}] if success else [],
            "effects_del": [],
        },
        "provenance": {
            "corpus": corpus,
            "benchmark": corpus,
            "modality": "browser",
            "bank_kind": "episode",
            "cluster_key": cluster_key,
            "source_episode": str(episode_path) if episode_path else None,
            "n_steps": n_steps,
            "lift_mode": "per_episode_no_llm",
        },
        "tags": [corpus, "browser", "sequence", "per_episode", cluster_key],
        "n_instances": 1,
        "sub_episodes": [],
        "source_type": "sequence_per_episode",
    }

    report = {
        "skill_id": skill_id,
        "n_instances": 1,
        "n_steps": n_steps,
        "overall_pass_rate": 1.0 if success else 0.0,
        "eff_add_success_rate": 1.0 if success else 0.0,
        "eff_del_success_rate": None,
        "eff_event_rate": None,
        "failure_signatures": [],
        "worst_segments": [],
        "reward_total": reward_total,
        "judge_correct": success,
    }

    return {"report": report, "skill": skill}


def aggregate_to_archetypes(
    records: List[Dict[str, Any]],
    corpus: str,
) -> List[Dict[str, Any]]:
    """Group per-episode records by cluster_key → archetype records."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        ck = (r.get("skill") or {}).get("provenance", {}).get("cluster_key", "unknown")
        groups[ck].append(r)

    archetypes = []
    for ck in sorted(groups.keys()):
        members = groups[ck]
        best = max(members, key=lambda m: (
            float((m.get("report") or {}).get("overall_pass_rate") or 0),
            -(m.get("report") or {}).get("n_steps", 999),
        ))
        best_skill = best.get("skill", {})
        best_report = best.get("report", {})

        member_ids = [m.get("skill", {}).get("skill_id", "") for m in members]
        n_success = sum(1 for m in members if m.get("report", {}).get("judge_correct"))
        mean_steps = sum(m.get("report", {}).get("n_steps", 0) for m in members) / max(1, len(members))
        mean_pass = sum(float(m.get("report", {}).get("overall_pass_rate") or 0) for m in members) / max(1, len(members))

        archetype_id = f"archetype.{corpus}.{ck}"

        arch_skill = {
            "skill_id": archetype_id,
            "name": best_skill.get("name", archetype_id),
            "strategic_description": best_skill.get("strategic_description", ""),
            "applicable_domains": ["browser"],
            "feasible_tasks": [corpus, ck],
            "feasible_domains": ["browser"],
            "verified_domains": ["browser"] if n_success > 0 else [],
            "verified_tasks": [corpus] if n_success > 0 else [],
            "evidence_role": "COMMIT",
            "execution_hint": best_skill.get("execution_hint"),
            "expected_tag_pattern": best_skill.get("expected_tag_pattern"),
            "protocol": best_skill.get("protocol", []),
            "protocol_history": [],
            "protocol_raw": best_skill.get("protocol_raw"),
            "contract": {
                "preconditions": list((best_skill.get("contract") or {}).get("preconditions", [])),
                "success_criteria": list((best_skill.get("contract") or {}).get("success_criteria", [])),
                "abort_criteria": [],
                "effects_add": [{"type": "task_completed", "args": {}, "from_phrase": "archetype_union"}],
                "effects_del": [],
            },
            "provenance": {
                "corpus": corpus,
                "benchmark": corpus,
                "modality": "browser",
                "bank_kind": "archetype",
                "cluster_key": ck,
                "n_members": len(members),
                "member_skill_ids": member_ids,
                "representative_skill_id": best_skill.get("skill_id"),
                "representative_pass_rate": float(best_report.get("overall_pass_rate") or 0),
                "aggregation": "direct",
                "aggregated_at": datetime.now(timezone.utc).isoformat(),
            },
            "tags": sorted(set([corpus, "browser", "archetype", ck])),
            "n_instances": len(members),
            "sub_episodes": [],
            "source_type": "MINED",
        }

        arch_report = {
            "skill_id": archetype_id,
            "n_instances": len(members),
            "overall_pass_rate": mean_pass,
            "eff_add_success_rate": mean_pass,
            "eff_del_success_rate": None,
            "eff_event_rate": None,
            "failure_signatures": [],
            "worst_segments": [],
            "lift_stats": {
                "n_members": len(members),
                "n_success": n_success,
                "mean_steps": round(mean_steps, 1),
                "representative_skill_id": best_skill.get("skill_id"),
            },
        }

        archetypes.append({"report": arch_report, "skill": arch_skill})

    return archetypes


def process_miniwob(output_dir: Path) -> Tuple[int, int, int]:
    """Lift + aggregate miniwob tasks."""
    miniwob_dirs = sorted([
        d for d in BROWSERGYM_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("miniwob.")
    ])
    logger.info("Found %d miniwob task directories", len(miniwob_dirs))

    per_episode_records = []
    for task_dir in miniwob_dirs:
        game_name = task_dir.name
        family = _miniwob_family(game_name)
        ep_files = sorted(task_dir.glob("episode_*.json"))
        ep_files = [f for f in ep_files if "buffer" not in f.name]

        for ep_file in ep_files:
            try:
                episode = json.loads(ep_file.read_text())
            except Exception:
                continue
            record = lift_episode(
                episode, corpus="miniwob", cluster_key=family, episode_path=ep_file,
            )
            if record:
                per_episode_records.append(record)

    logger.info("Lifted %d miniwob episodes", len(per_episode_records))

    archetypes = aggregate_to_archetypes(per_episode_records, "miniwob")
    logger.info("Aggregated into %d miniwob archetypes", len(archetypes))

    out_dir = output_dir / "miniwob"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "skill_bank.jsonl"
    with bank_path.open("w") as f:
        for a in archetypes:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    return len(miniwob_dirs), len(per_episode_records), len(archetypes)


def process_webshop(output_dir: Path) -> Tuple[int, int, int]:
    """Lift + aggregate webshop tasks."""
    webshop_root = BROWSERGYM_ROOT / "webshop_50task_low"
    if not webshop_root.exists():
        logger.warning("webshop_50task_low not found at %s", webshop_root)
        return 0, 0, 0

    task_dirs = sorted([
        d for d in webshop_root.iterdir()
        if d.is_dir() and d.name.startswith("webshop.")
    ])
    logger.info("Found %d webshop task directories", len(task_dirs))

    per_episode_records = []
    for task_dir in task_dirs:
        ep_files = sorted(task_dir.glob("episode_*.json"))
        ep_files = [f for f in ep_files if "buffer" not in f.name]

        for ep_file in ep_files:
            try:
                episode = json.loads(ep_file.read_text())
            except Exception:
                continue
            n_steps = len(episode.get("experiences", []))
            bucket = _webshop_step_bucket(n_steps)
            record = lift_episode(
                episode, corpus="webshop", cluster_key=bucket, episode_path=ep_file,
            )
            if record:
                per_episode_records.append(record)

    logger.info("Lifted %d webshop episodes", len(per_episode_records))

    archetypes = aggregate_to_archetypes(per_episode_records, "webshop")
    logger.info("Aggregated into %d webshop archetypes", len(archetypes))

    out_dir = output_dir / "webshop"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "skill_bank.jsonl"
    with bank_path.open("w") as f:
        for a in archetypes:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    return len(task_dirs), len(per_episode_records), len(archetypes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "frontier_data" / "output" / "per_task_banks"),
        help="Output directory for per-task skill banks",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Building web task skill banks (miniwob + webshop)")
    logger.info("=" * 60)

    mw_tasks, mw_eps, mw_arch = process_miniwob(output_dir)
    ws_tasks, ws_eps, ws_arch = process_webshop(output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  miniwob: %d tasks → %d episodes → %d archetypes", mw_tasks, mw_eps, mw_arch)
    logger.info("  webshop: %d tasks → %d episodes → %d archetypes", ws_tasks, ws_eps, ws_arch)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
