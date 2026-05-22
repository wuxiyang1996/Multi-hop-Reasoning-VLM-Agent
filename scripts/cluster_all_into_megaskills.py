"""Cluster Stage 1 + Stage 2 skills into unified mega-skill families.

Approach:
  1. Load the existing mega-skill labels (Stage 1 only) as anchors.
  2. Load Stage 1 per-task banks (Phase 1 source games).
  3. Load Stage 2 merged banks (4 holdout games, post-GRPO).
  4. For each skill, assign to an existing mega-family by matching:
     a) Protocol compressed-plan signature similarity
     b) Skill name / description text similarity
  5. Skills without a strong match form new mega-families.
  6. Output a unified mega_skill_labels.json and summary.

Usage:
    python scripts/cluster_all_into_megaskills.py [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.judge_plan_level_similarity import (
    DOMAIN_OF, compressed_plan,
)

STAGE1_BANKS = {
    "gymv_thunder_force_iii": REPO / "frontier_data/output/per_task_banks/gymv_thunder_force_iii/skill_bank.jsonl",
    "gymv_streets_of_rage_2": REPO / "frontier_data/output/per_task_banks/gymv_streets_of_rage_2/skill_bank.jsonl",
    "gymv_strider":           REPO / "frontier_data/output/per_task_banks/gymv_strider/skill_bank.jsonl",
    "gymv_columns":           REPO / "frontier_data/output/per_task_banks/gymv_columns/skill_bank.jsonl",
    "candy_crush":            REPO / "frontier_data/output/per_task_banks/candy_crush/skill_bank.jsonl",
}

STAGE2_BANKS = {
    "gymv_airstriker":       REPO / "frontier_data/output/stage2_merged_banks/gymv_airstriker/skill_bank.jsonl",
    "gymv_altered_beast":    REPO / "frontier_data/output/stage2_merged_banks/gymv_altered_beast/skill_bank.jsonl",
    "gymv_dynamite_headdy":  REPO / "frontier_data/output/stage2_merged_banks/gymv_dynamite_headdy/skill_bank.jsonl",
    "gymv_space_harrier_ii": REPO / "frontier_data/output/stage2_merged_banks/gymv_space_harrier_ii/skill_bank.jsonl",
}

EXISTING_LABELS = REPO / "frontier_data/output/mega_skill_labels_final.json"
MEGASKILLS_2STAGE = REPO / "frontier_data/output/megaskills_2stage/mega_skills.jsonl"
DEFAULT_OUTPUT = REPO / "frontier_data/output/unified_mega_skills"

GENRE_OF = {
    "gymv_thunder_force_iii": "shooter", "gymv_airstriker": "shooter",
    "gymv_space_harrier_ii": "shooter",
    "gymv_streets_of_rage_2": "brawler", "gymv_altered_beast": "brawler",
    "gymv_strider": "platformer", "gymv_dynamite_headdy": "platformer",
    "gymv_columns": "puzzle", "candy_crush": "puzzle",
}


@dataclass
class SkillEntry:
    task: str
    skill_id: str
    name: str
    description: str
    protocol_steps: List[str]
    intent_seq: List[str]
    plan_sig: str
    domain: str
    genre: str
    stage: int  # 1 or 2
    raw_entry: dict = field(repr=False, default_factory=dict)

    def key(self) -> str:
        return f"{self.task}::{self.skill_id}"


def _as_steps(protocol) -> List[str]:
    if not protocol:
        return []
    if isinstance(protocol, list):
        return [str(x).strip() for x in protocol if x and str(x).strip()]
    if isinstance(protocol, dict):
        s = protocol.get("steps") or []
        if isinstance(s, list):
            return [str(x).strip() for x in s if x and str(x).strip()]
    return []


def _norm_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"^(seed\.|bootstrap\.)", "", name)
    name = re.sub(r"^(early|mid|late)[:_]\s*", "", name)
    name = re.sub(r"[_/]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_skills(bank_paths: Dict[str, Path], stage: int) -> List[SkillEntry]:
    entries = []
    for task, path in bank_paths.items():
        if not path.exists():
            print(f"  WARN: {path} not found, skipping")
            continue
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sk = d.get("skill", d)
                if isinstance(sk, str):
                    sk = json.loads(sk)
                sid = sk.get("skill_id", "")
                protocol = sk.get("protocol") or {}
                protocol_raw = sk.get("protocol_raw") or {}
                steps = _as_steps(protocol) or _as_steps(protocol_raw)
                intent = compressed_plan(steps[:8])
                contract = sk.get("contract") or {}
                desc = str(sk.get("strategic_description", "") or
                          contract.get("description", ""))

                entries.append(SkillEntry(
                    task=task,
                    skill_id=sid,
                    name=sk.get("name", sid),
                    description=desc,
                    protocol_steps=steps[:8],
                    intent_seq=intent,
                    plan_sig=" → ".join(intent) if intent else "",
                    domain=DOMAIN_OF.get(task, "GAME"),
                    genre=GENRE_OF.get(task, "other"),
                    stage=stage,
                    raw_entry=d,
                ))
                n += 1
        print(f"  {task:30s}  {n:3d} skills  (stage {stage})")
    return entries


@dataclass
class MegaFamily:
    family_id: str
    display_name: str
    plan_sig: str
    members: List[SkillEntry] = field(default_factory=list)
    source: str = "existing"  # "existing" or "new"

    @property
    def tasks(self) -> Set[str]:
        return {m.task for m in self.members}

    @property
    def stages(self) -> Set[int]:
        return {m.stage for m in self.members}


def _sig_similarity(a: str, b: str) -> float:
    """Similarity of two plan signatures."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_name(a), _norm_name(b)).ratio()


def _step_text_similarity(steps_a: List[str], steps_b: List[str]) -> float:
    if not steps_a or not steps_b:
        return 0.0
    text_a = " ".join(s[:100] for s in steps_a[:6]).lower()
    text_b = " ".join(s[:100] for s in steps_b[:6]).lower()
    return SequenceMatcher(None, text_a, text_b).ratio()


def load_existing_families(path: Path) -> Dict[str, dict]:
    """Load existing mega_skill_labels_final.json families."""
    if not path.exists():
        return {}
    data = json.load(open(path))
    return data.get("mega_skills", {})


# Temporal_ ↔ gymv_ name mapping for resolving existing labels
_TEMPORAL_TO_GYMV = {
    "Temporal_ThunderForceIII-v0": "gymv_thunder_force_iii",
    "Temporal_StreetsOfRage2-v0": "gymv_streets_of_rage_2",
    "Temporal_Strider-v0": "gymv_strider",
    "Temporal_Columns-v0": "gymv_columns",
    "Temporal_SpaceHarrierII-v0": "gymv_space_harrier_ii",
    "Temporal_Airstriker-v0": "gymv_airstriker",
    "Temporal_AlteredBeast-v0": "gymv_altered_beast",
    "Temporal_DynamiteHeaddy-v0": "gymv_dynamite_headdy",
}


def _resolve_task(task: str) -> str:
    """Normalise Temporal_* to gymv_* for index lookups."""
    return _TEMPORAL_TO_GYMV.get(task, task)


def classify_skill(
    skill: SkillEntry,
    families: Dict[str, MegaFamily],
    threshold: float = 0.45,
) -> Optional[str]:
    """Find best matching mega-family for a skill.

    Returns family_id or None.

    Scoring:
      - 0.4 * plan_signature_similarity
      - 0.35 * name_similarity (normalised)
      - 0.25 * protocol_step_text_similarity
    """
    best_fam = None
    best_score = 0.0

    for fid, fam in families.items():
        if not fam.members:
            continue
        rep = fam.members[0]

        sig_sim = _sig_similarity(skill.plan_sig, fam.plan_sig or rep.plan_sig)
        name_sim = max(
            _name_similarity(skill.name, m.name)
            for m in fam.members[:5]
        )
        step_sim = _step_text_similarity(skill.protocol_steps, rep.protocol_steps)

        score = 0.40 * sig_sim + 0.35 * name_sim + 0.25 * step_sim

        if score > best_score:
            best_score = score
            best_fam = fid

    if best_score >= threshold:
        return best_fam
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Classification threshold (default 0.45)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Unified Mega-Skill Clustering: Stage 1 + Stage 2")
    print("=" * 70)

    # 1. Load all skills
    print("\n[1] Loading Stage 1 banks...")
    s1_skills = load_skills(STAGE1_BANKS, stage=1)
    print(f"    Total Stage 1: {len(s1_skills)} skills")

    print("\n[2] Loading Stage 2 merged banks...")
    s2_skills = load_skills(STAGE2_BANKS, stage=2)
    print(f"    Total Stage 2: {len(s2_skills)} skills")

    all_skills = s1_skills + s2_skills
    print(f"\n    Grand total: {len(all_skills)} skills across "
          f"{len(set(s.task for s in all_skills))} tasks")

    # 2. Load the 10 LLM-judge mega-skills as anchors
    print(f"\n[3] Loading LLM-judge mega-skills from {MEGASKILLS_2STAGE.name}...")
    mega_records = []
    with open(MEGASKILLS_2STAGE) as f:
        for line in f:
            if line.strip():
                mega_records.append(json.loads(line))
    print(f"    Found {len(mega_records)} mega-skill families")

    # Build skill index for lookup
    skill_index: Dict[str, SkillEntry] = {}
    for s in all_skills:
        skill_index[s.key()] = s
        norm_key = f"{_resolve_task(s.task)}::{s.skill_id}"
        if norm_key not in skill_index:
            skill_index[norm_key] = s

    # Bootstrap families from the 10 LLM mega-skills, populating with
    # real SkillEntry objects from our loaded banks.
    families: Dict[str, MegaFamily] = {}
    pre_assigned: Set[str] = set()

    for rec in mega_records:
        fam_id = rec["mega_skill_id"]
        plan_sig = rec.get("template_signature", "")

        members: List[SkillEntry] = []
        for m in rec.get("members", []):
            task = m.get("task", "")
            sid = m.get("skill_id", "")
            entry = (skill_index.get(f"{task}::{sid}") or
                     skill_index.get(f"{_resolve_task(task)}::{sid}"))
            if entry:
                members.append(entry)
                pre_assigned.add(entry.key())

        families[fam_id] = MegaFamily(
            family_id=fam_id,
            display_name=rec.get("representative", {}).get("name", fam_id),
            plan_sig=plan_sig,
            members=members,
            source="llm_judge",
        )

    n_pre = sum(len(f.members) for f in families.values())
    n_fam_pop = sum(1 for f in families.values() if f.members)
    print(f"    Pre-assigned {n_pre} Stage 1 skills to {n_fam_pop}/{len(families)} families")

    remaining_skills = [s for s in all_skills if s.key() not in pre_assigned]
    print(f"    Remaining to classify: {len(remaining_skills)} "
          f"({sum(1 for s in remaining_skills if s.stage == 1)} S1 + "
          f"{sum(1 for s in remaining_skills if s.stage == 2)} S2)")

    # 4. Classify remaining skills
    print(f"\n[4] Classifying {len(remaining_skills)} remaining skills into families "
          f"(threshold={args.threshold})...")

    assigned = 0
    new_families_created = 0
    unassigned: List[SkillEntry] = []

    for skill in remaining_skills:
        fam_id = classify_skill(skill, families, threshold=args.threshold)
        if fam_id:
            families[fam_id].members.append(skill)
            assigned += 1
        else:
            unassigned.append(skill)

    print(f"    Assigned to existing: {assigned}")
    print(f"    Unassigned: {len(unassigned)}")

    # 5. Cluster unassigned skills into new families
    # Group by (normalised_name, plan_sig)
    if unassigned:
        print(f"\n[5] Clustering {len(unassigned)} unassigned skills into new families...")
        new_groups: Dict[str, List[SkillEntry]] = defaultdict(list)
        for skill in unassigned:
            norm = _norm_name(skill.name)
            cluster_key = f"{norm}|{skill.plan_sig}"

            merged = False
            for existing_key, group in new_groups.items():
                rep = group[0]
                if (_name_similarity(skill.name, rep.name) >= 0.65 and
                    _sig_similarity(skill.plan_sig, rep.plan_sig) >= 0.5):
                    new_groups[existing_key].append(skill)
                    merged = True
                    break

            if not merged:
                new_groups[cluster_key].append(skill)

        next_idx = len(mega_records) + 1
        for group_key, group in sorted(new_groups.items(),
                                        key=lambda x: -len(x[1])):
            rep = group[0]
            slug = re.sub(r"[^a-z0-9]+", "_",
                         _norm_name(rep.name))[:30].strip("_")
            fam_id = f"NEW_{slug}" if slug else f"NEW_{next_idx:03d}"

            if fam_id in families:
                fam_id = f"{fam_id}_{next_idx:03d}"

            families[fam_id] = MegaFamily(
                family_id=fam_id,
                display_name=rep.name,
                plan_sig=rep.plan_sig,
                members=group,
                source="new",
            )
            new_families_created += 1
            next_idx += 1

        print(f"    Created {new_families_created} new families")

    # 6. Summary
    print(f"\n{'=' * 70}")
    print("UNIFIED MEGA-SKILL SUMMARY")
    print(f"{'=' * 70}")

    total_families = len(families)
    existing_with_members = sum(1 for f in families.values()
                                if f.source == "llm_judge" and f.members)
    new_fams = sum(1 for f in families.values() if f.source == "new")
    multi_task = sum(1 for f in families.values() if len(f.tasks) >= 2)
    multi_stage = sum(1 for f in families.values()
                      if 1 in f.stages and 2 in f.stages)

    print(f"  Total families:          {total_families}")
    print(f"    Existing (with skills):{existing_with_members:3d}")
    print(f"    New:                   {new_fams:3d}")
    print(f"  Multi-task (≥2 tasks):   {multi_task}")
    print(f"  Cross-stage (S1+S2):     {multi_stage}")
    print(f"  Total skills:            {sum(len(f.members) for f in families.values())}")

    # Print per-family breakdown
    print(f"\n{'─' * 70}")
    print(f"{'Family':45s} {'Src':5s} {'#Sk':>4s} {'#Tasks':>6s} {'Stages':>8s}  Tasks")
    print(f"{'─' * 70}")
    for fid, fam in sorted(families.items(),
                            key=lambda x: -len(x[1].members)):
        if not fam.members:
            continue
        stages_str = ",".join(str(s) for s in sorted(fam.stages))
        tasks_short = ",".join(sorted(t.replace("gymv_", "") for t in fam.tasks))
        if len(tasks_short) > 50:
            tasks_short = tasks_short[:47] + "..."
        print(f"  {fid:43s} {fam.source:5s} {len(fam.members):4d} "
              f"{len(fam.tasks):6d} {stages_str:>8s}  {tasks_short}")

    # 7. Write outputs
    print(f"\n[7] Writing outputs to {out_dir}...")

    # unified_mega_skills.json
    output_data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_skills": sum(len(f.members) for f in families.values()),
        "total_families": total_families,
        "existing_families_used": existing_with_members,
        "new_families_created": new_fams,
        "cross_stage_families": multi_stage,
        "stage1_banks": list(STAGE1_BANKS.keys()),
        "stage2_banks": list(STAGE2_BANKS.keys()),
        "mega_skills": {},
    }

    for fid, fam in sorted(families.items(),
                            key=lambda x: -len(x[1].members)):
        if not fam.members:
            continue
        output_data["mega_skills"][fid] = {
            "display_name": fam.display_name,
            "plan_signature": fam.plan_sig,
            "source": fam.source,
            "n_members": len(fam.members),
            "n_tasks": len(fam.tasks),
            "tasks": sorted(fam.tasks),
            "stages": sorted(fam.stages),
            "skills": [
                {
                    "task": m.task,
                    "skill_id": m.skill_id,
                    "name": m.name,
                    "plan_signature": m.plan_sig,
                    "genre": m.genre,
                    "stage": m.stage,
                    "domain": m.domain,
                }
                for m in sorted(fam.members, key=lambda m: (m.task, m.skill_id))
            ],
        }

    labels_path = out_dir / "unified_mega_skill_labels.json"
    with open(labels_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  → {labels_path}")

    # Per-skill flat index (for downstream lookups)
    flat_index = {}
    for fid, fam in families.items():
        for m in fam.members:
            flat_index[m.key()] = {
                "family_id": fid,
                "family_source": fam.source,
                "plan_signature": m.plan_sig,
                "stage": m.stage,
            }

    index_path = out_dir / "skill_to_family_index.json"
    with open(index_path, "w") as f:
        json.dump(flat_index, f, indent=2, ensure_ascii=False)
    print(f"  → {index_path}")

    # Markdown summary
    md_lines = [
        "# Unified Mega-Skills: Stage 1 + Stage 2",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Total families: **{total_families}** "
        f"({existing_with_members} existing + {new_fams} new)",
        f"- Total skills: **{sum(len(f.members) for f in families.values())}**",
        f"- Cross-stage families (S1+S2): **{multi_stage}**",
        f"- Stage 1 games: {', '.join(STAGE1_BANKS.keys())}",
        f"- Stage 2 games: {', '.join(STAGE2_BANKS.keys())}",
        "",
        "## Families by size",
        "",
    ]
    for fid, fam in sorted(families.items(),
                            key=lambda x: -len(x[1].members)):
        if not fam.members:
            continue
        stages_str = "+".join(f"S{s}" for s in sorted(fam.stages))
        md_lines.append(f"### `{fid}` ({fam.source}) — {len(fam.members)} skills, "
                       f"{len(fam.tasks)} tasks [{stages_str}]")
        md_lines.append(f"- Plan signature: `{fam.plan_sig}`")
        md_lines.append(f"- Tasks: {', '.join(sorted(fam.tasks))}")
        md_lines.append("- Members:")
        for m in sorted(fam.members, key=lambda m: (m.stage, m.task, m.skill_id)):
            stage_tag = "S1" if m.stage == 1 else "S2"
            md_lines.append(f"  - [{stage_tag}] `{m.task}::{m.skill_id}` — {m.name}")
        md_lines.append("")

    md_path = out_dir / "UNIFIED_MEGA_SKILLS_SUMMARY.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  → {md_path}")

    print(f"\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
