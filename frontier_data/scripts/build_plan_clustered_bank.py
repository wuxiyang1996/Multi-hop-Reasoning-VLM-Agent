#!/usr/bin/env python
"""Build the shared abstract bank using plan-level LLM judge clustering.

DEFAULT clustering method: groups skills by shared reasoning procedure
as determined by plan-level LLM-as-judge similarity scores, rather than
by skill name stems or structural signatures.

Algorithm:
  1. Load all per-task skills and the plan-level judge results
  2. Build a similarity graph: edge between (skill_A, skill_B) if
     plan-level judge scored them >= threshold (default 4)
  3. Find connected components — each component is one mega-skill
  4. For skills with no judge edges, fall back to collapsed-signature
     grouping so nothing is orphaned
  5. Elect a representative per cluster, infer template signature,
     and emit the standard TwoLayerSkillStore layout

Output (same format as build_shared_bank.py):
  frontier_data/output/shared_skill_bank/
    abstract.jsonl
    by_task/<task>/bindings.jsonl
    SUMMARY.json

Usage:
    python frontier_data/scripts/build_plan_clustered_bank.py
    python frontier_data/scripts/build_plan_clustered_bank.py --threshold 3
    python frontier_data/scripts/build_plan_clustered_bank.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("build_plan_clustered_bank")

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

COHORT_MAP = {
    "Temporal_": "gymv_game",
    "tetris": "env_wr_game", "super_mario": "env_wr_game",
    "candy_crush": "env_wr_game", "twenty_forty_eight": "env_wr_game",
    "browsergym": "web", "miniwob": "web", "webshop": "web",
    "osworld": "web",
    "siv_bench": "vr_video", "video_holmes": "vr_video",
    "tir_bench": "vr_image", "visual_toolbench": "vr_image",
}

DOMAIN_MAP = {
    "gymv_game": "GAME", "env_wr_game": "GAME",
    "web": "WEB",
    "vr_image": "VR", "vr_video": "VR",
}

TEMPLATE_OPS = {
    "PERCEIVE", "RECALL", "COMPARE", "FILTER",
    "DECIDE", "COMMIT", "VERIFY", "RECOVER",
}


def task_to_cohort(task: str) -> str:
    for prefix, cohort in COHORT_MAP.items():
        if task.startswith(prefix) or task == prefix:
            return cohort
    return "unknown"


def task_to_domain(task: str) -> str:
    return DOMAIN_MAP.get(task_to_cohort(task), "OTHER")


# ── Load skills ──────────────────────────────────────────────────────

def extract_skill_core(d: dict) -> dict:
    if "skill" in d and isinstance(d["skill"], dict):
        return d["skill"]
    return d


def load_per_task_skills(per_task_root: Path) -> Dict[str, Dict[str, dict]]:
    """Returns {task: {skill_id: skill_record}}."""
    result: Dict[str, Dict[str, dict]] = {}
    for task_dir in sorted(per_task_root.iterdir()):
        if not task_dir.is_dir():
            continue
        sb = task_dir / "skill_bank.jsonl"
        if not sb.exists():
            continue
        skills = {}
        with open(sb) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = extract_skill_core(d)
                sid = s.get("skill_id", s.get("name", ""))
                if sid:
                    skills[sid] = d
        result[task_dir.name] = skills
    return result


# ── Load judge results ───────────────────────────────────────────────

def load_plan_level_judgments(
    path: Path, threshold: int = 4,
) -> List[Dict[str, Any]]:
    """Load plan-level judge results, return edges above threshold."""
    with open(path) as f:
        data = json.load(f)

    edges = []
    for result in data.get("results", []):
        target_id = result["target_skill_id"]
        target_task = result["target_task"]
        target_domain = result["target_domain"]

        for match in result.get("judgment", {}).get("matches", []):
            score = match.get("score", 0)
            if score < threshold:
                continue
            cand_id = match.get("candidate_skill_id", "")
            cand_task = match.get("candidate_task", "")
            if not cand_id:
                continue
            edges.append({
                "skill_a": target_id,
                "task_a": target_task,
                "domain_a": target_domain,
                "skill_b": cand_id,
                "task_b": cand_task,
                "domain_b": "GAME" if target_domain != "GAME" else match.get("candidate_domain", "GAME"),
                "score": score,
                "shared_reasoning": match.get("shared_reasoning", ""),
                "transfer_value": match.get("transfer_value", ""),
            })

    return edges


def load_sig_level_judgments(
    path: Path, threshold: int = 4,
) -> List[Dict[str, Any]]:
    """Load signature-level judge results as additional edges."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)

    edges = []
    for j in data.get("judgments", []):
        score = j.get("judgment", {}).get("score", 0)
        if score < threshold:
            continue
        edges.append({
            "skill_a": j.get("skill_id_a", ""),
            "task_a": j.get("task_a", ""),
            "domain_a": j.get("domain_a", ""),
            "skill_b": j.get("skill_id_b", ""),
            "task_b": j.get("task_b", ""),
            "domain_b": j.get("domain_b", ""),
            "score": score,
            "shared_reasoning": j.get("judgment", {}).get("shared_reasoning", ""),
            "transfer_value": j.get("judgment", {}).get("transfer_value", ""),
        })
    return edges


# ── Union-Find for connected components ──────────────────────────────

class UnionFind:
    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for x in self._parent:
            groups[self.find(x)].append(x)
        return dict(groups)


# ── Cluster skills ───────────────────────────────────────────────────

def build_skill_key(task: str, skill_id: str) -> str:
    return f"{task}::{skill_id}"


def parse_skill_key(key: str) -> Tuple[str, str]:
    parts = key.split("::", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else ("", key)


def cluster_by_plan_judge(
    all_tasks: Dict[str, Dict[str, dict]],
    edges: List[Dict[str, Any]],
) -> Dict[str, List[Tuple[str, str]]]:
    """Cluster skills using plan-level judge edges + union-find.

    Returns {cluster_id: [(task, skill_id), ...]}
    """
    uf = UnionFind()

    all_keys: Set[str] = set()
    for task, skills in all_tasks.items():
        for sid in skills:
            key = build_skill_key(task, sid)
            all_keys.add(key)
            uf.find(key)

    n_edges_used = 0
    for edge in edges:
        key_a = build_skill_key(edge["task_a"], edge["skill_a"])
        key_b = build_skill_key(edge["task_b"], edge["skill_b"])
        if key_a in all_keys and key_b in all_keys:
            uf.union(key_a, key_b)
            n_edges_used += 1

    logger.info("Union-Find: %d edges used from %d total", n_edges_used, len(edges))

    raw_components = uf.components()
    clusters: Dict[str, List[Tuple[str, str]]] = {}
    for root, members in raw_components.items():
        cluster_id = f"plan_cluster_{root}"
        clusters[cluster_id] = [parse_skill_key(m) for m in members]

    return clusters


def fallback_signature_clusters(
    all_tasks: Dict[str, Dict[str, dict]],
    already_clustered: Set[str],
) -> Dict[str, List[Tuple[str, str]]]:
    """For skills not reached by judge edges, group by collapsed_signature."""
    by_csig: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for task, skills in all_tasks.items():
        for sid, raw in skills.items():
            key = build_skill_key(task, sid)
            if key in already_clustered:
                continue
            s = extract_skill_core(raw)
            csig = s.get("collapsed_signature", s.get("template_signature", ""))
            stem = normalise_skill_id(sid)
            group_key = csig if csig else f"stem_{stem}"
            by_csig[group_key].append((task, sid))

    return {f"sig_cluster_{k}": members for k, members in by_csig.items()}


# ── Build shared bank from clusters ──────────────────────────────────

def infer_template_signature(skill: dict) -> str:
    ts = skill.get("template_signature", "")
    if ts and "→" in ts:
        return ts

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

    return "PERCEIVE → DECIDE → COMMIT"


def build_template_steps(signature: str) -> List[dict]:
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
    return [{"op": op, "predicate": predicates.get(op, f"Execute {op.lower()} step")}
            for op in ops]


def build_protocol_steps(signature: str) -> List[dict]:
    ops = [op.strip() for op in signature.split("→")]
    return [{
        "op": op,
        "payload": {f"{op.lower()}_target": f"${{{op.lower()}_target}}"},
        "slot_types": {f"{op.lower()}_target": "tracked_entity"},
        "preconditions": [],
        "effects_add": [],
        "effects_del": [],
        "evidence_role": op.lower(),
        "notes": "",
    } for op in ops]


def elect_cluster_representative(
    members: List[Tuple[str, str]],
    all_tasks: Dict[str, Dict[str, dict]],
) -> Tuple[str, str, dict]:
    """Pick the richest skill record as cluster representative."""
    best_task, best_sid, best_raw = "", "", {}
    best_size = 0
    for task, sid in members:
        raw = all_tasks.get(task, {}).get(sid, {})
        size = len(json.dumps(raw))
        if size > best_size:
            best_task, best_sid, best_raw = task, sid, raw
            best_size = size
    return best_task, best_sid, best_raw


def derive_cluster_name(
    members: List[Tuple[str, str]],
    all_tasks: Dict[str, Dict[str, dict]],
    edges: List[Dict[str, Any]],
    cluster_id: str,
) -> str:
    """Derive a human-readable name for the cluster from the best
    shared_reasoning description across its edges."""
    member_keys = {build_skill_key(t, s) for t, s in members}
    best_desc = ""
    best_score = 0
    for edge in edges:
        ka = build_skill_key(edge["task_a"], edge["skill_a"])
        kb = build_skill_key(edge["task_b"], edge["skill_b"])
        if ka in member_keys or kb in member_keys:
            if edge["score"] > best_score:
                best_score = edge["score"]
                best_desc = edge.get("shared_reasoning", "")

    if best_desc:
        return best_desc[:120]

    _, _, rep_raw = elect_cluster_representative(members, all_tasks)
    rep = extract_skill_core(rep_raw)
    return str(rep.get("name", rep.get("skill_id", cluster_id)))[:120]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--threshold", type=int, default=4,
                    help="Minimum judge score to form an edge (default: 4)")
    ap.add_argument("--plan-judgments",
                    default=str(REPO_ROOT / "frontier_data" / "output" / "plan_level_similarity_judgments.json"))
    ap.add_argument("--sig-judgments",
                    default=str(REPO_ROOT / "frontier_data" / "output" / "plan_similarity_judgments.json"))
    ap.add_argument("--per-task-root",
                    default=str(REPO_ROOT / "frontier_data" / "output" / "per_task_banks"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "frontier_data" / "output" / "shared_skill_bank"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    per_task_root = Path(args.per_task_root)
    out_root = Path(args.out)

    # ── 1. Load all per-task skills ──────────────────────────────
    logger.info("Loading per-task banks from %s", per_task_root)
    all_tasks = load_per_task_skills(per_task_root)
    total_skills = sum(len(v) for v in all_tasks.values())
    logger.info("Loaded %d skills across %d tasks", total_skills, len(all_tasks))

    # ── 2. Load judge results ────────────────────────────────────
    plan_path = Path(args.plan_judgments)
    sig_path = Path(args.sig_judgments)

    edges = []
    if plan_path.exists():
        plan_edges = load_plan_level_judgments(plan_path, args.threshold)
        logger.info("Plan-level judge: %d edges (score >= %d)", len(plan_edges), args.threshold)
        edges.extend(plan_edges)
    else:
        logger.warning("Plan-level judgments not found at %s", plan_path)

    if sig_path.exists():
        sig_edges = load_sig_level_judgments(sig_path, args.threshold)
        logger.info("Signature-level judge: %d edges (score >= %d)", len(sig_edges), args.threshold)
        edges.extend(sig_edges)

    if not edges:
        logger.error("No judge results found — run judge_plan_level_similarity.py first")
        return 1

    # ── 3. Cluster via union-find on judge edges ─────────────────
    plan_clusters = cluster_by_plan_judge(all_tasks, edges)

    already_clustered: Set[str] = set()
    for members in plan_clusters.values():
        for task, sid in members:
            already_clustered.add(build_skill_key(task, sid))

    multi_member_clusters = {k: v for k, v in plan_clusters.items() if len(v) >= 2}
    singleton_judge = {k: v for k, v in plan_clusters.items() if len(v) == 1}

    orphan_skills = set()
    for task, skills in all_tasks.items():
        for sid in skills:
            key = build_skill_key(task, sid)
            if key not in already_clustered:
                orphan_skills.add(key)

    for members in singleton_judge.values():
        for task, sid in members:
            orphan_skills.add(build_skill_key(task, sid))

    fallback = fallback_signature_clusters(
        all_tasks,
        already_clustered - {build_skill_key(t, s)
                             for members in singleton_judge.values()
                             for t, s in members},
    )

    logger.info("Plan-judge clusters: %d (%d multi-member, %d singleton)",
                len(plan_clusters), len(multi_member_clusters), len(singleton_judge))
    logger.info("Fallback sig clusters: %d (covering %d orphan skills)",
                len(fallback), sum(len(v) for v in fallback.values()))

    all_clusters = {**multi_member_clusters, **fallback}
    for cid, members in singleton_judge.items():
        task, sid = members[0]
        s = extract_skill_core(all_tasks.get(task, {}).get(sid, {}))
        csig = s.get("collapsed_signature", s.get("template_signature", ""))
        stem = normalise_skill_id(sid)
        fkey = f"sig_cluster_{csig}" if csig else f"sig_cluster_stem_{stem}"
        if fkey in all_clusters:
            all_clusters[fkey].append((task, sid))
        else:
            all_clusters[cid] = members

    logger.info("Total clusters: %d", len(all_clusters))

    if args.dry_run:
        logger.info("DRY-RUN: would write %d abstracts to %s", len(all_clusters), out_root)
        for cid, members in sorted(all_clusters.items(), key=lambda x: -len(x[1]))[:20]:
            domains = sorted({task_to_domain(t) for t, _ in members})
            tasks = sorted({t for t, _ in members})
            logger.info("  %s: %d members, domains=%s, tasks=%s",
                        cid[:60], len(members), domains, tasks[:5])
        return 0

    # ── 4. Build SharedAbstractSkill + BoundConcreteSkill ────────
    out_root.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    abstracts: List[SharedAbstractSkill] = []
    all_bindings: Dict[str, List[BoundConcreteSkill]] = defaultdict(list)

    for cluster_id, members in sorted(all_clusters.items()):
        rep_task, rep_sid, rep_raw = elect_cluster_representative(members, all_tasks)
        rep = extract_skill_core(rep_raw)

        sig = infer_template_signature(rep)
        cluster_name = derive_cluster_name(members, all_tasks, edges, cluster_id)

        abstract_id = cluster_id
        if cluster_id.startswith("plan_cluster_"):
            suffix = cluster_id.removeprefix("plan_cluster_")
            _, first_sid = parse_skill_key(suffix)
            abstract_id = f"plan.{normalise_skill_id(first_sid)}"
        elif cluster_id.startswith("sig_cluster_"):
            abstract_id = cluster_id.removeprefix("sig_cluster_")

        lineage_entries = []
        cohorts_seen: Set[str] = set()
        for task, sid in members:
            raw = all_tasks.get(task, {}).get(sid, {})
            s = extract_skill_core(raw)
            cohort = task_to_cohort(task)
            cohorts_seen.add(cohort)
            lineage_entries.append(LineageEntry(
                task=task,
                concrete_skill_id=normalise_skill_id(sid),
                raw_skill_id=sid,
                cohort=cohort,
                discovered_via="plan_judge_clustering",
                is_native=True,
                n_uses=0,
                n_success=0,
                n_translated_uses=0,
                contract_hash=hashlib.md5(
                    json.dumps(s.get("contract", {}), sort_keys=True).encode()
                ).hexdigest()[:8],
            ))

        abstract = SharedAbstractSkill(
            abstract_skill_id=abstract_id,
            name=cluster_name,
            template_signature=sig,
            template_steps=[TemplateStep.from_dict(ts)
                            for ts in build_template_steps(sig)],
            protocol_steps=[ProtocolStep.from_dict(ps)
                            for ps in build_protocol_steps(sig)],
            lineage=lineage_entries,
            cohorts_seen=sorted(cohorts_seen),
            discovered_via="plan_judge_clustering",
            schema_version=1,
            created_at=now_iso,
            updated_at=now_iso,
        )
        abstracts.append(abstract)

        for task, sid in members:
            raw = all_tasks.get(task, {}).get(sid, {})
            s = extract_skill_core(raw)
            contract = s.get("contract", {})
            if isinstance(contract, str):
                try:
                    contract = json.loads(contract)
                except json.JSONDecodeError:
                    contract = {}

            bound = BoundConcreteSkill(
                concrete_skill_id=normalise_skill_id(sid),
                task=task,
                abstract_skill_id=abstract_id,
                name=str(s.get("name", s.get("skill_id", ""))),
                protocol=[ProtocolStep.from_dict(ps) for ps in build_protocol_steps(sig)],
                contract=contract,
                sub_episodes=[],
                binding_status="VALIDATED",
                binding_source="plan_judge_clustering",
                raw_skill_id=sid,
                schema_version=2,
                created_at=now_iso,
                updated_at=now_iso,
            )
            all_bindings[task].append(bound)

    logger.info("Created %d abstract mega-skills", len(abstracts))

    # ── 5. Write TwoLayerSkillStore ──────────────────────────────
    abstract_path = out_root / "abstract.jsonl"
    with open(abstract_path, "w") as f:
        for a in abstracts:
            f.write(json.dumps(a.to_dict()) + "\n")
    logger.info("Wrote %d abstracts to %s", len(abstracts), abstract_path)

    for task, bindings in sorted(all_bindings.items()):
        task_dir = out_root / "by_task" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        bind_path = task_dir / "bindings.jsonl"
        with open(bind_path, "w") as f:
            for b in bindings:
                f.write(json.dumps(b.to_dict()) + "\n")

    # ── 6. Summary ───────────────────────────────────────────────
    sig_counts = Counter(a.template_signature for a in abstracts)
    cohort_counts = Counter()
    for a in abstracts:
        for c in a.cohorts_seen:
            cohort_counts[c] += 1

    multi_task = [a for a in abstracts if a.n_bound_tasks >= 2]
    cross_domain = [a for a in abstracts
                    if len({DOMAIN_MAP.get(c, "OTHER") for c in a.cohorts_seen}) >= 2]

    domain_coverage = defaultdict(set)
    for a in cross_domain:
        for L in a.lineage:
            domain_coverage[task_to_domain(L.task)].add(L.concrete_skill_id)

    summary = {
        "generated_utc": now_iso,
        "clustering_method": "plan_level_llm_judge",
        "judge_threshold": args.threshold,
        "n_input_skills": total_skills,
        "n_abstracts": len(abstracts),
        "n_multi_task_abstracts": len(multi_task),
        "n_cross_domain_abstracts": len(cross_domain),
        "n_tasks_with_bindings": len(all_bindings),
        "total_bindings": sum(len(v) for v in all_bindings.values()),
        "top_signatures": sig_counts.most_common(15),
        "cohort_coverage": dict(cohort_counts),
        "cross_domain_skill_coverage": {
            d: len(sids) for d, sids in domain_coverage.items()
        },
        "multi_task_examples": [
            {
                "id": a.abstract_skill_id,
                "name": a.name[:100],
                "signature": a.template_signature,
                "domains": sorted({DOMAIN_MAP.get(c, "OTHER") for c in a.cohorts_seen}),
                "tasks": sorted({L.task for L in a.lineage}),
                "n_members": len(a.lineage),
            }
            for a in sorted(cross_domain, key=lambda x: -x.n_bound_tasks)[:25]
        ],
    }
    with open(out_root / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print report ─────────────────────────────────────────────
    logger.info("═" * 65)
    logger.info("PLAN-CLUSTERED SHARED BANK BUILT SUCCESSFULLY")
    logger.info("  Clustering: plan-level LLM judge (threshold=%d)", args.threshold)
    logger.info("  %d abstract mega-skills", len(abstracts))
    logger.info("  %d multi-task (span ≥2 tasks)", len(multi_task))
    logger.info("  %d cross-domain (span ≥2 domains)", len(cross_domain))
    logger.info("  %d tasks with concrete bindings", len(all_bindings))
    logger.info("  %d total bindings", sum(len(v) for v in all_bindings.values()))
    logger.info("  Top 5 template signatures:")
    for sig, cnt in sig_counts.most_common(5):
        logger.info("    %3d  %s", cnt, sig)
    logger.info("  Cohort coverage: %s", dict(cohort_counts))

    if cross_domain:
        logger.info("  Top cross-domain mega-skills:")
        for a in sorted(cross_domain, key=lambda x: -x.n_bound_tasks)[:10]:
            domains = sorted({DOMAIN_MAP.get(c, "OTHER") for c in a.cohorts_seen})
            logger.info("    %s (%d tasks, %s): %s",
                        a.abstract_skill_id[:40], a.n_bound_tasks,
                        "+".join(domains), a.name[:60])

    logger.info("  Output: %s", out_root)
    logger.info("═" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
