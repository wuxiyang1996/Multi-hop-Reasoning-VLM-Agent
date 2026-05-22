#!/usr/bin/env python
"""Build mega-skills from plan-level LLM-judge similarity output.

Replaces ``frontier_data/scripts/build_plan_clustered_bank.py`` because that
script:

  * forces every cluster's ``template_signature`` to a hardcoded
    ``PERCEIVE → DECIDE → COMMIT`` fallback (real plan info is lost),
  * does NOT dedupe bindings within a cluster (same skill instance can
    appear twice),
  * uses non-unique ``abstract_skill_id`` (multiple clusters share
    ``plan.COMMIT/CLEAR``), and
  * cohort-tags ``gymv_*`` tasks as ``unknown``.

This script:

  1. Loads source skill banks (5 best GRPO banks by default — same files
     ``judge_plan_level_similarity.py`` consumed).
  2. Reads the judge's similarity edges (``plan_level_similarity_judgments.json``).
  3. Runs union-find on edges >= --threshold to form clusters.
  4. For each cluster: dedupes by ``(task, skill_id)``, picks a
     representative as the member with the most complete protocol,
     extracts the *consensus* plan signature (most-frequent compressed
     intent sequence across members), and packs:

       - ``mega_skill_id``       : unique id derived from cluster index + rep
       - ``representative``      : pointer to rep skill
       - ``template_signature``  : consensus signature (e.g. ``EVALUATE → ACT → PERCEIVE → ACT``)
       - ``template_steps``      : rep's ACTUAL protocol.steps (verbatim NL)
       - ``step_checks``         : rep's protocol.step_checks if any
       - ``preconditions``       : rep's protocol.preconditions
       - ``members``              : deduped list of binding records
                                    (task, skill_id, name, plan_signature, contract_hash)
       - ``icl_exemplar``        : 1-shot exemplar pulled from rep.protocol_raw.steps
       - ``cohorts``             : set of task cohorts spanned (game cohort is fine here)
       - ``judge_evidence``      : top-3 (shared_reasoning, transfer_value)
                                    excerpts from the judge

Output (default):
  frontier_data/output/megaskills_from_judge/
    mega_skills.jsonl          one mega-skill per line
    mega_skills_SUMMARY.md     human-readable summary
    mega_skills_lineage.json   per-mega-skill member lineage (full)

Usage::

    python scripts/build_megaskills_from_judge.py \\
        --judgments frontier_data/output/plan_level_similarity_judgments.json \\
        --threshold 4 \\
        --out frontier_data/output/megaskills_from_judge
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the same intent classifier + bank loader from the judge script
from scripts.judge_plan_level_similarity import (  # noqa: E402
    BEST_GRPO_RUNS, ALL_STAGES, DOMAIN_OF,
    SkillRec, compressed_plan, load_bank,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("build_megaskills_from_judge")


# ── Union-Find ───────────────────────────────────────────────────────

class UnionFind:
    def __init__(self) -> None:
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.p:
            self.p[x] = x
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = defaultdict(list)
        for k in self.p:
            out[self.find(k)].append(k)
        return dict(out)


# ── Skill record loader (full payload, not just summary) ─────────────

@dataclass
class FullSkill:
    task: str
    skill_id: str
    name: str
    description: str
    domain: str
    protocol_steps: List[str]
    protocol_raw_steps: List[str]
    step_checks: List[Any]
    preconditions: List[str]
    contract: Dict[str, Any]
    contract_hash: str

    def key(self) -> str:
        return f"{self.task}::{self.skill_id}"

    def plan_signature(self) -> str:
        return " → ".join(compressed_plan(self.protocol_steps))


_VALID_ACTIONS_RE = re.compile(
    r"\s*[\n\r]+\s*Valid actions:[^\n]*\.\s*Choose one\.?", re.I
)
_PREDICATE_LEAK_RE = re.compile(
    r"world\.[a-z_]+=[\w.-]+", re.I
)


def _sanitize_step(s: str) -> str:
    """Strip ``Valid actions:`` and other prompt-template leaks that the
    GRPO actor concatenated into protocol step text.  Also collapse
    runaway whitespace.
    """
    s = _VALID_ACTIONS_RE.sub("", s)
    s = re.sub(r"\s*[\n\r]+\s*", " ", s).strip()
    # If the step is now just predicate fragments like "Achieve:
    # world.lives=3, world.score=80" with nothing else, keep it but
    # collapse multiple commas / spaces.
    s = re.sub(r"\s{2,}", " ", s).strip(",; ")
    return s


def _as_steps(p) -> List[str]:
    if not p:
        return []
    raw: List[str] = []
    if isinstance(p, list):
        raw = [str(x) for x in p if x]
    elif isinstance(p, dict):
        s = p.get("steps") or []
        if isinstance(s, list):
            raw = [str(x) for x in s if x]
    return [_sanitize_step(x) for x in raw if x and str(x).strip()]


def _short_hash(d: dict) -> str:
    import hashlib
    payload = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def load_full_skills(bank_paths: Dict[str, Path]) -> Dict[str, FullSkill]:
    """Returns {task::skill_id : FullSkill}."""
    out: Dict[str, FullSkill] = {}
    for task, path in bank_paths.items():
        if not path.exists():
            logger.warning("bank not found: %s", path)
            continue
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = d.get("skill", d)
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("skill_id", s.get("name", "")))
                if not sid:
                    continue
                protocol = s.get("protocol") or {}
                protocol_raw = s.get("protocol_raw") or {}
                contract = s.get("contract") or {}
                rec = FullSkill(
                    task=task,
                    skill_id=sid,
                    name=str(s.get("name", "")),
                    description=str(
                        s.get("strategic_description", "")
                        or contract.get("description", "")
                    ),
                    domain=DOMAIN_OF.get(task, "OTHER"),
                    protocol_steps=_as_steps(protocol),
                    protocol_raw_steps=_as_steps(protocol_raw),
                    step_checks=[
                        _sanitize_step(str(x))
                        for x in (protocol.get("step_checks") or [])
                        if isinstance(protocol, dict) and x
                    ] if isinstance(protocol, dict) else [],
                    preconditions=[
                        _sanitize_step(str(x))
                        for x in (protocol.get("preconditions") or [])
                        if isinstance(protocol, dict) and x
                    ] if isinstance(protocol, dict) else [],
                    contract=contract,
                    contract_hash=_short_hash(contract),
                )
                out[rec.key()] = rec
                n += 1
        logger.info("  %-30s %3d skills", task, n)
    return out


# ── Cluster builder ──────────────────────────────────────────────────

def cluster_skills(
    judgments: dict,
    full_skills: Dict[str, FullSkill],
    threshold: int,
    *,
    core_threshold: int = 5,
    require_mutual_core: bool = True,
    attach_min_score: int = 4,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], List[dict]]]:
    """Two-stage cluster builder driven by judge scores.

    Stage 1 — CORE clusters.
        Build union-find on edges with score ≥ ``core_threshold``.  When
        ``require_mutual_core=True`` we additionally require *both*
        directions of a pair to satisfy the threshold (A judged B ≥ T
        AND B judged A ≥ T) before merging.  This prevents the
        score-4 edge explosion that collapses every skill into one
        giant component.

    Stage 2 — SATELLITE attach.
        Skills not in any multi-member core cluster are attached to the
        core cluster against which they have the highest-scoring outgoing
        edge, provided that score is ≥ ``attach_min_score``.  Skills with
        no such edge remain singletons.

    Returns ``(clusters, edge_evidence)`` where edge_evidence keeps ALL
    edges with score ≥ ``threshold`` (the original loose filter) so the
    final mega-skill records can include strong-and-weak rationales.
    """
    # ── Build the directed edge index from the judge file ────────────
    # edge_score[(a, b)] = score from a judging b
    edge_score: Dict[Tuple[str, str], int] = {}
    edge_payload: Dict[Tuple[str, str], dict] = {}

    n_seen = 0
    for result in judgments.get("results", []):
        t_key = f"{result['target_task']}::{result['target_skill_id']}"
        if t_key not in full_skills:
            continue
        for m in result.get("judgment", {}).get("matches", []):
            n_seen += 1
            score = int(m.get("score", 0))
            c_key = f"{m['candidate_task']}::{m['candidate_skill_id']}"
            if c_key not in full_skills:
                continue
            edge_score[(t_key, c_key)] = score
            edge_payload[(t_key, c_key)] = {
                "score": score,
                "shared_reasoning": m.get("shared_reasoning", ""),
                "transfer_value": m.get("transfer_value", ""),
                "from": t_key, "to": c_key,
            }

    # Collect undirected evidence (kept ≥ loose threshold) for output
    edge_evidence: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    n_kept = 0
    for (a, b), score in edge_score.items():
        if score >= threshold:
            edge_evidence[tuple(sorted([a, b]))].append(edge_payload[(a, b)])
            n_kept += 1

    logger.info("edges: %d total / %d ≥ loose-threshold %d",
                n_seen, n_kept, threshold)
    logger.info("score=5 directed edges: %d",
                sum(1 for s in edge_score.values() if s >= 5))

    # ── Stage 1: build CORE clusters (mutual high-score) ─────────────
    uf = UnionFind()
    for k in full_skills:
        uf.find(k)

    n_core_edges = 0
    for (a, b), s_ab in edge_score.items():
        if s_ab < core_threshold:
            continue
        if require_mutual_core:
            s_ba = edge_score.get((b, a), 0)
            if s_ba < core_threshold:
                continue
        uf.union(a, b)
        n_core_edges += 1
    logger.info("core edges used (mutual=%s, ≥%d): %d",
                require_mutual_core, core_threshold, n_core_edges)

    raw_groups = uf.groups()
    core_clusters = {
        root: sorted(set(members))
        for root, members in raw_groups.items()
        if len(members) >= 2
    }
    in_core: set = set()
    for members in core_clusters.values():
        in_core.update(members)
    logger.info("Stage 1: %d core clusters, %d skills in cores",
                len(core_clusters), len(in_core))

    # ── Stage 2: attach satellites to nearest core cluster ───────────
    # For each skill not in any core, find the highest-scoring outgoing
    # edge to a core-cluster member.  Tie-break: more cluster members at
    # high score wins, then by cluster's representative key.
    cluster_assign = dict(core_clusters)  # root -> members
    member_to_root: Dict[str, str] = {}
    for root, members in cluster_assign.items():
        for m in members:
            member_to_root[m] = root

    satellites_attached = 0
    satellites_orphan = 0
    for skill_key in full_skills:
        if skill_key in in_core:
            continue
        # Outgoing edges from this skill
        best_root: Optional[str] = None
        best_score = 0
        best_root_total = 0
        for (a, b), s in edge_score.items():
            if a != skill_key:
                continue
            if s < attach_min_score:
                continue
            tgt_root = member_to_root.get(b)
            if tgt_root is None:
                continue
            # Score weighted by how strong this single edge is; ties broken
            # by counting how many edges this skill has to the cluster
            members_set = set(cluster_assign[tgt_root])
            n_strong = sum(
                1 for (a2, b2), s2 in edge_score.items()
                if a2 == skill_key and b2 in members_set and s2 >= attach_min_score
            )
            if (s, n_strong) > (best_score, best_root_total):
                best_score = s
                best_root_total = n_strong
                best_root = tgt_root
        if best_root is not None:
            cluster_assign[best_root] = sorted(set(cluster_assign[best_root] + [skill_key]))
            member_to_root[skill_key] = best_root
            satellites_attached += 1
        else:
            cluster_assign[skill_key] = [skill_key]  # singleton
            satellites_orphan += 1
    logger.info("Stage 2: %d satellites attached, %d orphans",
                satellites_attached, satellites_orphan)

    return cluster_assign, edge_evidence


def pick_representative(members: List[FullSkill]) -> FullSkill:
    """Pick the member with the most complete protocol_steps.

    Tie-break by length of description, then by skill_id alphabetic order
    so the choice is deterministic.
    """
    def score(s: FullSkill) -> Tuple[int, int, int, str]:
        n_steps = len(s.protocol_steps)
        n_chars = sum(len(x) for x in s.protocol_steps)
        n_desc = len(s.description or "")
        return (-n_steps, -n_chars, -n_desc, s.key())
    return sorted(members, key=score)[0]


def consensus_signature(members: List[FullSkill]) -> str:
    """Most-frequent compressed plan signature across cluster members."""
    sigs = [m.plan_signature() for m in members if m.protocol_steps]
    if not sigs:
        return ""
    counts = Counter(sigs)
    return counts.most_common(1)[0][0]


_NAME_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slug(text: str) -> str:
    """Snake-case a free-form name; drops repeated underscores."""
    s = _NAME_SLUG_RE.sub("_", text or "").strip("_").lower()
    return s


def derive_mega_skill_id(
    cluster_idx: int,
    rep: FullSkill,
    members: List[FullSkill],
    used_ids: set,
) -> str:
    """Produce a stable, human-readable mega-skill id.

    Strategy: ``mega.NNN.<slug>`` where ``slug`` is derived from the
    representative skill's ``name`` field (e.g. ``"Commit/Match4Stripe"``
    → ``commit_match4stripe``), stripping the ``early:``/``late:``/
    ``mid:`` prefixes which are stage markers, not concept labels.
    If the slug is empty (rare), fall back to a short verb-bag derived
    from rep's first protocol step.  Numeric ``NNN`` prefix preserves
    stable ordering by cluster size.
    """
    raw = rep.name or rep.skill_id
    # Note: keep stage markers (early/mid/late) in the *full* slug so that
    # multiple stage variants of the same concept don't collide.  The
    # short slug below first tries the bare name and only adds the stage
    # qualifier when needed.
    bare = re.sub(r"^(early|mid|late)\s*[:_]\s*", "", raw, flags=re.I)
    bare_slug = _slug(bare)[:40]
    full_slug = _slug(raw)[:40]
    if not bare_slug or bare_slug.startswith("skill_"):
        # Hash-style name → derive verb bag from first protocol step
        # description (Commit/Match4Stripe etc. won't hit this branch
        # since they have proper names).
        seed = rep.protocol_steps[0] if rep.protocol_steps else (rep.description or "")
        tokens = [
            t for t in re.findall(r"[A-Za-z]+", seed.lower())
            if t not in {"the","a","an","is","to","of","and","or","for","in","on","at","be","will"}
        ][:4]
        bare_slug = "_".join(tokens) or "mega"
        full_slug = bare_slug

    # Try bare slug first; if name collides with another mega's slug,
    # promote to full_slug (which keeps the early/late qualifier).
    cand = f"mega.{cluster_idx:03d}.{bare_slug}"
    if any(uid.endswith(f".{bare_slug}") for uid in used_ids):
        if full_slug and full_slug != bare_slug:
            cand = f"mega.{cluster_idx:03d}.{full_slug}"
        else:
            cand = f"{cand}.{rep.contract_hash[:4]}"
    return cand


def select_icl_exemplar(rep: FullSkill, members: List[FullSkill]) -> Optional[dict]:
    """Pick a 1-shot ICL exemplar.

    Priority:
      1. rep's protocol_raw.steps (the GPT-5.4 NL reasoning trace) if non-empty.
      2. another member's protocol_raw.steps if rep's is empty.
      3. rep's protocol.steps (the abstracted NL).
    """
    if rep.protocol_raw_steps:
        return {
            "source_task": rep.task,
            "source_skill_id": rep.skill_id,
            "source_kind": "protocol_raw",
            "steps": rep.protocol_raw_steps[:12],
        }
    for m in members:
        if m.protocol_raw_steps:
            return {
                "source_task": m.task,
                "source_skill_id": m.skill_id,
                "source_kind": "protocol_raw",
                "steps": m.protocol_raw_steps[:12],
            }
    if rep.protocol_steps:
        return {
            "source_task": rep.task,
            "source_skill_id": rep.skill_id,
            "source_kind": "protocol",
            "steps": rep.protocol_steps[:12],
        }
    return None


def build_mega_skill(
    cluster_idx: int,
    members: List[FullSkill],
    edge_evidence_for_cluster: List[dict],
    used_ids: set,
) -> dict:
    rep = pick_representative(members)
    sig = consensus_signature(members) or rep.plan_signature()
    mega_id = derive_mega_skill_id(cluster_idx, rep, members, used_ids)
    used_ids.add(mega_id)

    # Member binding records (deduped by key already; cluster_skills used a set)
    member_bindings = []
    for m in sorted(members, key=lambda s: s.key()):
        member_bindings.append({
            "task": m.task,
            "skill_id": m.skill_id,
            "name": m.name,
            "domain": m.domain,
            "plan_signature": m.plan_signature(),
            "n_protocol_steps": len(m.protocol_steps),
            "contract_hash": m.contract_hash,
            "is_representative": (m.key() == rep.key()),
        })

    # Domain / task spans
    tasks_in = sorted({m.task for m in members})
    domains_in = sorted({m.domain for m in members})

    # Top-3 judge evidence
    top_evidence = sorted(
        edge_evidence_for_cluster,
        key=lambda e: (-int(e.get("score", 0)), e.get("from", "")),
    )[:3]
    judge_excerpts = [
        {
            "score": int(e["score"]),
            "shared": e["shared_reasoning"][:200],
            "transfer": e["transfer_value"][:200],
            "from": e["from"],
            "to": e["to"],
        }
        for e in top_evidence
    ]

    return {
        "mega_skill_id": mega_id,
        "cluster_idx": cluster_idx,
        "template_signature": sig,
        "template_steps": rep.protocol_steps[:10],
        "step_checks": rep.step_checks[:10],
        "preconditions": rep.preconditions[:8],
        "representative": {
            "task": rep.task,
            "skill_id": rep.skill_id,
            "name": rep.name,
            "description": rep.description[:600],
        },
        "n_members": len(members),
        "n_tasks": len(tasks_in),
        "n_domains": len(domains_in),
        "tasks": tasks_in,
        "domains": domains_in,
        "members": member_bindings,
        "icl_exemplar": select_icl_exemplar(rep, members),
        "judge_evidence": judge_excerpts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--judgments",
                    default=str(REPO_ROOT / "frontier_data/output/plan_level_similarity_judgments.json"))
    ap.add_argument("--threshold", type=int, default=4,
                    help="loose score threshold for keeping edge evidence (default 4)")
    ap.add_argument("--core-threshold", type=int, default=5,
                    help="strict score for Stage-1 core cluster edges (default 5)")
    ap.add_argument("--attach-min-score", type=int, default=4,
                    help="minimum score to attach a satellite to a core cluster (default 4)")
    ap.add_argument("--no-mutual-core", action="store_true",
                    help="don't require mutual high-score for core merging")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "frontier_data/output/megaskills_from_judge"))
    ap.add_argument("--bank-set", default="best_grpo")
    ap.add_argument("--include-singletons", action="store_true",
                    help="emit mega-skill entries for orphaned skills too (default: skip)")
    args = ap.parse_args()

    if args.bank_set == "best_grpo":
        bank_paths = BEST_GRPO_RUNS
    elif args.bank_set == "all_stages":
        bank_paths = ALL_STAGES
    else:
        raise SystemExit(f"unknown bank-set: {args.bank_set}")

    logger.info("loading judge file: %s", args.judgments)
    judgments = json.loads(Path(args.judgments).read_text())
    logger.info("  judge results: %d targets",
                len(judgments.get("results", [])))

    logger.info("loading full skill banks (%s)", args.bank_set)
    full = load_full_skills(bank_paths)
    logger.info("  %d skills loaded total", len(full))

    clusters, edge_evidence = cluster_skills(
        judgments, full, args.threshold,
        core_threshold=args.core_threshold,
        require_mutual_core=not args.no_mutual_core,
        attach_min_score=args.attach_min_score,
    )

    # Build mega-skills, sorted by cluster size (biggest first)
    ordered_clusters = sorted(
        clusters.values(),
        key=lambda members: (-len(members), members[0]),
    )

    mega_skills: List[dict] = []
    used_ids: set = set()
    n_singletons = 0
    for idx, member_keys in enumerate(ordered_clusters):
        members = [full[k] for k in member_keys if k in full]
        if not members:
            continue
        if len(members) < 2 and not args.include_singletons:
            n_singletons += 1
            continue
        # Collect edge evidence inside this cluster
        member_set = set(member_keys)
        cluster_edges: List[dict] = []
        for pair, evlist in edge_evidence.items():
            if pair[0] in member_set and pair[1] in member_set:
                cluster_edges.extend(evlist)
        mega = build_mega_skill(idx, members, cluster_edges, used_ids)
        mega_skills.append(mega)

    logger.info(
        "built %d mega-skills (%d singletons %s)",
        len(mega_skills), n_singletons,
        "kept" if args.include_singletons else "skipped",
    )

    # Stats
    n_multi_task = sum(1 for ms in mega_skills if ms["n_tasks"] >= 2)
    n_multi_domain = sum(1 for ms in mega_skills if ms["n_domains"] >= 2)
    sig_counter = Counter(ms["template_signature"] for ms in mega_skills)
    logger.info("multi-task (≥2 tasks): %d", n_multi_task)
    logger.info("multi-domain (≥2 domains): %d", n_multi_domain)
    logger.info("top template signatures:")
    for sig, n in sig_counter.most_common(10):
        logger.info("  %2d  %s", n, sig)

    # Write outputs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "mega_skills.jsonl"
    with open(jsonl_path, "w") as f:
        for ms in mega_skills:
            f.write(json.dumps(ms, ensure_ascii=False) + "\n")
    logger.info("wrote %s", jsonl_path)

    # Lineage detail
    lineage_path = out_dir / "mega_skills_lineage.json"
    lineage_path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "judgments_path": args.judgments,
        "threshold": args.threshold,
        "n_mega_skills": len(mega_skills),
        "mega_skills_summary": [
            {
                "mega_skill_id": ms["mega_skill_id"],
                "template_signature": ms["template_signature"],
                "n_members": ms["n_members"],
                "n_tasks": ms["n_tasks"],
                "tasks": ms["tasks"],
                "representative": ms["representative"],
                "members": ms["members"],
            } for ms in mega_skills
        ],
    }, indent=2, ensure_ascii=False))
    logger.info("wrote %s", lineage_path)

    # Markdown summary
    md_path = out_dir / "mega_skills_SUMMARY.md"
    lines: List[str] = []
    lines.append(f"# Mega-skills from LLM-judge clustering")
    lines.append("")
    lines.append(f"- generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- judgments: `{args.judgments}`")
    lines.append(f"- threshold: ≥ {args.threshold}")
    lines.append(f"- source banks: {len(bank_paths)} GRPO-validated games "
                 f"({sum(1 for _ in full)} skills)")
    lines.append(f"- mega-skills: **{len(mega_skills)}** "
                 f"({n_multi_task} multi-task, {n_multi_domain} multi-domain)")
    lines.append("")
    lines.append("## Mega-skills (sorted by member count)")
    lines.append("")
    for ms in mega_skills:
        lines.append(f"### `{ms['mega_skill_id']}` — {ms['template_signature']}")
        lines.append("")
        lines.append(f"- members: **{ms['n_members']}** "
                     f"across {ms['n_tasks']} task(s): "
                     f"{', '.join(ms['tasks'])}")
        rep = ms["representative"]
        lines.append(f"- representative: `{rep['task']}::{rep['skill_id']}` — {rep['name']}")
        if rep.get("description"):
            lines.append(f"  - description: {rep['description'][:240]}")
        lines.append("")
        lines.append(f"- template steps (from representative):")
        for i, s in enumerate(ms["template_steps"]):
            lines.append(f"  {i + 1}. {s[:200]}")
        lines.append("")
        if ms["judge_evidence"]:
            lines.append("- judge rationale (top-3 edges):")
            for e in ms["judge_evidence"]:
                lines.append(f"  - score={e['score']}: {e['shared']}")
            lines.append("")
        ex = ms.get("icl_exemplar")
        if ex:
            lines.append(f"- ICL exemplar source: `{ex['source_task']}::{ex['source_skill_id']}` ({ex['source_kind']})")
        lines.append("")
        lines.append("- members detail:")
        for mb in ms["members"]:
            star = " ★" if mb["is_representative"] else ""
            lines.append(
                f"  - `{mb['task']:30s}::{mb['skill_id']:30s}` "
                f"sig=`{mb['plan_signature']}` n_steps={mb['n_protocol_steps']}{star}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    logger.info("wrote %s", md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
