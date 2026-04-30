#!/usr/bin/env python3
"""Decorate extractor outputs with ``SkillRecord``-shape future-proofing fields.

The two skill-bank extractors (``extract_skillbank_gpt54.py`` for env_wrappers
and ``extract_skillbank_gymv_gpt54.py`` for gym-v) produce per-env outputs:

  <env_dir>/skill_bank.jsonl     — one row per skill (Skill.to_dict())
  <env_dir>/skill_catalog.json   — {"skills": [...], "n_skills": ...}

Today these rows do *not* yet carry the canonical ``SkillRecord`` fields
required by the unified gate (PLAN-UNIFIED-SKILL-GATE §3.1). This script
walks the extractor output tree after the run finishes and adds the five
fields needed for the future ``lifecycle.ingest_draft`` import shim — no
gate evaluation is performed, the skills only get *shaped* so they can
be ingested later without a destructive migration.

Fields added (all five are idempotent):

  source_type         : "mined_from_trace"           (SkillSourceType.MINED)
  applicable_domains  : ["gymv"]                     (game-foundry source)
  verified_domains    : []                           (filled by gate Stage 3 later)
  evidence_role       : <GATHER|VERIFY|REASON|COMMIT>  (from skill.tags / segment op)
  status              : "draft"                      (SkillStatus.DRAFT)

Plus a top-level ``corpus`` ("gym_v" | "env_wrappers") for traceability.

The script also adds a sibling ``_lifecycle_meta.json`` to each env folder
describing the provenance (``cold_start_run``, ``intentions_run``,
``model``, ``skills_extracted``, ``decorator_version``).

Idempotency: if ``source_type`` is already present on a row, that row is
left untouched. Re-running the decorator is safe.

Usage:

    python labeling/_decorate_skill_records.py \\
        --root labeling/skill_bank_out/run_<timestamp> \\
        --intentions_run run_dualaxis_<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants — match `common/enums.py` exactly so the decorated output is
# load-able by `lifecycle.ingest_draft` without translation.
# ---------------------------------------------------------------------------
DECORATOR_VERSION = "skillrecord_shape_v1"

SOURCE_TYPE_MINED = "mined_from_trace"
STATUS_DRAFT = "draft"
APPLICABLE_DOMAIN_GAME = ["gymv"]   # both gym_v and env_wrappers are game-foundry

# Map dual-axis INTENT_OPERATOR → EVIDENCE_ROLE.
#
#   EVIDENCE_ROLES = (GATHER, VERIFY, REASON, COMMIT)
#
# RECOVER is a corrective action (still a COMMIT-style write to the env);
# COMPARE involves cross-step reasoning. INSPECT/TRACK both gather signal.
_OPERATOR_TO_EVIDENCE_ROLE: Dict[str, str] = {
    "INSPECT": "GATHER",
    "TRACK":   "GATHER",
    "COMPARE": "REASON",
    "COMMIT":  "COMMIT",
    "VERIFY":  "VERIFY",
    "RECOVER": "COMMIT",
}
_DEFAULT_EVIDENCE_ROLE = "COMMIT"


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _evidence_role_from_skill(skill: Dict[str, Any]) -> str:
    """Pull the operator out of skill.tags / expected_tag_pattern and
    map it to the four-value EVIDENCE_ROLES taxonomy."""
    candidates: List[str] = []
    for key in ("tags", "expected_tag_pattern"):
        v = skill.get(key)
        if isinstance(v, list):
            candidates.extend(str(x).upper() for x in v if x)
        elif isinstance(v, str) and v:
            candidates.append(v.upper())

    skill_id = str(skill.get("skill_id") or "").upper()
    if skill_id:
        # gym-v skills have ids like "mid:NAVIGATE" or "skill_<game>_<tag>_<idx>"
        for op in _OPERATOR_TO_EVIDENCE_ROLE:
            if op in skill_id:
                candidates.append(op)
                break

    for c in candidates:
        if c in _OPERATOR_TO_EVIDENCE_ROLE:
            return _OPERATOR_TO_EVIDENCE_ROLE[c]
    return _DEFAULT_EVIDENCE_ROLE


def _decorate_skill_dict(
    entry: Dict[str, Any],
    *,
    corpus: str,
    source_name: str,
) -> Tuple[Dict[str, Any], bool]:
    """Add SkillRecord-shape fields to one skill dict. Returns (entry, mutated)."""
    skill = entry.get("skill") if isinstance(entry.get("skill"), dict) else entry
    if not isinstance(skill, dict):
        return entry, False

    if "source_type" in skill and "applicable_domains" in skill and "status" in skill:
        return entry, False  # already decorated

    skill["source_type"] = skill.get("source_type") or SOURCE_TYPE_MINED
    skill["applicable_domains"] = skill.get("applicable_domains") or list(APPLICABLE_DOMAIN_GAME)
    skill.setdefault("verified_domains", [])
    skill["evidence_role"] = skill.get("evidence_role") or _evidence_role_from_skill(skill)
    skill["status"] = skill.get("status") or STATUS_DRAFT

    skill.setdefault("provenance", {})
    if isinstance(skill["provenance"], dict):
        skill["provenance"].setdefault("corpus", corpus)
        skill["provenance"].setdefault("source_name", source_name)
        skill["provenance"].setdefault("decorator_version", DECORATOR_VERSION)

    if "skill" in entry and isinstance(entry["skill"], dict):
        entry["skill"] = skill
    else:
        entry = skill
    return entry, True


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def _decorate_skill_bank_jsonl(
    path: Path, *, corpus: str, source_name: str
) -> Tuple[int, int]:
    """Rewrite a skill_bank.jsonl in place, adding fields. Returns (n_rows, n_decorated)."""
    if not path.exists():
        return 0, 0
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    n_decorated = 0
    for row in rows:
        _, mutated = _decorate_skill_dict(row, corpus=corpus, source_name=source_name)
        if mutated:
            n_decorated += 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)
    return len(rows), n_decorated


def _decorate_skill_catalog_json(
    path: Path, *, corpus: str, source_name: str
) -> Tuple[int, int]:
    """Rewrite a skill_catalog.json in place. Returns (n_skills, n_decorated)."""
    if not path.exists():
        return 0, 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return 0, 0
    if not isinstance(doc, dict):
        return 0, 0

    skills = doc.get("skills") or []
    if not isinstance(skills, list):
        return 0, 0

    n_decorated = 0
    new_skills: List[Dict[str, Any]] = []
    for s in skills:
        if not isinstance(s, dict):
            new_skills.append(s)
            continue
        decorated, mutated = _decorate_skill_dict(
            s, corpus=corpus, source_name=source_name
        )
        if mutated:
            n_decorated += 1
        new_skills.append(decorated)

    doc["skills"] = new_skills
    doc.setdefault("corpus", corpus)
    doc.setdefault("source_name", source_name)
    doc.setdefault("decorator_version", DECORATOR_VERSION)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)
    return len(skills), n_decorated


def _write_lifecycle_meta(
    env_dir: Path,
    *,
    corpus: str,
    source_name: str,
    n_skills: int,
    intentions_run: Optional[str] = None,
    cold_start_run: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    meta = {
        "decorator_version": DECORATOR_VERSION,
        "corpus": corpus,
        "source_name": source_name,
        "n_skills": n_skills,
        "applicable_domains": list(APPLICABLE_DOMAIN_GAME),
        "default_status": STATUS_DRAFT,
        "default_source_type": SOURCE_TYPE_MINED,
        "intentions_run": intentions_run,
        "cold_start_run": cold_start_run,
        "model": model,
        "notes": (
            "These rows are SkillRecord-shape but un-gated. "
            "Use `skill_bank.lifecycle.ingest_draft` plus `gate_service.evaluate` "
            "with stages={STATIC, REPLAY} to import this bank into draft_store/. "
            "See PLAN-UNIFIED-SKILL-GATE §6 for the canonical import path."
        ),
    }
    out = env_dir / "_lifecycle_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------

def decorate_corpus_tree(
    root: Path,
    *,
    corpus_name: str,
    intentions_run: Optional[str],
    cold_start_run: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    """Walk root/<source_name>/ and decorate each env's outputs.

    ``root`` is e.g. ``labeling/skill_bank_out/run_<ts>/gym_v`` or
    ``.../env_wrappers``. Each immediate subfolder is one env / game.
    """
    summary: Dict[str, Any] = {
        "corpus": corpus_name,
        "root": str(root),
        "envs": [],
        "totals": {"envs": 0, "skills_decorated": 0, "skills_total": 0},
    }
    if not root.exists():
        return summary

    for env_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        bank_path = env_dir / "skill_bank.jsonl"
        cat_path = env_dir / "skill_catalog.json"

        n_rows, n_dec_rows = _decorate_skill_bank_jsonl(
            bank_path, corpus=corpus_name, source_name=env_dir.name
        )
        n_cat, n_dec_cat = _decorate_skill_catalog_json(
            cat_path, corpus=corpus_name, source_name=env_dir.name
        )
        n_skills = max(n_rows, n_cat)
        _write_lifecycle_meta(
            env_dir,
            corpus=corpus_name,
            source_name=env_dir.name,
            n_skills=n_skills,
            intentions_run=intentions_run,
            cold_start_run=cold_start_run,
            model=model,
        )
        summary["envs"].append({
            "source_name": env_dir.name,
            "skill_bank_rows": n_rows,
            "skill_bank_rows_decorated": n_dec_rows,
            "skill_catalog_rows": n_cat,
            "skill_catalog_rows_decorated": n_dec_cat,
        })
        summary["totals"]["envs"] += 1
        summary["totals"]["skills_total"] += n_skills
        summary["totals"]["skills_decorated"] += max(n_dec_rows, n_dec_cat)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Decorate skill-discovery outputs with SkillRecord-shape fields.",
    )
    p.add_argument(
        "--root", required=True, type=str,
        help="Root containing gym_v/ and/or env_wrappers/ subfolders.",
    )
    p.add_argument("--intentions_run", type=str, default=None)
    p.add_argument("--cold_start_run", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        return 2

    print("=" * 62)
    print(f"  decorate_skill_records  ({DECORATOR_VERSION})")
    print("=" * 62)
    print(f"  Root            : {root}")
    print(f"  Intentions run  : {args.intentions_run}")
    print(f"  Cold-start run  : {args.cold_start_run}")
    print(f"  Model           : {args.model}")
    print()

    overall: Dict[str, Any] = {"corpora": []}
    for corpus in ("gym_v", "env_wrappers"):
        corpus_root = root / corpus
        if not corpus_root.exists():
            continue
        print(f"  -- {corpus} --")
        s = decorate_corpus_tree(
            corpus_root,
            corpus_name=corpus,
            intentions_run=args.intentions_run,
            cold_start_run=args.cold_start_run,
            model=args.model,
        )
        for env in s["envs"]:
            print(
                f"    {env['source_name']:34s} "
                f"jsonl={env['skill_bank_rows']:>3d} "
                f"(decorated={env['skill_bank_rows_decorated']:>3d}) "
                f"catalog={env['skill_catalog_rows']:>3d} "
                f"(decorated={env['skill_catalog_rows_decorated']:>3d})"
            )
        print(
            f"    -> total envs={s['totals']['envs']}  "
            f"skills={s['totals']['skills_total']}  "
            f"decorated={s['totals']['skills_decorated']}"
        )
        print()
        overall["corpora"].append(s)

    out_summary = root / "_decorator_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Wrote summary: {out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
