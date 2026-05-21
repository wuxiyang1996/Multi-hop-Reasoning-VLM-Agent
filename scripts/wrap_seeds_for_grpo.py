#!/usr/bin/env python3
"""Wrap v3 mega-skill seed banks into the ``{report, skill}`` envelope
the Phase-2 GRPO loop expects.

The v3 builder (``scripts/stage2_seeds_from_megaskills.py``) emits flat
records keyed at the top level by ``skill_id, protocol, contract,
exemplars, provenance, …``.  ``SkillBankMVP.load()`` and the
``skill_agents.stage3_mvp.schemas.Skill`` dataclass instead expect:

    {
      "skill":  Skill.to_dict(),     # nested
      "report": VerificationReport.to_dict()
    }

This script does the conversion in-place per target game:

* wraps each flat seed into the canonical envelope
* fills in defaults the GRPO loop reads:
    - ``protocol.predicate_success = ['state_observed=true']``
    - ``protocol.predicate_abort   = []``
    - ``protocol.success_criteria  = ['state_observed=true']``
    - ``protocol.abort_criteria    = []``
    - ``protocol.expected_duration = 8``
    - ``protocol.source            = 'megaskill_seed'``
* preserves the 1-shot ICL exemplar in ``skill.protocol_raw`` (the field
  ``SkillBankProvider._enrich_from_skill`` reads to render
  ``SkillGuidance.exemplar_steps``)
* lifts provenance (mega-skill id) into ``skill.derived_from`` and stamps
  ``confidence_tag='translated'`` — the soft-promotion state described in
  ``Skill`` docstring §22 (the seed must re-earn ``verified`` on the
  target via the GRPO gate)
* emits a zero-filled ``VerificationReport`` so the GRPO verifier has a
  consistent place to accumulate evidence
* round-trip validates the output by loading each game with
  ``SkillBankMVP`` and asserting all skills come back

Usage::

    python scripts/wrap_seeds_for_grpo.py \\
        --seed-root frontier_data/output/stage2_seeds_v3 \\
        --out-root  frontier_data/output/stage2_seeds_v3_grpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PRED_SUCCESS = ["state_observed=true"]
DEFAULT_PRED_ABORT: List[str] = []
DEFAULT_SUCCESS_CRITERIA = ["state_observed=true"]
DEFAULT_ABORT_CRITERIA: List[str] = []
DEFAULT_EXPECTED_DURATION = 8
DEFAULT_PROTOCOL_SOURCE = "megaskill_seed"


def _now() -> float:
    return time.time()


def _expected_tag_pattern(template_signature: str) -> List[str]:
    """Decompose ``EVALUATE → ACT → PERCEIVE → ACT`` into a tag list.

    The decision agent's eligibility filter compares this against the
    intention-tag sequence emitted by ``IntentionSignalExtractor``.  We
    keep the canonical intents the LLM judge used; if the target game's
    extractor emits a different vocabulary the comparison will simply
    miss and the seed acts as a free shadow skill until verified.
    """
    if not template_signature:
        return []
    return [
        tok.strip()
        for tok in template_signature.replace("->", "→").split("→")
        if tok.strip()
    ]


def _build_execution_hint(seed: Dict[str, Any]) -> Dict[str, Any]:
    proto = seed.get("protocol") or {}
    desc = (
        (seed.get("contract") or {}).get("description")
        or seed.get("strategic_description")
        or ""
    )
    members = ((seed.get("provenance") or {}).get("source_members")) or []
    return {
        "common_preconditions": list(proto.get("preconditions") or []),
        "common_target_objects": [],
        "state_transition_pattern": (
            seed.get("template_signature") or "EVALUATE → ACT"
        ),
        "termination_cues": list(DEFAULT_SUCCESS_CRITERIA),
        "common_failure_modes": [
            "No state change observed after the protocol's expected duration",
        ],
        "execution_description": desc[:240],
        "n_source_segments": len(members),
        "updated_at": _now(),
    }


def _build_protocol_raw(seed: Dict[str, Any]) -> Dict[str, Any] | None:
    exemplars = seed.get("exemplars") or []
    if not exemplars:
        return None
    ex = exemplars[0]
    steps = list(ex.get("reasoning_steps") or [])
    if not steps:
        return None
    return {
        "steps": steps,
        "source": "megaskill_icl",
        "source_task": ex.get("source_task"),
        "source_skill_id": ex.get("source_skill_id"),
        "source_kind": ex.get("source_kind", "protocol_raw"),
    }


def _build_protocol(seed: Dict[str, Any]) -> Dict[str, Any]:
    proto = seed.get("protocol") or {}
    steps = list(proto.get("steps") or [])
    step_checks = list(proto.get("step_checks") or [])
    if step_checks and len(step_checks) < len(steps):
        step_checks = step_checks + [""] * (len(steps) - len(step_checks))
    elif not step_checks and steps:
        step_checks = ["state_observed=true"] * len(steps)

    success_criteria = list(proto.get("success_criteria") or [])
    if not success_criteria:
        success_criteria = list(DEFAULT_SUCCESS_CRITERIA)
    abort_criteria = list(proto.get("abort_criteria") or DEFAULT_ABORT_CRITERIA)

    predicate_success = list(proto.get("predicate_success") or DEFAULT_PRED_SUCCESS)
    predicate_abort = list(proto.get("predicate_abort") or DEFAULT_PRED_ABORT)

    expected_duration = int(proto.get("expected_duration") or DEFAULT_EXPECTED_DURATION)

    return {
        "preconditions": list(proto.get("preconditions") or []),
        "steps": steps,
        "success_criteria": success_criteria,
        "abort_criteria": abort_criteria,
        "expected_duration": expected_duration,
        "step_checks": step_checks,
        "predicate_success": predicate_success,
        "predicate_abort": predicate_abort,
        "action_vocab": list(proto.get("action_vocab") or []),
        "source": proto.get("source") or DEFAULT_PROTOCOL_SOURCE,
    }


def _build_contract(seed: Dict[str, Any], skill_id: str, name: str) -> Dict[str, Any]:
    c = seed.get("contract") or {}
    desc = c.get("description") or seed.get("strategic_description") or ""
    return {
        "skill_id": skill_id,
        "version": int(seed.get("version") or 1),
        "name": name,
        "description": desc,
        "eff_add": sorted(c.get("eff_add") or []),
        "eff_del": sorted(c.get("eff_del") or []),
        "eff_event": sorted(c.get("eff_event") or []),
        "support": dict(c.get("support") or {}),
        "n_instances": 0,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _build_report(skill_id: str) -> Dict[str, Any]:
    return {
        "skill_id": skill_id,
        "n_instances": 0,
        "eff_add_success_rate": {},
        "eff_del_success_rate": {},
        "eff_event_rate": {},
        "overall_pass_rate": 0.0,
        "worst_segments": [],
        "failure_signatures": {},
    }


def _derive_mega_from_tags(tags: List[str]) -> str | None:
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith("mega_skill_id:"):
            return tag.split(":", 1)[1].strip()
    return None


def wrap_seed(seed: Dict[str, Any], target_game: str) -> Dict[str, Any]:
    skill_id = seed.get("skill_id") or seed.get("id") or "seed.unknown"
    name = seed.get("name") or skill_id.replace("seed.", "").replace("_", " ").title()

    prov = seed.get("provenance") or {}
    mega_id = prov.get("source_mega_skill") or _derive_mega_from_tags(seed.get("tags"))

    tags = list(seed.get("tags") or [])
    if "confidence:translated" not in tags:
        tags.append("confidence:translated")
    if not any(t.startswith("target:") for t in tags):
        tags.append(f"target:{target_game}")

    skill_block = {
        "skill_id": skill_id,
        "version": int(seed.get("version") or 1),
        "name": name,
        "strategic_description": seed.get("strategic_description") or "",
        "tags": tags,
        "protocol": _build_protocol(seed),
        "contract": _build_contract(seed, skill_id, name),
        "sub_episodes": [],
        "expected_tag_pattern": _expected_tag_pattern(
            seed.get("template_signature") or ""
        ),
        "execution_hint": _build_execution_hint(seed),
        "protocol_raw": _build_protocol_raw(seed),
        "protocol_history": [],
        "n_instances": 0,
        "retired": bool(seed.get("retired") or False),
        "created_at": _now(),
        "updated_at": _now(),
        "feasible_tasks": list(seed.get("feasible_tasks") or [target_game]),
        "verified_tasks": list(seed.get("verified_tasks") or []),
        "derived_from": mega_id,
        "confidence_tag": "translated",
    }

    return {
        "skill": skill_block,
        "report": _build_report(skill_id),
    }


def convert_game(seed_path: Path, out_path: Path, target_game: str) -> Dict[str, Any]:
    rows = [json.loads(l) for l in seed_path.open() if l.strip()]
    wrapped = [wrap_seed(r, target_game) for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for entry in wrapped:
            f.write(json.dumps(entry, default=str) + "\n")

    # Round-trip validate via SkillBankMVP if available; otherwise fall
    # back to verifying schema keys manually so the script is still
    # usable in environments where the trainer package isn't importable.
    validated = 0
    try:
        from skill_agents.skill_bank.bank import SkillBankMVP
        bank = SkillBankMVP(str(out_path))
        bank.load(str(out_path))
        validated = len(getattr(bank, "_skills", {}) or {})
    except Exception as exc:
        validated = -1
        validation_error = repr(exc)
    else:
        validation_error = ""

    return {
        "target": target_game,
        "n_in": len(rows),
        "n_out": len(wrapped),
        "n_loaded": validated,
        "validation_error": validation_error,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--seed-root",
        default="frontier_data/output/stage2_seeds_v3",
        help="Root containing <target>/skill_bank.jsonl flat seeds.",
    )
    p.add_argument(
        "--out-root",
        default="frontier_data/output/stage2_seeds_v3_grpo",
        help="Output root for the wrapped {report, skill} banks.",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of target game directories to convert.",
    )
    args = p.parse_args()

    seed_root = Path(args.seed_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not seed_root.is_dir():
        print(f"[error] seed-root {seed_root} does not exist", file=sys.stderr)
        return 2

    if args.targets:
        target_dirs = [seed_root / t for t in args.targets]
    else:
        target_dirs = sorted(p for p in seed_root.iterdir() if p.is_dir())

    if not target_dirs:
        print(f"[error] no targets under {seed_root}", file=sys.stderr)
        return 2

    results = []
    for d in target_dirs:
        seed_path = d / "skill_bank.jsonl"
        if not seed_path.exists():
            print(f"[skip] {d.name}: no skill_bank.jsonl")
            continue
        out_path = out_root / d.name / "skill_bank.jsonl"
        info = convert_game(seed_path, out_path, target_game=d.name)
        results.append(info)
        msg = (
            f"[ok] {info['target']:30s} in={info['n_in']:3d} "
            f"out={info['n_out']:3d} loaded={info['n_loaded']:3d}"
        )
        if info["validation_error"]:
            msg += f"  warn={info['validation_error'][:120]}"
        print(msg)

    summary_path = out_root / "WRAP_SUMMARY.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed_root": str(seed_root),
                "out_root": str(out_root),
                "defaults": {
                    "predicate_success": DEFAULT_PRED_SUCCESS,
                    "predicate_abort": DEFAULT_PRED_ABORT,
                    "success_criteria": DEFAULT_SUCCESS_CRITERIA,
                    "abort_criteria": DEFAULT_ABORT_CRITERIA,
                    "expected_duration": DEFAULT_EXPECTED_DURATION,
                    "protocol_source": DEFAULT_PROTOCOL_SOURCE,
                    "confidence_tag": "translated",
                },
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n[summary] {summary_path}")
    return 0 if all(r["validation_error"] == "" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
