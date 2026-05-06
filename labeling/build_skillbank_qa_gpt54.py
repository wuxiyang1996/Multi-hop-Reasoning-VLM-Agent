#!/usr/bin/env python
"""GPT-5.4-driven skill-bank construction for QA + MiniWob (Option β).

This script replaces the SkillBankAgent SEGMENT/CONTRACT/CURATOR LoRA
pipeline with a single LLM-driven pipeline that:

1. **CLUSTER (deterministic)** — For each ``(source, op, sg)`` triple,
   collect every labeled hop / step across all four models
   (gpt-5.4 / claude / gemini / qwen).  Drops singleton clusters
   (n_instances < 2) since one example is not a skill yet.

2. **CONTRACT (gpt-5.4 LLM)** — For each cluster sample N representative
   ``(operator, subgoal, note, context_signature)`` tuples and ask
   gpt-5.4 to derive a SkillRecord-shape contract:

     - name                          (≤ 6 words, human-readable)
     - strategic_description         (2-3 sentences)
     - eff_add / eff_del             (predicate names that change)
     - common_preconditions          (≤ 6 short bullets)
     - common_postconditions         (≤ 6)
     - common_pitfalls               (≤ 4)
     - example_predicates            (3-6 short predicate-like strings)

3. **CURATOR (gpt-5.4 LLM, optional)** — Compute Jaccard similarity over
   ``(eff_add ∪ eff_del ∪ common_preconditions)`` token sets.  Pairs
   with Jaccard ≥ ``--curator-jaccard`` (default 0.6) are sent in a
   batch to gpt-5.4 with a "MERGE / KEEP_BOTH" decision prompt.  Merges
   are applied conservatively (only when LLM says MERGE).

Output (mirrors ``labeling/skill_bank_out/run_<ts>/<corpus>/<game>/skill_bank.jsonl``
shape produced by extract_skillbank_gymv_gpt54.py so downstream readers
in skill_agents.boundary_proposal / scripts/build_multimodal_decision_sft
ingest it without any code changes):

    labeling/skill_bank_qa/run_<ts>/<source>/skill_bank.jsonl
        {"skill": {skill_id, version, name, strategic_description,
                   contract, execution_hint, applicable_domains,
                   feasible_tasks, n_instances, sub_episodes, ...},
         "report": {...}}
    labeling/skill_bank_qa/run_<ts>/<source>/_summary.json
    labeling/skill_bank_qa/run_<ts>/_run_summary.json

Sources handled
~~~~~~~~~~~~~~~

* QA  (uses ``labeling/qa_multihop_out/run_<ts>/<src>/<model>/samples_with_hops.jsonl``):
  ``video_holmes``, ``siv_bench``, ``tir_bench``, ``visual_toolbench``
* MiniWob (uses ``labeling/qa_miniwob_labeled/run_<ts>/miniwob/<model>/<game>/rollouts.jsonl``,
  steps already carry ``intention_operator/intention_subgoal/intention_note``).

Usage
~~~~~

    python -m labeling.build_skillbank_qa_gpt54 \\
        --multihop-run labeling/qa_multihop_out/run_<ts>     \\
        --miniwob-run  labeling/qa_miniwob_labeled/run_20260506_070722 \\
        --output-dir   labeling/skill_bank_qa/run_<ts>      \\
        --workers 8

The output is consumed by the skill query labeler
``labeling/label_skill_actions_qa_gpt54.py`` which writes the per-step
``skill_query.candidates`` field that
``scripts/build_multimodal_decision_sft.py`` then uses to emit
``skill_selection`` SFT rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path / API key bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key
    if getattr(_ak, "openai_api_key", "") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _ak.openai_api_key
except Exception:
    pass

try:
    from API_func import ask_model  # type: ignore
except ImportError:
    ask_model = None

logger = logging.getLogger("labeling.build_skillbank_qa")

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

DEFAULT_LABEL_MODEL = "gpt-5.4"
DEFAULT_WORKERS = 8
DEFAULT_CURATOR_JACCARD = 0.6

CONTRACT_LLM_MAX_TOKENS = 1100
CURATOR_LLM_MAX_TOKENS = 400
LLM_TEMPERATURE = 0.15

# Per-cluster: how many representative examples to send to gpt-5.4
CLUSTER_EXEMPLAR_LIMIT = 12
# Per-cluster: minimum instances to count as a real skill
MIN_CLUSTER_SIZE = 3
# Hard cap on bank size per source (curator drops the smallest clusters past this)
MAX_SKILLS_PER_SOURCE = 32
# How many sub-episode pointers to keep per skill for SFT joining
SUB_EPISODE_KEEP = 16

QA_SOURCES = ("video_holmes", "siv_bench", "tir_bench", "visual_toolbench")
MINIWOB_SOURCE = "miniwob"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HopInstance:
    """One ``(operator, subgoal, note)`` instance found in the corpus."""
    source: str          # "video_holmes" | "miniwob" | ...
    model: str           # "gpt-5.4" | "claude" | "gemini" | "qwen"
    sample_id: str       # episode_id or sample_id, used for sub_episodes
    bucket: str          # game_name (miniwob) or "qa" (other sources)
    step_idx: int        # hop step / experience step
    operator: str
    subgoal: str
    note: str
    evidence: str        # "frame_x" | "options" | "passage" | "derived" | ...
    tool_call: str       # "" or e.g. "frame_pick"
    action: str          # for miniwob: chosen action; for QA: gold answer or "(answer)"
    context: str         # short string capturing question or task
    correct: Optional[bool] = None   # only for QA
    reward: Optional[float] = None   # only for miniwob

    def signature(self) -> str:
        """Short textual signature of the hop for prompting."""
        ev = f" [{self.evidence}]" if self.evidence else ""
        tc = f" tool={self.tool_call}" if self.tool_call else ""
        ctx = f" ctx={self.context[:80]}" if self.context else ""
        return f"{self.note}{ev}{tc}{ctx}"


# ---------------------------------------------------------------------------
# Source iterators — emit HopInstance streams
# ---------------------------------------------------------------------------

def _safe_iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def _iter_qa_hops(
    multihop_run: Path, source: str, models: Tuple[str, ...],
) -> Iterable[HopInstance]:
    """Walk a QA source × all models, emit one HopInstance per labelled hop."""
    src_dir = multihop_run / source
    if not src_dir.is_dir():
        return
    for mdl in models:
        f = src_dir / mdl / "samples_with_hops.jsonl"
        if not f.exists():
            continue
        for sample in _safe_iter_jsonl(f):
            sample_id = str(
                sample.get("sample_id")
                or sample.get("id")
                or sample.get("question_id")
                or hashlib.md5(
                    (str(sample.get("question", "")) + mdl).encode()
                ).hexdigest()[:12]
            )
            ctx = (sample.get("question") or sample.get("query") or "")[:160]
            correct = sample.get("correct")
            for hop in sample.get("hops") or []:
                yield HopInstance(
                    source=source,
                    model=mdl,
                    sample_id=sample_id,
                    bucket="qa",
                    step_idx=int(hop.get("step", 0)),
                    operator=str(hop.get("operator", "COMMIT")),
                    subgoal=str(hop.get("subgoal", "ANSWER")),
                    note=str(hop.get("note", "")),
                    evidence=str(hop.get("evidence", "")),
                    tool_call=str(hop.get("tool_call", "")),
                    action=str(sample.get("answer") or sample.get("answer_raw") or ""),
                    context=ctx,
                    correct=bool(correct) if isinstance(correct, bool) else None,
                )


def _iter_miniwob_steps(
    miniwob_run: Path, models: Tuple[str, ...],
) -> Iterable[HopInstance]:
    """Walk miniwob × all models × all games, emit step-level instances."""
    mw_dir = miniwob_run / "miniwob"
    if not mw_dir.is_dir():
        return
    for mdl in models:
        mdir = mw_dir / mdl
        if not mdir.is_dir():
            continue
        for game_dir in sorted(mdir.iterdir()):
            if not game_dir.is_dir():
                continue
            game = game_dir.name
            for f in sorted(game_dir.glob("*.jsonl")):
                for ep in _safe_iter_jsonl(f):
                    eid = str(ep.get("episode_id") or hashlib.md5(
                        (game + mdl + str(ep.get("query", ""))).encode()
                    ).hexdigest()[:12])
                    ctx = str(ep.get("task") or ep.get("query") or "")[:160]
                    outcome = ep.get("outcome")
                    reward = (
                        float(ep.get("rollout_metadata", {}).get("total_reward", 0.0))
                        if isinstance(ep.get("rollout_metadata"), dict)
                        else None
                    )
                    for i, exp in enumerate(ep.get("experiences") or []):
                        op = str(
                            exp.get("intention_operator")
                            or exp.get("intention_tag")
                            or "COMMIT"
                        )
                        sg = str(exp.get("intention_subgoal") or "EXECUTE")
                        note = str(exp.get("intention_note") or "")
                        if not note:
                            continue
                        yield HopInstance(
                            source=MINIWOB_SOURCE,
                            model=mdl,
                            sample_id=eid,
                            bucket=game,
                            step_idx=i,
                            operator=op,
                            subgoal=sg,
                            note=note,
                            evidence="step",
                            tool_call="",
                            action=str(exp.get("action") or exp.get("action_text") or ""),
                            context=ctx,
                            correct=bool(outcome) if isinstance(outcome, bool) else None,
                            reward=reward,
                        )


# ---------------------------------------------------------------------------
# Cluster & CONTRACT prompt
# ---------------------------------------------------------------------------

@dataclass
class SkillCluster:
    source: str
    operator: str
    subgoal: str
    instances: List[HopInstance] = field(default_factory=list)

    @property
    def skill_id(self) -> str:
        return f"{self.operator}/{self.subgoal}"


def _cluster_instances(
    instances: Iterable[HopInstance], *, source: str,
) -> Dict[Tuple[str, str], SkillCluster]:
    out: Dict[Tuple[str, str], SkillCluster] = {}
    for inst in instances:
        key = (inst.operator, inst.subgoal)
        c = out.get(key)
        if c is None:
            c = SkillCluster(source=source, operator=inst.operator, subgoal=inst.subgoal)
            out[key] = c
        c.instances.append(inst)
    return out


def _exemplars_for_prompt(cluster: SkillCluster, *, limit: int) -> List[HopInstance]:
    """Pick a diverse set of exemplars across (model, sample, bucket)."""
    seen_keys: set = set()
    picked: List[HopInstance] = []
    for inst in cluster.instances:
        k = (inst.model, inst.sample_id, inst.bucket)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        picked.append(inst)
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for inst in cluster.instances:
            if inst in picked:
                continue
            picked.append(inst)
            if len(picked) >= limit:
                break
    return picked


_CONTRACT_SYSTEM = (
    "You are a skill-bank curator distilling reusable cognitive/tool skills from "
    "multi-model reasoning traces.  For each skill cluster you receive, output "
    "a SINGLE strict-JSON object describing the skill — no prose, no markdown."
)


def _build_contract_prompt(cluster: SkillCluster, exemplars: List[HopInstance]) -> str:
    examples_block: List[str] = []
    for i, h in enumerate(exemplars, 1):
        line = (
            f"  {i}. [{h.source}/{h.bucket}/{h.model}#{h.sample_id[:6]}@{h.step_idx}] "
            f"{h.note}"
        )
        if h.evidence and h.evidence != "step":
            line += f"  (evidence={h.evidence})"
        if h.tool_call:
            line += f"  (tool={h.tool_call})"
        if h.context:
            line += f"\n     context: {h.context[:140]}"
        examples_block.append(line)
    n_models = len({h.model for h in cluster.instances})
    n_samples = len({h.sample_id for h in cluster.instances})

    return "\n".join([
        f"Skill cluster: source={cluster.source}, operator={cluster.operator}, "
        f"subgoal={cluster.subgoal}.",
        f"Total instances: {len(cluster.instances)} across {n_models} models "
        f"and {n_samples} samples.",
        "",
        "Representative exemplars (note + context):",
        *examples_block,
        "",
        "TASK: Distil the cluster into a reusable skill record.  Output strict JSON",
        "with EXACTLY these keys:",
        "{",
        '  "name": "<≤ 6 words, human-readable skill name>",',
        '  "strategic_description": "<2-3 sentence essence: when to invoke, what it achieves>",',
        '  "eff_add": ["<predicate1>", ...],   # state predicates that become true',
        '  "eff_del": ["<predicate1>", ...],   # state predicates that become false',
        '  "common_preconditions": ["<bullet>", ...],   # ≤ 6, ≤ 12 words each',
        '  "common_postconditions": ["<bullet>", ...],  # ≤ 6',
        '  "common_pitfalls": ["<bullet>", ...],         # ≤ 4',
        '  "example_predicates": ["<short_predicate>", ...],  # 3-6 short strings',
        '  "applicable_modalities": ["<modality>"]       # subset of {video, image, text, web, mixed}',
        "}",
        "",
        "Constraints:",
        "  - Predicate strings are short snake_case-ish names (e.g.",
        "    'evidence_grounded', 'option_pruned', 'tool_invoked', 'answer_committed').",
        "  - Preconditions are concrete observable facts ('reasoning chain has',",
        "    'frame contains', 'task instructs'), NOT vague guidance.",
        "  - Pitfalls are common failure modes you observed in the exemplars.",
        "  - applicable_modalities reflects what the exemplars show.",
    ])


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _strip_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    return text


def _extract_top_level_object(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_fence(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = -1
    return None


def _str_list(raw: Any, *, max_items: int, max_chars: int = 140) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for x in raw[:max_items]:
        s = str(x).strip()
        if s:
            out.append(s[:max_chars])
    return out


def _build_contract_record(
    *, cluster: SkillCluster, parsed: Dict[str, Any],
) -> Dict[str, Any]:
    """Coerce a CONTRACT LLM output into a SkillRecord-shape dict."""
    skill_id = cluster.skill_id
    name_raw = str(parsed.get("name") or skill_id).strip()
    if len(name_raw.split()) > 8:
        name_raw = " ".join(name_raw.split()[:8])
    desc = str(parsed.get("strategic_description") or "").strip()
    if not desc:
        desc = (
            f"{cluster.operator}/{cluster.subgoal}: a reusable skill mined "
            f"from {cluster.source} traces (n={len(cluster.instances)})."
        )
    eff_add = _str_list(parsed.get("eff_add"), max_items=10, max_chars=64)
    eff_del = _str_list(parsed.get("eff_del"), max_items=10, max_chars=64)
    pre = _str_list(parsed.get("common_preconditions"), max_items=6, max_chars=160)
    post = _str_list(parsed.get("common_postconditions"), max_items=6, max_chars=160)
    pit = _str_list(parsed.get("common_pitfalls"), max_items=4, max_chars=160)
    pred = _str_list(parsed.get("example_predicates"), max_items=6, max_chars=64)
    modalities = [m.strip().lower() for m in
                  _str_list(parsed.get("applicable_modalities"), max_items=4, max_chars=24)
                  if m.strip()]
    if not modalities:
        modalities = ["mixed"]

    n_models = len({i.model for i in cluster.instances})
    n_samples = len({i.sample_id for i in cluster.instances})
    feasible_tasks = sorted({i.bucket for i in cluster.instances})

    sub_episodes = []
    seen_sub: set = set()
    for inst in cluster.instances:
        key = (inst.sample_id, inst.bucket)
        if key in seen_sub:
            continue
        seen_sub.add(key)
        sub_episodes.append({
            "episode_id": inst.sample_id,
            "seg_start": inst.step_idx,
            "seg_end": inst.step_idx + 1,
            "rollout_source": inst.model,
            "summary": inst.note[:120],
            "intention_tags": [skill_id],
            "outcome": "success" if inst.correct else (
                "failure" if inst.correct is False else "unknown"
            ),
            "cumulative_reward": float(inst.reward) if inst.reward is not None else 0.0,
            "quality_score": 0.0,
            "task": inst.bucket,
        })
        if len(sub_episodes) >= SUB_EPISODE_KEEP:
            break

    now = time.time()
    contract = {
        "skill_id": skill_id,
        "version": 1,
        "name": name_raw,
        "description": desc,
        "eff_add": eff_add,
        "eff_del": eff_del,
        "preconditions": pre,
        "postconditions": post,
        "example_predicates": pred,
    }
    # Field names mirror ``skill_agents.stage3_mvp.schemas.ExecutionHint``
    # (common_failure_modes, not common_pitfalls; termination_cues, not
    # common_postconditions) so the loaded ExecutionHint preserves the
    # GPT-5.4-derived signal end-to-end.  Unknown keys are silently
    # dropped by ExecutionHint.from_dict, but the JSON-on-disk record
    # keeps them for human inspection.
    execution_hint = {
        "common_preconditions": pre,
        "termination_cues": post,
        "common_failure_modes": pit,
        # Auxiliary (preserved on disk; ExecutionHint ignores extras):
        "applicable_modalities": modalities,
        "common_postconditions": post,
        "common_pitfalls": pit,
    }
    skill = {
        "skill_id": skill_id,
        "version": 1,
        "name": name_raw,
        "strategic_description": desc,
        "tags": [],
        "protocol": [],
        "contract": contract,
        "execution_hint": execution_hint,
        "expected_tag_pattern": [skill_id],
        "applicable_domains": ["visual_reasoning"] if cluster.source != MINIWOB_SOURCE else ["browsergym"],
        "verified_domains": [],
        "evidence_role": cluster.operator,
        "feasible_tasks": feasible_tasks,
        "verified_tasks": [],
        "n_instances": len(cluster.instances),
        "retired": False,
        "created_at": now,
        "updated_at": now,
        "source_type": "mined_from_trace_qa" if cluster.source != MINIWOB_SOURCE
                       else "mined_from_trace_browser",
        "status": "draft",
        "provenance": {
            "corpus": "visual_reasoning" if cluster.source != MINIWOB_SOURCE else "browsergym",
            "source_name": cluster.source,
            "decorator_version": "skillrecord_shape_v2",
            "n_models_contributing": n_models,
            "n_samples_contributing": n_samples,
            "build_pipeline": "gpt54_contract_curator_v1",
        },
        "sub_episodes": sub_episodes,
        "protocol_history": [],
        "protocol_raw": {"preconditions": pre, "postconditions": post, "pitfalls": pit},
    }
    # Match the exact ``VerificationReport`` schema expected by
    # ``skill_agents.stage3_mvp.schemas`` so ``SkillBankMVP.load`` can
    # deserialise the bank without modification.  We can't carry our
    # extra fields here without breaking that loader, so the auxiliary
    # provenance is stored under ``skill.provenance`` instead.
    report = {
        "skill_id": skill_id,
        "n_instances": len(cluster.instances),
        "eff_add_success_rate": {p: 1.0 for p in eff_add},
        "eff_del_success_rate": {p: 1.0 for p in eff_del},
        "eff_event_rate": {},
        "overall_pass_rate": 1.0,
        "worst_segments": [],
        "failure_signatures": {},
    }
    skill["provenance"]["aux_report"] = {
        "models": sorted({i.model for i in cluster.instances}),
        "tasks": feasible_tasks,
        "n_models": n_models,
        "n_samples": n_samples,
    }
    return {"skill": skill, "report": report}


# ---------------------------------------------------------------------------
# CONTRACT call
# ---------------------------------------------------------------------------

def _contract_one(
    cluster: SkillCluster, *, label_model: str,
) -> Dict[str, Any]:
    exemplars = _exemplars_for_prompt(cluster, limit=CLUSTER_EXEMPLAR_LIMIT)
    prompt = _build_contract_prompt(cluster, exemplars)

    parsed: Dict[str, Any] = {}
    err: Optional[str] = None
    if ask_model is not None:
        try:
            raw = ask_model(
                f"{_CONTRACT_SYSTEM}\n\n{prompt}",
                model=label_model,
                temperature=LLM_TEMPERATURE,
                max_tokens=CONTRACT_LLM_MAX_TOKENS,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raw = None
        if raw and not str(raw).startswith("Error"):
            obj = _extract_top_level_object(str(raw))
            if obj is not None:
                parsed = obj
            else:
                err = "json_parse_failed"
        elif raw is not None and not err:
            err = f"llm_error: {str(raw)[:120]}"
    else:
        err = "ask_model_unavailable"

    record = _build_contract_record(cluster=cluster, parsed=parsed)
    # Preserve any LLM error message under provenance.aux_report so the
    # on-disk record stays informative without polluting the strict
    # ``VerificationReport`` schema (whose ``from_dict`` is a hard
    # ``cls(**d)``).
    record["skill"]["provenance"]["aux_report"]["llm_error"] = err
    return record


# ---------------------------------------------------------------------------
# CURATOR — Jaccard pre-pass + LLM merge decision
# ---------------------------------------------------------------------------

def _token_set(items: List[str]) -> set:
    out: set = set()
    for s in items:
        for tok in re.findall(r"[a-z0-9_]+", s.lower()):
            if len(tok) >= 3:
                out.add(tok)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


_CURATOR_SYSTEM = (
    "You are a skill-bank dedup judge.  Given two skill candidates from the "
    "same source, decide if they describe the SAME reusable skill or two "
    "distinct skills.  Output strict JSON: {\"decision\":\"MERGE\"|\"KEEP_BOTH\","
    "\"keep_id\":\"<id of preferred>\",\"reason\":\"<≤ 20 words>\"}"
)


def _build_curator_prompt(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    return "\n".join([
        "SKILL A:",
        f"  id: {a['skill_id']}",
        f"  name: {a['name']}",
        f"  desc: {a['strategic_description']}",
        f"  eff_add: {a['contract']['eff_add']}",
        f"  preconditions: {a['execution_hint']['common_preconditions']}",
        f"  n_instances: {a['n_instances']}",
        "",
        "SKILL B:",
        f"  id: {b['skill_id']}",
        f"  name: {b['name']}",
        f"  desc: {b['strategic_description']}",
        f"  eff_add: {b['contract']['eff_add']}",
        f"  preconditions: {b['execution_hint']['common_preconditions']}",
        f"  n_instances: {b['n_instances']}",
        "",
        "Decide: MERGE if same skill differently named; KEEP_BOTH if distinct.",
        "Note: if op/subgoal pair differs (e.g. INSPECT/EVIDENCE vs REASON/DEDUCE),",
        "they are USUALLY distinct skills even if exemplars overlap; only MERGE",
        "when descriptions truly converge.",
    ])


def _curator_pass(
    skills: List[Dict[str, Any]], *, jaccard_thresh: float, label_model: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run pairwise dedup; return (kept_skills, merge_decisions)."""
    n = len(skills)
    keep_mask = [True] * n
    decisions: List[Dict[str, Any]] = []

    for i in range(n):
        if not keep_mask[i]:
            continue
        a = skills[i]
        a_set = (
            _token_set(a["contract"]["eff_add"])
            | _token_set(a["execution_hint"]["common_preconditions"])
        )
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            b = skills[j]
            b_set = (
                _token_set(b["contract"]["eff_add"])
                | _token_set(b["execution_hint"]["common_preconditions"])
            )
            jac = _jaccard(a_set, b_set)
            if jac < jaccard_thresh:
                continue
            # LLM call
            decision = "KEEP_BOTH"
            keep_id = a["skill_id"]
            reason = "similar but distinct (default)"
            if ask_model is not None:
                try:
                    raw = ask_model(
                        f"{_CURATOR_SYSTEM}\n\n{_build_curator_prompt(a, b)}",
                        model=label_model,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=CURATOR_LLM_MAX_TOKENS,
                    )
                except Exception:
                    raw = None
                if raw and not str(raw).startswith("Error"):
                    obj = _extract_top_level_object(str(raw))
                    if obj is not None:
                        decision = str(obj.get("decision") or "KEEP_BOTH").upper()
                        keep_id = str(obj.get("keep_id") or a["skill_id"])
                        reason = str(obj.get("reason") or "")[:120]
            decisions.append({
                "a_id": a["skill_id"], "b_id": b["skill_id"],
                "jaccard": round(jac, 3), "decision": decision,
                "keep_id": keep_id, "reason": reason,
            })
            if decision == "MERGE":
                # Drop the smaller-instance one; merge sub_episodes.
                drop_idx = j if a["n_instances"] >= b["n_instances"] else i
                keep_idx = i if drop_idx == j else j
                keeper = skills[keep_idx]
                dropper = skills[drop_idx]
                keeper["n_instances"] = keeper["n_instances"] + dropper["n_instances"]
                # Merge sub_episodes (preserve dedup by sample_id)
                seen = {se["episode_id"] for se in keeper["sub_episodes"]}
                for se in dropper["sub_episodes"]:
                    if se["episode_id"] not in seen and len(keeper["sub_episodes"]) < SUB_EPISODE_KEEP:
                        keeper["sub_episodes"].append(se)
                        seen.add(se["episode_id"])
                # Merge feasible_tasks
                keeper["feasible_tasks"] = sorted(
                    set(keeper["feasible_tasks"]) | set(dropper["feasible_tasks"])
                )
                keeper["provenance"]["merged_from"] = (
                    keeper["provenance"].get("merged_from") or []
                ) + [dropper["skill_id"]]
                keep_mask[drop_idx] = False
                if drop_idx == i:
                    break  # skill i was dropped, skip rest of inner loop

    kept = [s for k, s in zip(keep_mask, skills) if k]
    return kept, decisions


# ---------------------------------------------------------------------------
# Per-source pipeline
# ---------------------------------------------------------------------------

def _process_source(
    *,
    source: str,
    instances: List[HopInstance],
    output_dir: Path,
    label_model: str,
    workers: int,
    curator_jaccard: float,
    skip_curator: bool,
) -> Dict[str, Any]:
    """Cluster → CONTRACT (parallel) → CURATOR → write skill_bank.jsonl."""
    out_subdir = output_dir / source
    out_subdir.mkdir(parents=True, exist_ok=True)

    n_total = len(instances)
    if n_total == 0:
        logger.warning("[%s] no instances; skipping", source)
        return {"source": source, "n_instances": 0, "n_skills": 0, "skipped": True}

    # 1) Cluster.
    clusters = _cluster_instances(instances, source=source)
    raw_n_clusters = len(clusters)
    # Drop tiny clusters.
    eligible = [c for c in clusters.values() if len(c.instances) >= MIN_CLUSTER_SIZE]
    eligible.sort(key=lambda c: -len(c.instances))
    eligible = eligible[:MAX_SKILLS_PER_SOURCE]

    logger.info(
        "[%s] %d instances → %d clusters → %d eligible (>=%d, top-%d)",
        source, n_total, raw_n_clusters, len(eligible),
        MIN_CLUSTER_SIZE, MAX_SKILLS_PER_SOURCE,
    )
    if not eligible:
        return {"source": source, "n_instances": n_total,
                "n_clusters": raw_n_clusters, "n_skills": 0,
                "skipped": True}

    # 2) CONTRACT in parallel.
    t0 = time.time()
    contract_records: List[Dict[str, Any]] = []
    contract_errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_contract_one, c, label_model=label_model): c
                for c in eligible}
        for fu in as_completed(futs):
            try:
                rec = fu.result()
                contract_records.append(rec)
                if rec["report"].get("llm_error"):
                    contract_errors += 1
            except Exception as exc:
                logger.warning("CONTRACT failed: %s", exc)
                contract_errors += 1
    contract_elapsed = time.time() - t0

    skills = [r["skill"] for r in contract_records]
    skills.sort(key=lambda s: -s["n_instances"])

    # 3) CURATOR.
    if skip_curator:
        kept_skills = skills
        decisions: List[Dict[str, Any]] = []
        curator_elapsed = 0.0
    else:
        t1 = time.time()
        kept_skills, decisions = _curator_pass(
            skills, jaccard_thresh=curator_jaccard, label_model=label_model,
        )
        curator_elapsed = time.time() - t1

    # 4) Write skill_bank.jsonl
    skill_bank_path = out_subdir / "skill_bank.jsonl"
    with skill_bank_path.open("w") as f:
        for r in contract_records:
            if r["skill"]["skill_id"] in {s["skill_id"] for s in kept_skills}:
                # Replace the kept skill (which may have been merged-into) with current state
                live = next(s for s in kept_skills if s["skill_id"] == r["skill"]["skill_id"])
                f.write(json.dumps({"skill": live, "report": r["report"]},
                                   ensure_ascii=False) + "\n")

    summary = {
        "source": source,
        "n_instances_total": n_total,
        "n_clusters_raw": raw_n_clusters,
        "n_clusters_eligible": len(eligible),
        "n_contracts": len(contract_records),
        "n_skills_kept": len(kept_skills),
        "contract_errors": contract_errors,
        "curator_decisions": decisions,
        "contract_elapsed_seconds": round(contract_elapsed, 1),
        "curator_elapsed_seconds": round(curator_elapsed, 1),
        "label_model": label_model,
        "curator_jaccard_threshold": curator_jaccard,
        "skill_bank_path": str(skill_bank_path),
        "kept_skill_ids": [s["skill_id"] for s in kept_skills],
        "operator_distribution": dict(Counter(
            i.operator for i in instances
        ).most_common()),
        "subgoal_distribution": dict(Counter(
            i.subgoal for i in instances
        ).most_common()),
    }
    (out_subdir / "_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "[%s] DONE — %d skills kept (contract=%.1fs, curator=%.1fs, errs=%d)",
        source, len(kept_skills), contract_elapsed, curator_elapsed, contract_errors,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

DEFAULT_MODELS = ("gpt-5.4", "claude", "gemini", "qwen")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPT-5.4-driven skill bank construction (Option β).",
    )
    p.add_argument("--multihop-run", type=Path, required=True,
                   help="Output dir of label_qa_multihop_gpt54 "
                        "(labeling/qa_multihop_out/run_<ts>).")
    p.add_argument("--miniwob-run", type=Path, default=None,
                   help="qa_miniwob_labeled run dir; if omitted, miniwob "
                        "skipped.  Default: same as --multihop-run if it has "
                        "a 'miniwob/' folder, else None.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output dir; default: labeling/skill_bank_qa/run_<utc-ts>.")
    p.add_argument("--sources", type=str, nargs="+",
                   default=list(QA_SOURCES) + [MINIWOB_SOURCE])
    p.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--label-model", type=str, default=DEFAULT_LABEL_MODEL)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--curator-jaccard", type=float, default=DEFAULT_CURATOR_JACCARD,
                   help="Jaccard ≥ this triggers an LLM merge-decision call.")
    p.add_argument("--skip-curator", action="store_true",
                   help="Skip the dedup pass entirely (faster smoke test).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap instances per source (smoke test).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    multihop_run: Path = args.multihop_run.resolve()
    if not multihop_run.is_dir():
        print(f"[build_skillbank_qa] multihop run missing: {multihop_run}",
              file=sys.stderr)
        return 2
    miniwob_run: Optional[Path] = (
        args.miniwob_run.resolve() if args.miniwob_run else None
    )
    if miniwob_run and not miniwob_run.is_dir():
        print(f"[build_skillbank_qa] miniwob run missing: {miniwob_run}",
              file=sys.stderr)
        miniwob_run = None

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = (CODEBASE_ROOT / "labeling" / "skill_bank_qa" / f"run_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "multihop_run": str(multihop_run),
        "miniwob_run": str(miniwob_run) if miniwob_run else None,
        "output_dir": str(output_dir),
        "sources": list(args.sources),
        "models": list(args.models),
        "label_model": args.label_model,
        "workers": args.workers,
        "curator_jaccard": args.curator_jaccard,
        "skip_curator": args.skip_curator,
        "limit": args.limit,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "argv": sys.argv,
    }
    (output_dir / "_run_meta.json").write_text(json.dumps(run_meta, indent=2))

    summaries: List[Dict[str, Any]] = []
    models_tup = tuple(args.models)

    for source in args.sources:
        logger.info("Loading instances for source=%s ...", source)
        if source == MINIWOB_SOURCE:
            if miniwob_run is None:
                logger.warning("miniwob skipped (no --miniwob-run).")
                continue
            instances = list(_iter_miniwob_steps(miniwob_run, models=models_tup))
        elif source in QA_SOURCES:
            instances = list(_iter_qa_hops(multihop_run, source, models=models_tup))
        else:
            logger.warning("unknown source %s — skipping", source)
            continue

        if args.limit is not None:
            instances = instances[: args.limit]

        try:
            summary = _process_source(
                source=source,
                instances=instances,
                output_dir=output_dir,
                label_model=args.label_model,
                workers=args.workers,
                curator_jaccard=args.curator_jaccard,
                skip_curator=args.skip_curator,
            )
        except Exception as exc:
            logger.error("source %s failed: %s", source, exc)
            traceback.print_exc()
            summary = {"source": source, "error": f"{type(exc).__name__}: {exc}"}
        summaries.append(summary)

    aggregate = {
        "run_meta": run_meta,
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "n_sources": len(summaries),
        "n_skills_total": sum(s.get("n_skills_kept", 0) for s in summaries),
        "n_instances_total": sum(s.get("n_instances_total", 0) for s in summaries),
        "per_source": summaries,
    }
    (output_dir / "_run_summary.json").write_text(json.dumps(aggregate, indent=2))

    print()
    print("=" * 70)
    print(
        f"[build_skillbank_qa] DONE — {len(summaries)} sources, "
        f"{aggregate['n_skills_total']} skills, "
        f"{aggregate['n_instances_total']} instances"
    )
    print(f"[build_skillbank_qa] output: {output_dir}")
    print(f"[build_skillbank_qa] summary: {output_dir / '_run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
