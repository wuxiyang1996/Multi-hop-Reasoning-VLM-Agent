#!/usr/bin/env python
"""Plan-level LLM-judge for skill similarity (Qwen3.5-35B-A3B).

For each target skill (across the configured per-task skill banks),
shortlist K candidate skills from OTHER tasks using a cheap canonical
intent-signature heuristic, then ask the 35B judge to score each
candidate on a 1-5 plan-level similarity scale.

Output JSON file matches the contract consumed by
``frontier_data/scripts/build_plan_clustered_bank.py``::

    {
      "meta": {...},
      "results": [
        {
          "target_skill_id": "...",
          "target_task": "...",
          "target_domain": "GAME|VR|WEB|OTHER",
          "judgment": {
            "matches": [
              {
                "candidate_skill_id": "...",
                "candidate_task": "...",
                "candidate_domain": "GAME|VR|WEB|OTHER",
                "score": 4,
                "shared_reasoning": "...",
                "transfer_value": "..."
              }, ...
            ]
          }
        }, ...
      ]
    }

Usage::

    # 1. start 35B-A3B server on :8001, then in this shell:
    source scripts/use_35b_judge.sh

    # 2. run the judge over the 5 best Stage-1 GRPO banks
    python scripts/judge_plan_level_similarity.py \\
        --bank-set best_grpo \\
        --top-k 15 \\
        --workers 6 \\
        --thinking

    # output: frontier_data/output/plan_level_similarity_judgments.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("judge_plan_level_similarity")


# ── Bank configurations ──────────────────────────────────────────────

BEST_GRPO_RUNS: Dict[str, Path] = {
    "candy_crush":             REPO_ROOT / "runs/candy_crush_coevo_v4_20260519_093912/skillbank/candy_crush/skill_bank.jsonl",
    "gymv_columns":            REPO_ROOT / "runs/gymv_columns_coevo_v4_20260519_001840/skillbank/gymv_columns/skill_bank.jsonl",
    "gymv_strider":            REPO_ROOT / "runs/gymv_strider_coevo_v5_20260519_184613/skillbank/gymv_strider/skill_bank.jsonl",
    "gymv_thunder_force_iii":  REPO_ROOT / "runs/gymv_thunder_force_iii_coevo_v9_grpo_unclip/skillbank/gymv_thunder_force_iii/skill_bank.jsonl",
    "gymv_streets_of_rage_2":  REPO_ROOT / "runs/gymv_streets_of_rage_2_coevo_v5_20260520_010806/skillbank/gymv_streets_of_rage_2/skill_bank.jsonl",
}


DOMAIN_OF = {
    # GAMES (env_wrappers + retro gymv)
    "candy_crush": "GAME", "tetris": "GAME", "twenty_forty_eight": "GAME", "super_mario": "GAME",
    "gymv_columns": "GAME", "gymv_strider": "GAME", "gymv_thunder_force_iii": "GAME",
    "gymv_streets_of_rage_2": "GAME", "gymv_airstriker": "GAME", "gymv_altered_beast": "GAME",
    "gymv_dynamite_headdy": "GAME", "gymv_space_harrier_ii": "GAME",
    # WEB / desktop
    "miniwob": "WEB", "webshop": "WEB", "browsergym": "WEB", "osworld": "WEB",
    # VR
    "siv_bench": "VR", "tir_bench": "VR", "video_holmes": "VR", "visual_toolbench": "VR",
}


# ── Canonical intent classifier (inline; 7 intents) ──────────────────

CANONICAL_INTENTS = ("PERCEIVE", "RECALL", "EVALUATE", "DECIDE", "NAVIGATE", "ACT", "VERIFY")

_INTENT_PATTERNS: List[Tuple[str, List[re.Pattern]]] = [
    ("PERCEIVE", [
        re.compile(r"\b(observe|locate|identify|detect|scan|read|see|spot|note|notice|inspect|examine|look(\s+at)?|find|determine\s+the\s+(?:position|location)|check\s+the\s+(?:screen|board|state|position))\b", re.I),
    ]),
    ("RECALL", [
        re.compile(r"\b(remember|recall|reference\s+the|consult|prior\s+knowledge|memory|use\s+the\s+(?:rules|knowledge))\b", re.I),
    ]),
    ("EVALUATE", [
        re.compile(r"\b(assess|evaluate|measure|weigh|compute|calculate|count|estimate|score|rank|compare|tally|aggregate|reason\s+about)\b", re.I),
    ]),
    ("DECIDE", [
        re.compile(r"\b(decide|choose|select|pick|determine\s+(?:which|whether|what)|prioritize|opt\s+for|elect|plan|strategize)\b", re.I),
    ]),
    ("NAVIGATE", [
        re.compile(r"\b(navigate|move\s+(?:to|toward)|approach|go\s+to|travel|head\s+(?:to|toward)|advance|retreat|dodge|sidestep|reposition|relocate)\b", re.I),
    ]),
    ("VERIFY", [
        re.compile(r"\b(verify|confirm|check\s+(?:that|whether|if|the\s+(?:result|outcome|effect))|validate|ensure|ascertain|monitor|wait\s+for|observe\s+the\s+(?:effect|result|outcome))\b", re.I),
    ]),
    # ACT is fallback — covers click/press/execute/swap/attack/etc
]


def classify_intent(step: str) -> str:
    s = (step or "").strip()
    if not s:
        return "ACT"
    for intent, patterns in _INTENT_PATTERNS:
        for pat in patterns:
            if pat.search(s):
                return intent
    return "ACT"


def compressed_plan(steps: List[str]) -> List[str]:
    """De-duplicate adjacent intents and drop empty."""
    intents = [classify_intent(s) for s in steps if s and str(s).strip()]
    out: List[str] = []
    for it in intents:
        if not out or out[-1] != it:
            out.append(it)
    return out


# ── Skill record helpers ─────────────────────────────────────────────

@dataclass
class SkillRec:
    task: str
    skill_id: str
    name: str
    description: str
    steps: List[str]                    # raw protocol steps
    intent_seq: List[str]               # compressed canonical-intent plan
    domain: str
    contract_desc: str
    eff_add: List[str]
    eff_del: List[str]

    def key(self) -> str:
        return f"{self.task}::{self.skill_id}"

    def signature(self) -> str:
        return " → ".join(self.intent_seq)


def _as_list_steps(protocol) -> List[str]:
    if not protocol:
        return []
    if isinstance(protocol, list):
        return [str(x) for x in protocol if x]
    if isinstance(protocol, dict):
        steps = protocol.get("steps") or []
        if isinstance(steps, list):
            return [str(x) for x in steps if x]
    return []


def load_bank(task: str, path: Path) -> List[SkillRec]:
    recs: List[SkillRec] = []
    if not path.exists():
        logger.warning("bank not found: %s", path)
        return recs
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
            steps = _as_list_steps(s.get("protocol"))
            if not steps:
                steps = _as_list_steps(s.get("protocol_raw"))
            contract = s.get("contract") or {}
            recs.append(SkillRec(
                task=task,
                skill_id=str(s.get("skill_id", s.get("name", ""))),
                name=str(s.get("name", "")),
                description=str(s.get("strategic_description", "") or contract.get("description", "")),
                steps=steps[:8],
                intent_seq=compressed_plan(steps[:8]),
                domain=DOMAIN_OF.get(task, "OTHER"),
                contract_desc=str(contract.get("description", ""))[:400],
                eff_add=[str(x) for x in (contract.get("eff_add") or [])][:8],
                eff_del=[str(x) for x in (contract.get("eff_del") or [])][:8],
            ))
    return recs


def load_banks(bank_paths: Dict[str, Path]) -> List[SkillRec]:
    all_recs: List[SkillRec] = []
    for task, path in bank_paths.items():
        recs = load_bank(task, path)
        logger.info("  %-30s %3d skills (%s)", task, len(recs), path.relative_to(REPO_ROOT))
        all_recs.extend(recs)
    return all_recs


# ── Shortlist (content-based, NOT signature-based) ───────────────────
# We shortlist by overlap of the actual NL reasoning content (protocol
# steps + description + skill name), so candidates picked for the LLM
# reflect procedural-content similarity, not the cheap canonical
# intent label.  Implementation: bag-of-stem cosine over a simple
# tokenizer; no external deps.

import math
from collections import Counter

_STOP = frozenset({
    "the","a","an","is","are","was","were","be","been","being","of","to",
    "in","on","at","by","for","with","from","into","onto","or","and","not",
    "if","then","that","this","these","those","it","its","they","them",
    "their","there","as","so","when","while","than","do","does","did",
    "doing","done","has","have","had","having","will","would","can","could",
    "may","might","should","shall","must","just","step","steps","check",
    "checks","ensure","make","sure","next","first","second","third","last",
    "between","over","under","onto","off","out","up","down","very","also",
    "any","all","both","each","every","one","two","three","four","five",
    "i","you","we","he","she","my","your","our","his","her",
})

_WORD_RE = re.compile(r"[a-z][a-z\-]+")


def _stem(tok: str) -> str:
    # Tiny rule-based stemmer (drop common English suffixes).  We don't
    # need linguistic accuracy — only that "evaluate"/"evaluation"/"evaluating"
    # collapse to the same key so they vote together in cosine.
    for suf in ("ization","izations","ational","izations","izing","ized","izes","ize",
                "ation","ations","tions","tion","ments","ment","ness","ously","ously",
                "ically","ical","ally","ies","ing","ies","ied","ers","er","ed","es","s"):
        if len(tok) - len(suf) >= 4 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def _content_tokens(rec: SkillRec) -> Counter:
    """Tokenize the *actual procedural content* of a skill, dropping
    stopwords and stemming.  The features used are:

      - protocol.steps   (the real NL reasoning steps)
      - description      (strategic_description / contract.description)
      - skill name       (verb/object hints)
      - eff_add/eff_del  (effect predicates if any)

    Notably we do NOT use the canonical-intent signature here, since the
    whole point is to score similarity on content, not on label sequence.
    """
    parts: List[str] = []
    parts.extend(rec.steps)
    if rec.description:
        parts.append(rec.description)
    if rec.name:
        parts.append(rec.name.replace("_", " "))
    parts.extend(rec.eff_add or [])
    parts.extend(rec.eff_del or [])
    text = " ".join(parts).lower()
    bag: Counter = Counter()
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        if tok in _STOP:
            continue
        bag[_stem(tok)] += 1
    return bag


def _bag_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def shortlist_for(target: SkillRec, pool: List[SkillRec], k: int) -> List[SkillRec]:
    """Rank candidates by NL-content cosine; cross-task only.

    The ranking is intentionally content-driven so the LLM judge gets
    candidates that *look like* the target in actual procedure, not in
    its canonical label sequence.  Same-task skills are still excluded
    (we only care about cross-task transfer candidates).
    """
    t_bag = _content_tokens(target)
    scored: List[Tuple[float, SkillRec]] = []
    for cand in pool:
        if cand.task == target.task:
            continue
        if cand.key() == target.key():
            continue
        c_bag = _content_tokens(cand)
        score = _bag_cosine(t_bag, c_bag)
        scored.append((score, cand))
    scored.sort(key=lambda x: (-x[0], x[1].key()))
    return [c for _, c in scored[:k]]


# ── LLM prompt ───────────────────────────────────────────────────────

_PROMPT_HEADER = """You are an offline judge scoring how similar two agent skills are
based on the ACTUAL reasoning steps an agent executes — the substantive
"what does the agent reason about and do, in what order" — not on
abstract intent labels.

Read each skill's `protocol_steps` (its actual natural-language reasoning
procedure) carefully. Two skills are similar when the AGENT'S REASONING
PROCESS is similar — same kind of observations, same decision points,
same verification logic — even when the concrete actions or game
mechanics differ.

The `plan_signature` field is a noisy auto-generated label; **ignore it
when judging** and rely on the natural-language steps. The skill name
is also unreliable.

Score each candidate 1-5 by how transferable the reasoning procedure is:
  5 = the reasoning procedure is essentially the same (same checks, same
      decision logic, same verification) — a model trained on one would
      reason the same way on the other
  4 = strong overlap in reasoning content; 1-2 steps differ but the
      underlying procedure transfers
  3 = partial overlap; share one or two reasoning ideas but diverge
      in how they decide / verify
  2 = weak; only the broadest motivation overlaps (e.g. "both react to
      threats") with otherwise distinct procedures
  1 = unrelated reasoning procedures

Respond with EXACTLY one JSON object, no prose, no markdown fences. Be terse.
Use the CANDIDATE_N labels as candidate identifiers. Score every candidate.

{"matches":[
  {"candidate":"CANDIDATE_1","score":1-5,
   "shared":"<≤20 words: what reasoning step/check/decision is shared>",
   "transfer":"<≤20 words: what a new task could reuse from this procedure>"},
  ...
]}
"""


def _fmt_skill(rec: SkillRec, label: str) -> str:
    # Emphasize the ACTUAL reasoning steps; keep the signature only as a
    # bottom-of-card "auto label" hint the prompt explicitly tells the
    # judge to disregard.
    steps_str = "\n".join(
        f"    {i + 1}. {s[:340]}" for i, s in enumerate(rec.steps[:8])
    ) or "    (no steps)"
    contract_line = (
        f"  contract_description: {rec.contract_desc[:280]}\n" if rec.contract_desc else ""
    )
    eff_line = ""
    if rec.eff_add or rec.eff_del:
        eff_line = f"  effects: add={rec.eff_add[:5]} del={rec.eff_del[:5]}\n"
    return (
        f"{label}: {rec.skill_id}\n"
        f"  task: {rec.task}  (domain={rec.domain})\n"
        f"  name: {rec.name}\n"
        f"  description: {rec.description[:320]}\n"
        f"  protocol_steps:\n{steps_str}\n"
        f"{eff_line}"
        f"{contract_line}"
        f"  auto_label (noisy, ignore): {rec.signature()}\n"
    )


def build_prompt(target: SkillRec, candidates: List[SkillRec]) -> str:
    target_block = _fmt_skill(target, "TARGET")
    cand_blocks = []
    for i, c in enumerate(candidates):
        cand_blocks.append(_fmt_skill(c, f"CANDIDATE_{i + 1}"))
    cand_block_str = "\n\n".join(cand_blocks)
    return (
        _PROMPT_HEADER
        + "\n\n=== TARGET SKILL ===\n"
        + target_block
        + "\n\n=== CANDIDATES ===\n"
        + cand_block_str
        + "\n\n=== END ===\nReturn the JSON object now."
    )


# ── Response parsing ─────────────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


_CAND_LABEL_RE = re.compile(r"CANDIDATE[_\s]*(\d+)", re.I)
_MATCH_BLOCK_RE = re.compile(
    r'\{\s*"candidate"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*(\d+)'
    r'(?:\s*,\s*"shared"\s*:\s*"([^"]*)")?'
    r'(?:\s*,\s*"transfer"\s*:\s*"([^"]*)")?',
    re.S,
)


def _candidate_lookup(label: str, candidates: List[SkillRec]) -> Optional[SkillRec]:
    """Resolve either a CANDIDATE_N label or a literal skill_id."""
    if not label:
        return None
    m = _CAND_LABEL_RE.search(label)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    by_id = {c.skill_id: c for c in candidates}
    return by_id.get(label.strip())


def parse_judge_response(text: str, candidates: List[SkillRec]) -> List[dict]:
    """Extract list of {candidate_skill_id, score, shared_reasoning, transfer_value}.

    Tolerates truncated / partial JSON by falling back to regex extraction
    of individual match blocks.
    """
    if not text:
        return []
    txt = text.strip()
    # Strip Qwen thinking block if present
    if "</think>" in txt:
        txt = txt.split("</think>", 1)[1].strip()
    # Strip fences
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1] if "\n" in txt else txt
        if txt.endswith("```"):
            txt = txt.rsplit("```", 1)[0].strip()

    out: List[dict] = []
    obj = None
    try:
        obj = json.loads(txt)
    except Exception:
        m = _JSON_OBJ_RE.search(txt)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                pass

    if isinstance(obj, dict):
        matches = obj.get("matches") or []
        if isinstance(matches, list):
            for m in matches:
                if not isinstance(m, dict):
                    continue
                cid_raw = (m.get("candidate") or m.get("candidate_skill_id")
                           or m.get("candidate_id") or "")
                cand = _candidate_lookup(str(cid_raw), candidates)
                if cand is None:
                    continue
                try:
                    score = int(m.get("score", 0))
                except Exception:
                    continue
                if not (1 <= score <= 5):
                    continue
                out.append({
                    "candidate_skill_id": cand.skill_id,
                    "candidate_task": cand.task,
                    "candidate_domain": cand.domain,
                    "score": score,
                    "shared_reasoning": str(m.get("shared") or m.get("shared_reasoning") or "")[:400],
                    "transfer_value": str(m.get("transfer") or m.get("transfer_value") or "")[:400],
                })
            if out:
                return out

    # Fallback: salvage individual blocks (handles truncated JSON)
    for m in _MATCH_BLOCK_RE.finditer(txt):
        label = m.group(1)
        try:
            score = int(m.group(2))
        except Exception:
            continue
        if not (1 <= score <= 5):
            continue
        cand = _candidate_lookup(label, candidates)
        if cand is None:
            continue
        out.append({
            "candidate_skill_id": cand.skill_id,
            "candidate_task": cand.task,
            "candidate_domain": cand.domain,
            "score": score,
            "shared_reasoning": (m.group(3) or "")[:400],
            "transfer_value": (m.group(4) or "")[:400],
        })
    # Dedupe by candidate
    seen = set()
    deduped: List[dict] = []
    for r in out:
        if r["candidate_skill_id"] in seen:
            continue
        seen.add(r["candidate_skill_id"])
        deduped.append(r)
    return deduped


# ── Single judge call (with retry) ───────────────────────────────────

def _is_openai_reasoning_model(model: str) -> bool:
    """Models that use ``max_completion_tokens`` and a reasoning budget."""
    m = model.lower().replace("openai/", "")
    return (
        m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
    )


def _call_openrouter_reasoning(prompt: str, *, model: str, max_completion_tokens: int) -> str:
    """Direct OpenRouter call for OpenAI reasoning models (gpt-5*, o1*, o3*, o4*)."""
    import openai as _openai
    import API_func as _af
    or_key = (_af.open_router_api_key or "").strip()
    if not or_key:
        raise RuntimeError("OPENROUTER_API_KEY not configured")
    client = _openai.OpenAI(base_url=_af.OPENROUTER_BASE, api_key=or_key)
    mid = model if "/" in model else f"openai/{model}"
    resp = client.chat.completions.create(
        model=mid,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_completion_tokens,
    )
    return resp.choices[0].message.content or ""


def judge_one(
    target: SkillRec, candidates: List[SkillRec],
    *, model: str, temperature: float, max_tokens: int, thinking: bool,
    retries: int = 1,
) -> dict:
    prompt = build_prompt(target, candidates)
    raw = ""
    err = None
    matches: List[dict] = []
    use_reasoning = _is_openai_reasoning_model(model)
    for attempt in range(retries + 1):
        try:
            if use_reasoning:
                raw = _call_openrouter_reasoning(
                    prompt, model=model, max_completion_tokens=max_tokens,
                ) or ""
            else:
                from API_func import ask_model
                raw = ask_model(
                    prompt, model=model, temperature=temperature,
                    max_tokens=max_tokens, enable_thinking=thinking,
                ) or ""
            matches = parse_judge_response(raw, candidates)
            if matches:
                err = None
                break
            err = "parse_empty"
        except Exception as exc:  # noqa: BLE001
            err = f"call_error:{exc}"
            time.sleep(2 * (attempt + 1))
    return {
        "target_skill_id": target.skill_id,
        "target_task": target.task,
        "target_domain": target.domain,
        "target_signature": target.signature(),
        "n_candidates": len(candidates),
        "candidate_keys": [c.skill_id for c in candidates],
        "judgment": {"matches": matches},
        "error": err,
        "raw_len": len(raw or ""),
        "raw_preview": (raw or "")[:2500] if not matches else "",
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bank-set", choices=["best_grpo"], default="best_grpo",
                    help="which bank configuration to load")
    ap.add_argument("--top-k", type=int, default=15,
                    help="shortlist size per target (default 15)")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel LLM calls (default 6)")
    ap.add_argument("--model", default="gpt-5-mini",
                    help="judge model (gpt-5-mini is the default; Qwen/Qwen3.5-35B-A3B works too)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=2500,
                    help="response budget; bump for Qwen3 thinking")
    ap.add_argument("--thinking", action="store_true",
                    help="enable Qwen3 <think> reasoning blocks; ignored by gpt-5-mini")
    ap.add_argument("--out", default=str(REPO_ROOT / "frontier_data" / "output" / "plan_level_similarity_judgments.json"))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of target skills (debug; 0 = no cap)")
    ap.add_argument("--retry-failed-from", default="",
                    help="path to a previous judgments JSON; only re-runs targets that "
                         "had 0 matches, merges results, and overwrites --out.")
    args = ap.parse_args()

    if args.bank_set == "best_grpo":
        bank_paths = BEST_GRPO_RUNS
    else:
        raise SystemExit(f"unknown bank-set: {args.bank_set}")

    logger.info("loading skill banks (%s)", args.bank_set)
    all_recs = load_banks(bank_paths)
    logger.info("total skills loaded: %d", len(all_recs))
    if not all_recs:
        logger.error("no skills loaded; aborting")
        return 1

    if args.limit > 0:
        all_recs_targets = all_recs[: args.limit]
    else:
        all_recs_targets = all_recs

    # Retry-only mode: keep prior good results, re-run only the failed ones.
    prior_results: List[dict] = []
    if args.retry_failed_from:
        prior_path = Path(args.retry_failed_from)
        if not prior_path.exists():
            logger.error("retry-failed-from path not found: %s", prior_path)
            return 1
        prior_data = json.loads(prior_path.read_text())
        prior_results = prior_data.get("results", [])
        keep_keys = set()
        retry_keys = set()
        for r in prior_results:
            key = f"{r['target_task']}::{r['target_skill_id']}"
            if r.get("judgment", {}).get("matches"):
                keep_keys.add(key)
            else:
                retry_keys.add(key)
        all_recs_targets = [r for r in all_recs_targets if r.key() in retry_keys]
        logger.info("retry mode: keeping %d good results, retrying %d failed targets",
                    len(keep_keys), len(all_recs_targets))

    # Distribution of plan signatures
    from collections import Counter
    sig_count = Counter(r.signature() for r in all_recs)
    logger.info("distinct plan signatures: %d", len(sig_count))
    for sig, n in sig_count.most_common(10):
        logger.info("  %-50s %d", sig, n)

    # Build shortlists
    shortlists: Dict[str, List[SkillRec]] = {}
    for tgt in all_recs_targets:
        shortlists[tgt.key()] = shortlist_for(tgt, all_recs, args.top_k)

    logger.info("calling judge: model=%s thinking=%s workers=%d targets=%d k=%d",
                args.model, args.thinking, args.workers, len(all_recs_targets), args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    t0 = time.monotonic()
    completed = 0

    def _worker(tgt: SkillRec) -> dict:
        return judge_one(
            tgt, shortlists[tgt.key()],
            model=args.model, temperature=args.temperature,
            max_tokens=args.max_tokens, thinking=args.thinking,
            retries=1,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(_worker, tgt): tgt for tgt in all_recs_targets}
        for fut in concurrent.futures.as_completed(futs):
            tgt = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {
                    "target_skill_id": tgt.skill_id,
                    "target_task": tgt.task,
                    "target_domain": tgt.domain,
                    "judgment": {"matches": []},
                    "error": f"worker_exc:{exc}",
                }
            results.append(res)
            completed += 1
            n_matches = len(res.get("judgment", {}).get("matches", []))
            err = res.get("error")
            elapsed = time.monotonic() - t0
            logger.info(
                "[%3d/%3d] %s::%s → %d matches (%s) | err=%s | %.1fs",
                completed, len(all_recs_targets),
                tgt.task, tgt.skill_id[:40], n_matches,
                tgt.signature()[:50] or "-",
                err or "ok",
                elapsed,
            )

    # Merge prior good results in retry mode
    if args.retry_failed_from:
        good_prior = [r for r in prior_results if r.get("judgment", {}).get("matches")]
        merged = good_prior + results
        n_total = len(merged)
        logger.info("retry mode: merged %d prior-good + %d retried = %d total",
                    len(good_prior), len(results), n_total)
        results = merged

    # Final payload
    payload = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bank_set": args.bank_set,
            "n_banks": len(bank_paths),
            "n_skills": len(all_recs),
            "n_targets": len(all_recs_targets),
            "n_results": len(results),
            "top_k": args.top_k,
            "model": args.model,
            "thinking": args.thinking,
            "max_tokens": args.max_tokens,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "tasks": list(bank_paths.keys()),
            "retry_from": args.retry_failed_from or None,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("wrote %s (%d results)", out_path, len(results))

    # Summary stats
    score_hist = Counter()
    n_edges_per_thresh = {t: 0 for t in (3, 4, 5)}
    cross_domain_edges = 0
    for r in results:
        for m in r.get("judgment", {}).get("matches", []):
            score_hist[m["score"]] += 1
            for t in n_edges_per_thresh:
                if m["score"] >= t:
                    n_edges_per_thresh[t] += 1
            if m["candidate_domain"] != r["target_domain"]:
                if m["score"] >= 4:
                    cross_domain_edges += 1
    logger.info("=== judge summary ===")
    logger.info("  score histogram: %s", dict(sorted(score_hist.items())))
    logger.info("  edges@>=3: %d  edges@>=4: %d  edges@>=5: %d",
                n_edges_per_thresh[3], n_edges_per_thresh[4], n_edges_per_thresh[5])
    logger.info("  cross-domain edges @>=4: %d", cross_domain_edges)
    n_err = sum(1 for r in results if r.get("error"))
    logger.info("  results with error: %d / %d", n_err, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
