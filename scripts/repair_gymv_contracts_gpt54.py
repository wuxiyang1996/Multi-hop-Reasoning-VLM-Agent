#!/usr/bin/env python
"""Backfill the missing GPT-5.4 contract fields on legacy gym_v skill banks.

The 8 gym_v + 5 retro skill banks under
``labeling/skill_bank_out/run_20260430_030637/gym_v/<game>/skill_bank.jsonl``
were produced by an older event-mining pipeline that only filled
``contract.eff_add`` / ``eff_del`` from event-trigger statistics.  The
``preconditions``, ``postconditions``, ``example_predicates`` (and
``execution_hint`` / ``protocol_raw``) fields were left empty, which
makes the resulting skills incomparable to env_wrappers / qa skills
(those have ~20 predicates each thanks to ``build_skillbank_qa_gpt54``).

This script repairs that asymmetry **in place per-record but to a
fresh run dir**: for every skill in every gym_v bank we ask gpt-5.4 to
distil:

  * ``preconditions``         (≤ 6 short bullets)
  * ``postconditions``        (≤ 6 short bullets)
  * ``example_predicates``    (3-6 snake_case predicates)
  * ``eff_add`` / ``eff_del`` (only when the original is empty)

…using the existing ``name`` + ``strategic_description`` +
``protocol[*].notes`` + ``support`` event keys + (optional)
``sub_episodes[*].summary`` as context.  The resulting JSON is patched
onto the original skill record; every other field (``protocol``,
``sub_episodes``, ``report``, ``provenance``, …) is preserved verbatim
so ``SkillBankMVP.load`` keeps working.

Output layout (mirrors the input):

    labeling/skill_bank_out/run_repair_<utc-ts>/gym_v/<game>/skill_bank.jsonl
    labeling/skill_bank_out/run_repair_<utc-ts>/_repair_summary.json

Run::

    python scripts/repair_gymv_contracts_gpt54.py        # repair all 13 banks
    python scripts/repair_gymv_contracts_gpt54.py --games Temporal_Strider-v0
    python scripts/repair_gymv_contracts_gpt54.py --limit 2 -v   # smoke test

Idempotent: the script never touches the source ``run_20260430_030637``
directory, only writes a new sibling.  After verifying the repaired
bank, point ``sft_data_inventory/build_inventory.py``'s
``GYMV_BANK_RUN`` constant at the new run dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
for p in [str(WORKSPACE_ROOT), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Pull in API keys (mirrors the pattern used by other GPT-5.4 driver scripts)
try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key  # type: ignore
    if getattr(_ak, "openai_api_key", "") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _ak.openai_api_key  # type: ignore
except Exception:  # pragma: no cover
    pass


logger = logging.getLogger("repair_gymv_contracts")

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_WORKERS = 16
DEFAULT_BANK_RUN = REPO_ROOT / "labeling/skill_bank_out/run_20260430_030637/gym_v"


# ---------------------------------------------------------------------------
# OpenRouter / OpenAI client (same pattern as other GPT-5.4 driver scripts)
# ---------------------------------------------------------------------------
def _get_openai_client():
    from openai import OpenAI  # type: ignore
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# JSON parsing helpers (lifted from build_skillbank_qa_gpt54)
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
                    return json.loads(text[start: i + 1])
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


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert at distilling player-skill contracts from observed "
    "trajectories in retro action / arcade games (Sega Genesis era: "
    "platformers, beat-em-ups, shooters, puzzle games).  You are given a "
    "single mined skill — its name, strategic description, observed action "
    "protocol, the trigger events that recur in its trajectories, and a "
    "couple of episode summaries.  Your job is to fill in the missing "
    "contract fields (preconditions, postconditions, predicate vocabulary, "
    "and effect predicates) using ONLY what the evidence supports.  Output "
    "must be a single STRICT JSON object — nothing else."
)


def _build_repair_prompt(
    *, game: str, skill: Dict[str, Any], existing: Dict[str, Any],
) -> str:
    """Compose the user-side prompt for one skill."""
    skill_id = skill.get("skill_id", "")
    name = skill.get("name", "")
    desc = (skill.get("strategic_description") or "").strip()

    # Pull protocol behavioural cues.  Two on-disk shapes exist in legacy
    # gym_v banks:
    #   * Type A — ``protocol: list[dict]`` produced by the original
    #     event-mining pipeline, each entry has ``op`` + ``notes`` +
    #     ``evidence_role`` + ``payload``.
    #   * Type B — ``protocol: dict`` produced by the later crafter /
    #     promotion-writeback pipeline, with ``steps`` (list[str]),
    #     ``preconditions``, ``success_criteria``, ``abort_criteria``,
    #     and ``expected_duration`` keys.
    proto_lines: List[str] = []
    proto = skill.get("protocol")
    aux_proto: Dict[str, Any] = {}
    if isinstance(proto, list):
        for i, step in enumerate(proto, 1):
            if not isinstance(step, dict):
                continue
            op = step.get("op", "?")
            notes = (step.get("notes") or "").strip()
            evidence = step.get("evidence_role", "")
            payload = step.get("payload") or {}
            if notes:
                head = f" [op={op}, role={evidence}"
                if payload:
                    head += f", payload={json.dumps(payload, ensure_ascii=False)}"
                head += "]"
                proto_lines.append(f"  {i}. {notes}{head}")
    elif isinstance(proto, dict):
        steps = proto.get("steps") or []
        for i, s in enumerate(steps, 1):
            if isinstance(s, str) and s.strip():
                # Crafter steps are snake_case_phrases; render them readable.
                pretty = s.replace("_", " ").strip()
                proto_lines.append(f"  {i}. {pretty}  [from crafter:{proto.get('source','?')}]")
        for k in ("preconditions", "success_criteria", "abort_criteria"):
            v = proto.get(k) or []
            if v:
                aux_proto[k] = v
        dur = proto.get("expected_duration")
        if dur is not None:
            aux_proto["expected_duration_steps"] = dur

    # Trigger events from the legacy event-mining pipeline.  Keys look
    # like "event.e11[type_appeared" — preserve them verbatim, the LLM
    # can interpret them.
    support = skill.get("support") or existing.get("support") or {}
    if isinstance(support, dict) and support:
        support_lines = [
            f"  - {k}  (count={v})" for k, v in list(support.items())[:8]
        ]
    else:
        support_lines = ["  (no event support recorded)"]

    # A couple of sub-episode goal/summary lines for grounding
    summaries: List[str] = []
    for ep in (skill.get("sub_episodes") or [])[:3]:
        s = (ep.get("summary") or "").strip()
        if s:
            summaries.append(f"  - {s[:160]}")
    if not summaries:
        summaries = ["  (no episode summaries available)"]

    n_inst = skill.get("n_instances") or 0
    feasible = skill.get("feasible_tasks") or [game]
    evidence_role = skill.get("evidence_role", "")

    # Crafter-synthesised skills (Type B) have informative *names* like
    # ``compose__COMMIT__EXPLORE__then__RECOVER__EVADE`` or
    # ``patch__Recover/Evade`` — surface that for the LLM since their
    # ``strategic_description`` is empty.
    name_hint = name.replace("__", " · ").replace("_", " ")
    if not desc:
        desc = (
            f"(synthesised skill — name decoded: {name_hint!r}; "
            f"evidence_role={evidence_role!r}; no human description recorded)"
        )

    keep_eff_add = bool(_str_list(existing.get("eff_add"), max_items=10))
    keep_eff_del = bool(_str_list(existing.get("eff_del"), max_items=10))

    aux_proto_lines: List[str] = []
    for k, v in aux_proto.items():
        aux_proto_lines.append(f"  {k}: {v}")

    parts = [
        f"GAME: {game}  (Sega Genesis arcade environment, video modality)",
        f"SKILL_ID: {skill_id}",
        f"NAME: {name}",
        f"FEASIBLE_TASKS: {feasible}",
        f"N_INSTANCES: {n_inst}",
        f"EVIDENCE_ROLE: {evidence_role}",
        "",
        "STRATEGIC_DESCRIPTION:",
        f"  {desc}",
        "",
        "OBSERVED PROTOCOL (the actions the player took when this skill was triggered):",
        *(proto_lines or ["  (no protocol notes recorded)"]),
        *(["", "AUX PROTOCOL META:", *aux_proto_lines] if aux_proto_lines else []),
        "",
        "TRIGGER EVENTS (from event-mining over trajectories):",
        *support_lines,
        "",
        "SAMPLE EPISODE SUMMARIES:",
        *summaries,
        "",
        "EXISTING contract fields you should NOT overwrite if they have content:",
        f"  eff_add (existing, {'KEEP' if keep_eff_add else 'EMPTY → fill in'}): "
        f"{existing.get('eff_add') or []}",
        f"  eff_del (existing, {'KEEP' if keep_eff_del else 'EMPTY → fill in'}): "
        f"{existing.get('eff_del') or []}",
        "",
        "TASK: Output STRICT JSON with EXACTLY these keys:",
        "{",
        '  "preconditions": ["<bullet>", ...],     // ≤ 6, ≤ 16 words each, observable game-state facts',
        '  "postconditions": ["<bullet>", ...],    // ≤ 6, observable outcomes after the skill commits',
        '  "example_predicates": ["<predicate>", ...],   // 3-6 short snake_case names',
        '  "eff_add": ["<predicate>", ...],        // 1-6 state predicates becoming TRUE',
        '  "eff_del": ["<predicate>", ...],        // 1-6 state predicates becoming FALSE',
        '  "common_pitfalls": ["<bullet>", ...]    // ≤ 4 likely failure modes',
        "}",
        "",
        "CONSTRAINTS:",
        "  - Predicates are short snake_case names (e.g. 'enemy_dispatched',",
        "    'player_grounded', 'jump_window_closed', 'screen_scrolled').",
        "  - Preconditions must be CONCRETE observable facts about the game",
        "    state at the moment the skill is invoked — not vague guidance.",
        "  - Postconditions describe the immediate observable outcome (1-2",
        "    frames later) — not long-horizon goals.",
        "  - If the skill is fundamentally an INSPECT/ANALYSIS skill (no",
        "    action commitment), eff_add / eff_del may describe knowledge",
        "    predicates (e.g. 'hazard_localized', 'next_target_chosen').",
        "  - Tie predicates to the protocol notes & strategic description.",
        "    Do NOT invent game mechanics not implied by the evidence.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def _call_gpt54(client, *, model: str, prompt: str, attempt: int) -> Optional[Dict[str, Any]]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=900,
        )
    except Exception as exc:
        logger.warning("LLM call failed (attempt %d): %s", attempt, exc)
        return None
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    parsed = _extract_top_level_object(text)
    if parsed is None:
        logger.warning("LLM returned unparsable JSON (attempt %d): %s", attempt, text[:200])
    return parsed


# ---------------------------------------------------------------------------
# Repair one skill record (in-memory)
# ---------------------------------------------------------------------------
def _repair_record(
    rec: Dict[str, Any], *, game: str, client, model: str,
) -> Tuple[Dict[str, Any], str]:
    """Patch contract / execution_hint on ``rec`` (a single bank line dict).

    Returns ``(record, status)``.  Status ∈ {"ok", "skipped_complete",
    "llm_fail", "parse_fail"}.
    """
    sk = rec.get("skill") or rec
    contract = sk.get("contract") or {}
    pre = _str_list(contract.get("preconditions"), max_items=6)
    post = _str_list(contract.get("postconditions"), max_items=6)
    pred = _str_list(contract.get("example_predicates"), max_items=6)
    if pre and post and pred:
        return rec, "skipped_complete"

    prompt = _build_repair_prompt(game=game, skill=sk, existing=contract)
    parsed: Optional[Dict[str, Any]] = None
    for attempt in (1, 2):
        parsed = _call_gpt54(client, model=model, prompt=prompt, attempt=attempt)
        if parsed:
            break
    if parsed is None:
        return rec, "llm_fail"

    new_pre = _str_list(parsed.get("preconditions"), max_items=6, max_chars=160)
    new_post = _str_list(parsed.get("postconditions"), max_items=6, max_chars=160)
    new_pred = _str_list(parsed.get("example_predicates"), max_items=6, max_chars=64)
    new_pit = _str_list(parsed.get("common_pitfalls"), max_items=4, max_chars=160)
    new_eff_add = _str_list(parsed.get("eff_add"), max_items=6, max_chars=64)
    new_eff_del = _str_list(parsed.get("eff_del"), max_items=6, max_chars=64)

    if not (new_pre or new_post or new_pred):
        return rec, "parse_fail"

    # Patch contract — preserve existing eff_add/eff_del when they already
    # have content (the legacy event-miner produced those from real
    # trigger statistics, so they're more trustworthy than an LLM guess).
    contract.setdefault("skill_id", sk.get("skill_id"))
    contract.setdefault("name", sk.get("name"))
    contract.setdefault("description", sk.get("strategic_description", ""))
    contract["preconditions"] = new_pre
    contract["postconditions"] = new_post
    contract["example_predicates"] = new_pred
    if not _str_list(contract.get("eff_add"), max_items=10):
        contract["eff_add"] = new_eff_add
    if not _str_list(contract.get("eff_del"), max_items=10):
        contract["eff_del"] = new_eff_del

    # Patch execution_hint (the bank loader / SkillQueryEngine reads this)
    eh = sk.get("execution_hint") or {}
    if isinstance(eh, list):
        # Some legacy records store execution_hint as an empty list;
        # convert to dict.
        eh = {}
    eh["common_preconditions"] = new_pre
    eh["termination_cues"] = new_post
    eh["common_failure_modes"] = new_pit
    eh.setdefault("applicable_modalities", ["video", "pixel_frame"])
    eh.setdefault("common_postconditions", new_post)
    eh.setdefault("common_pitfalls", new_pit)

    sk["contract"] = contract
    sk["execution_hint"] = eh
    raw = sk.get("protocol_raw") or {}
    raw["preconditions"] = new_pre
    raw["postconditions"] = new_post
    raw["pitfalls"] = new_pit
    sk["protocol_raw"] = raw
    rec["skill"] = sk
    return rec, "ok"


# ---------------------------------------------------------------------------
# Per-game driver
# ---------------------------------------------------------------------------
def _process_game(
    *, src_bank: Path, dst_bank: Path, game: str,
    model: str, workers: int, limit: Optional[int],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for line in src_bank.open():
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    if limit is not None:
        records = records[:limit]

    started = time.time()
    client = _get_openai_client()
    statuses = ["pending"] * len(records)
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, rec in enumerate(records):
            futures[ex.submit(_repair_record, rec, game=game, client=client, model=model)] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                rec, status = fut.result()
            except Exception as exc:
                logger.error("[%s/%d] repair raised: %s", game, idx, exc)
                rec, status = records[idx], "exc"
            records[idx] = rec
            statuses[idx] = status

    dst_bank.parent.mkdir(parents=True, exist_ok=True)
    with dst_bank.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elapsed = time.time() - started

    counts = {
        "ok": statuses.count("ok"),
        "skipped_complete": statuses.count("skipped_complete"),
        "llm_fail": statuses.count("llm_fail"),
        "parse_fail": statuses.count("parse_fail"),
        "exc": statuses.count("exc"),
    }
    return {
        "game": game,
        "src_bank": str(src_bank),
        "dst_bank": str(dst_bank),
        "n_records": len(records),
        "elapsed_s": round(elapsed, 1),
        **counts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bank-run", type=Path, default=DEFAULT_BANK_RUN,
                    help=f"Source gym_v bank dir (default {DEFAULT_BANK_RUN}).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output dir (default: labeling/skill_bank_out/run_repair_<utc-ts>/gym_v).")
    ap.add_argument("--games", nargs="+", default=None,
                    help="Optional subset of game names to repair.  Default: all.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"OpenRouter model (default {DEFAULT_MODEL}).")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap skills per game (smoke test).")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    src_root: Path = args.bank_run.resolve()
    if not src_root.is_dir():
        print(f"[repair_gymv_contracts] source dir missing: {src_root}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        out_root = args.output_dir.resolve()
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = (REPO_ROOT / "labeling" / "skill_bank_out" /
                    f"run_repair_{ts}" / "gym_v")
    out_root.mkdir(parents=True, exist_ok=True)

    games = args.games or [d.name for d in sorted(src_root.iterdir()) if d.is_dir()]
    summaries: List[Dict[str, Any]] = []
    started_all = time.time()
    for g in games:
        src = src_root / g / "skill_bank.jsonl"
        if not src.is_file():
            logger.warning("missing source for %s: %s", g, src)
            continue
        dst = out_root / g / "skill_bank.jsonl"
        logger.info(">> repairing %s (n=%d)", g, sum(1 for _ in src.open()))
        s = _process_game(
            src_bank=src, dst_bank=dst, game=g,
            model=args.model, workers=args.workers, limit=args.limit,
        )
        summaries.append(s)
        logger.info(
            "   done: ok=%d skipped=%d llm_fail=%d parse_fail=%d  "
            "(%.1fs) -> %s",
            s["ok"], s["skipped_complete"], s["llm_fail"], s["parse_fail"],
            s["elapsed_s"], dst,
        )

    elapsed = time.time() - started_all
    summary_path = out_root / "_repair_summary.json"
    summary_path.write_text(json.dumps({
        "src_run": str(src_root),
        "dst_run": str(out_root),
        "model": args.model,
        "workers": args.workers,
        "limit": args.limit,
        "n_games": len(summaries),
        "elapsed_s": round(elapsed, 1),
        "totals": {
            "ok": sum(s["ok"] for s in summaries),
            "skipped_complete": sum(s["skipped_complete"] for s in summaries),
            "llm_fail": sum(s["llm_fail"] for s in summaries),
            "parse_fail": sum(s["parse_fail"] for s in summaries),
            "exc": sum(s["exc"] for s in summaries),
            "n_records": sum(s["n_records"] for s in summaries),
        },
        "per_game": summaries,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2))
    logger.info("=== summary -> %s (elapsed %.1fs) ===", summary_path, elapsed)

    print()
    print("=" * 70)
    print(f"[repair_gymv_contracts] DONE — {len(summaries)} games")
    print(f"  source : {src_root}")
    print(f"  output : {out_root}")
    print(f"  ok={sum(s['ok'] for s in summaries)}  "
          f"skipped={sum(s['skipped_complete'] for s in summaries)}  "
          f"llm_fail={sum(s['llm_fail'] for s in summaries)}  "
          f"parse_fail={sum(s['parse_fail'] for s in summaries)}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
