#!/usr/bin/env python3
"""Crafter v2 (offline): rebuild *real* failure traces from a live GRPO run's
already-collected rollout artifacts and ask the 35B teacher to propose novel
skills.

This script *only reads* from the live run directory and writes to
``<run_dir>/crafter_v2_offline/``. It never touches the live skill bank, the
live ``crafter_proposals_out/`` tree, or any process owned by the trainer.

Stage-1 prototype: extract enriched failures, run the proposer once on a
small sample, dump everything for visual inspection. If the proposer
produces non-redundant proposals we'll wire up the full pipeline (proposer
across all steps + offline judge).

Usage::

    python scripts/crafter_v2_extract_and_probe.py \\
        --run-dir runs/Qwen3.5-9B_20260507_192810 \\
        --game gymv_thunder_force_iii \\
        --max-failures 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


USELESS_ACTIONS_GYMV = {"MODE", "START", "X", "Y", "Z"}

STATE_TAG_RE = re.compile(r"<state>(.*?)</state>", re.DOTALL)
ACTIVE_SKILL_RE = re.compile(r"^Active skill:\s*(.+?)\s*(?:—|$)", re.MULTILINE)
RECENT_BLOCK_RE = re.compile(
    r"Recent actions and rewards:\n((?:\s+\S+\s*->\s*reward\s+-?\d+(?:\.\d+)?\s*\n?)+)"
)


# -------------------------- artifact loaders -----------------------------


def load_action_taking(run_dir: Path, game: str) -> list[dict]:
    rows = []
    for step_dir in sorted((run_dir / "grpo_data").glob("step_*")):
        f = step_dir / "action_taking.jsonl"
        if not f.exists():
            continue
        outer_step = int(step_dir.name.split("_")[1])
        with open(f) as fh:
            for L in fh:
                d = json.loads(L)
                if d.get("game") != game:
                    continue
                d["_outer_step"] = outer_step
                rows.append(d)
    return rows


def load_skill_selection(run_dir: Path, game: str) -> dict[tuple[str, int], dict]:
    """Return ``{(episode_id, step): row}``."""
    out: dict[tuple[str, int], dict] = {}
    for step_dir in sorted((run_dir / "grpo_data").glob("step_*")):
        f = step_dir / "skill_selection.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for L in fh:
                d = json.loads(L)
                if d.get("game") != game:
                    continue
                out[(d["episode_id"], int(d["step"]))] = d
    return out


def load_reward_log(run_dir: Path, game: str) -> dict[tuple[str, int], dict]:
    """Return ``{(episode_id, step): metadata}`` with chosen_action +
    raw_env_reward + active_skill."""
    out: dict[tuple[str, int], dict] = {}
    f = run_dir / "rewards" / "reward_log.jsonl"
    if not f.exists():
        return out
    with open(f) as fh:
        for L in fh:
            d = json.loads(L)
            if d.get("kind") != "grpo_step":
                continue
            if d.get("adapter") != "action_taking":
                continue
            if d.get("game") != game:
                continue
            out[(d["episode_id"], int(d["step"]))] = d.get("metadata", {}) | {
                "_grpo_reward": d.get("reward"),
            }
    return out


def load_episode_outcomes(run_dir: Path, game: str) -> dict[str, dict]:
    """Return ``{episode_id: {total_reward, steps, max_steps_seen}}`` from
    ``rewards/step_NNNN.jsonl`` files."""
    out: dict[str, dict] = {}
    for f in sorted((run_dir / "rewards").glob("step_*.jsonl")):
        with open(f) as fh:
            for L in fh:
                try:
                    d = json.loads(L)
                except Exception:
                    continue
                if "total_reward" not in d:
                    continue
                if d.get("game") != game:
                    continue
                out[d["episode_id"]] = {
                    "total_reward": d.get("total_reward"),
                    "steps": d.get("steps"),
                    "wall_time_s": d.get("wall_time_s"),
                }
    return out


def load_skill_bank(run_dir: Path, game: str) -> list[dict]:
    """Return slim records ``[{skill_id, name, desc, preconds, ...}]``."""
    f = run_dir / "skillbank" / game / "skill_bank.jsonl"
    out = []
    if not f.exists():
        return out
    with open(f) as fh:
        for L in fh:
            try:
                d = json.loads(L)
            except Exception:
                continue
            s = d.get("skill") or {}
            proto = s.get("protocol") or {}
            out.append({
                "skill_id": s.get("skill_id") or s.get("id"),
                "name": s.get("name"),
                "desc": (s.get("strategic_description") or "")[:200],
                "preconds": (proto.get("preconditions") or [])[:3],
            })
    return out


# -------------------------- prompt parsers -------------------------------


def parse_prompt_state(prompt: str) -> str | None:
    """Extract the ``<state>...</state>`` block from an action_taking
    prompt, if present."""
    m = STATE_TAG_RE.search(prompt)
    if m:
        return f"<state>{m.group(1)}</state>"
    return None


def parse_active_skill_name(prompt: str) -> str | None:
    m = ACTIVE_SKILL_RE.search(prompt)
    return m.group(1).strip() if m else None


def parse_recent_block(prompt: str) -> list[tuple[str, float]]:
    """Return ``[(action, reward), ...]`` parsed from the prompt, or []."""
    m = RECENT_BLOCK_RE.search(prompt)
    if not m:
        return []
    out = []
    for line in m.group(1).strip().split("\n"):
        mm = re.match(r"\s*(\S+)\s*->\s*reward\s+(-?\d+(?:\.\d+)?)", line)
        if mm:
            out.append((mm.group(1), float(mm.group(2))))
    return out


def parse_completion(completion: str) -> dict:
    out = {"subgoal": None, "reasoning": None, "action_num": None}
    sm = re.search(r"SUBGOAL:\s*(.+?)(?=\n[A-Z]+:|\Z)", completion, re.DOTALL)
    if sm:
        out["subgoal"] = sm.group(1).strip()[:300]
    rm = re.search(r"REASONING:\s*(.+?)(?=\n[A-Z]+:|\Z)", completion, re.DOTALL)
    if rm:
        out["reasoning"] = rm.group(1).strip()[:500]
    am = re.search(r"ACTION[:\s]*(\d+)", completion)
    if am:
        out["action_num"] = int(am.group(1))
    return out


def parse_action_map(prompt: str) -> dict[int, str]:
    am = re.search(
        r"Available actions[^\n]*\n((?:\s*\d+\.\s*\w+\s*\n?)+)", prompt
    )
    out = {}
    if am:
        for line in am.group(1).strip().split("\n"):
            mm = re.match(r"\s*(\d+)\.\s*(\w+)", line)
            if mm:
                out[int(mm.group(1))] = mm.group(2)
    return out


# -------------------------- failure detectors ----------------------------


def detect_failures(
    rows: list[dict],
    reward_log: dict[tuple[str, int], dict],
    episode_outcomes: dict[str, dict],
) -> list[dict]:
    """Return a list of enriched failure records.

    Failure classes detected:

    * ``USELESS_ACTION_WASTE`` — actor selected MODE/START/X/Y/Z and got 0
      raw env reward.
    * ``ZERO_REWARD_STREAK`` — ≥6 consecutive 0-reward steps in the same
      episode (reported once at the *start* of the streak).
    * ``EARLY_DEATH`` — episode total_reward < 200 and steps < 70.
    * ``SHARP_NEGATIVE`` — single-step raw env reward ≤ -50.
    """
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)
    for ep_rows in by_ep.values():
        ep_rows.sort(key=lambda r: int(r["step"]))

    failures: list[dict] = []

    for ep_id, ep_rows in by_ep.items():
        outcome = episode_outcomes.get(ep_id, {})
        total_reward = outcome.get("total_reward")
        episode_len = outcome.get("steps")
        early_death = (
            total_reward is not None
            and episode_len is not None
            and float(total_reward) < 200
            and int(episode_len) < 70
        )

        # Pre-compute per-step raw_env_reward + chosen_action lookup
        per_step: list[tuple[int, str, float]] = []
        for r in ep_rows:
            md = reward_log.get((ep_id, int(r["step"])), {})
            chosen = md.get("chosen_action")
            raw_r = md.get("raw_env_reward", 0.0) or 0.0
            per_step.append((int(r["step"]), chosen or "?", float(raw_r)))

        # ---- streak detection ----
        streak_start = None
        streak_len = 0
        streaks = []
        for i, (_step, _act, r) in enumerate(per_step):
            if r == 0.0:
                if streak_start is None:
                    streak_start = i
                streak_len += 1
            else:
                if streak_len >= 6:
                    streaks.append((streak_start, streak_len))
                streak_start = None
                streak_len = 0
        if streak_len >= 6:
            streaks.append((streak_start, streak_len))

        for (start_idx, length) in streaks:
            ref = ep_rows[start_idx]
            md = reward_log.get((ep_id, int(ref["step"])), {})
            failures.append(_build_record(
                kind="ZERO_REWARD_STREAK",
                row=ref,
                meta=md,
                episode_outcome=outcome,
                extra={
                    "streak_length": length,
                    "streak_actions": [a for (_s, a, _r) in per_step[start_idx:start_idx+length]],
                },
            ))

        # ---- per-step detectors ----
        for i, (step, act, raw_r) in enumerate(per_step):
            ref = ep_rows[i]
            md = reward_log.get((ep_id, step), {})

            if act in USELESS_ACTIONS_GYMV and raw_r == 0.0:
                failures.append(_build_record(
                    kind="USELESS_ACTION_WASTE",
                    row=ref, meta=md, episode_outcome=outcome,
                    extra={"useless_action": act},
                ))

            if raw_r <= -50.0:
                failures.append(_build_record(
                    kind="SHARP_NEGATIVE",
                    row=ref, meta=md, episode_outcome=outcome,
                    extra={"penalty": raw_r},
                ))

        # ---- whole-episode detector ----
        if early_death and ep_rows:
            ref = ep_rows[max(0, len(ep_rows) - 5)]  # pick a step near the end
            md = reward_log.get((ep_id, int(ref["step"])), {})
            failures.append(_build_record(
                kind="EARLY_DEATH",
                row=ref, meta=md, episode_outcome=outcome,
                extra={
                    "total_reward": total_reward,
                    "episode_length": episode_len,
                    "trailing_actions": [a for (_s, a, _r) in per_step[-10:]],
                },
            ))

    return failures


def _build_record(*, kind: str, row: dict, meta: dict,
                  episode_outcome: dict, extra: dict) -> dict:
    prompt = row.get("prompt", "")
    completion = row.get("completion", "")
    state = parse_prompt_state(prompt)
    active_skill = parse_active_skill_name(prompt)
    recent = parse_recent_block(prompt)
    parsed = parse_completion(completion)
    action_map = parse_action_map(prompt)
    chosen_action = (
        meta.get("chosen_action")
        or action_map.get(parsed["action_num"]) if parsed["action_num"] else None
    )
    return {
        "failure_class": kind,
        "episode_id": row["episode_id"],
        "outer_step": row["_outer_step"],
        "in_episode_step": int(row["step"]),
        "active_skill_name": active_skill,
        "state_markup": state,
        "subgoal": parsed["subgoal"],
        "reasoning": parsed["reasoning"],
        "chosen_action": chosen_action,
        "raw_env_reward": meta.get("raw_env_reward"),
        "grpo_reward": meta.get("_grpo_reward"),
        "recent_actions_rewards": recent,
        "episode_outcome": episode_outcome,
        "extra": extra,
    }


# -------------------------- 35B proposer ---------------------------------


PROPOSER_SYSTEM = """You are a skill-extraction expert for a video-game agent. \
You are reviewing FAILURE EVIDENCE from a Gym-V Thunder Force III (Genesis \
horizontal-scrolling shoot-em-up) reinforcement learning run, and proposing \
NEW reusable skills the agent should learn. The agent already has a skill \
bank — you must propose ONLY skills that are CONCRETELY DIFFERENT from \
existing ones.

DEFINITIONS:
* A skill is a reusable button-level decision policy: precondition → action \
pattern → effect.
* CONCRETE skills name specific buttons (B, A, C, UP, DOWN, LEFT, RIGHT) and \
specific game state predicates (e.g. "enemy projectile in same row as player").
* AVOID generic skills like "advance phase from early to mid" — those exist \
already and are useless. Be specific to TF3 mechanics.

OUTPUT FORMAT — strict JSON (no prose):
{
  "proposals": [
    {
      "name": "FIRE_RAPID_DURING_FORMATION",
      "rationale_evidence": "ep abc123 step 14: actor selected MODE while \
enemy ship at e10 was in fire range; lost the kill window.",
      "preconditions": ["phase in {opening, midgame}", "≥1 entity with type=object label contains 'enemy ship'"],
      "action_pattern": "press B every step until current target removed",
      "effects_add": ["score increases", "current target despawned"],
      "non_redundant_reason": "Existing 'COMMIT/ATTACK' is abstract; this skill names the actual button (B) and the specific scene (enemy ship in fire range).",
      "source_failure_ids": ["fail_xxx"]
    }
  ],
  "no_novel_skills": false,
  "review_notes": "≤200 words on patterns observed across the failures."
}

If after reviewing the failures you find NO truly novel skill (all patterns \
are already covered), set "proposals": [], "no_novel_skills": true, and \
explain in review_notes which existing skill covers each pattern.

CONSTRAINTS:
* Maximum 4 proposals.
* Each proposal MUST cite at least one failure_id from the input.
* Output ONLY the JSON object. Do not include ```json fences."""


def render_skill_bank_for_prompt(bank: list[dict]) -> str:
    lines = ["EXISTING SKILL BANK (do NOT propose duplicates):"]
    for s in bank:
        lines.append(
            f"- {s['skill_id']} ({s['name']}): {s['desc'][:150]}"
        )
    return "\n".join(lines)


def render_failures_for_prompt(failures: list[dict]) -> str:
    """Compress each failure to ~600-800 chars while keeping evidence."""
    lines = ["FAILURE EVIDENCE BATCH:"]
    for i, f in enumerate(failures):
        fid = f"fail_{f['episode_id'][-8:]}_{f['outer_step']:02d}_{f['in_episode_step']:02d}_{f['failure_class'][:6]}"
        f["_fid"] = fid
        recent_str = ", ".join(
            f"{a}->{r:+.0f}" for a, r in (f.get("recent_actions_rewards") or [])
        )
        outcome = f.get("episode_outcome") or {}
        state_excerpt = (f.get("state_markup") or "")[:1100]
        lines.append(
            f"\n[{fid}] class={f['failure_class']} outer={f['outer_step']} "
            f"in_ep={f['in_episode_step']} action={f.get('chosen_action')} "
            f"raw_r={f.get('raw_env_reward')} "
            f"active_skill={f.get('active_skill_name')} "
            f"ep_total={outcome.get('total_reward')} ep_steps={outcome.get('steps')}"
        )
        if state_excerpt:
            lines.append(f"  STATE: {state_excerpt}")
        if recent_str:
            lines.append(f"  RECENT: {recent_str}")
        if f.get("subgoal"):
            lines.append(f"  PLANNED_SUBGOAL: {f['subgoal'][:140]}")
        if f.get("extra"):
            lines.append(f"  EXTRA: {json.dumps(f['extra'])[:300]}")
    return "\n".join(lines)


def call_35b_proposer(system: str, user: str, max_tokens: int = 3500) -> tuple[str, dict]:
    import openai
    client = openai.OpenAI(
        base_url=os.environ.get("PROBE_JUDGE_URL", "http://localhost:8001/v1"),
        api_key="dummy",
    )
    resp = client.chat.completions.create(
        model="Qwen/Qwen3.5-35B-A3B",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        # disable thinking phase (we already established this works in
        # production via _ask_judge_blocking → see probe_35b_vision_schema.py)
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    return (msg.content or ""), {
        "finish_reason": resp.choices[0].finish_reason,
        "raw_len": len(msg.content or ""),
        "reasoning_len": len(getattr(msg, "reasoning", None) or ""),
    }


# -------------------------- main -----------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--game", default="gymv_thunder_force_iii")
    ap.add_argument("--max-failures", type=int, default=12,
                    help="how many failures to send to 35B in the probe")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "crafter_v2_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "enriched_failures").mkdir(exist_ok=True)
    (out_dir / "proposals").mkdir(exist_ok=True)

    print(f"[1/5] loading rollout artifacts for game={args.game} from {run_dir}…")
    rows = load_action_taking(run_dir, args.game)
    skill_sel = load_skill_selection(run_dir, args.game)
    reward_log = load_reward_log(run_dir, args.game)
    outcomes = load_episode_outcomes(run_dir, args.game)
    bank = load_skill_bank(run_dir, args.game)
    print(f"   action_taking rows: {len(rows)}")
    print(f"   skill_selection rows: {len(skill_sel)}")
    print(f"   reward_log rows: {len(reward_log)}")
    print(f"   episode outcomes: {len(outcomes)}")
    print(f"   skill bank size: {len(bank)}")

    print(f"\n[2/5] detecting failures…")
    failures = detect_failures(rows, reward_log, outcomes)
    by_kind: dict[str, int] = defaultdict(int)
    for f in failures:
        by_kind[f["failure_class"]] += 1
    print(f"   total failures: {len(failures)}")
    for k, n in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"      {k}: {n}")

    # Persist all failures (gives us a real corpus for follow-up runs)
    fpath = out_dir / "enriched_failures" / "all_failures.jsonl"
    with open(fpath, "w") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"   wrote {fpath} ({fpath.stat().st_size:,} bytes)")

    print(f"\n[3/5] sampling {args.max_failures} diverse failures for 35B…")
    random.seed(args.seed)
    by_kind_buckets: dict[str, list[dict]] = defaultdict(list)
    for f in failures:
        by_kind_buckets[f["failure_class"]].append(f)
    # Round-robin sample so all classes are represented.
    sample: list[dict] = []
    for kind, bucket in by_kind_buckets.items():
        random.shuffle(bucket)
    pos = 0
    while len(sample) < args.max_failures and any(by_kind_buckets.values()):
        for kind, bucket in by_kind_buckets.items():
            if bucket and len(sample) < args.max_failures:
                sample.append(bucket.pop())
        if all(not b for b in by_kind_buckets.values()):
            break
        pos += 1
        if pos > 100:  # safety
            break

    sample_classes: dict[str, int] = defaultdict(int)
    for f in sample:
        sample_classes[f["failure_class"]] += 1
    print(f"   sampled distribution: {dict(sample_classes)}")

    print(f"\n[4/5] calling 35B proposer on {len(sample)} failures…")
    bank_block = render_skill_bank_for_prompt(bank)
    fail_block = render_failures_for_prompt(sample)
    user_msg = f"{bank_block}\n\n{fail_block}\n\nReview the failures and emit your JSON."
    print(f"   prompt size: system={len(PROPOSER_SYSTEM)} chars, "
          f"user={len(user_msg)} chars")

    raw, meta = call_35b_proposer(PROPOSER_SYSTEM, user_msg)
    print(f"   meta: {json.dumps(meta)}")
    print(f"\n--- 35B raw output (first 3000 chars) ---")
    print(raw[:3000])
    print()

    print(f"\n[5/5] parsing JSON…")
    parsed = None
    parse_err = None
    try:
        # Strip code fences if model added them despite instructions.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(),
                         flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
    except Exception as e:
        parse_err = str(e)
    if parsed is not None:
        n_prop = len(parsed.get("proposals", []))
        print(f"   ✓ parsed {n_prop} proposals (no_novel={parsed.get('no_novel_skills')})")
        print(f"   review_notes: {(parsed.get('review_notes') or '')[:300]}")
        for i, p in enumerate(parsed.get("proposals", [])):
            print(f"\n   [proposal {i+1}] {p.get('name')}")
            print(f"      rationale: {(p.get('rationale_evidence') or '')[:200]}")
            print(f"      precond:   {p.get('preconditions')}")
            print(f"      pattern:   {p.get('action_pattern')}")
            print(f"      novel:     {(p.get('non_redundant_reason') or '')[:200]}")
    else:
        print(f"   ✗ JSON parse failed: {parse_err}")

    # Persist
    out_blob = {
        "run_dir": str(run_dir),
        "game": args.game,
        "n_failures_total": len(failures),
        "by_kind": dict(by_kind),
        "n_sampled": len(sample),
        "sampled_distribution": dict(sample_classes),
        "raw": raw,
        "meta": meta,
        "parsed": parsed,
        "parse_err": parse_err,
    }
    out_file = out_dir / "proposals" / "probe_run.json"
    with open(out_file, "w") as fh:
        json.dump(out_blob, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_file}")

    return 0 if parsed and not parsed.get("no_novel_skills", True) else 1


if __name__ == "__main__":
    sys.exit(main())
