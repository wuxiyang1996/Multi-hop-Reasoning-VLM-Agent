"""Bootstrap mega-skill contracts from teacher demonstrations.

Pipeline:
  1. Load high-reward teacher episodes (Gemini + Claude)
  2. Segment episodes by reward spikes (natural skill boundaries)
  3. Use Gemini Flash to label each segment with a mega-skill + extract effects
  4. Aggregate effects across episodes → populate contracts
  5. Write enriched skill_bank.jsonl for each game

Usage:
  python scripts/bootstrap_contracts.py \
      --games gymv_altered_beast gymv_airstriker \
      --output-dir frontier_data/output/stage2_seeds_v3_bootstrapped
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEACHER_DATA_ROOT = Path("/workspace/SFT_Data/gymv_games")
SEED_BANK_ROOT = Path("frontier_data/output/stage2_seeds_v3_grpo")
CODEBOOK_PATH = Path("frontier_data/output/mega_skill_codebook_v2.json")

GAME_MAP = {
    "gymv_altered_beast": "Temporal_AlteredBeast-v0",
    "gymv_airstriker": "Temporal_Airstriker-v0",
    "gymv_dynamite_headdy": "Temporal_DynamiteHeaddy-v0",
    "gymv_space_harrier_ii": "Temporal_SpaceHarrierII-v0",
}

MODELS = ["gemini", "claude"]
MIN_REWARD_FOR_EPISODE = 50  # skip low-reward episodes


# ---------------------------------------------------------------------------
# 1. Episode loading
# ---------------------------------------------------------------------------

def load_episodes(game_slug: str) -> List[Dict[str, Any]]:
    """Load all high-reward teacher episodes for a game."""
    env_name = GAME_MAP.get(game_slug)
    if not env_name:
        raise ValueError(f"Unknown game slug: {game_slug}")

    episodes = []
    for model in MODELS:
        ep_dir = TEACHER_DATA_ROOT / model / env_name
        if not ep_dir.exists():
            print(f"  [WARN] {ep_dir} not found, skipping")
            continue
        for ep_file in sorted(ep_dir.glob("episode_*.json")):
            ep = json.loads(ep_file.read_text())
            total_r = float(ep.get("metadata", {}).get("total_reward", 0))
            if total_r < MIN_REWARD_FOR_EPISODE:
                continue
            ep["_source_model"] = model
            ep["_source_file"] = str(ep_file)
            episodes.append(ep)

    print(f"  Loaded {len(episodes)} episodes for {game_slug} "
          f"(models: {MODELS}, min_reward={MIN_REWARD_FOR_EPISODE})")
    return episodes


# ---------------------------------------------------------------------------
# 2. Reward-based segmentation
# ---------------------------------------------------------------------------

def segment_episode(ep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split an episode into segments at reward spike boundaries.

    Segments are contiguous runs of steps. Boundaries are placed:
      - Before the first step with positive reward in a streak
      - At significant entity/phase changes in the schema
    """
    exps = ep.get("experiences", [])
    if not exps:
        return []

    segments = []
    seg_start = 0
    seg_reward = 0.0
    zero_run = 0

    for i, e in enumerate(exps):
        r = float(e.get("reward", 0))
        seg_reward += r

        if r > 0:
            zero_run = 0
        else:
            zero_run += 1

        is_last = (i == len(exps) - 1)
        # Cut when: reward spike followed by ≥3 zero-reward steps, or end
        should_cut = (
            is_last
            or (seg_reward > 0 and zero_run >= 3 and i - seg_start >= 3)
            or (i - seg_start >= 25)  # max segment length
        )

        if should_cut and i > seg_start:
            seg = _build_segment(ep, seg_start, i + 1, seg_reward)
            segments.append(seg)
            seg_start = i + 1
            seg_reward = 0.0
            zero_run = 0

    return segments


def _build_segment(
    ep: Dict[str, Any], start: int, end: int, total_reward: float,
) -> Dict[str, Any]:
    """Package a segment with state snapshots for LLM labeling."""
    exps = ep["experiences"]
    start_schema = exps[start].get("metadata", {}).get("schema", "")
    end_schema = exps[min(end - 1, len(exps) - 1)].get("metadata", {}).get("schema", "")

    actions = [exps[i].get("action", "?") for i in range(start, min(end, len(exps)))]
    rewards = [float(exps[i].get("reward", 0)) for i in range(start, min(end, len(exps)))]
    reward_steps = [i for i, r in enumerate(rewards) if r > 0]

    return {
        "episode_id": ep.get("episode_id", ""),
        "source_model": ep.get("_source_model", ""),
        "game": ep.get("game_name", ""),
        "start_step": start,
        "end_step": end,
        "n_steps": end - start,
        "total_reward": total_reward,
        "actions": actions,
        "reward_steps": reward_steps,
        "start_schema": start_schema[:1500],
        "end_schema": end_schema[:1500],
        "action_summary": _summarize_actions(actions),
    }


def _summarize_actions(actions: List[str]) -> str:
    """Compact action sequence summary."""
    from collections import Counter
    counts = Counter(actions)
    top = counts.most_common(5)
    return ", ".join(f"{a}×{c}" for a, c in top)


# ---------------------------------------------------------------------------
# 3. LLM labeling with Gemini Flash
# ---------------------------------------------------------------------------

MEGA_SKILLS = [
    "Recover/Survive",
    "Explore",
    "Inspect/Setup",
    "Execute",
    "Commit/Evade",
    "Attack/Engage",
    "Navigate/Position",
    "Collect/Gather",
    "Defend/Hold",
    "Optimize/Improve",
]

_LABEL_PROMPT_TEMPLATE = """\
You are analyzing a game agent's behavior segment to extract transferable skill patterns.

GAME: {game}
SEGMENT: steps {start}-{end} ({n_steps} steps, reward={reward:.0f})
ACTIONS: {action_summary}

STATE AT START:
{start_schema}

STATE AT END:
{end_schema}

MEGA-SKILL CATEGORIES (pick the BEST match):
{skill_list}

Respond with EXACTLY one JSON object (no markdown fences):
{{
  "mega_skill": "<category name from list above>",
  "preconditions": ["<observable condition that must be true BEFORE this skill>", ...],
  "effects_add": ["<state predicate that becomes true AFTER this skill>", ...],
  "effects_del": ["<state predicate that becomes false AFTER this skill>", ...],
  "tactical_description": "<1-2 sentence description of WHAT the agent did and WHY>",
  "confidence": <0.0-1.0>
}}

Rules for predicates:
- Use format: category.predicate=value (e.g. player.health=low, enemy.count=0, game.phase=boss)
- Preconditions describe WHEN this skill should activate
- Effects describe WHAT changes after successful execution
- Be specific to the game state, not generic
- Include 2-4 preconditions and 2-4 effects
"""


def label_segment_with_llm(
    segment: Dict[str, Any],
    model: str = "google/gemini-2.5-flash",
) -> Optional[Dict[str, Any]]:
    """Call Gemini Flash to label a segment with mega-skill + effects."""
    skill_list = "\n".join(f"  - {s}" for s in MEGA_SKILLS)

    prompt = _LABEL_PROMPT_TEMPLATE.format(
        game=segment["game"],
        start=segment["start_step"],
        end=segment["end_step"],
        n_steps=segment["n_steps"],
        reward=segment["total_reward"],
        action_summary=segment["action_summary"],
        start_schema=segment["start_schema"][:800],
        end_schema=segment["end_schema"][:800],
        skill_list=skill_list,
    )

    try:
        import openai
        base_url = os.environ.get("BOOTSTRAP_VLLM_URL", "http://localhost:8200/v1")
        model_name = os.environ.get("BOOTSTRAP_VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
        client = openai.OpenAI(base_url=base_url, api_key="na")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"    [LLM ERROR] {exc}")
        return None

    return _parse_json_response(raw)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from potentially noisy LLM response."""
    if not raw:
        return None
    raw = raw.strip()
    # Strip <think>...</think> if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# 4. Aggregate into contracts
# ---------------------------------------------------------------------------

def _normalize_predicate(pred: str) -> str:
    """Normalize a predicate for dedup: lowercase, collapse synonyms."""
    p = pred.strip().lower()
    p = re.sub(r"\s+", "", p)
    p = p.replace("_", ".").replace("position", "pos").replace("status", "state")
    p = p.replace("enemy.ship", "enemy").replace("player.ship", "player")
    p = p.replace("player.avatar", "player")
    return p


def _dedup_predicates(preds: Dict[str, int]) -> Dict[str, int]:
    """Merge predicates that normalize to the same key, keep best wording."""
    canonical: Dict[str, Tuple[str, int]] = {}
    for pred, count in preds.items():
        norm = _normalize_predicate(pred)
        if norm in canonical:
            old_pred, old_count = canonical[norm]
            canonical[norm] = (old_pred if old_count >= count else pred,
                               old_count + count)
        else:
            canonical[norm] = (pred, count)
    return {pred: count for pred, count in canonical.values()}


def _collect_predicates(
    segments: List[Dict[str, Any]],
    min_conf: float = 0.3,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int],
           List[str], List[float]]:
    """Extract and deduplicate predicates from labeled segments."""
    raw_preconds: Dict[str, int] = defaultdict(int)
    raw_eff_add: Dict[str, int] = defaultdict(int)
    raw_eff_del: Dict[str, int] = defaultdict(int)
    descriptions: List[str] = []
    rewards: List[float] = []

    for seg in segments:
        label = seg.get("label", {})
        conf = float(label.get("confidence", 0))
        if conf < min_conf:
            continue

        for p in label.get("preconditions", []):
            raw_preconds[p] += 1
        for e in label.get("effects_add", []):
            raw_eff_add[e] += 1
        for e in label.get("effects_del", []):
            raw_eff_del[e] += 1

        desc = label.get("tactical_description", "")
        if desc:
            descriptions.append(desc)
        rewards.append(seg.get("total_reward", 0))

    return (
        _dedup_predicates(raw_preconds),
        _dedup_predicates(raw_eff_add),
        _dedup_predicates(raw_eff_del),
        descriptions,
        rewards,
    )


def _top_predicates(preds: Dict[str, int], n_instances: int,
                    top_k: int = 5, min_frac: float = 0.1) -> List[str]:
    """Select top-k predicates exceeding a minimum frequency fraction."""
    min_count = max(2, int(n_instances * min_frac))
    ranked = sorted(preds.items(), key=lambda x: -x[1])
    return [p for p, c in ranked if c >= min_count][:top_k]


def aggregate_contracts(
    labeled_segments: List[Dict[str, Any]],
    seed_skills: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate labeled segments into enriched skill bank entries."""

    name_map = {
        "Recover/Survive": ["seed.recover_survive", "seed.early_recover_survive",
                            "seed.late_recover_survive"],
        "Explore": ["seed.explore", "seed.explore.f63b"],
        "Inspect/Setup": ["seed.inspect_setup"],
        "Execute": ["seed.mid_execute"],
        "Commit/Evade": ["seed.commit_evade"],
        "Attack/Engage": ["seed.commit_evade"],
        "Navigate/Position": ["seed.explore", "seed.explore.f63b",
                              "seed.commit_explore"],
        "Collect/Gather": ["seed.commit_explore"],
        "Defend/Hold": ["seed.recover_survive"],
        "Optimize/Improve": ["seed.recover_reshuffle"],
    }

    by_mega: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seg in labeled_segments:
        label = seg.get("label", {})
        mega = label.get("mega_skill", "")
        if mega:
            by_mega[mega].append(seg)

    sid_to_mega: Dict[str, str] = {}
    for mega, sids in name_map.items():
        for sid in sids:
            if sid not in sid_to_mega:
                sid_to_mega[sid] = mega

    enriched = []
    used_megas = set()

    for s in seed_skills:
        sk = s["skill"]
        sid = sk["skill_id"]
        matching_mega = sid_to_mega.get(sid)

        segments = by_mega.get(matching_mega, []) if matching_mega else []
        if matching_mega:
            used_megas.add(matching_mega)

        preconds, eff_add, eff_del, descriptions, rewards = \
            _collect_predicates(segments)
        n_instances = len(segments)

        top_precond = _top_predicates(preconds, n_instances)
        top_eff_add = _top_predicates(eff_add, n_instances)
        top_eff_del = _top_predicates(eff_del, n_instances)

        enriched_skill = dict(sk)
        enriched_skill["contract"] = {
            "eff_add": top_eff_add,
            "eff_del": top_eff_del,
            "n_instances": n_instances,
        }
        enriched_skill["preconditions"] = top_precond

        if descriptions:
            enriched_skill["tactical_description"] = descriptions[0]
        if rewards:
            enriched_skill["mean_reward"] = round(statistics.mean(rewards), 1)

        eff_add_sr = {e: round(eff_add.get(e, 0) / max(1, n_instances), 2)
                      for e in top_eff_add}
        eff_del_sr = {e: round(eff_del.get(e, 0) / max(1, n_instances), 2)
                      for e in top_eff_del}

        enriched.append({
            "skill": enriched_skill,
            "report": {
                "skill_id": sid,
                "n_instances": n_instances,
                "eff_add_success_rate": eff_add_sr,
                "eff_del_success_rate": eff_del_sr,
                "mean_reward": round(statistics.mean(rewards), 1) if rewards else 0,
                "source": "teacher_bootstrap",
            },
        })

    # Add NEW skills for mega categories that weren't mapped to any seed
    for mega, segments in by_mega.items():
        if mega in used_megas:
            continue
        if not segments:
            continue

        preconds, eff_add, eff_del, descriptions, rewards = \
            _collect_predicates(segments)
        n_instances = len(segments)

        sid = f"bootstrap.{mega.lower().replace('/', '_')}"
        top_eff_add = _top_predicates(eff_add, n_instances)
        top_eff_del = _top_predicates(eff_del, n_instances)
        top_precond = _top_predicates(preconds, n_instances)

        enriched.append({
            "skill": {
                "skill_id": sid,
                "version": 1,
                "name": mega,
                "strategic_description": descriptions[0] if descriptions else mega,
                "contract": {
                    "eff_add": top_eff_add,
                    "eff_del": top_eff_del,
                    "n_instances": n_instances,
                },
                "preconditions": top_precond,
                "mean_reward": round(statistics.mean(rewards), 1) if rewards else 0,
            },
            "report": {
                "skill_id": sid,
                "n_instances": n_instances,
                "eff_add_success_rate": {e: round(eff_add.get(e, 0) / max(1, n_instances), 2)
                                         for e in top_eff_add},
                "eff_del_success_rate": {e: round(eff_del.get(e, 0) / max(1, n_instances), 2)
                                         for e in top_eff_del},
                "source": "teacher_bootstrap_new",
            },
        })

    return enriched


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    game_slug: str,
    output_dir: Path,
    use_llm: bool = True,
    llm_model: str = "google/gemini-2.5-flash",
    max_segments: int = 100,
) -> Dict[str, Any]:
    """Run the full bootstrap pipeline for one game."""
    print(f"\n{'='*60}")
    print(f"BOOTSTRAPPING CONTRACTS: {game_slug}")
    print(f"{'='*60}")

    # 1. Load episodes
    episodes = load_episodes(game_slug)
    if not episodes:
        print(f"  No episodes found for {game_slug}")
        return {"game": game_slug, "status": "no_data"}

    # 2. Segment
    all_segments = []
    for ep in episodes:
        segs = segment_episode(ep)
        all_segments.extend(segs)
    print(f"  Segmented into {len(all_segments)} segments "
          f"(from {len(episodes)} episodes)")

    # Sort by reward descending, take top segments
    all_segments.sort(key=lambda s: -s["total_reward"])
    segments_to_label = all_segments[:max_segments]
    print(f"  Labeling top {len(segments_to_label)} segments (by reward)")

    # 3. LLM labeling
    labeled = []
    if use_llm:
        for i, seg in enumerate(segments_to_label):
            print(f"  [{i+1}/{len(segments_to_label)}] "
                  f"steps {seg['start_step']}-{seg['end_step']} "
                  f"reward={seg['total_reward']:.0f} ... ", end="", flush=True)
            label = label_segment_with_llm(seg, model=llm_model)
            if label:
                seg["label"] = label
                labeled.append(seg)
                print(f"→ {label.get('mega_skill', '?')} "
                      f"(conf={label.get('confidence', 0):.2f})")
            else:
                print("→ FAILED")
            time.sleep(0.2)  # rate limit
    else:
        print("  [DRY RUN] Skipping LLM labeling")

    print(f"  Successfully labeled: {len(labeled)}/{len(segments_to_label)}")

    # 4. Load seed skills
    seed_path = SEED_BANK_ROOT / game_slug / "skill_bank.jsonl"
    seed_skills = []
    if seed_path.exists():
        with open(seed_path) as f:
            seed_skills = [json.loads(l) for l in f if l.strip()]
    print(f"  Seed skills loaded: {len(seed_skills)}")

    # 5. Aggregate contracts
    enriched = aggregate_contracts(labeled, seed_skills)

    # 6. Write output
    out_game_dir = output_dir / game_slug
    out_game_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_game_dir / "skill_bank.jsonl"
    with open(out_path, "w") as f:
        for entry in enriched:
            f.write(json.dumps(entry) + "\n")

    # Also save raw labeled segments for debugging
    debug_path = out_game_dir / "labeled_segments.json"
    with open(debug_path, "w") as f:
        json.dump(labeled, f, indent=2, default=str)

    # Summary
    n_with_effects = sum(
        1 for e in enriched
        if e["skill"]["contract"]["eff_add"] or e["skill"]["contract"]["eff_del"]
    )
    total_instances = sum(e["report"]["n_instances"] for e in enriched)

    summary = {
        "game": game_slug,
        "episodes": len(episodes),
        "segments_total": len(all_segments),
        "segments_labeled": len(labeled),
        "skills_total": len(enriched),
        "skills_with_effects": n_with_effects,
        "total_instances": total_instances,
        "output": str(out_path),
    }

    print(f"\n  RESULT: {n_with_effects}/{len(enriched)} skills have effects "
          f"({total_instances} total instances)")
    for e in enriched:
        sk = e["skill"]
        contract = sk.get("contract", {})
        print(f"    {sk['skill_id']:35s} "
              f"inst={contract.get('n_instances', 0):3d}  "
              f"eff_add={contract.get('eff_add', [])}  "
              f"eff_del={contract.get('eff_del', [])}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Bootstrap mega-skill contracts from teacher demos")
    parser.add_argument("--games", nargs="+", default=["gymv_altered_beast", "gymv_airstriker"])
    parser.add_argument("--output-dir", type=str,
                        default="frontier_data/output/stage2_seeds_v3_bootstrapped")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, just segment")
    parser.add_argument("--max-segments", type=int, default=80,
                        help="Max segments to label per game")
    parser.add_argument("--llm-model", type=str, default="google/gemini-2.5-flash")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for game in args.games:
        result = run_pipeline(
            game,
            output_dir,
            use_llm=not args.dry_run,
            llm_model=args.llm_model,
            max_segments=args.max_segments,
        )
        results.append(result)

    # Write summary
    summary_path = output_dir / "BOOTSTRAP_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL DONE. Summary: {summary_path}")
    for r in results:
        print(f"  {r.get('game', '?')}: {r.get('skills_with_effects', 0)} skills with effects")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
