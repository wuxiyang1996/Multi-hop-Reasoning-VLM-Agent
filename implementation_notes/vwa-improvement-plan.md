# VWA improvement plan — closing the gap to 20–30 % pass-rate

> **Status (2026-05-03 04:50 AM):** 🟡 **PLAN — diagnostic complete,
> changes not yet landed.** Real-agent smoke on 1 task (gpt-5.5 low /
> max_steps=12) finished cleanly (12/12 steps, 16.1 s/step, watchdog
> ok) but reward=0.00 with classic *navigation-thrashing* behaviour.
> This memo records what the literature says we *should* be hitting,
> why our current config is well below the bar, and a tier-ranked set
> of improvements the 4-model 200-task baseline should ship with.

> **Cross-refs:**
> - [`implementation_notes/osworld-vwa-200-baseline-plan.md`](osworld-vwa-200-baseline-plan.md) — the parent baseline plan; its §13 already flagged `max_steps=12` vs. literature 30 as **Open Q #3**. This memo resolves that question with data.
> - [`implementation_notes/osworld-4model-baseline-plan.md`](osworld-4model-baseline-plan.md) — sister plan; we re-use the same 4 providers + skill-bank pattern.
> - [`Cold-start-out-browsergym/vwa_real_agent_smoke/visualwebarena.92/`](../Cold-start-out-browsergym/vwa_real_agent_smoke/) — the diagnostic episode this plan is grounded in.
> - [`cold_start/generate_cold_start_actor_browsergym.py`](../cold_start/generate_cold_start_actor_browsergym.py) — driver to patch.
> - [`browsergym_wrapper/tools.py`](../browsergym_wrapper/tools.py) lines 421-448 — `_h_list_valid_actions` is one of the bottleneck (see §3 below).

---

## 1. Where the literature is

Authoritative VWA leaderboard (AWorld README, Sept 2025; VWA paper
Koh et al. 2024). 910-task overall pass-rate, plain agent first then
scaffolded:

| Method (model) | Classifieds | Reddit | Shopping | **Overall** |
|---|---:|---:|---:|---:|
| VWA paper baseline (Gemini-Pro Image+Caps+Acc.Tree) | 3.4 | 4.3 | 8.2 | **6.0** |
| VWA paper (GPT-4 text-only) | 5.6 | 4.8 | 9.2 | **7.3** |
| VWA paper (GPT-4 + BLIP-2 captions) | 8.6 | 8.6 | 16.7 | **12.8** |
| VWA paper (GPT-4V multimodal Image+Caps+Acc.Tree) | 8.1 | 12.4 | 19.7 | **15.1** |
| **VWA paper best (GPT-4V SoM Image+Caps+SoM)** | 9.8 | 17.1 | 19.3 | **16.4** |
| WebDreamer (Qwen2-VL-7B) | 17.9 | 11.1 | 20.2 | **17.2** |
| WebDreamer (Qwen2-VL-72B) | 19.6 | 15.9 | 24.6 | **21.0** |
| WebDreamer (Dreamer-7B in-domain) | 25.0 | 15.9 | 26.3 | **23.2** |
| WebDreamer (GPT-4o) | 23.2 | 17.5 | 26.3 | **23.2** |
| ICAL (GPT-4o) | — | — | — | **23.4** |
| TreeSearch + SoM (GPT-4o) | 26.5 | 20.5 | 29.0 | **26.4** |
| ExAct R-MCTS MAD + SoM + Caption + Image (GPT-4o) | **41.0** | **28.7** | 32.3 | **33.7** |
| Recon-Act (GPT-5-Chat) | 39.3 | 27.1 | **39.3** | **36.5** |
| **Human** | 91.1 | 87.1 | 88.4 | **88.7** |

Less-authoritative but suggestive (MCP App Store leaderboard, raw
frontier-model numbers, no scaffold):

| Model | Overall |
|---|---:|
| Gemini 2.5 Flash | 54 % |
| GPT-5.4 | 52.9 % |
| GPT-4o | 19.8 % |
| Gemini 1.5 Pro | 12.0 % |

**Two anchors that matter for us:**

1. **Plain frontier-model bar (no scaffold) ≈ 16–25 %.** GPT-4V SoM
   was 16.4 %. GPT-4o WebDreamer 23.2 %. GPT-5.4 *raw* 52.9 % is
   plausible only with current-gen browser-aware tooling (the MCP
   leaderboard's methodology is opaque so treat as upper bound).
2. **Cheap-scaffold sweet spot ≈ 23–27 %.** GPT-4o + ICAL or
   TreeSearch sits there. R-MCTS / Recon-Act add real cost.

**Our publication bar.** The Phase-5/6 cross-domain transfer story
(see [osworld-4model-baseline-plan.md §1](osworld-4model-baseline-plan.md))
needs a *defensible teacher pass-rate per provider*. To not embarrass
ourselves vs. published frontier numbers we need to be in the
**~20-30 %** band on the 200-task subset for 2026-era models
(gpt-5.4/5, claude-sonnet-4.6, gemini-2.5-pro, qwen3-vl-235b). Below
~12 % is publication-poison.

## 2. Where we are (and why)

The 1-task diagnostic on `visualwebarena.92` ("Find the most
expensive TV from Maryland that displays an ongoing NFL game",
classifieds, url_match) finished 12/12 steps with reward=0. The
trace:

| step | action | comment |
|---|---|---|
| 0–1 | `scroll(+300)` / `scroll(-300)` | NOOP (page didn't visibly change) |
| 2–5 | `go_back` / `go_forward` × 2 | **thrashing** — agent landed on `about:blank` and tried to recover via history |
| 6–7 | `scroll(+300)` / `scroll(-300)` | NOOP again |
| 8–9 | `go_back` / `go_forward` | more thrashing |
| 10 | `click("157")` | first real click — went to `about:blank#blocked` |
| 11 | `go_back` | recovery |

**Action freq:** `scroll: 4, go_back: 4, go_forward: 3, click: 1,
fill: 0, keyboard_type: 0`.

The goal text (*"Find the most expensive TV from Maryland …"*) is a
multi-constraint search — the canonical solve path is
`fill(<search_box>, "TV") → click(<filter_state_Maryland>) →
sort(price desc) → click(top result)`. The agent **never emitted a
single `fill()` action** because:

1. `_h_list_valid_actions` (the candidate-action helper in
   [`browsergym_wrapper/tools.py:421`](../browsergym_wrapper/tools.py))
   surfaced only **5 actions** to the agent for step 3:
   `scroll(0,300) / scroll(0,-300) / go_back() / go_forward() /
   noop()`. Zero `fill(...)`. Zero `click("<bid>")`. The reason: its
   filter requires `props.get("set_of_marks") or
   props.get("clickable")`, and at the step-3 obs (URL =
   `about:blank`) no element has those flags set.
2. The system prompt does not aggressively promote `fill(...)` for
   search tasks, so even though the model could in principle emit
   one (step-10's `click("157")` came outside the candidate
   shortlist), it doesn't.
3. `reasoning_effort=low` makes gpt-5.5 myopic — it picks "scroll
   first to see what's around" and then loses the thread.
4. `max_steps=12` is half the literature default of 30 (and a third
   of WebArena's typical budget) — even a non-thrashing agent has
   barely enough room for a 5-step canonical solution + 5-step
   recovery + 2-step verify.

In other words: **our config is closer to the GPT-4 text-only baseline
(7.3 %) than to the 16-25 % band the same gpt-5.5-class model is
capable of.** That's the gap.

## 3. Improvements ranked by ROI

ROI = (expected pass-rate lift in pp) / (LOC + extra cost). Cost
column is **per-provider** marginal cost on top of the 200-task
baseline (~$60-130/provider).

| # | Change | LOC | Extra cost | Δ wall-clock | Expected lift | Tier |
|---|---|---:|---:|---:|---:|---:|
| **A** | Bump `max_steps=12 → 30` | 1 | ~+$60/prov | ×2.0 (~28 hr) | **+5-12 pp** | 1 |
| **B** | Fix `_h_list_valid_actions` to surface `fill(...)` even without SoM flag | ~10 | $0 | ~0 % | **+3-7 pp** | 1 |
| **C** | Add a "use search box first" hint to the system prompt for textbox-bearing pages | ~5 | $0 | ~0 % | **+2-4 pp** | 1 |
| **D** | Verify SoM overlay actually renders (the candidate list looks SoM-blind even when `use_som=True`) | ~20 | $0 | ~0 % | **+2-5 pp** | 1 |
| **E** | Caption-augmented prompt for image-bearing tasks (72/200): pre-caption `goal.image` with gpt-4o-mini and inject as text | ~60 | ~$0.40 | +1 % | **+3-6 pp on 72 tasks (~1-2 pp overall)** | 2 |
| **F** | `reasoning_effort=medium` for shopping/classifieds tasks (multi-constraint), keep low for reddit | ~10 | ~+$50/prov | ×1.3 | **+4-8 pp** | 2 |
| **G** | Skill retrieval (BM25 from VWA cold-start trajectories of *this* run, refresh after every 50 tasks) | ~120 | ~$0 | +5 % | **+1-3 pp** | 2 |
| **H** | Memory + reflection steering (port of OSWorld modules) | ~150 | ~+$30/prov | ×1.4 | **+2-5 pp**, OSWorld lesson says watch for slowdown | 3 |
| **I** | TreeSearch / N-tree sampling (pick-best over 3 rollouts) | ~250 | ×3 cost | ×3 | **+5-10 pp** | 3 |
| **J** | R-MCTS / Recon-Act-style tool-augmented agent | ~1500 | ×5-10 cost | ×5-10 | **+12-18 pp** (33 → 36 % band) | — |

**Recommended scope for the 200-task baseline:** **Tier 1 (A+B+C+D)**
must-have. **Tier 2 (E+F)** if budget allows. **Tier 3** explicitly
out-of-scope — that's a "publication v2" investment, not the teacher
baseline.

### Tier-1 expected outcome

Stack the lifts (with diminishing-returns discount, so 0.7×
multiplicative on overlap):

```
   baseline (current config)           ≈   8 %     (pessimistic for gpt-5.5)
   + A  (max_steps 30)                 ≈ +8 pp
   + B  (fill in candidate list)       ≈ +5 pp
   + C  (search-first prompt hint)     ≈ +3 pp     (overlaps B → discount to +2)
   + D  (SoM verify / fix)             ≈ +3 pp     (overlaps B partially → +2)
   ───────────────────────────────────────────────
   Tier-1 stacked target                ≈ 22-26 %  on 200-set, gpt-5.4 low
```

Lands us in the GPT-4o WebDreamer / TreeSearch band — the right
publication zone.

### Tier-2 stretch

```
   + E  (image caption augmentation)   ≈ +2 pp overall
   + F  (reasoning_effort=med)         ≈ +5 pp     but +50 % cost
   ───────────────────────────────────────────────
   Tier-1+2 stacked target              ≈ 28-32 %  on 200-set, gpt-5.4 mixed effort
```

That puts us solidly between TreeSearch (26.4 %) and ExAct (33.7 %)
— a defensible "frontier-model + light scaffolding" report.

## 4. Concrete code prereqs

Sized to land in **one short session** (< 250 LOC total) before the
4-model launch. Match the sister-plan budget shape.

| Change | File | LOC | Tier |
|---|---|---:|---:|
| `--max_steps` default 12 → 30 | `cold_start/generate_cold_start_actor_browsergym.py` | 1 | 1 |
| `--max_steps` default 8 → 30 in launcher | `cold_start/run_coldstart_actor_browsergym.sh` | 1 | 1 |
| Relax `_h_list_valid_actions` filter; emit `fill(<bid>, "...")` for any visible textbox/searchbox/combobox regardless of `set_of_marks` flag | `browsergym_wrapper/tools.py:421-448` | ~15 | 1 |
| System-prompt hint: "If the goal involves searching for an item, prefer `fill(<search_box_bid>, "<query>")` over scrolling." | `cold_start/generate_cold_start_actor_browsergym.py` (build_system_prompt block) | ~10 | 1 |
| SoM-overlay verification: log `len(extras)` + `n_set_of_marks` once per episode; raise `[WATCHDOG WARN]` if SoM-on but no flags set | `cold_start/generate_cold_start_actor_browsergym.py` | ~30 | 1 |
| `--enable_image_caption` flag + `_caption_input_image` helper (gpt-4o-mini, lazy-cached per `(site, task_id, input_index)`) | new helper in same driver | ~80 | 2 |
| `--reasoning_effort_per_site` (allow `low,low,medium,medium,low` per site) | `cold_start/generate_cold_start_actor_browsergym.py` | ~20 | 2 |
| Regression test: `tests/test_browsergym_action_candidates.py` — fixture-based check that `_h_list_valid_actions` surfaces `fill` for a synthetic page with one `<input type="search">` | new | ~50 | 1 |
| **Total Tier-1** | | **~107** | |
| **Total Tier-2** | | **~100** | |

## 5. Validation pipeline

Two bring-up smokes before the full 200-task launch:

1. **1-task re-smoke on `visualwebarena.92`** with all Tier-1 changes
   on. Pass criterion: `available_actions` includes ≥1 `fill(...)`
   on the listings page; agent emits ≥1 `fill()` step within first 8
   actions; reward may still be 0 (this task is hard) but the
   thrashing pattern must disappear (no more `go_back/go_forward × 4`).
2. **5-task mixed-difficulty smoke** (`92, 351, 96, 268, 433`) — 1 each
   from {classifieds easy, reddit easy, classifieds medium, reddit
   medium, reddit-image medium}. Pass criterion: ≥1 task with
   `reward=1.0` AND mean `[WATCHDOG]` status across 5 = `ok` or
   `FAST`. Estimated cost: ~$0.30, ~25 min. **This is the calibration
   point** that decides whether to launch the 200-task run or
   continue tuning Tier-1.

## 6. 200-task launch acceptance (revised from baseline-plan §11)

| Check | How to verify | Pre-Tier1 expected | Post-Tier1 expected |
|---|---|---:|---:|
| Overall pass-rate ≥ 12 % per provider | `compare/pass_at_1_overall.csv` | ~8 % | **≥18 %** |
| No "thrashing" run (≥4 consecutive nav-only actions) on >20 % of tasks | scan logs for the pattern | ~50 % | **≤15 %** |
| Per-step `fill()` rate ≥ 5 % across all steps | `compare/action_distribution.csv` (new) | ~0 % | **≥10 %** |
| `[WATCHDOG SLOW]` count = 0 across run | `grep -c '\[WATCHDOG SLOW\]' runs/vwa_baseline_200/*.log` | unknown | 0 |

If post-Tier-1 pass-rate is < 18 % overall, halt the 200-task run
after the 1st provider (~12 hr investment, not 4×) and add Tier-2
changes before launching the rest.

## 7. Open questions

1. **Should we ship Tier-1 + Tier-2 together, or stage them?** Stage:
   1-provider Tier-1 run (~12 hr, ~$60) → calibration → decide on
   Tier-2 → 4-provider full run. **Recommendation: stage.**
2. **Image captioning model:** gpt-4o-mini (~$0.005/caption,
   sufficient for vision-light captioning of input images) vs. our
   own qwen3-vl-235b (free locally, ~$0.07 cost-equivalent in GPU
   time). **Recommendation: gpt-4o-mini for the baseline run** —
   quality is well-known, parity across providers is cleaner; revisit
   for skill-bank cold-start where local model preferred.
3. **Should we record per-step trajectories with screenshots for
   later skill-bank ingestion?** Default `--no_save_frames` is on for
   disk pressure, but for the *first* 200-task run it might be worth
   `--save_frames` (4 GB extra → 800 episodes × 5 MB) to feed the
   VWA skill bank initialization. **Recommendation: yes, save frames
   for the Tier-1 calibration run; revisit for the 4-provider full
   run based on disk room.**
4. **Cross-machine vs single machine for the 4-model run?** The OSWorld
   plan §6 supports both. Given the new `max_steps=30` budget bumps
   per-provider wall-clock to ~28 hr, **single-machine 4-process
   parallel is now ~28 hr (max-clock), not 15 hr**. Consider 2 or 4
   machines if other workloads contend for the box.

## 8. What this resolves vs. defers

**Resolves** (becomes part of the 200-task baseline):
- VWA action-space bug surfaced by the diagnostic (no `fill()` in
  candidates).
- `max_steps=12` vs literature 30 (open question §13.3 of the parent
  plan).
- The "are we close to literature?" question — yes for Tier-1, no for
  Tier-3-class scores, and that's fine for the publication.

**Defers** (post-publication / paper v2):
- Tree-search / MCTS rollouts.
- Recon-Act style tool-augmented agents.
- Trainer co-evolution loop on VWA (orthogonal — goes through the
  same Phase-5/6 cross-domain matrix downstream).
