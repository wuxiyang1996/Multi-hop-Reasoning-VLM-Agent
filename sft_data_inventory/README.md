# SFT Data Inventory

Consolidated, **read-only** index of every SFT-eligible corpus we currently
have in this repo, organized as **12 games + 6 non-game tasks**.

Nothing here is a copy — every file is a symlink that points back to the
canonical location elsewhere in the repo.  Re-run `build_inventory.py` to
refresh links / counts after a new run.

```
sft_data_inventory/
├── README.md            ← you are here
├── INVENTORY.json       ← machine-readable rollup (totals + per-task manifests)
├── build_inventory.py   ← idempotent generator (symlinks + MANIFEST.json)
├── games/               ← 12 tasks
└── non_game/            ← 6 tasks
```

Per task you'll find:

```
<task>/
├── MANIFEST.json                ← task-level metadata (counts, status, notes)
├── skill_bank.jsonl             ← bank-extracted skills (if any)
├── sft/
│   ├── action_taking.jsonl      ← step-level decisions
│   └── skill_selection.jsonl    ← skill-query rows (k candidates → 1 chosen)
└── rollouts/
    ├── gpt54/                   ← per-teacher raw rollouts
    ├── claude/
    ├── gemini/
    └── qwen/
```

---

## Games (12)

| Task                          | Corpus       | Teachers | Bank skills | action_taking | skill_selection |
| ----------------------------- | ------------ | :------: | :---------: | :-----------: | :-------------: |
| Temporal_Airstriker-v0        | gym_v        |    4     |     31      |     3 920     |      3 920      |
| Temporal_AlteredBeast-v0      | gym_v        |    4     |     21      |     3 600     |      3 600      |
| Temporal_Columns-v0           | gym_v        |    4     |     27      |     3 855     |      3 855      |
| Temporal_DynamiteHeaddy-v0    | gym_v        |    4     |     32      |     3 920     |      3 920      |
| Temporal_SpaceHarrierII-v0    | gym_v        |    4     |     33      |     3 920     |      3 920      |
| Temporal_StreetsOfRage2-v0    | gym_v        |    4     |     34      |     2 880     |      2 880      |
| Temporal_Strider-v0           | gym_v        |    4     |     45      |     3 920     |      3 920      |
| Temporal_ThunderForceIII-v0   | gym_v        |    4     |     32      |     2 960     |      2 960      |
| tetris                        | env_wrappers |  1 *     |     12      |     1 573     |      1 573      |
| super_mario                   | env_wrappers |  1 *     |      7      |       326     |        326      |
| candy_crush                   | env_wrappers |  1 *     |      3      |     1 000     |      1 000      |
| twenty_forty_eight            | env_wrappers |  1 *     |      8      |     3 187     |      3 187      |

\* env_wrappers were only run with the **gpt-5.4** teacher — claude / gemini /
qwen do not have rollouts here.

For all 12 games **`active_skill` and `skill_query.candidates` are
bank-derived** (`OPERATOR/SUBGOAL` IDs from the skill bank, populated by
`labeling/label_skill_actions_gpt54.py`).  The `action_taking` and
`skill_selection` files agree on the chosen skill for every step.

---

## Non-game (6)

| Task              | Modality     | Teachers | Bank skills | action_taking | skill_selection | Status |
| ----------------- | ------------ | :------: | :---------: | :-----------: | :-------------: | ------ |
| miniwob           | web (browsergym) |  4   |     29      |       905     |      2 529      | full pipeline |
| webshop           | web (browsergym) |  4   |     16      |        —      |        —        | bank ✓ + skill_query ✓ (decision-SFT pending) |
| video_holmes      | video QA     |    4     |     31      |     1 271     |     18 451      | bank ✓, SFT ✓ (mixed IDs) |
| siv_bench         | video QA     |    4     |     23      |       803     |      6 818      | bank ✓, SFT ✓ (mixed IDs) |
| tir_bench         | image QA     |    4     |     32      |       302     |      5 247      | bank ✓, SFT ✓ (mixed IDs) |
| visual_toolbench  | image QA     |    4     |     32      |        74     |      6 356      | bank ✓, SFT ✓ (mixed IDs) |

### ⚠ Non-game `active_skill` mismatch

For miniwob and the 4 visual reasoning benches the **two SFT files use
different skill schemes for the same sample**:

* `skill_selection.jsonl` — `active_skill` is **bank-derived**
  (`OPERATOR/SUBGOAL`, with `candidates` from `SkillQueryEngine`)
* `action_taking.jsonl` — `active_skill` is a **synthetic, hardcoded ID**
  (e.g. `web/CLICK`, `video_qa/CAUSE_INFERENCE`,
  `visual_toolbench/OPEN_QA`) and `skill_pass_rate=1.0`,
  `skill_n_instances=1`

This is a known consequence of `scripts/build_multimodal_decision_sft.py`
emitting synthetic IDs while
`labeling/qa_skill_selection_sft/run_20260506_194156/` wrote bank-aware
selection rows.  See each task's `MANIFEST.json → sft.mismatch_warning` for
details.

### Webshop status (updated 2026-05-10)

Webshop now has the **full skill-extraction pipeline** wired up — the only
missing piece is the decision-SFT JSONL.

| stage | output | status |
| ----- | ------ | :----: |
| 1. label intentions (`OPERATOR/SUBGOAL`) | `labeling/qa_miniwob_labeled/run_webshop_20260510_043615/` | ✓ 200 episodes / 2 538 steps |
| 2. extract skill bank | `labeling/skill_bank_qa/run_webshop_20260510_044000/webshop/skill_bank.jsonl` | ✓ 16 skills |
| 3. attach `skill_query` (top-5 candidates + selected) | `labeling/skill_actions_qa_out/run_webshop_20260510_044300/webshop/` | ✓ 100 % coverage |
| 4. emit decision SFT (`action_taking.jsonl` + `skill_selection.jsonl`) | – | ⏳ pending |

The 16 webshop skills (most → least common): `COMMIT/POSITION` (601),
`COMMIT/NAVIGATE` (426), `COMMIT/EXPLORE` (328), `COMMIT/EXECUTE` (304),
`COMPARE/EXPLORE` (284), `COMMIT/BUILD` (273), `TRACK/EXPLORE` (139),
`INSPECT/EXPLORE` (60), `VERIFY/EXPLORE` (30), `COMMIT/OPTIMIZE` (28),
`RECOVER/NAVIGATE` (15), `VERIFY/POSITION` (12), `RECOVER/EXPLORE` (8),
`VERIFY/EXECUTE` (8), `RECOVER/POSITION` (6), `TRACK/POSITION` (3).

Per-step `skill_query` blocks now sit alongside per-task rollouts in
`non_game/webshop/skill_query/<teacher>/webshop.<idx>/rollouts_with_skill_query.jsonl`.
Run `scripts/build_multimodal_decision_sft.py` over those (or feed the
labeled rollouts at `qa_miniwob_labeled/run_webshop_*/`) to materialise
the SFT JSONLs.

---

## Totals

* **18 skill banks** (12 games + miniwob + webshop + 4 visual reasoning)
* **38 416 action_taking rows**
* **74 462 skill_selection rows**
* **18 tasks** (12 games + 6 non-game)

See `INVENTORY.json` for the machine-readable view, and each task's
`MANIFEST.json` for source paths and per-teacher rollout counts.

---

## Cross-task skill overlap

How many of the ~290 unique `OPERATOR/SUBGOAL` skill IDs in this
inventory are shared across tasks?  Skill banks are **highly
domain-specialised** — there is **no full skill ID present in all 18
tasks**.  The only universally-shared element is the `COMMIT` operator.

### Closest-to-universal full skills

| skill_id           | task coverage | tasks missing it |
| ------------------ | :-----------: | ---------------- |
| `INSPECT/SETUP`    | **15 / 18**   | `Temporal_Airstriker-v0`, `visual_toolbench`, `webshop` |
| `COMMIT/POSITION`  | 11 / 18       | most pure-reasoning benches (no on-screen targets) |
| `RECOVER/EVADE`    | 10 / 18       | non-action benches |

214 / 290 skill IDs (~74 %) appear in exactly **one** task.

### OPERATOR axis (ignoring subgoal)

| Operator | Coverage |
| -------- | :------: |
| `COMMIT` | **18 / 18** ← only universal element |
| `INSPECT`| 17 / 18 |
| `RECOVER`| 15 / 18 |
| `COMPARE`| 13 / 18 |
| `VERIFY` | 6 / 18 |
| `REASON` | 4 / 18 |
| `TRACK`  | 3 / 18 |

### SUBGOAL axis

No subgoal is present in all 18 tasks.  Top 5: `SETUP` (16/18),
`POSITION` (16/18), `EVADE` (12/18), `EXPLORE` (10/18), `NAVIGATE` (9/18).

### Within-cohort consistency is much higher

| Cohort | Full skill_id intersection |
| ------ | -------------------------- |
| 12 games (gym_v + env_wrappers) | 0 (per-game skills diverge: shooting vs. board vs. platformer) |
| 6 non-game benches              | 0 |
| **4 visual-reasoning benches** (`video_holmes`, `siv_bench`, `tir_bench`, `visual_toolbench`) | **13 skills** — `COMMIT/ANSWER`, `INSPECT/{EVIDENCE, IDENTIFY, POSITION}`, `COMPARE/{DEDUCE, IDENTIFY, OPTIMIZE, RULE_OUT}`, `REASON/{DEDUCE, IDENTIFY, LOOKUP, RULE_OUT}`, `VERIFY/EVIDENCE` |
| **2 web tasks** (`miniwob ∩ webshop`) | **15 skills** — webshop's bank is essentially miniwob's web-action vocabulary minus 14 miniwob-specific entries |

### Implications

1. **Cross-task transfer of a single skill is rare.**  Don't expect a
   policy that uses, say, `REASON/DEDUCE` on `tir_bench` to find that
   skill in any game bank.
2. **Cohort-level pooling is well-supported.**  The 4 visual-reasoning
   benches share 13 reasoning skills → pooling SFT across them should
   transfer cleanly.  The 2 web tasks (miniwob + webshop) share 15 →
   same.
3. **`COMMIT/SETUP` is the closest thing to a universal skill** (15/18
   has it; the 3 missing tasks are Airstriker which jumps straight into
   action, and the two `EXPLORE`-dominated web/QA tasks).  Use it as a
   sanity baseline when measuring cross-task skill reuse.

---

## Refreshing

```bash
python sft_data_inventory/build_inventory.py            # rebuild in place
python sft_data_inventory/build_inventory.py --dry-run  # plan only
```

The script always re-points the symlinks at the *currently configured*
canonical runs (set as constants at the top of `build_inventory.py`).  If
you produce a new frontier-distill run, edit those constants and re-run.
