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

There are two layers to this question:

* **(A) ID-level overlap** — does the same `OPERATOR/SUBGOAL` literal
  appear in two banks?  Cheap to answer, but the answer is misleading
  (the same ID frequently means very different things).
* **(B) Functional overlap** — do the two skills share predicate
  vocabulary (preconditions / postconditions / eff_add / eff_del /
  example_predicates)?  This is what you actually need to know if you
  want to reuse one skill on another task.

We compute (B) using a **token Jaccard** between the lower-cased
predicate sets, after stop-word removal.  Two skills with `J_tok ≥ 0.30`
share enough vocabulary that a frontier policy fine-tuned on one is
very likely to recognise the situation in the other; `0.10–0.30` is
"thematic similarity, not interchangeable"; `<0.10` is a name collision.

> Note: this analysis became possible only after running
> `scripts/repair_gymv_contracts_gpt54.py` (May 10).  The legacy
> `run_20260430_030637/gym_v/` bank had empty `preconditions /
> postconditions / example_predicates` for all 416 gym_v skills, so
> only 10 / 18 banks had usable contracts before that repair.

### Layer A — ID-level overlap (still useful for sanity)

* **0** full skill IDs appear in all 18 tasks.
* **Closest-to-universal**: `INSPECT/SETUP` (15/18, missing on
  Airstriker, visual_toolbench, webshop), `COMMIT/POSITION` (11/18),
  `RECOVER/EVADE` (10/18).
* **Operator universals**: only `COMMIT` is in all 18 banks; `INSPECT`
  17/18; `RECOVER` 15/18; `COMPARE` 13/18.
* **214 / 448 skill IDs are unique to a single task** (~48 %).

### Layer B — Functional overlap (cohort matrix)

Token-Jaccard means across all skill pairs (lower-triangle is symmetric):

|              | env_wr_game | gymv_game | vr_image | vr_video | web   |
| ------------ | :---------: | :-------: | :------: | :------: | :---: |
| env_wr_game  | **0.056**   | 0.035     | 0.024    | 0.024    | 0.030 |
| gymv_game    |             | **0.082** | 0.016    | 0.016    | 0.035 |
| vr_image     |             |           | **0.082**| 0.080    | 0.034 |
| vr_video     |             |           |          | **0.108**| 0.033 |
| web          |             |           |          |          | **0.073** |

Every diagonal cell (within-cohort) is 1.5–7× larger than the
off-diagonal, so cohort-level pooling really is the natural unit of
SFT transfer.

### Layer B — top cross-task functional twins

| J_tok | J_pred | task A → task B  | skill A ↔ skill B | interpretation |
| :---: | :---: | ---------------- | ----------------- | -------------- |
| 0.40 | 0.14 | siv_bench → video_holmes  | `COMPARE/ANSWER` ↔ `COMPARE/ANSWER` | "Option Evidence Matching" — same function, true twin |
| 0.40 | 0.04 | SpaceHarrierII → ThunderForceIII | shooter dodge skills | both are forced-perspective shooters with similar projectile patterns |
| 0.39 | 0.17 | AlteredBeast → StreetsOfRage2  | beat-em-up combat | both are beat-em-ups; predicates around enemy stagger / combo / on-screen mob |
| 0.35 | 0.04 | Airstriker → ThunderForceIII | shoot-em-up skills | both are scrolling shooters |
| 0.33 | 0.09 | siv_bench → tir_bench | `COMPARE/ANSWER` ↔ `COMPARE/DEDUCE` | candidate-set pruning across modalities |
| 0.31 | 0.10 | tir_bench → video_holmes  | `COMPARE/DEDUCE` ↔ `COMPARE/ANSWER` | same as above, vr_image ↔ vr_video |
| 0.31 | 0.03 | siv_bench → video_holmes  | `RECOVER/SETUP` ↔ `RECOVER/SETUP` | "reasoning_trace_required" backstop |
| 0.29 | 0.06 | tir_bench → visual_toolbench | `COMMIT/ANSWER` ↔ `COMMIT/ANSWER` | answer commitment in image QA |

* The 4 visual-reasoning benches (`siv_bench`, `video_holmes`,
  `tir_bench`, `visual_toolbench`) form by far the **densest transfer
  graph**: 25+ pairs at `J_tok ≥ 0.20`, all the way up to 0.40.  This is
  a real, pool-able cohort.
* The 2 web tasks (`miniwob`, `webshop`) cluster tightly with each
  other (e.g. `RECOVER/NAVIGATE` ↔ `RECOVER/EXPLORE` at J_tok=0.26)
  but neither matches anything outside the web cohort.
* Genre-mate gym_v games (shooters; beat-em-ups; platformers) reach
  J_tok 0.30–0.40 — comparable to within-VR cohort.  Cross-genre gym_v
  pairs sit closer to 0.10.

### Layer B — same ID, sorted by *functional coherence*

For every `OPERATOR/SUBGOAL` literal that appears in ≥4 tasks, the
average pairwise token-Jaccard across the tasks holding it.  Top of
this list = "same name actually means same thing across tasks":

| skill_id          | #tasks | avg J_tok | max J_tok |
| ----------------- | :----: | :-------: | :-------: |
| `INSPECT/POSITION`|  4     |  0.22     | 0.28      |
| `COMMIT/ANSWER`   |  4     |  0.22     | 0.29      |
| `INSPECT/IDENTIFY`|  4     |  0.21     | 0.29      |
| `COMPARE/IDENTIFY`|  4     |  0.20     | 0.27      |
| `REASON/RULE_OUT` |  4     |  0.18     | 0.23      |
| `INSPECT/EVIDENCE`|  4     |  0.16     | 0.27      |
| `COMPARE/DEDUCE`  |  4     |  0.15     | 0.23      |
| `VERIFY/EVIDENCE` |  4     |  0.14     | 0.20      |
| ...               |        |           |           |
| `RECOVER/EVADE`   | 10     |  **0.08** | 0.17      |
| `COMMIT/EXPLORE`  |  9     |  **0.08** | 0.18      |
| `COMMIT/POSITION` | 11     |  **0.06** | 0.19      |
| `COMMIT/EVADE`    |  8     |  **0.06** | 0.16      |
| `COMMIT/SETUP`    |  4     |  **0.05** | 0.08      |

Crucial observation: the most-shared IDs (`COMMIT/POSITION`, 11 tasks;
`RECOVER/EVADE`, 10 tasks) are also the **least functionally coherent**.
Their high task coverage is the unified vocabulary working as designed
— a generic "commit to a positioning move" — but the actual
preconditions and effects diverge wildly between selecting a webshop
size variant, jumping over a Strider hazard, and clicking a miniwob
button.  Treat the ID as an *intent label*, not as a transferable
contract.

### Implications for SFT / curriculum design

1. **Transfer at the cohort level, not the skill level.**  Pooled SFT
   on the 4 VR benches, or on miniwob+webshop, or on
   shooter/beat-em-up gym_v subsets, is well-supported by predicate
   overlap.  Cross-cohort pooling is not.
2. **Most "shared IDs" lie about transferability.**  Of the 206
   cross-task pairs sharing an ID across the contract-rich banks, only
   4 score `J_tok ≥ 0.30`; 84 score `<0.10`.
3. **The 4-VR cohort is the only example of true cross-task skill
   reuse**: 25+ cross-bench pairs at `J_tok ≥ 0.20`, including 13
   `OPERATOR/SUBGOAL` literals that are functionally honest across the
   cohort (`INSPECT/POSITION`, `INSPECT/IDENTIFY`, `COMPARE/IDENTIFY`,
   `COMMIT/ANSWER`, etc.).
4. **No skill is universal across all 18 tasks** — neither by ID nor
   by predicate vocabulary.  Use cohort-conditioned skill banks.

---

## Refreshing

```bash
python sft_data_inventory/build_inventory.py            # rebuild in place
python sft_data_inventory/build_inventory.py --dry-run  # plan only
```

The script always re-points the symlinks at the *currently configured*
canonical runs (set as constants at the top of `build_inventory.py`).  If
you produce a new frontier-distill run, edit those constants and re-run.
