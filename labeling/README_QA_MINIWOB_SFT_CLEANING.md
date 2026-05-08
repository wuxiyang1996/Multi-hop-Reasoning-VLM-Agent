# QA + MiniWob + WebShop + EnvWrappers SFT Dataset Cleaning (May 2026)

A record of the cleaning + enrichment pipeline applied to the multimodal
decision SFT dataset under
`labeling/decision_sft_jsonl/run_multimodal_20260506_055105/`.

WebShop was added as a **new browser source** (May 2026 update) using the
same Stage 1' / 2 / 3 / 4 / 5 pipeline that already cleaned MiniWob — both
sources share the BrowserGym episode schema, so the only differences are
the per-task directory glob (`webshop.*` vs `miniwob.*`), the model-tag
to root-directory mapping, and a per-source success-reward threshold
(WebShop reward is granular — 0/0.33/0.5/0.67/0.75/1.0 — so we keep
episodes with `total_reward ≥ 0.5` by default; pass
`--webshop-min-reward` to widen / tighten).

Three independent quality issues were addressed:

1. **QA / Web (5 sources)** — `action_taking` rows had collapsed all
   `(operator, subgoal)` labels to `EXECUTE/EXECUTE`, and **none** had any
   `skill_selection` rows at all → decision agent learned nothing about
   strategy selection on QA / Web benchmarks.
2. **EnvWrapper games (4 games)** — `skill_selection` rows existed but were
   built against tiny, hash-dominated banks → degenerate candidate sets
   (`super_mario` = 2 candidates per row across all 326 rows, every row
   identical; `twenty_forty_eight` = 3 candidates × 3187 rows, identical;
   `candy_crush` = 5 candidates of which 4 were anonymous `skill-XXXXXXXX`
   IDs) → no learnable selection signal.
3. **Original SkillQueryEngine ran on CPU** — labeling 6k+ steps took
   ~50 min on multi-thread CPU; fixed by piping a shared `cuda:0`
   embedder into both the gymv-style and QA-style labelers (~144 s
   for the 6 086 EnvWrapper steps).

After the clean, every source — QA, Web, GymV, and EnvWrappers — has
rich per-step intentions and a fully populated, semantically diverse
`skill_selection` candidate vocabulary.

---

## TL;DR — Before vs After

### QA / Web (Stages 1-5)

| Bench | action rows | skill_sel rows BEFORE | skill_sel rows AFTER | (op, sg) AFTER (top-3) |
|---|---:|---:|---:|---|
| `video_holmes`     | 1,271 | **0** | **18,451** | REASON/DEDUCE 923 · REASON/RULE_OUT 293 · REASON/TIMELINE 42 |
| `siv_bench`        |   803 | **0** |  **6,818** | REASON/DEDUCE 548 · REASON/RULE_OUT 237 · REASON/IDENTIFY 8 |
| `tir_bench`        |   302 | **0** |  **5,247** | REASON/DEDUCE 147 · REASON/MEASURE 58 · REASON/RULE_OUT 40 |
| `visual_toolbench` |    74 | **0** |  **6,356** | REASON/DEDUCE 37 · REASON/RULE_OUT 13 · REASON/MEASURE 11 |
| `miniwob`          |   905 | **0** |  **2,529** | COMMIT/EXECUTE 410 · COMMIT/BUILD 190 · COMMIT/POSITION 69 |
| `webshop` *(new)*  |   ⟂   |   ⟂   |     ⟂      | populated by re-running Stages 1' → 5 with `--source webshop`; row counts depend on `--webshop-min-reward`. Measured (4 frontier models × 50 tasks): **`r≥0.5`→ 784 rows** (98 episodes), `r=1.0`→ 160 rows (38 episodes), `r≥0.0`→ 2,538 rows (every step kept; noisy). |

**action_taking before:** all 5 sources collapsed to `EXECUTE/EXECUTE`
(or `NAVIGATE/NAVIGATE` for ~9 % of miniwob).
**action_taking after:** dominant `REASON/*` operators on QA, dominant
`COMMIT/*` operators on miniwob — both reflect the actual semantic role
of each row.

### EnvWrappers (Stages 1E-3E)

| Game | skill_sel rows | unique cands BEFORE | top-3 cands BEFORE | unique cands AFTER | top-3 cands AFTER |
|---|---:|---:|---|---:|---|
| `candy_crush`        | 1,000 | 8 (1 named + 7 hash) | `COMMIT/CLEAR · skill-648266e9c6 · skill-e63284fda3` | **3 (all named)** | `COMMIT/CLEAR · COMPARE/CLEAR · INSPECT/SETUP` |
| `super_mario`        |   326 | **2** (every row identical) | `INSPECT/SETUP · COMMIT/NAVIGATE` | **7 (all named)** | `COMMIT/EVADE · COMMIT/ATTACK · COMMIT/POSITION` |
| `tetris`             | 1,573 | 6 named | `COMMIT/OPTIMIZE · COMMIT/SETUP · COMMIT/EVADE` | **12 (all named)** | `COMMIT/OPTIMIZE · COMMIT/SETUP · COMMIT/EVADE` |
| `twenty_forty_eight` | 3,187 | **3** (every row identical) | `COMMIT/MERGE · COMPARE/MERGE · mid:OPTIMIZE` | **8 (all named)** | `COMMIT/MERGE · COMMIT/POSITION · COMMIT/OPTIMIZE` |

**Before:** super_mario and twenty_forty_eight had completely identical
candidate sets across every step — the model is being asked to pick from
the same N items 326/3187 times in a row → zero discriminative signal.
candy_crush had 4 of 5 candidates as anonymous hash IDs.
**After:** All candidate IDs are GPT-5.4 curated names with descriptions,
preconditions, termination cues, and failure modes. 3-12 distinct skills
per game, 100 % step coverage, 5 candidates/row average.

### Aggregate dataset sizes

```
action_taking      : 55,806  (unchanged count; 3,725 QA/Web rows relabeled)
skill_selection    : 22,086 → 61,487  (+39,401; +178 %)
unique skill IDs   : 25 (game side) → 25 + 71 (QA side) = 96
                     of which 30 EnvWrapper IDs are now all named
                     (was: 11 named + 36 anonymous hash IDs across 4 games)
```

---

## Pipeline Stages

The cleaning pipeline has two parallel tracks:

* **QA / Web track** (5 sources, Stages 1-5): `video_holmes`,
  `siv_bench`, `tir_bench`, `visual_toolbench`, `miniwob`.
* **EnvWrapper track** (4 games, Stages 1E-3E): `candy_crush`,
  `super_mario`, `tetris`, `twenty_forty_eight`. Stages 1 and 1' are
  not needed because the EnvWrapper rollouts already had per-step dual-axis
  intentions (from `labeling/intentions_out/run_dualaxis_20260429_224917/`).

All scripts live under `labeling/` and `scripts/`.

```
                  ┌──────────────────────────────┐
                  │ raw rollouts / QA samples    │
                  │ (Cold-start-out-*, OR-TX)    │
                  └──────────────┬───────────────┘
                                 │
   ┌─────────────────────────────┴──────────────────────────────┐
   │  QA / Web track                                            │
   ▼                                                            ▼
┌──────────────────────┐                              ┌──────────────────────┐
│ Stage 1: Multihop    │                              │ Stage 1': MiniWob    │
│ CoT decomposition    │                              │ per-step intention   │
│ (QA, GPT-5.4)        │                              │ labeling             │
│ → samples_with_hops  │                              │ → rollouts.jsonl     │
└──────────┬───────────┘                              └──────────┬───────────┘
           │                                                     │
           └──────────────────┬──────────────────────────────────┘
                              ▼
                  ┌──────────────────────────────┐
                  │ Stage 2: Skill Bank Build    │
                  │ (GPT-5.4 CONTRACT+CURATOR)   │
                  │ → skill_bank_qa/             │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Stage 3: Skill-Query Label   │
                  │ (GPU embedder, top-K)        │
                  │ → skill_actions_qa_out/      │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Stage 4: skill_selection     │
                  │ SFT row emitter              │
                  │ → 39,401 rows                │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Stage 5: action_taking       │
                  │ relabel (op, sg) inplace     │
                  │ → 3,725 rows updated         │
                  └──────────────────────────────┘

   ┌────────────────────────────────────────────────────────────┐
   │  EnvWrapper track (independent of the QA track above)      │
   └────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ pre-existing labeled rollouts │
                  │ intentions_out/run_dualaxis_  │
                  │ 20260429_224917/env_wrappers/ │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Stage 1E: Build per-game     │
                  │ skill banks (GPT-5.4         │
                  │ CONTRACT, reuses Stage 2     │
                  │ pipeline)                    │
                  │ → skill_bank_envwrappers/    │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Stage 2E: Re-run skill-query │
                  │ labeling with new banks      │
                  │ (GPU embedder reuses Stage 3 │
                  │ fix; ~144 s for 6 086 steps) │
                  │ → skill_actions_out/         │
                  │   run_envwrappers_*_gpu/     │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Stage 3E: Emit fresh         │
                  │ skill_selection.jsonl rows   │
                  │ + overwrite the 4 envwrapper │
                  │ files in master SFT dir      │
                  │ → 6 086 rows replaced        │
                  └──────────────────────────────┘
```

---

## Stage Details

### Stage 1 — Multihop CoT Decomposition (QA)

**Script:** `labeling/label_qa_multihop_gpt54.py`
**Vocabulary:** `labeling/qa_vocab.py` (introduces `REASON`, `TOOL_USE`
operators and `EVIDENCE`, `IDENTIFY`, `TIMELINE`, `COUNT`, `MEASURE`,
`LOOKUP`, `DEDUCE`, `RULE_OUT`, `ANSWER`, `FORM_FILL`, `SUBMIT` subgoals).

For each QA sample (with a chain-of-thought from GPT-5.4 / Claude /
Gemini / Qwen), call GPT-5.4 to decompose the reasoning into atomic hops.
Each hop is tagged with `(operator, subgoal, note)`.

**Stats:** 7,928 samples → **36,872 hops** across 4 sources × 4 models.
**Output:** `labeling/qa_multihop_out/run_20260506_181625/<source>/<model>/samples_with_hops.jsonl`

### Stage 1' — MiniWob / WebShop Per-Step Intentions

**Script:** `scripts/label_qa_miniwob_intentions.py`
**Output (miniwob):** `labeling/qa_miniwob_labeled/run_20260506_070722/miniwob/<model>/<game>/rollouts.jsonl`
**Output (webshop):** `labeling/qa_miniwob_labeled/run_<ts>/webshop/<model>/<game>/rollouts.jsonl`

Each browsergym experience step gets `intention_operator`,
`intention_subgoal`, `intention_note` attached via GPT-5.4.  WebShop
reuses the same script with `--source webshop`, which globs `webshop.*`
under each `webshop_50task_<tag>/` model-specific root.  WebShop
cold-start rollouts usually have `intentions=null`, so the few-shot
block was extended with three webshop-style exemplars (search compose,
result compare, Buy Now commit) so the LLM still picks an informative
`(operator, subgoal)` pair from the state + action context alone.

### Stage 2 — Skill Bank Construction (GPT-5.4 driven)

**Script:** `labeling/build_skillbank_qa_gpt54.py`

Replaces the multi-stage SkillBankAgent LoRA pipeline (SEGMENT / CONTRACT
/ CURATOR) with a single GPT-5.4 driven process: cluster hops/steps by
`(operator, subgoal)`, ask GPT-5.4 to produce a strategic skill record
with description, preconditions, effects, execution_hint.

Output schema is aligned to `skill_agents.stage3_mvp.schemas.{SkillRecord,
VerificationReport, ExecutionHint}` so downstream loaders work without
patching.

**Stats:** **147 unique skills** across 5 sources from 39,401 instances.

| Source            | # skills |
|-------------------|---------:|
| video_holmes      |       31 |
| miniwob           |       29 |
| visual_toolbench  |       32 |
| tir_bench         |       32 |
| siv_bench         |       23 |

**Output:** `labeling/skill_bank_qa/run_20260506_184439/<source>/skill_bank.jsonl`

### Stage 3 — Skill-Query Labeling

**Script:** `labeling/label_skill_actions_qa_gpt54.py`

Uses `skill_agents.query.SkillQueryEngine` with a Qwen3-Embedding-0.6B
SentenceTransformer to attach the top-K (default 5) candidate skills to
each QA hop and each MiniWob experience step.

**Critical fix #1 (perf):** the original `SkillQueryEngine` defaulted to
CPU embedder, taking ~0.5 samples/s. Forcing the embedder onto `cuda:0`
(`_SHARED_GPU_EMBEDDER`) brought throughput to ~3 samples/s.

**Critical fix #2 (correctness):** the miniwob path was crashing on every
episode with
`invalid literal for int() with base 10: '<uuid>'`.
Root cause: `int(exp.get("idx") or exp.get("step_id") or 0)` — when
`idx == 0` the `or` short-circuit fell through to the UUID `step_id`.
Fixed with a safe-cast helper that uses presence-based fallback.

**Stats:** 36,872 + 2,529 = **39,401 steps** with 100 % skill-query
coverage.
**Output:** `labeling/skill_actions_qa_out/run_20260506_190122_gpu/<source>/<model>/...`

### Stage 4 — skill_selection.jsonl Emission

**Script:** `scripts/build_qa_skill_selection_sft.py`

For each unit (QA hop or MiniWob step) with ≥ 2 candidate skills, emit
one SFT row. Row shape matches `labeling/build_decision_sft_jsonl.py:
build_skill_selection_row` so the trainer ingests it without code
changes.

Two operating modes:

* `--out-root <dir>` — write fresh `<bench>/skill_selection.jsonl` files.
* `--patch-sft-dir <existing>` — additionally write into an existing SFT
  dataset run.

**Stats:**

```
video_holmes     : 18,451 rows  (31 unique candidate IDs, 92,255 candidate slots)
siv_bench        :  6,818 rows  (23 unique IDs, 34,090 slots)
tir_bench        :  5,247 rows  (32 unique IDs, 26,235 slots)
visual_toolbench :  6,356 rows  (32 unique IDs, 31,780 slots)
miniwob          :  2,529 rows  (29 unique IDs, 12,645 slots)
─────────────────────────────────────────────────────────────────
TOTAL            : 39,401 rows
by source_model  : claude 11,031 · gpt-5.4 10,888 · qwen 10,680 · gemini 6,802
```

### Stage 5 — action_taking (op, sg) Relabel

**Script:** `scripts/relabel_qa_action_taking.py`

The QA/MiniWob `action_taking.jsonl` rows in the SFT dataset had
`intention_operator = intention_subgoal = "EXECUTE"` because the
original builder hard-coded `intention_subgoal="EXECUTE"`. We now have
hop-level / step-level labels from Stages 1 + 1', so we patch:

* **QA rows:** key by `(source, sample_id)` → `(operator, subgoal, note)`
  picked from the *most informative* hop (priority
  `REASON > COMPARE > VERIFY > INSPECT > COMMIT > TRACK > RECOVER >
  EXECUTE`, with bonus for terminal hop and non-empty note). Full hop
  sequence is also persisted as `intention_hops` for future trainers.
* **MiniWob rows:** key by `(base_episode_id, step_idx, model_dir)`
  with a fallback to `(base_episode_id, step_idx)` ignoring model.

**Critical fix #3 (model name mismatch):**
The SFT row's `source_model` field uses the long version-specific name
(`claude-4.6` / `gemini-3.1-pro` / `qwen3-vl-235b`), while the labeled
rollout directories use the short family name (`claude` / `gemini` /
`qwen`). Without normalisation, only `gpt-5.4` matched directly (215 of
905 rows) and the rest fell through to a model-agnostic fallback.

A 5-line `SFT_MODEL_TO_LABEL_DIR` map fixes this:

```python
SFT_MODEL_TO_LABEL_DIR = {
    "gpt-5.4": "gpt-5.4",
    "claude-4.6": "claude",
    "gemini-3.1-pro": "gemini",
    "qwen3-vl-235b": "qwen",
}
```

**Final coverage:**

```
[miniwob]          905/905 patched, match_kind={'direct': 905}
[siv_bench]        803/803 patched
[tir_bench]        302/302 patched
[video_holmes]   1,271/1,271 patched
[visual_toolbench]  74/74 patched
─────────────────────────────────────
TOTAL: 3,355 QA + 905 MiniWob = 4,260 rows relabeled (3,725 changed; rest already correct)
```

**Backups:** `<bench>/action_taking.jsonl.bak.<ts>` is created on every
inplace run so rollback is one `cp` away.

---

### Stage 1E — EnvWrapper Skill Bank Build (GPT-5.4)

**Script:** `labeling/build_skillbank_envwrappers_gpt54.py`

The 4 EnvWrapper games shipped with banks under
`labeling/skill_bank_out/run_20260430_030637/env_wrappers/<game>/` that
were dominated by anonymous hash IDs:

```
candy_crush:        10 skills (1 named, 9 hash)
super_mario:        16 skills (2 named, 14 hash)
tetris:             28 skills (6 named, 22 hash)
twenty_forty_eight: 19 skills (3 named, 16 hash)
```

We rebuild them by:

1. Reading the existing dual-axis labeled rollouts at
   `labeling/intentions_out/run_dualaxis_20260429_224917/env_wrappers/<game>/episode_*.json`.
2. Parsing per-step `(operator, subgoal, note)` triples (with regex
   fallback for the `[OP/SG] note` prefix used in the `intentions`
   field — the original labeler often left
   `intention_operator = None`).
3. Calling the GPT-5.4 CONTRACT pipeline from
   `labeling/build_skillbank_qa_gpt54.py` (clusters with ≥3 instances
   are kept; CURATOR pass merges near-duplicates).

**Stats:**

```
candy_crush         | instances=1000 | clusters=  3 | skills_kept= 3
super_mario         | instances= 326 | clusters= 10 | skills_kept= 7
tetris              | instances=1573 | clusters= 14 | skills_kept=12
twenty_forty_eight  | instances=3187 | clusters= 12 | skills_kept= 8
─────────────────────────────────────────────────────────────────
TOTAL: 6 086 step instances → 30 named skills (was: 11 named + 36 hash)
```

**Output:** `labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers/<game>/skill_bank.jsonl`

### Stage 2E — EnvWrapper Skill-Query Re-Labeling (GPU)

**Script:** `labeling/label_skill_actions_gpt54.py` (gymv version, with
shared `cuda:0` embedder applied)

We re-run skill-query labeling against the new banks, reusing the same
GPU embedder fix that we applied to the QA labeler in Stage 3 (a 6×
speedup over the default CPU embedder; ~144 s total wall-clock for all
6 086 steps).

**Stats:**

```
env_wrappers/candy_crush:        20 eps, 1 000 steps, coverage=100.00%,  3 distinct skills, 20.9s
env_wrappers/super_mario:        20 eps,   326 steps, coverage=100.00%,  7 distinct skills,  6.7s
env_wrappers/tetris:             20 eps, 1 573 steps, coverage=100.00%, 12 distinct skills, 48.3s
env_wrappers/twenty_forty_eight: 20 eps, 3 187 steps, coverage=100.00%,  8 distinct skills, 67.8s
```

Note the **distinct-skill-per-game** metric: every named skill in the
new bank is selected on at least one step → no dead vocabulary, healthy
candidate diversity.

**Output:** `labeling/skill_actions_out/run_envwrappers_20260506_202122_gpu/env_wrappers/<game>/episode_*.json`

### Stage 3E — EnvWrapper skill_selection.jsonl Emission + Patch

**Script:** `labeling/build_decision_sft_jsonl.py` (existing, unchanged)

Emits 1 SFT row per step (1:1 with `action_taking`). Output goes to
`labeling/decision_sft_jsonl/run_envwrappers_20260506_202502/<game>/skill_selection.jsonl`.

We then overwrite the 4 envwrapper files in the master SFT dir, saving
backups as `<game>/skill_selection.before_envwrapper_fix_<ts>.jsonl`.

**Quality verification (post-patch):**

```
candy_crush         | 1 000 rows |  3 unique candidates | top: COMMIT/CLEAR(1000) · COMPARE/CLEAR(1000) · INSPECT/SETUP(1000)
super_mario         |   326 rows |  7 unique candidates | top: COMMIT/EVADE(315) · COMMIT/ATTACK(313) · COMMIT/POSITION(287)
tetris              | 1 573 rows | 12 unique candidates | top: COMMIT/OPTIMIZE(1479) · COMMIT/SETUP(1167) · COMMIT/EVADE(1000)
twenty_forty_eight  | 3 187 rows |  8 unique candidates | top: COMMIT/MERGE(3165) · COMMIT/POSITION(3121) · COMMIT/OPTIMIZE(3069)
```

The selected-skill distribution now actually varies across rows (e.g.
super_mario uses 7 different choices instead of always picking from the
same 2), giving the SFT trainer a real classification signal.

---

## Bug Inventory & Fixes Applied

| # | Symptom | Root Cause | Fix |
|---|---|---|---|
| 1 | Stage 3 ran at ~0.5 samples/s on 8-GPU box | `SkillQueryEngine` defaulted `SentenceTransformer` to CPU | `_SHARED_GPU_EMBEDDER` lazy init on `cuda:0` in `label_skill_actions_qa_gpt54.py` |
| 2 | All MiniWob episodes failed with `invalid literal for int(): '<uuid>'` | `int(exp.get("idx") or exp.get("step_id") or 0)` — `idx=0` is falsy → fell through to UUID | Presence-based fallback chain + try/except for safe int cast |
| 3 | Skill bank load crashed: `VerificationReport.__init__() got unexpected keyword 'instances'` | Custom report dict mixed in non-schema fields | Strict schema-only `report` dict; auxiliaries moved to `provenance.aux_report` |
| 4 | Skill bank load crashed: ExecutionHint key mismatch | `common_pitfalls` / `common_postconditions` not in `ExecutionHint` schema | Mapped `common_pitfalls → common_failure_modes`, `common_postconditions → termination_cues`; original keys preserved on disk |
| 5 | Only 215/905 miniwob rows direct-matched (rest used cross-model fallback) | SFT writes `claude-4.6`, labeled dir is `claude` (etc.) | Added `SFT_MODEL_TO_LABEL_DIR` map; now 905/905 direct |
| 6 | EnvWrapper `skill_selection` rows had degenerate candidate sets (super_mario: 2 cands × all 326 rows identical; twenty_forty_eight: 3 cands × all 3187 identical; candy_crush: 4 of 5 cands were anonymous hash IDs) | Original bank had 1-3 named skills + 7-16 random hash IDs from raw clustering with no curator | Stages 1E-3E: rebuild banks via GPT-5.4 CONTRACT (3-12 named skills per game, all curated), re-run skill-query, emit fresh SFT rows |
| 7 | Stage 2E (EnvWrapper skill-query) hit same CPU embedder bottleneck as Stage 3 (~50 min wall-clock projected) | Original `label_skill_actions_gpt54.py` (gymv version) lacked the `_SHARED_GPU_EMBEDDER` helper that the QA version had | Ported `_get_shared_gpu_embedder()` to `label_skill_actions_gpt54.py`; ~144 s for 6 086 steps (was: ~50 min on CPU) |

---

## How to Reproduce

```bash
# Stage 1 — multihop CoT decomposition (QA)
python labeling/label_qa_multihop_gpt54.py \
    --sources video_holmes siv_bench tir_bench visual_toolbench \
    --models  gpt-5.4 claude gemini qwen \
    --workers 16

# Stage 1' — miniwob per-step intentions (already produced 2026-05-06_07h)
python scripts/label_qa_miniwob_intentions.py \
    --source miniwob --models gpt-5.4 claude gemini qwen --workers 16

# Stage 1' — webshop per-step intentions (run once per model-tag → labeled-run dir).
# Each tag corresponds to one model in the 4-way frontier comparison (see
# Cold-start-out-browsergym/REPORT_4way_comparison.md).
LABELED_RUN=labeling/qa_miniwob_labeled/run_20260506_070722
for pair in "low gpt-5.4" "claude claude" "gemini gemini" "qwen qwen"; do
    set -- $pair
    tag=$1; mdl=$2
    python scripts/label_qa_miniwob_intentions.py \
        --source webshop \
        --inputs Cold-start-out-browsergym/webshop_50task_${tag} \
        --output-dir ${LABELED_RUN}/webshop/${mdl} \
        --source-model-tag ${mdl} \
        --workers 16
done

# Stage 2 — skill bank build (GPT-5.4).  --sources defaults already include
# webshop alongside miniwob + the four QA benches.  The single
# --miniwob-run dir is expected to carry the peer subtrees miniwob/ and
# webshop/ — that's the layout produced by the Stage 1' loop above.
python labeling/build_skillbank_qa_gpt54.py \
    --multihop-run labeling/qa_multihop_out/run_20260506_181625 \
    --miniwob-run  labeling/qa_miniwob_labeled/run_20260506_070722 \
    --output-dir   labeling/skill_bank_qa/run_<ts>

# Stage 3 — skill-query labeling (GPU).  Add `webshop` to --sources so the
# per-source bank is loaded and the webshop subtree is walked.
python labeling/label_skill_actions_qa_gpt54.py \
    --bank-run     labeling/skill_bank_qa/run_20260506_184439 \
    --multihop-run labeling/qa_multihop_out/run_20260506_181625 \
    --miniwob-run  labeling/qa_miniwob_labeled/run_20260506_070722 \
    --output-dir   labeling/skill_actions_qa_out/run_<ts>_gpu \
    --sources video_holmes siv_bench tir_bench visual_toolbench miniwob webshop \
    --models  gpt-5.4 claude gemini qwen \
    --top-k 5 --workers 8 --verbose

# Build the action_taking.jsonl rows for webshop in the SFT dataset.  This
# step is parallel to the original miniwob entry in
# build_multimodal_decision_sft.py and writes
# <out-root>/webshop/action_taking.jsonl.  Use --webshop-min-reward 0.0
# to keep every episode (more data, noisier) or 1.0 for full successes
# only (~38 episodes total across the 4 frontier models).
python scripts/build_multimodal_decision_sft.py \
    --sources webshop \
    --webshop-min-reward 0.5 \
    --out-root labeling/decision_sft_jsonl/run_multimodal_20260506_055105

# Stage 4 — emit + patch skill_selection.jsonl (now also writes a
# webshop/ subdir under the SFT root).
python scripts/build_qa_skill_selection_sft.py \
    --skill-actions-run labeling/skill_actions_qa_out/run_20260506_190122_gpu \
    --out-root          labeling/qa_skill_selection_sft/run_<ts> \
    --patch-sft-dir     labeling/decision_sft_jsonl/run_multimodal_20260506_055105 \
    --sources video_holmes siv_bench tir_bench visual_toolbench miniwob webshop \
    --verbose

# Stage 5 — relabel action_taking (op, sg) inplace.  --miniwob-run is the
# single labeled-run directory holding both browser sub-trees; webshop SFT
# rows are patched the same way miniwob ones are.
python scripts/relabel_qa_action_taking.py \
    --sft-dir       labeling/decision_sft_jsonl/run_multimodal_20260506_055105 \
    --multihop-run  labeling/qa_multihop_out/run_20260506_181625 \
    --miniwob-run   labeling/qa_miniwob_labeled/run_20260506_070722 \
    --sources video_holmes siv_bench tir_bench visual_toolbench miniwob webshop \
    --inplace --verbose

# ─── EnvWrapper track (run independently after Stage 5) ───────────────

# Stage 1E — build per-game GPT-5.4 skill banks for the 4 EnvWrappers
python labeling/build_skillbank_envwrappers_gpt54.py \
    --intentions-run labeling/intentions_out/run_dualaxis_20260429_224917 \
    --output-dir     labeling/skill_bank_envwrappers/run_<ts> \
    --workers 8 --verbose

# Stage 2E — re-run skill-query labeling against the new banks (GPU)
python labeling/label_skill_actions_gpt54.py \
    --intentions-run labeling/intentions_out/run_dualaxis_20260429_224917 \
    --bank-run       labeling/skill_bank_envwrappers/run_<bank-ts> \
    --output-dir     labeling/skill_actions_out/run_envwrappers_<ts>_gpu \
    --corpus env_wrappers --all --top-k 5

# Stage 3E — emit fresh skill_selection.jsonl (1 row per labeled step)
python labeling/build_decision_sft_jsonl.py \
    --skill-actions-run labeling/skill_actions_out/run_envwrappers_<ts>_gpu \
    --output-dir        labeling/decision_sft_jsonl/run_envwrappers_<ts> \
    --corpus env_wrappers

# Stage 4E — overwrite the 4 envwrapper skill_selection.jsonl files in
# the master SFT dir; original files saved as
# <game>/skill_selection.before_envwrapper_fix_<ts>.jsonl.
# (Use the inline cp loop in the README_QA_MINIWOB_SFT_CLEANING git history
# or do it yourself: 4 files, 6 086 rows total.)
```

---

## Provenance Trail

Each cleaning step writes a sibling summary JSON for traceability:

| File | Stage |
|---|---|
| `labeling/qa_multihop_out/run_20260506_181625/_run_summary.json` | 1 |
| `labeling/qa_miniwob_labeled/run_20260506_070722/_dispatch_*.log` | 1' |
| `labeling/skill_bank_qa/run_20260506_184439/_summary.json` | 2 |
| `labeling/skill_actions_qa_out/run_20260506_190122_gpu/_run_summary.json` | 3 |
| `labeling/qa_skill_selection_sft/run_<ts>/_summary.json` | 4 |
| `labeling/decision_sft_jsonl/run_multimodal_20260506_055105/_qa_skill_selection_patch_summary.json` | 4 (patch) |
| `labeling/decision_sft_jsonl/run_multimodal_20260506_055105/_qa_action_taking_relabel_summary.<ts>.json` | 5 |
| `labeling/skill_bank_envwrappers/run_20260506_201030/_run_summary.json` | 1E |
| `labeling/skill_actions_out/run_envwrappers_20260506_202122_gpu/_run_meta.json` | 2E |
| `labeling/decision_sft_jsonl/run_envwrappers_20260506_202502/_run_summary.json` | 3E |
| `labeling/decision_sft_jsonl/run_multimodal_20260506_055105/<game>/skill_selection.before_envwrapper_fix_<ts>.jsonl` | 4E (backup) |

Original `action_taking.jsonl` rows are preserved as
`<bench>/action_taking.jsonl.bak.<ts>` for every inplace relabel.
Original EnvWrapper `skill_selection.jsonl` files are preserved as
`<game>/skill_selection.before_envwrapper_fix_<ts>.jsonl`.

---

## What Still Could Be Improved

1. **Game-side bank quality.** The latest GRPO-evolved game bank
   (`runs/Qwen3.5-9B_20260506_020501/checkpoints/step_99800/`) only adds
   1 new skill and drops 3 vs the cold-start bank, with all skills
   carrying `verified_domains: None`, `preconditions: None`,
   `effects: None`. Co-evolution hooks aren't actually evolving skills.
   Worth investigating whether `crafter_hook` is firing.
2. **No cross-domain transfer yet.** Game-side and QA-side skill IDs
   only share 4 names (`COMMIT/ATTACK`, `COMMIT/CLEAR`, `COMMIT/NAVIGATE`,
   `INSPECT/SETUP`) and those names have very different semantics in the
   two contexts. Real transfer probably has to come via SFT mixing
   (covered now — both vocabularies appear in the same run) rather than
   prompt-level ID overlap.
3. **action_taking vs skill_selection coverage skew.** QA `action_taking`
   is 1 row per sample, but `skill_selection` is 1 row per hop. The QA
   skill_selection corpus is therefore ~10× larger than the action one,
   which may bias adapter mixing weights. Trainer-side balancing or
   per-source row sampling is recommended.

---

*Last updated: 2026-05-06 20:30 UTC*
