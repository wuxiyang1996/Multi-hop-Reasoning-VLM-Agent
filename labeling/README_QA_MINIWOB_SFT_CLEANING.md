# QA + MiniWob SFT Dataset Cleaning (May 2026)

A record of the cleaning + enrichment pipeline applied to the multimodal
decision SFT dataset under
`labeling/decision_sft_jsonl/run_multimodal_20260506_055105/`.

The motivation: the QA / web action_taking rows had collapsed all (operator,
subgoal) labels to `EXECUTE/EXECUTE`, and **none** of the 5 QA/Web sources
had any `skill_selection` rows at all — meaning the decision agent learned
nothing useful about strategy selection on those benchmarks. After the
clean, every QA/Web source has both rich per-step intentions and a fully
populated skill_selection candidate vocabulary.

---

## TL;DR — Before vs After

| Bench | action rows | skill_sel rows BEFORE | skill_sel rows AFTER | (op, sg) AFTER (top-3) |
|---|---:|---:|---:|---|
| `video_holmes`     | 1,271 | **0** | **18,451** | REASON/DEDUCE 923 · REASON/RULE_OUT 293 · REASON/TIMELINE 42 |
| `siv_bench`        |   803 | **0** |  **6,818** | REASON/DEDUCE 548 · REASON/RULE_OUT 237 · REASON/IDENTIFY 8 |
| `tir_bench`        |   302 | **0** |  **5,247** | REASON/DEDUCE 147 · REASON/MEASURE 58 · REASON/RULE_OUT 40 |
| `visual_toolbench` |    74 | **0** |  **6,356** | REASON/DEDUCE 37 · REASON/RULE_OUT 13 · REASON/MEASURE 11 |
| `miniwob`          |   905 | **0** |  **2,529** | COMMIT/EXECUTE 410 · COMMIT/BUILD 190 · COMMIT/POSITION 69 |

**action_taking before:** all 5 sources collapsed to `EXECUTE/EXECUTE`
(or `NAVIGATE/NAVIGATE` for ~9 % of miniwob).
**action_taking after:** dominant `REASON/*` operators on QA, dominant
`COMMIT/*` operators on miniwob — both reflect the actual semantic role
of each row.

Aggregate dataset sizes:

```
action_taking      : 55,806  (unchanged count; 3,725 QA/Web rows relabeled)
skill_selection    : 22,086 → 61,487  (+39,401; +178 %)
unique skill IDs   : 25 (game side) → 25 + 71 (QA side, all new) = 96
```

---

## Pipeline Stages

The cleaning pipeline is 5 stages plus one bug-fix step. All scripts live
under `labeling/` and `scripts/`.

```
                     ┌──────────────────────────────┐
                     │ raw rollouts / QA samples    │
                     │ (Cold-start-out-*, OR-TX)    │
                     └──────────────┬───────────────┘
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
┌──────────────────────┐                                 ┌──────────────────────┐
│ Stage 1: Multihop    │                                 │ Stage 1': MiniWob    │
│ CoT decomposition    │                                 │ per-step intention   │
│ (QA, GPT-5.4)        │                                 │ labeling             │
│                      │                                 │ (gpt-5.4 per step)   │
│ → samples_with_hops  │                                 │ → rollouts.jsonl     │
└──────────┬───────────┘                                 └──────────┬───────────┘
           │                                                        │
           └────────────────────────┬───────────────────────────────┘
                                    ▼
                     ┌──────────────────────────────┐
                     │ Stage 2: Skill Bank Build    │
                     │ (GPT-5.4 driven CONTRACT +   │
                     │  CURATOR; per source)        │
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

### Stage 1' — MiniWob Per-Step Intentions

**Script:** `scripts/label_qa_miniwob_intentions.py`
**Output:** `labeling/qa_miniwob_labeled/run_20260506_070722/miniwob/<model>/<game>/rollouts.jsonl`

Each browsergym experience step gets `intention_operator`,
`intention_subgoal`, `intention_note` attached via GPT-5.4.

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

## Bug Inventory & Fixes Applied

| # | Symptom | Root Cause | Fix |
|---|---|---|---|
| 1 | Stage 3 ran at ~0.5 samples/s on 8-GPU box | `SkillQueryEngine` defaulted `SentenceTransformer` to CPU | `_SHARED_GPU_EMBEDDER` lazy init on `cuda:0` in `label_skill_actions_qa_gpt54.py` |
| 2 | All MiniWob episodes failed with `invalid literal for int(): '<uuid>'` | `int(exp.get("idx") or exp.get("step_id") or 0)` — `idx=0` is falsy → fell through to UUID | Presence-based fallback chain + try/except for safe int cast |
| 3 | Skill bank load crashed: `VerificationReport.__init__() got unexpected keyword 'instances'` | Custom report dict mixed in non-schema fields | Strict schema-only `report` dict; auxiliaries moved to `provenance.aux_report` |
| 4 | Skill bank load crashed: ExecutionHint key mismatch | `common_pitfalls` / `common_postconditions` not in `ExecutionHint` schema | Mapped `common_pitfalls → common_failure_modes`, `common_postconditions → termination_cues`; original keys preserved on disk |
| 5 | Only 215/905 miniwob rows direct-matched (rest used cross-model fallback) | SFT writes `claude-4.6`, labeled dir is `claude` (etc.) | Added `SFT_MODEL_TO_LABEL_DIR` map; now 905/905 direct |

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

# Stage 2 — skill bank build (GPT-5.4)
python labeling/build_skillbank_qa_gpt54.py \
    --multihop-run labeling/qa_multihop_out/run_20260506_181625 \
    --miniwob-run  labeling/qa_miniwob_labeled/run_20260506_070722 \
    --output-dir   labeling/skill_bank_qa/run_<ts>

# Stage 3 — skill-query labeling (GPU)
python labeling/label_skill_actions_qa_gpt54.py \
    --bank-run     labeling/skill_bank_qa/run_20260506_184439 \
    --multihop-run labeling/qa_multihop_out/run_20260506_181625 \
    --miniwob-run  labeling/qa_miniwob_labeled/run_20260506_070722 \
    --output-dir   labeling/skill_actions_qa_out/run_<ts>_gpu \
    --sources video_holmes siv_bench tir_bench visual_toolbench miniwob \
    --models  gpt-5.4 claude gemini qwen \
    --top-k 5 --workers 8 --verbose

# Stage 4 — emit + patch skill_selection.jsonl
python scripts/build_qa_skill_selection_sft.py \
    --skill-actions-run labeling/skill_actions_qa_out/run_20260506_190122_gpu \
    --out-root          labeling/qa_skill_selection_sft/run_<ts> \
    --patch-sft-dir     labeling/decision_sft_jsonl/run_multimodal_20260506_055105 \
    --verbose

# Stage 5 — relabel action_taking (op, sg) inplace
python scripts/relabel_qa_action_taking.py \
    --sft-dir       labeling/decision_sft_jsonl/run_multimodal_20260506_055105 \
    --multihop-run  labeling/qa_multihop_out/run_20260506_181625 \
    --miniwob-run   labeling/qa_miniwob_labeled/run_20260506_070722 \
    --inplace --verbose
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

Original `action_taking.jsonl` rows are preserved as
`<bench>/action_taking.jsonl.bak.<ts>` for every inplace relabel.

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

*Last updated: 2026-05-06 19:50 UTC*
