# Non-game — 6 tasks

| # | Task              | Modality         | Teachers | SFT status |
| - | ----------------- | ---------------- | :------: | ---------- |
| 1 | miniwob           | web (browsergym) |    4     | bank ✓ + SFT ✓ (mixed IDs) |
| 2 | webshop           | web (browsergym) |    4     | bank ✓ (16 skills) + skill_query ✓ — decision-SFT pending |
| 3 | video_holmes      | video QA         |    4     | bank ✓ + SFT ✓ (mixed IDs) |
| 4 | siv_bench         | video QA         |    4     | bank ✓ + SFT ✓ (mixed IDs) |
| 5 | tir_bench         | image QA         |    4     | bank ✓ + SFT ✓ (mixed IDs) |
| 6 | visual_toolbench  | image QA         |    4     | bank ✓ + SFT ✓ (mixed IDs) |

## Mixed-IDs caveat

For miniwob + the four visual reasoning benches the same step has
**different `active_skill` values** in the two SFT files:

| File                       | `active_skill` scheme                    | Source |
| -------------------------- | ---------------------------------------- | ------ |
| `sft/skill_selection.jsonl`| **bank-derived** (`OPERATOR/SUBGOAL`)    | `labeling/qa_skill_selection_sft/run_20260506_194156/` driven by `SkillQueryEngine` |
| `sft/action_taking.jsonl`  | **synthetic** (`web/CLICK`, `video_qa/...`, etc.) | `scripts/build_multimodal_decision_sft.py` hardcoded mapping |

The `skill_selection` rows are the bank-aligned ones; if you want to use a
single skill scheme everywhere, retro-fit `build_multimodal_decision_sft`
to consume the bank-aware labels.

## Webshop pipeline (built 2026-05-10)

`webshop/` now contains the full bank-extraction pipeline output:

| stage | path inside `non_game/webshop/` | source |
| ----- | -------------------------------- | ------ |
| raw rollouts (4 teachers) | `rollouts/{gpt54,claude,gemini,qwen}/` | `Cold-start-out-browsergym/webshop_50task_*` |
| stage-1 intentions (`[OP/SG] note`) | `labeled/{gpt54,claude,gemini,qwen}/webshop.<idx>/rollouts.jsonl` | `labeling/qa_miniwob_labeled/run_webshop_20260510_043615/webshop/` |
| stage-2 skill bank | `skill_bank.jsonl` | `labeling/skill_bank_qa/run_webshop_20260510_044000/webshop/` |
| stage-3 skill_query (top-5 + selected) | `skill_query/{gpt54,claude,gemini,qwen}/webshop.<idx>/rollouts_with_skill_query.jsonl` | `labeling/skill_actions_qa_out/run_webshop_20260510_044300/webshop/` |

**16 skills** were extracted from **2 538 steps × 200 episodes × 4
teachers**.  Stage 3 covers **100 %** of those steps with a bank-derived
`selected_skill_id` plus a 5-item candidate list.

Decision SFT (`action_taking.jsonl` / `skill_selection.jsonl`) for
webshop is **not yet built** — `scripts/build_multimodal_decision_sft.py`
has webshop wired up (`success_threshold=0.5`, `skill_prefix='webshop'`)
but it was not run during the latest SFT corpus build.  Once it runs the
two SFT files will land at
`labeling/decision_sft_jsonl/run_*/webshop/{action_taking,skill_selection}.jsonl`
and this inventory will pick them up automatically on the next
`build_inventory.py` run.

## Source paths

* QA skill banks (miniwob + 4 visual reasoning):
  `labeling/skill_bank_qa/run_20260506_184439/<task>/`
* SFT (all non-game with SFT):
  `labeling/decision_sft_jsonl/run_envwrapper_native_20260507_025519/<task>/`
* miniwob & webshop rollouts (gpt-5.4):  `Cold-start-out-browsergym/`
* miniwob frontier rollouts:
  `openrouter-transfer-baselines-out/2026-05-01_08-06-44/<model>/browsergym/`
* visual reasoning rollouts (gpt-5.4): `Cold-start-out-visual-reasoning{,-video}/<bench>/samples.jsonl`
* visual reasoning frontier rollouts:
  `openrouter-transfer-baselines-out/2026-05-01_08-06-44/<model>/{vr_image,vr_video}/<bench>/samples.jsonl`
