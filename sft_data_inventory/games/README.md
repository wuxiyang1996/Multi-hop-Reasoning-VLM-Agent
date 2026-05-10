# Games — 12 tasks

| # | Task                          | Corpus       | Teachers |
| - | ----------------------------- | ------------ | :------: |
| 1 | Temporal_Airstriker-v0        | gym_v        |    4     |
| 2 | Temporal_AlteredBeast-v0      | gym_v        |    4     |
| 3 | Temporal_Columns-v0           | gym_v        |    4     |
| 4 | Temporal_DynamiteHeaddy-v0    | gym_v        |    4     |
| 5 | Temporal_SpaceHarrierII-v0    | gym_v        |    4     |
| 6 | Temporal_StreetsOfRage2-v0    | gym_v        |    4     |
| 7 | Temporal_Strider-v0           | gym_v        |    4     |
| 8 | Temporal_ThunderForceIII-v0   | gym_v        |    4     |
| 9 | tetris                        | env_wrappers |    1     |
|10 | super_mario                   | env_wrappers |    1     |
|11 | candy_crush                   | env_wrappers |    1     |
|12 | twenty_forty_eight            | env_wrappers |    1     |

## Common shape

For all 12 games the SFT pipeline is **fully bank-aligned**:

* `skill_bank.jsonl` — extracted by `labeling/extract_skill_bank.py`
  (gym_v) or `labeling/extract_skill_bank_envwrappers.py` (env_wrappers)
* `sft/action_taking.jsonl` & `sft/skill_selection.jsonl` —
  `active_skill` is the same `OPERATOR/SUBGOAL` ID in both files,
  populated by `labeling/label_skill_actions_gpt54.py` driving
  `SkillQueryEngine` against the bank
* `rollouts/<teacher>/episode_*.json` — labeled rollouts, each step
  contains a `skills` block (chosen) and a `skill_query` block
  (top-k candidates with `selected_skill_id`)

## Teacher coverage

| Teacher        | gym_v (8 games) | env_wrappers (4 games) |
| -------------- | :-------------: | :--------------------: |
| gpt-5.4        |        ✓        |           ✓            |
| claude-4.6     |        ✓        |           ✗            |
| gemini-3.1-pro |        ✓        |           ✗            |
| qwen3-vl-235b  |        ✓        |           ✗            |

## Source paths

* gym_v skill banks: `labeling/skill_bank_out/run_20260430_030637/gym_v/`
* env_wrappers banks: `labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers/`
* gym_v frontier SFT: `labeling/frontier_distill_jsonl/run_20260506_065830_skill_enriched/`
* env_wrappers SFT:   `labeling/decision_sft_jsonl/run_envwrapper_native_20260507_025519/`
* gym_v gpt-5.4 rollouts:    `labeling/skill_actions_out/run_20260430_064325/gym_v/`
* gym_v frontier rollouts:   `labeling/skill_actions_out/run_frontier_20260506_062027/<model>/gymv/`
* env_wrappers rollouts:     `labeling/skill_actions_out/run_20260430_064325/env_wrappers/`
