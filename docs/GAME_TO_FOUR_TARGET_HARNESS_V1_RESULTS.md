# Game → ALFWorld / WebShop / DiscoveryWorld / TIR Harness V1

日期：2026-08-14

> **Historical scope note.** This document tests whether one shared Sokoban
> binary-effect program transfers unchanged to all four targets; that claim remains
> negative/partial. Later work validates four exact routes using three distinct
> game-derived structures. See
> [`NEUROSYMBOLIC_TRANSFER_FOUR_DOMAIN_STATUS.html`](NEUROSYMBOLIC_TRANSFER_FOUR_DOMAIN_STATUS.html)
> and [`FOUR_DOMAIN_NEUROSYMBOLIC_TRANSFER_HARNESS_V1.md`](FOUR_DOMAIN_NEUROSYMBOLIC_TRANSFER_HARNESS_V1.md).

## 结论

同一个 fresh-confirmed real-Sokoban effect program 已通过统一的 source receipt、
target-native grounding、matched null/control、content hash 和 fail-closed gate 接入四个 target。
结果是 **partial transfer**，不是四域全部成立：

| target | evidence tier | target/null | authentic | paired result | verdict |
|---|---|---:|---:|---:|---|
| WebShop | fresh formal, 32 goals | 11 | 18 | 7W / 0L | validated |
| DiscoveryWorld Easy | consumed adaptation, 4 forks | 3 | 4 | 1W / 0L | mechanism positive only |
| ALFWorld | consumed qualification, 24 tasks | 16 | 13 | 1W / 4L | negative transfer |
| TIR | consumed development, 8 tasks | 5 | 5 | 1W / 1L | no attributable gain |

统一 machine-readable verdict 是：

```text
PARTIAL_TRANSFER_ONLY
validated mechanism cells: WebShop, DiscoveryWorld Easy
validated held-out cells: WebShop only
```

Compact audit：
[`results/game_to_four_target_harness_v1_summary.json`](results/game_to_four_target_harness_v1_summary.json)。

## 真正共享的 source object

四个 adapter 都绑定同一个 source program：

```text
artifact_sha256:
6b02dc1d7271bbd435e90539cedd7d56d04fcc1ad03798dd6dd06146d67f1fcd

fresh_confirmation_sha256:
d64606c916ce6e812ae1b920771d5175cb48983a2f141b2dab6f43d491a6c1ed
```

其可执行结构只有：

```text
positive intervention effect available -> COMMIT -> VERIFY
otherwise                              -> POSITION -> recompute
```

source 不读取 target action token、坐标、reward、gold answer 或 official success。每个 target
自己负责 neural grounding 和 native action realization：

- WebShop：`POSITION/COMMIT` → browse/buy；
- DiscoveryWorld：`POSITION/COMMIT` → exact spatial move/drop；
- ALFWorld：`POSITION/COMMIT` → search/current-stage advance；
- TIR：source `COMMIT` role → effectful TEST，source `POSITION` role → answer COMMIT。

最后一项是 role-preserving retargeting：source 的 `COMMIT` 表示执行有预测正效果的 intervention，
不是把 Sokoban 的 push token 搬到视觉域。

## ALFWorld：修了 concrete realization，仍然失败

旧 V2 的一个合理怀疑是 source effect score 同时覆盖 `ACQUIRE/TRANSFORM/PLACE` 后，又在
option 内错误排序 concrete action。V14 明确修正 authority：

1. source 只选 `POSITION` 或 `COMMIT/current-stage advance`；
2. ALFWorld masked neural policy 在选定 option 内独占 native action ranking；
3. raw、null、authentic、availability、inverted、position-prior 和 target-oracle 共享同一
   grounder、seed、initial state 和 70-step budget；
4. evaluator-only outcome 不进入 selector/Harness。

结果：

| condition | success | mean steps | changed-option rate |
|---|---:|---:|---:|
| raw target-only | 16/24 | 43.96 | 0 |
| null, same Harness | 16/24 | 43.96 | 0 |
| authentic Sokoban effect | 13/24 | 46.00 | 17.57% |
| availability | 0/24 | 70.00 | 88.81% |
| inverted | 0/24 | 70.00 | 88.87% |
| position prior | 0/24 | 70.00 | 8.75% |
| target oracle | 17/24 | 37.21 | 36.06% |

Authentic 严格超过三个 source controls，证明 source structure 不是 inert；但它相对 null 是
`1W/4L/19T`。因此 primary utility 与 zero-negative-transfer gates 均失败。这个结果与早期
binary-effect ALFWorld failure 一致，说明 blocker 不是 option 内 action ranking，而是：

> `POSITION/COMMIT` 无法表达 ALFWorld 的 `SEARCH → ACQUIRE → TRANSFORM → PLACE → VERIFY`
> typed stage executability。

Raw report（34 MB，保留在本地 runs）：
`runs/game_to_alfworld_stage_retargeting_v14_consumed/report.json`；其 self-hash 已绑定进 compact
four-target audit。任何 ALFWorld held-out 均未因 V14 打开。

## TIR：同 source 接通，但 target applicability 不够可靠

先完成的 TIR V3 使用 controlled synthetic-game value ensemble，在冻结的 8-task qualification
上 authentic、target-only、shuffled 都是 `2/8`，尽管有 5 个 action contrasts。24 个 held-out
保持未读。

为排除 source-lineage 不一致，V5 改用与 WebShop/DiscoveryWorld/ALFWorld 完全相同的 Sokoban
effect program，并用 `openai/gpt-4.1-mini` 的 target-native cross-fitted candidate grounder：

| condition | success | TEST count |
|---|---:|---:|
| raw answer | 4/8 | 0 |
| null, same active Harness | 5/8 | 4 |
| authentic Sokoban effect | 5/8 | 5 |
| availability / always TEST | 5/8 | 8 |
| inverted | 4/8 | 3 |
| position / always answer | 4/8 | 0 |
| target oracle | 6/8 | 6 |

Authentic 相对 null 是 `1W/1L/6T`。更强 target model 把 raw baseline/oracle 从 Qwen 的
`1/8, 3/8` 提高到 `4/8, 6/8`，但 source utility 仍未出现；shuffled controlled-game prior
甚至达到 `6/8`。因此问题不是简单加 frames 或换强模型，而是当前 binary effect predicate
不能预测哪种 TIR visual intervention 值得执行。

Shared-source compact report：
[`results/sokoban_tir_effect_v5_consumed_summary.json`](results/sokoban_tir_effect_v5_consumed_summary.json)。

## DiscoveryWorld 的边界

V21 在四个 consumed Easy forks 上，authentic `4/4`、target-native myopic `3/4`、availability
`2/4`、inverted/position `0/4`，并有一个 exact matched success rescue 和零 regressions。
这建立了 mechanism，但不是 held-out generalization。

V22 Normal formal 在 outcome-blind baseline 阶段只找到 0 个 eligible `DROP/PUT` forks；三份已开
task 加三份未开 task也不可能达到预注册的 minimum 4 forks，因此提前停止。Space Sick Normal
要求 NPC dialogue/feeding，Proteomics Normal 又没有在 96 steps 内到达 commit state。这是
applicability coverage failure，不是 negative-transfer outcome。

## WebShop 为什么通过

WebShop 与 source binary effect program 在同一控制层级：先浏览/定位满足约束的商品，只有 target
grounder预测 candidate 会带来 direct progress 时才 commit。V13 在 fresh 32 goals 上：

```text
target-only: 11/32
authentic:   18/32
paired:      7W / 0L / 25T
two-sided exact p = 0.015625
```

Authentic 同时严格超过 availability、inverted、position-prior 和 target-native myopic，所有
receipt、initial-state、source、grounder、runner 与 final-report hashes 均通过。

## 可部署的 selective Harness

统一 Harness 不应在四个 target 上强制启用同一个 skill。Fail-closed policy 是：

```text
target adaptation mechanism gate passed -> ENABLE_SOURCE
otherwise                               -> ABSTAIN_TO_MATCHED_TARGET
```

在现有异构 splits 上，它会对 WebShop 与 DiscoveryWorld Easy enable，对 ALFWorld 与 TIR
abstain。描述性总计从 target/null 的 `35` successes 变成 `43`，观测到的 gating 后 regression
为 `0`。这只是已有不同 benchmark/split 的 projection，不是 pooled statistical estimand，也不是
新的 held-out result。

## Bitter lesson 与下一步

这次结果反驳了“只要 target-native neural grounding 足够好，同一个二元 symbolic skill 就能迁移到
任何域”。正确边界是：

1. **WebShop / DiscoveryWorld Easy** 需要 intervention-effect guard，当前 source object 合适；
2. **ALFWorld** 需要从真实 source interventions 学出的 typed successor graph，至少区分
   `BIND/TRANSFORM/RELATE/VERIFY`，不能再把 target stages 全折叠为 COMMIT；
3. **TIR** 需要 topology-preserving vs localizing evidence-operation structure，而不是 generic
   TEST/COMMIT value prior；
4. 增加 source games 只有在提供新的、source-qualified typed intervention edges 时才有意义；把更多
   rollouts 塞进同一个 binary guard 不会修复 abstraction mismatch。

因此下一正式实验不应重开 ALFWorld/TIR held-out，也不应调低 gate。应先在 source 侧建立并 fresh
confirm 两个新 artifact：typed stage-transition graph（ALFWorld）和 spatial evidence-operation graph
（TIR），随后分别用 target-native neural binding 做 consumed adaptation gate。

## Reproduction

```bash
PYTHONPATH=src:. /fs/gamma-projects/vlm-robot/conda/envs/alfworld/bin/python \
  scripts/run_game_to_alfworld_stage_retargeting.py \
  --config configs/game_to_alfworld_stage_retargeting_v14_consumed.json \
  --output runs/game_to_alfworld_stage_retargeting_v14_consumed/report.json

PYTHONPATH=src:. python scripts/summarize_sokoban_tir_effect_development.py \
  --config configs/sokoban_tir_effect_v5_consumed.json \
  --output docs/results/sokoban_tir_effect_v5_consumed_summary.json

PYTHONPATH=src:. python scripts/summarize_game_to_four_target_harness.py \
  --alfworld-report runs/game_to_alfworld_stage_retargeting_v14_consumed/report.json \
  --output docs/results/game_to_four_target_harness_v1_summary.json
```
