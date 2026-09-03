# Source Micro-Controller v1

## 结论

`source micro-controller v1` 已完成真实 source replay，但没有通过 source gate。
它不能进入 ALFWorld 或其他 target domain 的 authentic-source treatment。

本轮检验的不是旧 `COMMIT/CLEAR/POSITION` 标签，而是两个 receipt-derived
pre-action event：

```text
PROGRESS := 上一步 official reward > 0
STALL    := 连续两步 reward = 0、未 terminal、native affordance set 未变化
```

控制器只允许两个符号分支：`PERSIST` 或 `SWITCH`。具体动作始终是 source/target
环境的 native action；skill name、bank description、protocol 和 reasoning text 均未进入
induction 或 runtime。

## Discovery-only induction

六个长 source rollout 共提取 963 个 event decision points。只读取 discovery episodes 的
observational h8 positive rate，并先在每个游戏内计算、再跨游戏平均，得到候选：

```text
PROGRESS -> SWITCH
STALL    -> PERSIST
```

| Event | PERSIST discovery rate | SWITCH discovery rate | Candidate |
|---|---:|---:|---|
| PROGRESS | 0.5926 | 0.9167 | SWITCH |
| STALL | 0.5855 | 0.5139 | PERSIST |

这些数字只用于 candidate generation；它们是行为策略下的观察性相关，不是因果证据。

## Frozen replay

旧 Orak wrapper 不支持 seeded reset，因此 Candy/Tetris 没有被伪装成 matched intervention。
真正 replay 的四个 GymV 游戏为 Columns、Streets of Rage 2、Strider 和 Thunder Force III。

冻结计划在 qualification 与 held-out 各选择 8 个 snapshots，每个 snapshot 执行：

```text
2 estimands × 5 treatments × h1/h2/h4/h8
```

Treatments：

```text
EVENT_CONTROLLER
SHUFFLED_EVENT_CONTROLLER
ALWAYS_PERSIST
ALWAYS_SWITCH
HASH_RANDOM
```

共 16 snapshots、32 complete matched cells、160 trajectories。全部得到
`INTERVENTION_OBSERVED`；无 replay mismatch、event mismatch 或 inadmissible action。

## h8 result

下表顺序为 `EVENT / SHUFFLED / PERSIST / SWITCH / RANDOM`：

| Split | Estimand | Mean return | Progress rate |
|---|---|---|---|
| qualification | common continuation | 10.5 / 15.875 / 23.125 / 3.25 / 0.75 | .500 / .375 / .625 / .250 / .125 |
| qualification | full regime | .625 / 13.375 / 18.125 / 3.25 / .75 | .125 / .250 / .375 / .250 / .125 |
| held-out | common continuation | 131.25 / 131.25 / 131.25 / 131.25 / 131.875 | .375 / .375 / .375 / .375 / .500 |
| held-out | full regime | 131.875 / 131.25 / 144.875 / 131.25 / 131.875 | .500 / .375 / .625 / .375 / .500 |

Primary raw-return gate 与不改变 gate 的 binary-progress diagnostic 得出相同方向：

- qualification/full 中 event controller 比 shuffled、always-persist、always-switch 都差；
- held-out/common 中 event、shuffled 与两个 static controls 完全相同；
- held-out/full 虽超过 shuffled 与 always-switch，但落后 always-persist，且等于 random；
- 因此 `SOURCE_MICROCONTROLLER_SUPPORTED=false`。

按游戏检查可看到 observation/causation reversal：Thunder qualification 的 `PROGRESS`
snapshot 中，discovery 候选 `SWITCH` 的 h8 return 为 0，而 `PERSIST` 为 100。Thunder held-out
的多个 snapshot 中所有 first-action treatments 都得到相同 delayed reward。Strider 的 40 条
treatment trajectories 全部为零 reward。当前 event alphabet 因而仍不足以区分可控进展与
游戏自身的 reward cadence。

## Interpretation

本轮支持：

- 高层 skill name 可以完全移除；
- event-level controller 的 discovery、冻结、真实 fork 和 fail-closed gate 可执行；
- target-native neural grounding + symbolic routing 的实验骨架已就绪。

本轮不支持：

- `PROGRESS/STALL -> PERSIST/SWITCH` 是稳定 source control law；
- 将该 controller、旧 skill 或 source probe 权重迁移到 target；
- “停滞后总是换动作”或“进展后总是保持”一类固定 heuristic。

下一候选不应继续增加高层标签。应把 event 从 absolute reward 改成同一 snapshot 内的
relative intervention effect，例如 `ALTERNATIVE_BETTER / PERSIST_BETTER / UNIDENTIFIED`，
再学习 `COMPARE -> SELECT / ABSTAIN`。这会把环境 reward cadence 从 transferred object 中移除。

## Artifacts

- `configs/source_microcontroller_v1.json`
- `src/motif_transfer/source_microcontroller.py`
- `scripts/prepare_source_microcontroller_v1.py`
- `scripts/run_source_microcontroller_v1.py`
- `runs/source_microcontroller_v1_gymv/plan.json`
- `runs/source_microcontroller_v1_gymv/observational_report.json`
- `runs/source_microcontroller_v1_gymv/execution/microcontroller_rows.jsonl`
- `runs/source_microcontroller_v1_gymv/execution/report.json`
- `runs/source_microcontroller_v1_gymv/execution/scale_robustness_diagnostic.json`
