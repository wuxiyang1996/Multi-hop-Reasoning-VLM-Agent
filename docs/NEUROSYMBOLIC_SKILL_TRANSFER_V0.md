# Neural-Symbolic Skill Transfer v0

## 当前状态

本仓库已加入一个最小可执行 vertical slice。它实现的是
**neural-symbolic adaptation-procedure transfer 接口**，尚未声称已经观察到跨域正迁移。

核心对象为：

```text
NeuroSymbolicProgram
  ├── BEFORE neural guard
  ├── deterministic symbolic route
  ├── Decision Agent 的 target-native action
  ├── immutable transition receipt
  ├── TRANSITION neural effect probe
  └── NEXT_NODE / REPLAN / ABSTAIN / TERMINATE
```

## 已实现的不变量

- `BEFORE` probe 只能读取当前 observation 与 native action set，不能读取 selected action
  或 after-state；
- `TRANSITION` probe 只能在真实 transition 之后运行；
- transition 必须通过 receipt hash、before/after state、native-action membership、terminal
  和 official-success 一致性检查；
- probe checkpoint、阈值和 source receipt lineage 都冻结在 content-addressed IR 中；
- SFT formatter 默认不再暴露稳定的 game ID hash，避免 game-identity shortcut；
- 缺失 score 或落入未校准区间时固定得到 `UNKNOWN → ABSTAIN`；
- neural score 只有 `NEURAL_PROPOSAL_ONLY` 权限；
- `TERMINATE` 必须同时得到 target official evaluator 的 success；
- monitor 没有 action 字段，也不能替换 Decision Agent 的动作。

主要代码：

- `src/motif_transfer/neurosymbolic_ir.py`
- `src/motif_transfer/neural_event_probes.py`
- `src/motif_transfer/program_monitor.py`
- `tests/test_neurosymbolic_program.py`

## Source-only supervision

`harness_training.py` 新增 `OPERATIONAL_EFFECT_PROBE` objective。Label 只来自 immutable
source transition receipt：

```text
observation_changed
admissible_set_changed
positive_native_reward
terminal
```

它不包含具名 predicate、source→target action mapping 或 Agent verdict。
`official_success` 不作为这个 neural objective 的 label；它始终由 target official evaluator
独立提供。

使用已有六游戏、36 episodes 的真实 source evidence 做过一次不落盘的 preflight：

```text
总 examples                         12,383
OPERATIONAL_EFFECT_PROBE             1,769
train / validation / source-heldout  4,144 / 4,123 / 4,116
```

当前 1,769 个 source transitions 中，`observation_changed` 为 1,769/1,769，不能单独提供
discriminative supervision；`admissible_set_changed` 为 616/1,153，`positive_native_reward`
为 733/1,036，`terminal` 为 36/1,733。正式训练前应对常量/稀有 label 加权，并加入 receipt
counterfactual 与 withheld-evidence negatives。否则模型可能只学习常数输出。

## 本地复现

```bash
python examples/smoke_neurosymbolic.py
pytest -q tests/test_neurosymbolic_program.py tests/test_harness_training.py
pytest -q
```

Smoke 的期望控制流为：

```text
SUPPORTED guard → ADMIT
official receipt + SUPPORTED effect + official success → TERMINATE
```

## 下一实验

1. 用现有 fresh source evidence 重新生成包含 `OPERATIONAL_EFFECT_PROBE` 的 dataset；
2. 分别评估 Base 与 game-receipt LoRA 在 source-held-out 上的 effect、UNKNOWN 和 abstention
   calibration；
3. 在冻结 ALFWorld adaptation split 上分别使用 0/1/2/4 examples 构建 target-native program；
4. 冻结 program，在 held-out task 上比较：

```text
BASE / target-induced
GAME_RECEIPT_TRAINED / target-induced
GAME_RECEIPT_TRAINED / authentic-source + target-induced
GAME_RECEIPT_TRAINED / shuffled-source + target-induced
```

第一主 contrast 是：

```text
GAME_RECEIPT_TRAINED / target-induced
    versus
BASE / target-induced
```

只有 game-trained Harness 在相同 target examples、token、tool 和 environment budgets 下稳定
超过 Base Harness，才能把结果升级为 neural-symbolic adaptation-procedure transfer。

## v0 offline transfer probe 结果

冻结协议见 `configs/neurosymbolic_transfer_probe_v0.json`，紧凑结果见
`docs/results/neurosymbolic_transfer_probe_v0_summary.json`。实验使用 source `train` split
训练一个只读取九个 domain-agnostic before/history features 的 12-unit MLP；ALFWorld task
offset `0–3` 只用于 k-shot adaptation，`4–7` 固定用于 internal evaluation。

Source-held-out 上 authentic probe 的 macro Brier 为 `0.0291`，优于 shuffled-label control
的 `0.0468`，说明模型确实学到了 source 内部结构。但 target primary gate 失败：

```text
            target-only  authentic  shuffled  marginal
k=1            .1938       .1994      .1594     .0989
k=2            .1560       .1363      .1789     .1265
k=4            .1211       .1429      .1488     .1096
```

Authentic 只通过预注册九项比较中的三项，并且没有在任何 k 上同时超过全部 controls。
当前结果支持：source probe 可学习；不支持：其 neural control structure 稳定迁移到 ALFWorld。

`k=4` 的 terminal balanced-Brier 在五个 optimization seeds 上都优于 target-only，但 target
internal evaluation 只有四个 terminal positives，而且该指标不是注册的 primary metric。因此它
只能作为下一轮 qualification replication candidate，不能从本轮结果中挑选后改写主张。

更进一步的原始日志诊断见
`docs/results/neurosymbolic_transfer_probe_v0_diagnosis.json`。失败的主要原因不是 MLP 没有
拟合 source，而是当前三个 operational symbols 被 game regime 决定：Tetris/Candy 几乎每步
都有 action-set change 和正 reward，另外四个游戏大多没有。模型即使看不到 game ID，也能从
native-action 数量与近期 effect history 恢复这个 domain fingerprint。ALFWorld 因而被错误外推为
几乎每步获得正 reward。

同一个历史 skill label 也不具有稳定行为，例如 `COMMIT/POSITION` 在 Tetris 的正 reward rate
为 1.0，在 Strider 为 0.071，在 Thunder 为 0.0。Fresh matched receipts 证明 authentic skill
context 会改变 source action。旧 one-step receipt 只能测 immediate reward，不能识别稳定的多步
skill value，因此当时仍只能证明 policy influence。

因此下一版本不应共享 absolute reward/affordance predictor。更合理的 neural-symbolic 边界是：

```text
source matched interventions
  → relative causal control graph（symbolic transfer object）

target adaptation receipts
  → target-native evidence/failure/completion probes（neural grounding）

symbolic graph + target-native probes
  → VERIFY / CONTINUE / REPLAN / ABSTAIN / OFFICIAL_STOP
```

Source neural supervision 应预测 authentic 相对 skill-masked/random controls 的 outcome delta，
而不是绝对游戏 reward。Source 和 target 默认不共享 raw probe weights，只检验控制结构能否迁移。

## Source multi-horizon value smoke v1

随后实现并运行了真实 `h1/h2/h4/h8` matched fork。冻结配置为
`configs/source_neurosymbolic_value_smoke_v1.json`，紧凑结果为
`docs/results/source_neurosymbolic_value_smoke_v1_summary.json`，Slurm job 为 `7230057`。
计划在读取 multi-horizon outcome 前冻结，只按连续 lineage 与 stable hash 分别选择一个
qualification 和 held-out Thunder Force III snapshot。

两个 snapshot 的 `2 estimands × 4 treatments` 共 16 个 cell 全部完成，112 次新的 closed-loop
continuation policy call 均返回有效 native action。h8 结果为：

```text
split          estimand                    B   G-S   G+S   G+Random
qualification common continuation         0     0     0          0
qualification full treatment regime       0     0   100          0
held-out      common continuation         0     0     0          0
held-out      full treatment regime       0     0     0          0
```

qualification/full-regime 的 authentic reward 在第 3 个 decision 出现，说明持续保留 authentic
context 可以在一个 snapshot 改变后续 action 并产生局部价值。但该效应既不属于 treatment-specific
first-action effect，也没有在 held-out snapshot 重现。冻结的 `SOURCE_H8_VALUE_SUPPORTED` 因此失败。

本轮支持 multi-horizon intervention 机制可执行；不支持把当前 `COMMIT/CLEAR` context 或由它直接
命名出的 graph 冻结为 transferable symbolic structure。下一 source candidate 必须从 discovery-only
relative outcome delta 中归纳，再在更多 blind snapshots 上验证 value recurrence；当前结果不授权
ALFWorld/VTB source-transfer treatment。
