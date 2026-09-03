# Sokoban → ALFWorld Neural-Symbolic Transfer V1/V2

日期：2026-08-11

## 结论

两轮 fresh ALFWorld qualification 都没有建立 real-game transfer：

| 条件 | V1 success | V2 success |
|---|---:|---:|
| target-only | 11/24 | **16/24** |
| authentic source + Harness | 14/24 | **10/24** |
| 最强错误 source control | **18/24** phase-permuted | 0/24 |
| target-native stage reference | 19/24 | 15/24 |

V1 看似把 success 从 `45.8%` 提到 `58.3%`，但 phase-permuted source 达到
`75.0%`，因此不能归因于真实 source structure。V2 删除 source option occupancy、重新训练
stage-masked target neural grounder，并把 transfer object 收窄为 intervention-effect guard；该 guard
严格胜过四个 source controls，却将强 target baseline 从 `66.7%` 降到 `41.7%`。

所以当前正确 verdict 是：

```text
REAL_GAME_TRANSFER_NOT_QUALIFIED
NEGATIVE_TRANSFER_OBSERVED
HELDOUT_UNREAD
```

这不是 neural-symbolic factorization 本身不成立。它说明当前 Sokoban source skill 的抽象层级仍
太粗，不能支撑 ALFWorld 的 typed multi-stage effects。机器可读 compact receipt 为
[`results/sokoban_alfworld_transfer_v1_v2_summary.json`](results/sokoban_alfworld_transfer_v1_v2_summary.json)，
summary SHA-256 为
`e36763f609a30dfb5f1182461b539b86b76a6f7e8044e98a17ad2d0bd609a934`。

## V1：为什么表面增益不算 transfer

V1 从真实 Sokoban logs 建立 deterministic transition model，在新 procedural boards 上确认
POSITION/COMMIT option predictor，然后由 ALFWorld Harness 做 target-native grounding。

Source 侧证据本身通过：

- legacy qualification：authentic accuracy `0.9697`；
- fresh source confirmation：`67/72 = 0.9306`，POSITION `59`、COMMIT `13`；
- transition validation `1.0`；
- fresh report SHA-256：
  `d442a1c72e8014c58a3ce53e036a8106ab21b9c09e53ca1bfef2b51f90eb8e90`。

但 target qualification 为：

| V1 condition | Success | Changed option | Paired authentic comparison |
|---|---:|---:|---:|
| target-only | 11/24 | 0 | 6 wins / 3 losses |
| authentic | 14/24 | 0.123 | — |
| option swap | 0/24 | 0.069 | 14 wins / 0 losses |
| source marginal | 7/24 | 0.101 | 9 wins / 2 losses |
| phase permuted | **18/24** | 0.119 | 0 wins / 4 losses |
| target-stage reference | 19/24 | 0.124 | reference only |

失败不是随机噪声。authentic 在 target 已是 COMMIT-ready 时仍选择 POSITION `485` 次；
phase-permuted 只有 `89` 次。Sokoban 最短路径中走位通常比推动多，option-value model 因而携带了
source-specific duration/occupancy prior。target 的阶段长度不同，错误 permutation 恰好抵消该 prior。

更根本的问题是 V1 canonicalization 只保证 feature vector 同宽，却没有保证 causal semantics 同构：
Sokoban `box_moved`、`assignment_improvement`、`deadlock` 被 ALFWorld completion/applicability
分数代填。字段名称相似不是 intervention-grounded transfer。

V1 qualification report SHA-256：
`002fe9fbc47cc4aa7f6591abe9d25c3f2fc0047672d14f9f91b1d26cbe080072`。

## V2：去掉 occupancy 后，为什么仍然负迁移

### Source effect program

V2 不再预测 next option，而冻结一个 Boolean intervention program：

```text
if DIRECT_PROGRESS_AVAILABLE or ASSIGNMENT_IMPROVEMENT_AVAILABLE:
    COMMIT
    VERIFY_EXPECTED_EFFECT
else:
    POSITION
    RECOMPUTE_EFFECT_PREDICATES

if EXPECTED_EFFECT_REFUTED:
    REPLAN_OR_ABSTAIN
```

它明确禁止迁移 source 坐标、action token、option frequency 和 duration。artifact SHA-256 为
`6b02dc1d7271bbd435e90539cedd7d56d04fcc1ad03798dd6dd06146d67f1fcd`。

在规则冻结后生成的 24 个新 procedural Sokoban layouts、96 个 snapshot 上：

| Source condition | Accuracy | Selected COMMIT |
|---|---:|---:|
| authentic effect guard | **0.9896** | 23 |
| availability only | 0.7396 | 49 |
| inverted effect | 0.5000 | 26 |
| POSITION occupancy prior | 0.7500 | 0 |

POSITION/COMMIT ground-truth support 为 `72/24`，不是单类准确率造成的假通过。fresh source report
SHA-256 为 `d64606c916ce6e812ae1b920771d5175cb48983a2f141b2dab6f43d491a6c1ed`。

### Stage-masked target neural grounding

为避免 Harness 偷带 target planner，V2 使用 36 个 adaptation_train episodes 重新训练四个 neural
heads，并对所有 head 屏蔽 `required_option` 输入。12 个 adaptation_validation episodes 上：

| Target grounding metric | Value |
|---|---:|
| applicability macro AUC | 0.9311 |
| completion balanced accuracy | 0.9098 |
| binding recall@3 | 1.0000 |
| neural policy AUC | 0.9176 |

target artifact SHA-256 为
`85af728b469a61cda7b9ab36aef90f03ada4b807230fa8aac62568a51ff12d4b`。
Effect threshold `0.10` 只由 adaptation_train 选择；adaptation_validation 上 coarse option accuracy
`0.9673`、balanced accuracy `0.9173`、COMMIT recall `0.8529`。Harness SHA-256 为
`80577ee194b08edc8b442181695184ef9ea87491ddd91eaa51782e79f149424f`。

Authentic path 的选择和 concrete realization 对 report 中 `required_option` 字段做 counterfactual
mutation 后保持 `100%` 不变，因此 V2 target outcome 不能解释为直接读取 target stage。

### Fresh target qualification

V2 manifest 在任何 selected task reset 前冻结，排除了此前 82 个 consumed/reserved task；
qualification 与 held-out 各 24 个。manifest SHA-256 为
`8377d8b8fb13e3095ed9220927e7828541e6fbed1a2218348656d6bd395024b4`。

| V2 condition | Success | Mean steps | Changed option |
|---|---:|---:|---:|
| target-only masked neural policy | **16/24** | 42.00 | 0 |
| authentic source effect Harness | **10/24** | 49.25 | 0.184 |
| availability control | 0/24 | 70.00 | 0.911 |
| inverted-effect control | 0/24 | 70.00 | 0.852 |
| POSITION occupancy control | 0/24 | 70.00 | 0.106 |
| effect-group permutation | 0/24 | 70.00 | 0.872 |
| target-native stage reference | 15/24 | 41.58 | 0.191 |

Authentic 相对 target-only 为 `1 win / 16 ties / 7 losses`，net `-6`。两种 effect predicate 值均被
实际触发、changed-option gate 通过、required-option invariance 通过；失败只发生在最重要的
official-success superiority gate。V2 qualification report SHA-256 为
`a16090f77514f9609b91571ef683e83e63b82cfbae28c320f42564675d25fa08`。

## 失败机制：binary COMMIT representation collapse

V2 authentic 的 option/stage diagnostic 为：

```text
POSITION -> POSITION   680
COMMIT   -> COMMIT     196
COMMIT   -> POSITION    23
POSITION -> COMMIT     283
```

因此 effect trigger 仍有大量 COMMIT false negatives。更重要的是，V2 把以下不同效果全部压成
一个 COMMIT：

```text
ACQUIRE(object)
TRANSFORM(object, clean)
TRANSFORM(object, cool)
TRANSFORM(object, heat)
PLACE(object, receptacle)
```

在 target-only 成功、authentic 失败的 clean tasks 中，错误 loop 包括：

- `cool pan 1 with fridge 1` 重复 14 或 37 次；
- `cool plate 4 with fridge 1` 重复 15 次；
- POSITION/COMMIT 阶段错配占满 70-step budget。

按 task family，authentic 相对 target-only 的成功数变化为：

| Family | target-only | authentic |
|---|---:|---:|
| clean then place | 7/9 | 3/9 |
| cool then place | 2/4 | 0/4 |
| simple place | 6/6 | 5/6 |
| two objects | 0/2 | 1/2 |
| heat then place | 1/1 | 1/1 |
| look under light | 0/2 | 0/2 |

四个破坏 source semantics 的 controls 均为 `0/24`，说明 authentic effect direction 确有内容，
但“非随机”不等于“有正 utility”。这是一次有信息量的 negative-transfer result。

## 为什么不能直接用 Thunder 补齐 V3

旧审计显示，当前带 no-human-hints exclusion receipt 的 Columns 和 Thunder 各只有 2 个 episode。
它们支持的结构主要是 idle detection、distinct control probe、delayed reward 与 verification；并没有
matched intervention forks 证明：

```text
possession added/removed
object property changed from x to y
object-receptacle relation added/removed
typed effect completed or refuted
```

将 Thunder 的 shoot/move/reward 事后命名成 ACQUIRE/TRANSFORM/PLACE，会再次制造 semantic
alignment，而不是发现 source structure。增加更多相同 coarse rollouts 也不会修复 representation
collapse。

## 下一轮的最小可行 source data

在再次运行 ALFWorld 前，应先补采一个具备对象状态的 source game/task suite。每个 snapshot 必须
执行 matched native intervention forks，并从 source transition 自动计算以下 typed deltas：

```text
BIND(entity):              possession false -> true
MUTATE(entity, p, x, y):   property p changes x -> y
RELATE(entity, r, target): relation r false -> true
VERIFY(effect):            expected typed delta observed/refuted
```

之后只能从 discovery split 抽取参数化程序：

```text
SEEK(x)
-> BIND(x)
-> optional MUTATE(x, property, target_value)
-> RELATE(x, destination)
-> VERIFY(goal_relation)
```

需要保留的 non-triviality gates：

1. source qualification/held-out 上，typed program 严格超过 option-label permutation、effect-type
   permutation、marginal 与 wrong-source controls；
2. source program 只输出带参数的 canonical effect，不排序 target action；
3. target-native neural grounder 从 observation/native actions 绑定 entity、property、relation；
4. same-Harness target-only/null/shuffled/authentic/target-stage reference 同轮 paired；
5. authentic 必须改善 official success，或在 success 完全不降的情况下改善预注册 efficiency；
6. 任何 qualification 失败都停止 held-out。

V1 与 V2 各 24 个 target held-out task 仍未 reset。它们可以在新的 V3 协议冻结后重新分配成
qualification/held-out，但在 typed source gate 通过之前不得读取。当前 valid_unseen 只剩 4 个未被
任何 manifest 保留的 task，因此不能继续无止境 outcome-driven retuning。

## Claim boundary

本轮支持：

- target-native neural grounding + frozen symbolic control 在代码上可隔离并可执行；
- Sokoban intervention forks 能产生稳定的 positive-effect guard；
- 错误 effect semantics 会产生显著 negative transfer；
- target success 需要 typed effect hierarchy，binary POSITION/COMMIT 不够。

本轮不支持：

- Sokoban/Thunder → ALFWorld 已经成功迁移；
- V1 的 `14/24 vs 11/24` 是 source-specific gain；
- V2 source accuracy `98.96%` 能预测 target utility；
- 继续调 effect threshold 就能修复 target；
- 可以读取任一 reserved held-out split。
