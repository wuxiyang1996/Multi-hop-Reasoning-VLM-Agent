# Sokoban → WebShop neural-symbolic transfer V12/V13

日期：2026-08-12

> **2026-08-14 independence correction（优先于下方历史结论）：** installed WebShop
> human-goal pool 只有 13 个 goals，bridge 将 task ID 对 13 取模。V13 `114–145` 与 replication
> `146–177` 各自都重复这同一组 13 个 semantics；两段 unique goal sets 完全重合。因此下方
> task-level `7W/0L, p=.015625` 仍是历史观察，但“32 个独立 fresh goals”与 independent
> confirmatory claim 撤回。按 semantic goal 聚类后，V13 为 5 positive / 0 negative clusters，
> `p=.0625`；replication 为 4/1，`p=.375`。V14 已冻结并 live 验证 56 个 unique-ASIN synthetic
> goals；32-goal formal outcome 仍 sealed。两组 development smoke 显示 authentic source authority=0，
> 尚未支持新 transfer claim。完整审计见
> [`WEBSHOP_V14_AND_VIDEO_COST_BOUNDARY.md`](WEBSHOP_V14_AND_VIDEO_COST_BOUNDARY.md)。

## 历史 V13 结论（已被上方 independence correction 限定）

V13 在预先冻结的 32 个 fresh WebShop goals 上通过了全部完整性、non-triviality 和 paired
performance gates：

```text
REAL_SOKOBAN_TO_WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_VALIDATED
```

在这个受限 setup 中，真实 Sokoban source 经 matched interventions 确认的 effect program，结合
WebShop-native neural grounding，把 strict success 从 target-only 的 `11/32` 提高到 `18/32`：
绝对提升 `+21.875pp`，paired `7 wins / 0 losses / 25 ties`，双侧 exact `p=.015625`。

这回答了此前 V11 留下的主要问题：不再只是 controlled hidden-rule synthetic source。Source
现在是 Sokoban；target action space、observation 和 goal semantics 则完全是 WebShop-native。

## 实际迁移的结构

没有迁移 Sokoban 坐标、动作 token、轨迹文本、option frequency 或 source duration。迁移对象是
一个 intervention-effect control relation：

```text
if positive irreversible effect is available:
    COMMIT
    VERIFY_EXPECTED_EFFECT
else:
    POSITION / PREPARE
    RECOMPUTE_EFFECT_PREDICATES

if expected effect is refuted:
    REPLAN_OR_ABSTAIN
```

Source program 只决定 `POSITION/PREPARE` 与 `COMMIT`。冻结的 WebShop 12-hidden-unit MLP 从
13 个 target-native state/action features 预测 `state_changed`、`terminated`、`reward` 和
`prerequisite_progress`，负责把 source predicates 落到 constraint click、navigation 和
`Buy Now` 等 native actions。Target planner 提供候选动作；source program 不接收 goal token、
accessibility-tree token 或 native action string。

因此这里的 neural-symbolic 分工是：

```text
Sokoban matched interventions
    -> frozen symbolic effect relation
    -> WebShop-native neural causal grounding
    -> typed POSITION/COMMIT decision
    -> one target-native action or abstention
```

这是 `intervention-grounded symbolic structure + target-native neural grounding`，不是把一个高层
skill description 塞进 prompt。

## Source gate

旧 V1 把 Sokoban 中 POSITION-heavy 的 option occupancy/value prior 搬到 ALFWorld，错误的
phase-permuted source 反而更强。V2 因此明确删除 occupancy、coordinates、action names 和 duration，
只保留正向 intervention effect guard。这个修改是在 ALFWorld negative result 后完成的，所以不能把
原 discovery 当 confirmatory evidence。

规则冻结后，代码生成 24 个新 procedural Sokoban layouts，共 96 个 fresh snapshots；每个 state
枚举 native actions 并用 shortest-solution first option 检验：

| source condition | accuracy | selected COMMIT/POSITION |
|---|---:|---:|
| authentic effect guard | **95/96 = 98.96%** | 23 / 73 |
| commit-availability only | 73.96% | 49 / 47 |
| inverted effect | 50.00% | 26 / 70 |
| POSITION occupancy prior | 75.00% | 0 / 96 |

Ground-truth support 为 `COMMIT 24 / POSITION 72`；coverage、accuracy 和 strict control-superiority
三个 gate 全部通过。Compact source receipt 是
[`results/sokoban_effect_program_v2_compact_receipt.json`](results/sokoban_effect_program_v2_compact_receipt.json)，
其 content hash 为
`86898bf50e77eca037213656f1fa1578ececfc938df79ca8eb99b7da11bc0a85`。

## Target protocol

V13 在读取 goal text 或 outcome 前冻结连续 goal IDs `114–145`。每个 goal 跑 6 个 matched
conditions，共 `32 × 6 = 192` final receipts：

1. target-only；
2. target-native myopic immediate reward；
3. authentic Sokoban effect program；
4. commit-availability control；
5. inverted-effect control；
6. POSITION-prior control。

所有条件共享 target planner/cache contract、target MLP、step budget 和 initial task state。协议在运行
前声明：只要任一 condition 出现 provider/schema/HTTP/timeout failure，就用 fresh shared cache 对完整
六条件组做最多一次 symmetric retry；禁止 selective retry。实际 V13 没有触发任何 retry。

V12 曾预留 goals `82–113`，但 runner 在第一个 `env.step` 前因 legacy interface dictionary 缺少
`uncertainty_scale` 而中止。`webshop.82` 已 reset，`webshop.83` 保守视为可能 reset；没有执行 action，
也没有观察 outcome。我们没有复用 V12 range，而是在仅补两个 selector 明确丢弃的 compatibility
scalars 后冻结全新的 V13 `114–145`。V12 abort 被保留在
[`../configs/webshop_sokoban_effect_transfer_v12_frozen.json`](../configs/webshop_sokoban_effect_transfer_v12_frozen.json)。

## V13 formal result

| condition | strict success | rate | mean reward | mean steps | source decisions |
|---|---:|---:|---:|---:|---:|
| target-only | 11/32 | 34.4% | .5599 | 8.16 | 0 |
| target-native myopic | 2/32 | 6.3% | .5859 | 4.50 | 30 |
| authentic Sokoban + target | **18/32** | **56.3%** | **.7188** | **7.53** | 33 |
| commit availability | 2/32 | 6.3% | .5547 | 6.31 | 18 |
| inverted effect | 2/32 | 6.3% | .5547 | 6.31 | 18 |
| POSITION prior | 9/32 | 28.1% | .4583 | 8.59 | 33 |

Frozen paired gates 全部通过：

| authentic versus | strict W–L–T | success delta | reward delta | exact two-sided p | action-contrast goals |
|---|---:|---:|---:|---:|---:|
| target-only | 7–0–25 | **+21.875pp** | **+.1589** | **.015625** | 10 |
| target-native myopic | 16–0–16 | **+50.0pp** | **+.1328** | **.0000305** | 23 |
| commit availability | 16–0–16 | **+50.0pp** | **+.1641** | **.0000305** | 18 |
| inverted effect | 16–0–16 | **+50.0pp** | **+.1641** | **.0000305** | 18 |
| POSITION prior | 9–0–23 | **+28.125pp** | **+.2604** | **.003906** | 9 |

Receipt matrix `192/192` 完整，所有 receipt self-hash 有效、全部 initial-state hashes matched、
final failures 为 0、operational retries 为 0。Authentic 实际触发 33 次 source decisions，而不是
只靠 abstention 复现 target-only。

Availability 和 inverted controls 在本次 target 上收敛成 `32/32` 相同 action sequences，因此它们
是一个公开的 redundant control，不应当被描述成两份独立证据。更重要的互补 controls 仍然成立：
availability/inverted 检验错误 COMMIT timing，POSITION prior 检验“只要多做准备动作就会变好”，
target-only 检验 source 是否有增量，myopic 检验 target MLP 单独是否足够。

Machine-readable compact result 是
[`results/webshop_sokoban_effect_transfer_v13_summary.json`](results/webshop_sokoban_effect_transfer_v13_summary.json)，
content hash 为
`2bde00ae5b1bd8c99f416a55551ade0e78c39023ffb0db0d1fbd3eae0c6ab86a`。

## 为什么这是 non-trivial transfer，以及哪里仍是人工设计

Non-triviality 来自四点：

1. source rule 先在 fresh Sokoban intervention states 上严格超过破坏语义的 controls；
2. target MLP 只用 WebShop-native counterfactual receipts grounding，不读 source tokens；
3. authentic 在 10 个 goals 上改变 target-only action sequence，并且获得 7 个 strict wins、0 losses；
4. authentic 同时超过 semantic controls，说明效果不是任意 source prior、固定 POSITION prior 或
   最大化 target immediate reward 都能产生。

但 abstraction correspondence 仍是 typed-by-design：研究者指定 Sokoban 的 reversible positioning /
positive-effect commit 与 WebShop 的 prerequisite preparation / purchase commit 对应。Source rule 自身是
symbolic，不是神经网络自动发现；neural 部分位于 target causal grounding。运行前还已知该 source
program 在 V11 的 266 个 controlled-source WebShop states 上与原 selector `266/266` 同动作，这只是
interface-alignment diagnostic，不是 V13 outcome evidence。

所以合适的论文 claim 是：

> A fresh-confirmed intervention-effect program from Sokoban can improve success in a semantically
> distinct WebShop target when its predicates and actions are grounded by a target-native neural
> model.

不合适的 claim 包括：任意 game skill 都可迁移、raw rollout retrieval 已经足够、cross-domain analogy
已自动发现、或者对不同 WebShop implementation/model 也必然成立。

## TIR / Video-Holmes 是否现在需要

对“真实游戏 source 是否能跨域提高成功率”这个 primary question，V13 已给出 fresh、paired、带 source
controls 的 positive result，所以不需要用 TIR/Video-Holmes 救这个 claim。

它们仍然是下一阶段很有价值的 external-validity test，但现有 preflight 不能直接跑 formal split：

- TIR 的 evidence response 为正，但 authentic 与 target-only action contrast 为 `0/16`；
- Video-Holmes 只有 `2/8` states 出现 prediction/action contrast，且没有 positive evidence-accuracy
  response。

因此下一步若扩域，应该先补 target-native matched evidence interventions，训练 paired uplift / CATE
grounder，并要求 adaptation 上出现稳定的 TEST/COMMIT action contrast；在 preflight 修复前读取 formal
TIR/Video-Holmes test split只会重复旧的“source 看起来存在，但不改变 target policy”失败。

## Authoritative artifacts

- V12 abort protocol：
  [`../configs/webshop_sokoban_effect_transfer_v12_frozen.json`](../configs/webshop_sokoban_effect_transfer_v12_frozen.json)
- V13 frozen protocol：
  [`../configs/webshop_sokoban_effect_transfer_v13_frozen.json`](../configs/webshop_sokoban_effect_transfer_v13_frozen.json)
- Sokoban → WebShop core：
  [`../src/motif_transfer/webshop_sokoban_effect_transfer.py`](../src/motif_transfer/webshop_sokoban_effect_transfer.py)
- Frozen executable runner：
  [`../scripts/run_webshop_sokoban_effect_transfer_v12.py`](../scripts/run_webshop_sokoban_effect_transfer_v12.py)
- Independent summarizer：
  [`../scripts/summarize_webshop_sokoban_effect_v13.py`](../scripts/summarize_webshop_sokoban_effect_v13.py)
- Frozen target grounder：
  [`results/webshop_neural_symbolic_v9_frozen_grounder.json`](results/webshop_neural_symbolic_v9_frozen_grounder.json)
- Compact source receipt：
  [`results/sokoban_effect_program_v2_compact_receipt.json`](results/sokoban_effect_program_v2_compact_receipt.json)
- Compact V13 result：
  [`results/webshop_sokoban_effect_transfer_v13_summary.json`](results/webshop_sokoban_effect_transfer_v13_summary.json)

Full local receipts remain under `runs/webshop_sokoban_effect_transfer_v13/`; they are intentionally excluded
from Git because they contain 192 verbose trajectories and model completions. The compact result binds their
run-summary and final-report file hashes.

## Independent replication V1: positive direction, strict gate not replicated

After the original V13 result, a prospective replication froze the next 32
contiguous goals (`146–177`) before reading any goal text or outcome. The
source artifact, target MLP, six conditions, runner, model, thresholds, and
12-step budget were unchanged. All `32 × 6 = 192` receipts completed with zero
failures or retries and valid matched initial states.

| condition | strict success | mean reward | mean steps |
|---|---:|---:|---:|
| target-only | 9/32 | .5651 | 7.875 |
| target-native myopic | 3/32 | .6589 | 4.313 |
| authentic Sokoban + target | **14/32** | **.7057** | **7.688** |
| commit availability | 3/32 | .6068 | 6.813 |
| inverted effect | 3/32 | .6068 | 6.813 |
| POSITION prior | 8/32 | .5182 | 8.188 |

Authentic versus target-only is `6W/1L/25T`, a strict-success improvement of
`+15.625pp` and mean-reward improvement of `+.140625`. Thus the effect direction
replicates and paired wins still exceed losses, but the predeclared two-sided
exact test is `p=.125`, above `.05`. Every other comparator gate passes; the
single failed target-only significance gate makes the frozen route-level status
`REAL_SOKOBAN_TO_WEBSHOP_TRANSFER_V13_NOT_VALIDATED` for this replication.

This is not evidence that the mechanism has zero value, nor permission to pool
the two 32-task sets post hoc. It says the original large effect was smaller and
less stable on the independent range. The four-domain aggregate protocol may
use only its separately frozen directional estimand. The machine-readable
replication record is
[`results/webshop_sokoban_effect_replication_v1_summary.json`](results/webshop_sokoban_effect_replication_v1_summary.json).
