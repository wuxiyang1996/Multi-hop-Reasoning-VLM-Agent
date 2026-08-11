# Typed Multi-Source V3: source task expansion

## 结论

可以增加 source tasks，而且这一次增加的是互补的可干预结构，不是更多同类 rollout。
V3 使用三个真实 MiniGrid tasks：

| Source task | 必需效果 | 对 ALFWorld 的结构类比 |
|---|---|---|
| DoorKey 5x5 | `BIND`, `MUTATE` | 先获得工具，再改变对象状态 |
| UnlockPickup | `BIND`, `MUTATE` | 获得工具、解锁、再获得目标物 |
| PutNear | `BIND`, `RELATE` | 获得对象，再建立对象间空间关系 |

这不是 target action mapping。迁移对象只包含 typed effect graph；ALFWorld 仍必须使用
target-native neural grounding 来识别当前观察中哪个 native action 可能产生相应效果。

## 为什么不继续只扩 Thunder/Columns rollouts

旧日志的主要失败是高层 skill label 与 reward cadence 绑定：同一个
`COMMIT/POSITION` 在不同游戏没有稳定 outcome。增加相同类型 rollout 只会更精确地学习这种
domain fingerprint。V3 改为从同一 simulator state fork 所有 native actions，再由 before/after
state delta 机械标注：

```text
BIND   := carrier empty -> carrier bound
MUTATE := 同一对象的 open/locked property 改变
RELATE := carried entity 被释放，并新增与另一对象的 adjacency
```

action name、mission text 和 reward 都不参与 effect label。

## Frozen source protocol

配置为 `configs/typed_multisource_v3.json`。每个 task 使用 seeds `0/1/2`，通过稳定哈希分到
development / heldout / qualification。对每个 BFS 状态，每个 action fork 都重新执行：

```text
reset(seed) -> exact prefix replay -> one native intervention
```

source gate 要求每个 `task × split × required effect` cell 同时具有：

1. 至少一个 effect-positive state；
2. 同一 state 中至少一个不产生该 effect 的 native-action control；
3. replay state hash 零 mismatch。

## 实际结果

运行探索了 523 个去重 simulator states，并冻结 126 个 matched forks：

| 指标 | 结果 |
|---|---:|
| Gate cells | 18 / 18 passed |
| Replay mismatch | 0 |
| `BIND` positive forks | 9 |
| `MUTATE` positive forks | 6 |
| `RELATE` positive forks | 3 |
| `POSITION` positive forks | 3 |
| 官方正奖励 forks | 3 |

三个官方正奖励都来自 PutNear 的 `RELATE` transition，在 development / qualification /
heldout 分别为 `0.70 / 0.76 / 0.82`。同一个冻结命令重复运行得到 byte-identical report。

冻结 IR SHA256：

```text
1a19d0a7f9849b61fbb2178384fbf9fc9d9118617f2b2653f1a731b45ed8557e
```

IR 中得到：

```text
BIND --[CARRIER_BOUND]--> MUTATE
  support: DoorKey, UnlockPickup

BIND --[CARRIER_BOUND]--> RELATE
  support: PutNear
```

IR 不包含 source action ordinal、environment ID、坐标或 mission；它只有 symbolic routing
authority。具体 target action 始终由 target-native neural probes 提议，official evaluator
仍独占 termination authority。

## 这证明了什么、还没证明什么

本轮证明：可以从更多真实 source tasks 中稳定提取 intervention-grounded typed structure，且
`BIND -> MUTATE` 在两个 source tasks 上复现。它修复了旧方案最关键的“高层 label 无稳定行为语义”
问题。

本轮尚未证明：该 IR 能提高 ALFWorld / WebShop / visual reasoning 的 success rate。当前每个
task/split/effect 只有一个 positive group，source gate 是可运行性与因果可辨识性 gate，不是统计
显著性声明。`BIND -> RELATE` 也只有一个 source task 支持，应先加入 MiniWorld PutNext 复现，
再作为强 transfer edge 使用。

## 下一步：真正的 target transfer test

按旧 V2 结果，不能直接让 IR 覆盖 target policy。下一轮只测试 target-native candidate ranking：

```text
target-only neural grounding
vs
typed IR + target-native neural grounding
vs
edge-permuted IR + same grounding
vs
wrong-guard IR + same grounding
```

第一目标应继续用 ALFWorld，因为它原生包含 acquire / transform / place 三类 stage，可以直接
检验两个 edge 是否降低 stage-order error。主指标必须是 paired episode success；若 authentic IR
不能同时超过 target-only 和两个 causal controls，就不能声称 transferable 或 success-rate gain。

## 复现

使用已有 `gymv` conda 环境：

```bash
PYTHONPATH=src /fs/gamma-projects/vlm-robot/conda/envs/gymv/bin/python \
  scripts/collect_typed_source_tasks_v3.py \
  --output runs/typed_multisource_v3/report.json \
  --seeds 0 1 2

pytest -q tests/test_typed_source_tasks.py
ruff check src/motif_transfer/typed_source_tasks.py \
  scripts/collect_typed_source_tasks_v3.py tests/test_typed_source_tasks.py
```

紧凑、可提交结果见 `docs/results/typed_multisource_v3_summary.json`；完整 raw receipts 保留在
ignored `runs/typed_multisource_v3/report.json`。
