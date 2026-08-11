# Typed Multi-Source V4: cross-engine relation replication

## 结论

V4 已完成并通过 source gate。它没有改写冻结的 V3，而是新增真实 3D source task
`MiniWorld-PutNext-v0`，用来独立检验 V3 中只有 MiniGrid PutNear 支持的
`BIND -> RELATE` edge。

结果为：

```text
BIND --[CARRIER_BOUND]--> RELATE
  MiniGrid PutNear       PASS
  MiniWorld PutNext 3D  PASS
```

两个任务使用不同 simulator family、不同状态空间和不同导航动力学。迁移 IR 不包含两者的动作、
坐标、oracle policy 或环境 ID。

## MiniWorld intervention protocol

MiniWorld 是连续 3D 环境，不能复用 MiniGrid 的离散 BFS。V4 因此使用 source-only state oracle
寻找两个 fork point，但 oracle 只能产生 source native navigation prefix，不能进入 IR：

```text
empty carrier, red box in pickup range
  -> fork all 8 native actions

red box carried and geometrically near yellow box
  -> fork all 8 native actions
```

每个 fork 都重新执行 `reset(seed) -> exact prefix replay -> one intervention`。`BIND` 和
`RELATE` 仍由 before/after state delta 标注；reward 不参与标签。对于 carried object，空间关系只在
object 被放回 world 后才算 grounded，因此 drop transition 会产生新的 `RELATE` fact。

## Frozen gates

配置见 `configs/typed_multisource_v4.json`。除 V3 的 matched-control gate 外，V4 新增两个
预先机械化的条件：

1. edge replication gate：`BIND -> RELATE` 至少由两个 source tasks、两个 simulator
   families 支持；
2. effect value gate：`RELATE` 必须在两个 tasks 的 development / qualification / heldout
   六个 cells 中都同时获得官方正奖励并 terminal。

IR 仍只从 development source lineage 归纳，qualification 和 heldout 只验证冻结结构。

## 实际结果

| 指标 | 结果 |
|---|---:|
| Source tasks | 4 |
| Simulator families | 2 |
| 去重/规划状态 | 955 |
| Matched forks | 174 |
| Typed-effect gate | 24 / 24 passed |
| Replay mismatch | 0 |
| `BIND` positive forks | 12 |
| `MUTATE` positive forks | 6 |
| `RELATE` positive forks | 6 |
| Official positive + terminal forks | 6 |
| Edge replication gate | passed |
| Effect value gate | 6 / 6 passed |

MiniWorld PutNext 的 `RELATE` fork 在 qualification / development / heldout 上的官方奖励分别为
`0.8512 / 0.9344 / 0.8664`。完整 V4 重复运行得到 byte-identical report。

冻结 IR：

```text
SHA256 5a7a48a6cc083a9b8caffbcb02fa721971a28d2052f276c847bb9296641942ea

BIND --[CARRIER_BOUND]--> MUTATE
  support: DoorKey, UnlockPickup

BIND --[CARRIER_BOUND]--> RELATE
  support: PutNear, PutNext 3D
```

## 当前 claim 边界

本轮支持：

- typed effects 可以跨离散 2D 与连续 3D source environments 机械提取；
- `BIND -> RELATE` 不是单一 MiniGrid task 的偶然结构；
- `RELATE` 在两个引擎中都与官方任务成功一致；
- source action、坐标和 oracle 不需要进入 transfer artifact。

本轮仍不支持：

- IR 已提高 ALFWorld、WebShop 或 visual reasoning success rate；
- source oracle/navigation policy 可以迁移；
- 不经 target-native grounding 直接执行 source structure。

## 下一步

source coverage 已足够，不应继续无上限增加游戏。下一实验应冻结 V4 IR，只在 ALFWorld
adaptation split 上训练 target-native `BIND/MUTATE/RELATE` applicability 与 completion probes，
然后在 qualification episodes 做 paired comparison：

```text
TARGET_ONLY
AUTHENTIC_TYPED_IR
EDGE_PERMUTED_IR
WRONG_GUARD_IR
```

IR 只能重排或过滤 target-native candidates，不能生成 target action。主指标是 episode success，
次指标是 stage-order violation 和 repeated-wrong-effect loops。Authentic 必须超过 target-only 和
两个 causal controls，才允许读取 target heldout。

## 复现

```bash
PYTHONPATH=src PYGLET_HEADLESS=1 \
  /fs/gamma-projects/vlm-robot/conda/envs/gymv/bin/python \
  scripts/collect_typed_source_tasks_v4.py \
  --output runs/typed_multisource_v4/report.json

pytest -q tests/test_typed_source_tasks.py tests/test_real_source_interventions.py
ruff check src/motif_transfer/typed_source_tasks.py \
  scripts/collect_typed_source_tasks_v3.py \
  scripts/collect_typed_source_tasks_v4.py \
  tests/test_typed_source_tasks.py
```

紧凑结果见 `docs/results/typed_multisource_v4_summary.json`；完整 receipts 位于 ignored
`runs/typed_multisource_v4/`。
