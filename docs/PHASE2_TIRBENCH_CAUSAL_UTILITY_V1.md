# Phase 2：TIRBench neural-symbolic causal utility

## 结论

TIRBench 不需要再打开一个新 reserve。对已经 prospectively frozen、独立执行的 48 个
single-image maze tasks 做 raw-receipt 复核后，统一 Phase-2 结论为：

> **`PHASE2_TIRBENCH_CAUSAL_UTILITY_VALIDATED`**

| Condition | Success |
|---|---:|
| raw target-only neural answer | 19/48 |
| authentic source topology + target grounding | **41/48** |
| alpha-renamed authentic | 41/48 |
| direction-permuted source control | 19/48 |
| endpoint-only target control | 31/48 |
| path-length marginal control | 14/48 |

Authentic 相对 raw 为 **22W / 0L / 26T**，exact two-sided paired sign-test
`p = 4.76837158203125e-7`，success rate 提升 45.83 percentage points。相对最强的
endpoint-only control 仍为 10W / 0L，`p = .001953125`。

独立 audit 从 48 个 task receipts 和 288 个 condition traces 重算全部 success、paired
statistics 与 self-hash，**14/14 gates passed**：

```text
audit status  = PHASE2_TIRBENCH_V1_INDEPENDENT_AUDIT_PASSED
audit sha256  = d8175e3a4b432e72726a8a161f0d9ea1b3957dd38af607358fc1067efb3d4c49
compact sha256= 63f58b9fd092ceb01e107209d3d176218c0d473fe4cb69a34291911d0a823484
report sha256 = d5acf80ecca5d6f7f24c4934e0b0c4f7fc3aafc9c692400c450243dac13ba036
```

## 为什么可以归入 Phase 2

这不是把旧的 mechanism cell 改名。原 replication 在执行前已经冻结：

- 48 个从未分配过的 held-out maze IDs；
- source topology artifact 与 fresh source confirmation；
- GPT-4.1-mini target-native direction/color binder；
- target-native pixel graph executor；
- raw、authentic、alpha rename、direction permutation、endpoint-only 和 path-length controls；
- success endpoint、zero-negative-transfer 和 strict-control gates。

因此原实验本身就是 matched causal utility experiment；此前只是没有与 WebShop、
ALFWorld 的 Phase-2 命名和 audit schema 对齐。本轮没有重跑、换样本、调参或降低 gate，
只对 immutable raw receipts 做独立重算并生成 portable compact evidence。

## Neural-symbolic 边界

```text
target neural binder
  -> R/L/U/D semantics + start/goal color channels
target-native pixels
  -> anonymous maze graph + candidate path execution receipts
source symbolic controller
  -> execute / refute / verify unique goal / commit / abstain
target evaluator (after all arms)
  -> correctness only
```

source artifact 不含 TIR 坐标、颜色名、方向 delta、answer label、gold answer 或 target
path length。`alpha_renamed_authentic` 与 authentic 48/48 action outcome 等价，说明结构不依赖
symbol name；direction-permuted control 回落到 raw，说明不是任意 symbolic scaffold 都有效。

## Claim boundary

成立的是：

- source-derived anonymous topology execution structure 在 TIR single-image maze 上显著提高
  success；
- target-native neural grounding、source symbolic program 和 native executor 都实际参与；
- wrong/permuted、endpoint-only 和 marginal controls 不能解释完整增益。

不成立的是：

- 已经证明 RefCOCO、rotation、visual search 或整个 TIR 都有同样增益；
- 已经证明 source provenance 是唯一的——target-authored isomorphic controller 仍可能匹配；
- 已经估计六个 Phase-1 game lineage 各自的独立 success-rate effect。24/24 Phase-1 audit
  证明六 lineage 都能 online 驱动 TIR mechanism；这里的 utility estimate 来自独立合格的
  Sokoban topology artifact。两项证据不能合并成“六个 source 各自都提高 48-task success”。

## Evidence 与复核

- frozen config：`configs/tir_maze_topology_replication_v1_frozen.json`
- raw report：`runs/tir_maze_topology_replication_v1/heldout_report.json`
- raw receipts：`runs/tir_maze_topology_replication_v1/heldout_receipts.json`
- compact evidence：`docs/results/phase2_tirbench_utility_v1_compact.json`
- independent audit：`docs/results/phase2_tirbench_utility_v1_audit.json`

```bash
export PYTHONPATH=src:.
python scripts/audit_phase2_tirbench_utility_v1.py
```
