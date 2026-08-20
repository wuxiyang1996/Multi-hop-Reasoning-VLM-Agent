# Phase 17：V28 prospective target reserve

## 结论

Phase 16 最重要的剩余证据边界已经关闭：V28 从 fresh official-Tetris interventions 自动归纳的同一个
匿名 cyclic identity program，已在 **冻结之后才生成、qualification 通过之后才打开** 的新 MiniGrid
target population 上获得正向 formal result。

机器状态为：

```text
PROSPECTIVE_V28_TARGET_RESERVE_VALIDATED
```

Formal reserve 的核心结果：

| Condition | Native success |
|---|---:|
| V28 source-induced program | **48/48** |
| alpha-renamed source | **48/48** |
| target-written isomorphic ceiling | **48/48** |
| target-only neural direct policy（64 development labels） | 16/48 |
| copy-effect relation control | 26/48 |
| fixed-token control | 12/48 |
| shuffled effect-binding control | 13/48 |

Source 对 neural-only direct 为 **32 wins / 0 losses / 16 ties**，exact two-sided
`p=4.6566e-10`。但 target-written isomorphic 同样为 48/48，所以结论仍然是：

> 造成 execution gain 的是 source interventions 学到的正确 program content，不是 source provenance。
> Source 的价值在于提供了不依赖完整 target trajectory 的结构 acquisition route。

## 1. 为什么不用耗尽的 TIR rotation

TIR 的 75 个 rotation items 已被历史 V1/V2 protocol 分配。重新使用它们只能得到 retrospective bridge，
不能为 V28 提供 prospective target evidence。V31 因此使用新的 target family：

```text
official Tetris intervention forks
  → COMPOSE(PROBE_EFFECT, RECOVERY_EFFECT) == IDENTITY
  → target-native MiniGrid visual grounding
  → anonymous recovery macro
  → official terminated-on-goal success
```

Target 是 `MiniGrid-Empty-Random-6x6-v0` 上定义的 orientation-recovery MDP。它使用官方环境 dynamics
和 goal termination，但 recovery protocol 是本项目定义的，因此不冒充官方 MiniGrid benchmark task。

## 2. Target MDP 与统一 IR

每个 task 在 reset 时先生成一个到 native goal 的 BFS action suffix，然后施加 7–14 个左右转 probe。
四个候选 recovery macros 的 order-four effect 通过每个 seed 的 SHA-256 permutation 匿名绑定到
`A/B/C/D`。正确 recovery 必须恢复 reset 时 heading，否则同一个 suffix 通常无法到达 goal。

Neural grounder 只处理七幅 rendered panels：

- `I/P`：probe 前后 agent orientation；
- `C0`：独立 calibration state；
- `A/B/C/D`：四个匿名 macros 在 calibration state 上的 visual effect。

它输出的只有：

```text
probe_effect ∈ Z4
candidate_effect[token] ∈ Z4
```

V28 program 不读取 pixels、MiniGrid action name、goal、reward、source identity 或 target success；它只选择唯一满足：

```text
COMPOSE(probe_effect, candidate_effect) == IDENTITY
```

的 native token。零个或多个匹配都 abstain。独立 target-written ceiling 直接写同一个 target-side equation，
不调用 source artifact。

## 3. Grounder acquisition 与 baseline

最初两轮排除 pilot 使用 GPT-4.1-mini 读取全图/zoom panels，但分别只有 35/42 与 37/42 panel
direction 正确，而且错误仍可能高置信度。因此没有降低 threshold 或在正式 seeds 上追 prompt/model；这些
`750xxx` pilots 从所有 gate 排除，总 provider cost 约 `$0.0068624`。

正式协议改为 development-only target-native MLP：

- 从 pixels 定位红色 agent，方向保持的 crop 作为 neural input；
- 64 development tasks、448 orientation labels；
- 不读取 target success/reward 或完整 trajectory；
- 不读取 source program/identity；
- orientation confidence threshold `0.90` 在 development 前冻结；
- qualification/formal 只加载同一个 hash-bound artifact 做 inference。

同一 development split 还训练一个不使用 symbolic IR 的 direct MLP。它得到 64 个正确 recovery-token
labels，比 source route 多 64 个 target-side relation labels。它在 training tasks 上是 64/64，但在 fresh
qualification 为 5/24、formal 为 16/48。这是固定 architecture/data budget 下的 compositional generalization
contrast，不声称所有 neural policy 都无法学习模四 composition。

## 4. 冻结与开启顺序

在任何 `760xxx` task render 或 outcome 生成前，V31 固定：

- development：64 seeds；
- qualification：24 seeds；
- formal reserve：48 seeds；
- grounder architecture、random state、confidence threshold；
- source、neural-only、target-written 和三种 destructive controls；
- coverage/accuracy/success/negative-transfer/significance gates；
- runner、source program、MiniGrid implementation 与 grounder code hashes。

Stage 顺序及结果：

| Stage | Grounding | Source | Neural-only | Result |
|---|---:|---:|---:|---|
| development | 448/448 panels，64/64 exact bindings | 64/64 | 64/64 | passed |
| qualification | 168/168 panels，24/24 exact bindings | 24/24 | 5/24 | passed |
| formal reserve | 336/336 panels，48/48 exact bindings | 48/48 | 16/48 | passed |

Qualification report 通过并绑定 grounder artifact 后，runner 才允许打开 formal reserve。

## 5. 因果解释

V31 同时排除了三种过强解释：

1. **不是任何 action 都行。** fixed、shuffled 和错误 relation 显著低于 authentic；
2. **不是 source identity feature。** Neural inference 明确不接收 source identity，alpha rename 完全不变；
3. **不是只有 source 才能执行。** 独立 target-written isomorphic 与 source 都为 48/48。

最准确的结论是：

```text
source interventions acquire exact compositional content
  + target-native neural perception binds native effects
  → prospective target execution gain
```

不是：

```text
source provenance itself causes target success
```

## 6. 仍然保留的边界

- 这是一个新 target population，但不是官方 MiniGrid leaderboard task；
- target grounder 使用 64 development tasks 的方向监督，不是 zero-shot perception；
- neural-only 差距只适用于冻结的 MLP architecture 与 64-label budget；
- source 与 target simulator transition 的经济成本仍不可直接比较；
- 三个 program families 加一个 prospective target 不等于 arbitrary-domain universality；
- TIR rotation 本身仍没有新的 items；V31 是新的 oriented-navigation target，而不是重开 TIR。

## 复现

```bash
python scripts/analyze_minigrid_orientation_target_v31.py

python -m pytest -q \
  tests/test_minigrid_orientation_recovery.py \
  tests/test_minigrid_neural_grounder.py \
  tests/test_analyze_minigrid_orientation_target_v31.py
```

机器报告：

- [`results/minigrid_orientation_target_v31_summary.json`](results/minigrid_orientation_target_v31_summary.json)；
- `runs/minigrid_orientation_target_v31/{development,qualification,formal_reserve}_report.json`；
- `runs/minigrid_orientation_target_v31/grounder_artifact.json`。
