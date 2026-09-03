# Phase 10：DiscoveryWorld Normal 的 source information value 与在线迁移

## 结论

DiscoveryWorld Proteomics Normal 现在通过了一个新的 prospective、fresh-seed、在线
neural-symbolic transfer protocol。结果不是“source 让一个弱 target policy 变强”，而是更窄、
也更可靠的两点：

1. 游戏 source-only rollouts 归纳出的 `acquisition → positive binding → goal relation` program，
   在不读取完整 target 成功轨迹的条件下，可以由 target-native MLP grounder 绑定到 Normal；
2. 相同 target grounding data 下，target-only program induction 至少需要 1 条完整、有序、成功的
   target trajectory 才能恢复同构 program。source condition 的预算是 0 条，因此 measured source
   information value 是 **1 complete target demonstration**。

正式在线结果：

| Condition | Success | Abstaining tasks | 解释 |
|---|---:|---:|---|
| authentic source program | **24/24** | 0/24 | 24 条路径都完成 acquisition→binding→relation |
| neural-only | **24/24** | 0/24 | 同一个 source-blind target survey，无 symbolic monitor |
| source-permuted | **0/24** | 24/24 | 交换 binding/relation 后全部 fail closed |

Authentic 对 source-permuted 是 `24W/0L`，paired one-sided sign-test
`p = 2^-24 = 5.96e-8`。Authentic 对 neural-only 是 `0W/0L/24T`，所以 **没有 incremental
success-rate gain**。这个零差异必须保留在论文 claim 中。

独立机器审计见
[`results/discoveryworld_normal_source_transfer_v26_audit.json`](results/discoveryworld_normal_source_transfer_v26_audit.json)，
qualification、冻结 MLP 和 formal summary 分别见：

- [`results/discoveryworld_normal_source_value_v26_qualification.json`](results/discoveryworld_normal_source_value_v26_qualification.json)
- [`results/discoveryworld_normal_source_value_v26_grounder.json`](results/discoveryworld_normal_source_value_v26_grounder.json)
- [`results/discoveryworld_normal_source_transfer_v26_formal.json`](results/discoveryworld_normal_source_transfer_v26_formal.json)

## V25 为什么失败，V26 修了什么

V25 只在 final DROP fork 上测试，development 只有 2 tasks、1 matched fork；target-native 已经
1/1 成功，authentic 反而 0/1。更根本的问题是 binder 把任务关系写成 `same_location`，但实际
成功条件是 flag 与 statue **cardinally adjacent at distance 1**。因此 V25 正确保持 formal
reserve 关闭。

V26 没有修改 source artifact。它把 intervention 提前并扩大到完整 phase：

```text
target-native meter / measurement acquisition
              ↓
5 species measured + anomaly unique + flag held
              ↓
correct statue cardinally adjacent at distance 1
              ↓
exactly one positive relation intervention is executable
              ↓
DROP flag → goal relation update
```

Source program 仍然来自 Sokoban source-only `(state, action, effect, next_state)` induction：

- two repeatable anonymous acquisition operator families；
- one `POSITIVE_EFFECT_BINDING/CANDIDATE_CARDINALITY` onset；
- one `ENTITY_GOAL_RELATION/RELATION_COVERAGE` update；
- binding 不可观测、candidate cardinality 错误、effect 不符合时 fail closed。

Target 侧只增加 domain-native parsing 与 grounding：proteomics measurement memory、flag inventory、
statue species normalization、cardinal adjacency，以及 action candidate features。没有把 Sokoban action、
坐标或 entity name 搬进 DiscoveryWorld。

## Fresh qualification

Seed role 在任何 V26 reset 前冻结：

- development：301–308；
- qualification：309–316；
- untouched formal：401–424。

五个历史工作树的 assigned seed exposure scan 为 0。Development 与 qualification collector 都是
source-blind deterministic target survey；source program 不参与数据采集。结果：

| Split | Tasks | Official success | Transitions |
|---|---:|---:|---:|
| development | 8 | 8/8 | 185 |
| qualification | 8 | 8/8 | 197 |

MLP 只读取 public target facts、candidate action 和 survey memory；inference schema 明确禁止
`evaluation`、`official_success` 和 scorecard 字段。Fresh qualification 上四个 role 全部为
precision=recall=1.0，总 exact accuracy 为 197/197。Authentic 与 neural-predicted trace 均为
8/8 conforming；rotated trace 与 binding/relation-permuted trace 均为 0/8。

采集时还发现并修复了一个真实 grounding bug：teleport location token 使用
`Statue of a echojelly`，public object observation 使用 `statue of an echojelly`。修复只按 statue
type 与 species suffix 做 target-native normalization，不读取 formal success。

## Matched source-information experiment

两臂共享完全相同的 185 个 local grounding transitions 和冻结 MLP；变化的只有 global symbolic
program 的来源：

- source-induced：0 条 complete ordered successful target trajectories；
- target-only：预算曲线为 `K ∈ {0,1,2,4,8}`。

结果：K=0 时 target-only 按预注册规则 abstain；K=1 时第一次恢复与 source 相同的
two-acquisition-types + binding→relation program，并在 8/8 qualification trajectories 上有 support。
因此可以说 source program 替代 1 条完整 target demonstration。

不能从这个结果说“program 的 source provenance 是必要的”。如果人工写一个 extensionally identical
target program，它在行为上不可辨识；这里把它明确标为 oracle ceiling，而不是 learned baseline。

## Formal authority chain

Formal 在任何 401–424 reset 前固定以下内容哈希：source artifact、held-out source confirmation、
qualification report、MLP、runner 和 grounding module。每个 candidate action 的 authority chain 是：

```text
source-blind target survey proposes native action
        ↓
frozen target-native MLP predicts anonymous source role
        ↓
frozen source phase monitor authorizes or abstains
        ↓
one official DiscoveryWorld native action executes
```

独立审计重新验证：

- 72/72 episode self-hashes 与 72/72 monitor self-hashes；
- 三臂每个 seed 的 initial policy/audit state 完全相同；
- authentic 与 neural-only 的 native action traces 24/24 完全相同；
- authentic 24/24 trace 符合 source program、0 abstention、final phase 全是 `DONE`；
- permuted 24/24 的第一次 denial 都是 grounded `BINDING`；
- 72/72 policy runtime 没有 scorecard exposure。

## 正确 claim 与剩余边界

现在可以写：

> A source-induced anonymous acquisition–binding–relation program transferred online from game
> rollouts to fresh DiscoveryWorld Proteomics Normal through a frozen target-native neural
> grounder. It achieved 24/24 success, matched neural-only performance, rejected a destructive
> source-program permutation on 24/24 tasks, and replaced one complete ordered target
> demonstration in matched program induction.

仍不能写：

- source 提高了 Normal success rate：实际是 24/24 vs 24/24；
- provenance 可由 extensional behavior 单独识别；
- source program 自己选择 proteomics actions。当前 source-blind target survey 仍负责 candidate
  generation，source IR 是在线 structural monitor；
- 任意 game program 可迁移到任意 target；
- video MDP transfer 已因此成立。

最有价值的后续不是制造 heuristic headroom，而是在自然不满分的 target-native proposer 上保持同一
冻结 IR，预注册 error-recovery intervention，测试 source monitor 是否带来真实 success gain。

## 复现

```bash
PY=/fs/gamma-projects/vlm-robot/conda/bin/python

PYTHONPATH=src $PY scripts/qualify_discoveryworld_normal_source_value_v26.py \
  --output-dir runs/discoveryworld_normal_source_value_v26_qualification_artifacts

PYTHONPATH=src:/fs/gamma-projects/vlm-robot/discoveryworld-official $PY \
  scripts/run_discoveryworld_normal_source_transfer_v26.py \
  --config configs/discoveryworld_normal_source_transfer_v26_formal.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --output-dir runs/discoveryworld_normal_source_transfer_v26_formal

PYTHONPATH=src $PY scripts/audit_discoveryworld_normal_source_transfer_v26.py \
  --output docs/results/discoveryworld_normal_source_transfer_v26_audit.json

PYTHONPATH=src $PY -m pytest -q \
  tests/test_discoveryworld_normal_transfer.py \
  tests/test_qualify_discoveryworld_normal_source_value_v26.py \
  tests/test_audit_discoveryworld_normal_source_transfer_v26.py
```
