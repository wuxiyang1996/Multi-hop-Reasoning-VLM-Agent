# Phase 14–15：matched acquisition cost 与第二个 symbolic program family

> **Phase 16 update（2026-08-19）：**本页末尾列出的四个后续实验均已执行。完整 source fork cost、
> zero-trajectory target LLM baseline、第三个 algebraic family 与 fresh source reserve 见
> [`PHASE16_GAP_CLOSURE_RESULTS.md`](PHASE16_GAP_CLOSURE_RESULTS.md)。下面“还剩什么”保留为当时的
> preregistration/history，不再代表当前执行状态。

## 最终答案

原问题现在可以拆成两个可检验的结论：

1. **执行阶段真正有效的是 program content。** 人工写出的 extensionally isomorphic target
   controller 可以与 source-induced controller 完全等价；source provenance 本身没有额外执行效应。
2. **Source intervention 的可测价值在 program acquisition。** 它能从 source-only
   `(state, action, effect, next_state)` receipts 自动取得正确结构，而不需要完整 target 成功轨迹。

Phase 14 补齐了 ALFWorld 的 source acquisition curve；Phase 15 又在一个结构完全不同的有限程序族上
重复了同样的结论。因此当前结果不能再被解释为“只写了一套 canonical recurrent controller，然后换
source label”。

## Phase 14：ALFWorld matched acquisition audit

### 修复的小样本 induction flaw

旧 relation learner 会在少量 source episodes 上枚举所有成功终点恒定 scalar。这样会把
`entity_cardinality=2` 之类没有被 intervention 改变的偶然常量选成 terminal predicate；主排序需要
32 个 source episodes 才纠正为真正的 `ENTITY_GOAL_RELATION` terminal。

新规则只允许一个 terminal candidate：它必须在每条成功 source macro path 中都被 intervention 以正
方向改变。该规则：

- 不接收 feature name；
- 不读取 target observation、action、outcome 或 identity；
- 不能唯一识别 feature 时 fail closed；
- 之后仍调用原 V3 learner，不手工提供 `EXPLORE/BACKTRACK/COMMIT` 模板。

真实 source 数据上的主排序在 `K=4` 第一次恢复完整 recurrence、binding→relation、terminal 和
abstention structure，并同时通过 63 个 held-out relation episodes 与 69 条 held-out acquisition
trajectories；两个 shuffled-effect binding counts 都为 0。

### 冻结结果

| Evidence | K=0 | 第一次恢复 | Robustness | Held-out controls |
|---|---:|---:|---:|---:|
| Source-only Sokoban interventions | abstain | **K=4** | 64/64 orders；min 1、median 3、max 9 | relation 63/63；acquisition 69/69；shuffled 0 |
| Target-only ALFWorld complete trajectories | abstain | **K=1** | 9/9 eligible demos 分别独立恢复 | 每个 program 支持 11/11 later V14 paths；shuffled/permuted 0 |

主 source prefix 保留了 27 条成功路径 primitive transitions 和 5 条 relation macro transitions。
九条 target demos 的完整 episode 长度为 15–39 transitions；真正参与 post-first-relation induction 的
部分为 10–34 transitions。

这些数字**不是共同成本单位**：source collections 包含 simulator intervention forks，但旧 portable
receipt 没保留每个 fork 的全部 primitive steps；source simulator 与 ALFWorld environment interaction 的
实际成本也不同。因此本实验不声称 source 在统一 sample metric 上比 target 更 sample-efficient。

Phase 13 的 fresh compiler-valid execution 结论保持不变：source-induced 与 target-only K=1 都是
`9/14`，相对 raw 的 `4/14` 都是 `5W/0L/9T`，并且 14/14 action/state/effect traces 完全相同。

Phase 14 的准确解释是：

> An isomorphic target controller is sufficient once its content is supplied. Source interventions
> add acquisition value by recovering that content without a complete target trajectory; source and
> target evidence costs are not asserted to be commensurate.

机器报告：
[`results/alfworld_matched_acquisition_cost_v25.json`](results/alfworld_matched_acquisition_cost_v25.json)。

## Phase 15：第二个 distinct family

第二个实验复用已有 prospective 正结果，但把 acquisition analysis 扩展到 MiniGrid PutNear →
DiscoveryWorld Easy。这个 source program 是一个非 recurrent 的有限匿名 graph-edit sequence：

```text
ADD ENTITY_SLOT
  → REMOVE ENTITY_SLOT
  → target-native terminal check
```

它与 ALFWorld family 的差异不是名称：

| Property | ALFWorld family | PutNear → DiscoveryWorld family |
|---|---|---|
| IR kind | recurrent acquisition/relation program | finite structural-delta sequence |
| Recurrent control | yes | no |
| Core operators | control update, positive binding, relation update | add entity slot, remove entity slot |
| Target composition | complete recovery controller + target handle adapter | reusable ordered subgraph inside a larger target partial order |
| Terminal | relation coverage / remaining relations | target-native outcome check after finite sequence |

### Source acquisition

Portable receipt 包含原 source learner 真正读取的匿名 deltas、generic graph-cardinality guards 和 path
success labels，不包含 source action token，也不读取 target 数据。

- K=0：abstain；
- K=1：learner 找到候选 sequence，但因冻结的 `minimum_success_paths=2` 保持 abstain；
- K=2：恢复与既有 frozen PutNear artifact 相同的 execution normal form；
- source qualification 固定使用 2 个独立 collections；
- 4/4 fresh source success paths 支持 authentic sequence；
- reversed sequence、wrong-family sequence 和 shuffled binding 均为 0。

### Target-only acquisition

DiscoveryWorld target learner 不读取 source artifact。它从 caller-supplied complete target structural
path 归纳 counted partial-order program；source comparison 只在 induction 完成后执行。

- K=0：abstain；
- 三条 development success paths 分别单独作为 K=1：3/3 都恢复相同 semantic partial order；
- 3/3 都包含 `ADD ENTITY_SLOT → REMOVE ENTITY_SLOT` ordered subprogram；
- 每个 K=1 program 都支持 3/3 qualification paths；
- precedence-reversed、reversed-source 和 wrong-family controls 全部为 0 support；
- target program 还学习 target-native observation-relation counts，未复制 source body。

这条 route 的既有 prospective formal utility 为：source-induced `12/12`，neural-only `3/12`，
source-permuted `3/12`；source 对 neural 为 `9W/0L/3T`，exact two-sided `p=.00390625`。Phase 15
只复用该结果作为 utility context，没有打开新 seed 或制造新的 prospective claim。

机器报告：
[`results/put_near_discoveryworld_acquisition_v27.json`](results/put_near_discoveryworld_acquisition_v27.json)。

## 现在排除了什么解释

以下解释已被直接排除：

- **“source label 自带执行收益”**：同构 target-written/target-induced content 与 source trace 完全等价；
- **“任何 controller 都行”**：effect permutation、terminal permutation、precedence reversal 和 wrong-family
  controls 都失效或 fail closed；
- **“只有一套 canonical recurrence”**：第二个 finite graph-edit family 的 IR kind、operator signature、
  recurrence 和 terminal contract 都不同；
- **“K=1 只是挑中一个幸运 demo”**：ALFWorld 9/9、DiscoveryWorld 3/3 eligible demos 分别单独恢复。

仍然不能声称：

- source 优于带正确 domain prior 的人类、LLM synthesis 或 retrieval system；
- source collection 与 target trajectory 具有统一 sample cost；
- source provenance 可以从 extensionally identical target behavior 中识别；
- 两个 family 足以覆盖任意 game、target domain 或 video MDP；
- source 自动发现了 target ontology。Target-native neural grounding 与 domain-specific state variables 仍然必要。

## 当时预注册的后续实验（Phase 16 已执行）

最重要的剩余实验不再是增加同 family tasks，而是：

1. **Matched synthesis baseline**：给 target-only LLM 与 source inducer 相同的模型调用、token、wall-clock
   和 interaction budget，测其能否在没有完整成功轨迹时合成正确 program；
2. **完整 source fork cost**：未来 source collector 必须保留所有 candidate-fork primitive transitions，
   才能形成真正统一的 intervention-cost curve；
3. **第三个 algebraic/topological family**：TIR rotation 很适合，但旧 Tetris compiler 直接计算 inverse，
   尚未达到本轮同等级的 source-only rule induction；应先从 intervention tuples 学出 inverse law，再比较
   target acquisition；
4. **Independent source acquisition reserve**：Phase 14/15 是对既有 consumed receipts 的冻结 retrospective
   audit；可移植复跑已经完成，但新的 prospective source acquisition reserve 仍会增强证据。

## 复现

```bash
python scripts/analyze_alfworld_matched_acquisition_cost_v25.py
python scripts/analyze_put_near_discoveryworld_acquisition_v27.py

python -m pytest -q \
  tests/test_source_goal_relation_causal_budget.py \
  tests/test_analyze_alfworld_matched_acquisition_cost_v25.py \
  tests/test_analyze_put_near_discoveryworld_acquisition_v27.py
```

冻结协议与 portable source receipt：

- `configs/alfworld_matched_acquisition_cost_v25.json`；
- `configs/put_near_discoveryworld_acquisition_v27.json`；
- [`results/put_near_source_induction_receipts_v26.json`](results/put_near_source_induction_receipts_v26.json)。
