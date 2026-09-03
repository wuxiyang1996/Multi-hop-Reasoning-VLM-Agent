# Phase 2：四 target neural-symbolic causal utility status

## 当前结论

截至 2026-08-16，四个 target route 都有 source-derived symbolic controller 相对 matched
target-native neural baseline 的 causal utility evidence：

| Target route | N | Authentic | Matched raw | Paired W/L/T | Exact p | Status |
|---|---:|---:|---:|---:|---:|---|
| WebShop constrained search | 32 | 19 | 9 | 10/0/22 | .001953125 | validated |
| ALFWorld selective search | 75 | 57 | 37 | 21/1/53 | 1.0967e-5 | validated |
| DiscoveryWorld Proteomics Easy | 36 | 27 | 12 | 17/2/17 | .000728607 | validated, 1 abstention |
| TIR single-image maze | 48 | 41 | 19 | 22/0/26 | 4.7684e-7 | validated |

合计是 191 target tasks、authentic 144 successes、matched raw 77 successes、70W/3L/118T。
这个 aggregate 只作描述，不计算 pooled p-value：四个实验的 task、controller、eligibility 与
evaluator 不同，不能把 post-hoc pooled statistic 当作新的预注册 hypothesis test。

## 实际是怎样完成 transfer 的

我们没有迁移游戏按键、画面 token、object name、target action 或 raw trajectory，也没有把
“玩 Tetris”一类高层 skill label 塞进 target prompt。实际迁移链是：

```text
source game intervention rollouts
  -> typed state / event / effect artifact
  -> EXPLORE_UNTRIED / BACKTRACK_REPLAN / COMMIT_VERIFY controller
  -> target-native neural candidate and predicate grounding
  -> symbolic applicability, effect guard and attempt ledger
  -> official target-native action and observed outcome
  -> online symbolic-state update
```

其中 neural 与 symbolic 的职责严格分开：

| Component | 负责什么 | 明确不负责什么 |
|---|---|---|
| Source artifact | intervention 后的 state/effect relation、合法 symbolic transition、何时 explore/refute/commit | target object、坐标、ASIN、BID、native action 或答案 |
| Target neural grounder | 从当前 observation 与合法 action space 生成 target-native candidates，并绑定 object/event predicate | 决定跨步 search policy 或伪造 source effect |
| Symbolic executor | 根据 event binding、effect witness 和 attempt ledger 选择 candidate；不适用时 abstain | 产生 target semantics 或绕过 native validator |
| Official environment/evaluator | 执行动作并返回新 observation；在所有 matched arms 完成后计算 outcome | 给 controller 泄漏 gold outcome |

同一个 canonical controller 在不同 target 上有不同的 native realization：WebShop 对商品候选做
explore/backtrack/verified purchase；ALFWorld 对 grounded workflow 做有状态搜索；DiscoveryWorld 在
first-DROP/PUT fork 上要求 positive-effect witness；TIR 则在 target pixel graph 上执行匿名 topology
program。source-native action 从未直接成为 target action。

### Causal identification 是怎样做的

每个 formal manifest 都在 target outcome 前冻结 task、source assignment、controller、horizon、endpoint
与 gate。同一个 task 的 matched arms 共用 target reset、neural grounder、candidate schema、native
validator 和 evaluator，只替换 symbolic controller：

1. `raw`：target-native neural rank-zero 或 myopic choice；
2. `authentic`：正确 source state transition、event/effect binding 与跨步 ledger；
3. `wrong/permuted`：保留 source artifact 和执行成本，但破坏 event/effect binding；
4. `ledger-blind` 或 marginal control：保留局部 rule，但移除关键的跨步或结构信息；
5. `target-native isomorphic ceiling`（适用时）：检查 source artifact 是否精确实例化同构机制。

因此 wrong/permuted control 回落到 raw，排除了“多一段 prompt”“多调用一次模型”或“任意 symbolic
scaffold 都有效”的解释；isomorphic ceiling 与 authentic 相等则同时限制了结论：证明的是 transferred
structure 的 utility，不是 source provenance 的唯一性。

ALFWorld 与 DiscoveryWorld 还把 applicability 当成 controller 的一部分，而不是事后删题。无法表示
双对象 multiplicity 的 ALFWorld tasks，或没有预声明 commit fork 的 DiscoveryWorld task，都按冻结
规则 abstain 并继承 matched raw outcome。所有 start marker、cell receipt、action trace、failure、hash
与 negative-transfer case 均保留，独立 audit 从 raw receipts 重算 paired statistics。

## 这比 24/24 mechanism transfer 多证明了什么

Phase 1 的 `6 source games × 4 targets = 24/24` 证明每个 source lineage 都能在 fresh target
execution 中实际驱动 online symbolic route。Phase 2 进一步加入：

- 多 target tasks，而不是一个 task/cell；
- matched neural-only baseline；
- wrong/permuted、ledger-blind 或其他 symbolic controls；
- paired success endpoint 和 exact test；
- negative-transfer 与 abstention calibration；
- immutable manifest、cell receipts 和 independent audit。

因此现在可以说：

> 在四条已限定 target routes 上，正确的 intervention-grounded symbolic structure 与 target-native
> neural grounding 组合，显著提高 matched target success；错误或缺失的 symbolic structure 不能
> 解释该增益。

## 不能说什么

- 不是 WebShop、ALFWorld、DiscoveryWorld、TIRBench 全 benchmark 的普遍结论；每个结果都只对应
  文档里的具体 route/split/interface。
- 不是六个 game 各自都有 powered independent effect。WebShop、ALFWorld、DiscoveryWorld 平衡绑定
  六 lineage，但它们共享 canonical controller；TIR utility 使用独立 Sokoban topology artifact。
- 不是 source provenance 唯一性。WebShop/ALFWorld 的 target-authored isomorphic ceiling 可精确匹配
  authentic；证明的是 structure utility，不是只有从某个 game 才能得到该结构。
- 不是 zero negative transfer。ALFWorld 有 1 次、DiscoveryWorld 有 2 次 strict losses；选择性
  applicability 把总体风险控制在冻结阈值内。
- 不是 neural grounding 已解决。DiscoveryWorld acquisition 的 schema fallback 尤其严重，下一步
  应提升 target-native acquisition/grounding，而不是继续堆 source labels。

## Evidence map

| Target | Primary document | Independent audit status |
|---|---|---|
| WebShop | `PHASE2_WEBSHOP_CAUSAL_UTILITY_V4.md` | 17/17 gates |
| ALFWorld | `PHASE2_ALFWORLD_SELECTIVE_UTILITY_V3.md` | 15/15 gates |
| DiscoveryWorld | `PHASE2_DISCOVERYWORLD_SELECTIVE_UTILITY_V2.md` | 15/15 gates |
| TIRBench | `PHASE2_TIRBENCH_CAUSAL_UTILITY_V1.md` | 14/14 gates |

## 下一步：Phase 3 source-induced selective transfer

当前最重要的缺口不是 success effect，而是 **program provenance 与 learning**：六个 source artifacts
虽然独立、均在线执行，但共享一个 canonical controller；同时 target-authored isomorphic ceiling 可以
精确匹配 authentic。下一阶段应检验：controller 与 applicability 是否真的能由 source-only
interventions 学出，并在不读取 target outcome 的情况下迁移。

### Primary question

> Can a typed symbolic program and its abstention rule, induced only from source-game interventions,
> improve success on an untouched target reserve over matched neural-only and source-permuted controls,
> without a target-authored task policy?

### 建议的执行顺序

1. **先修 target grounding，不碰 formal transfer outcome。** 在 DiscoveryWorld development tasks 上把
   acquisition schema fallback 从当前 36.5% 降到预注册阈值，并确认 native-action validity；这一步只做
   grounding qualification，不能用 success 选择 symbolic controller。
2. **source-only program induction。** 从 intervention `(state, action, effect, next_state)` 自动归纳 typed
   precondition/effect、transition 和 abstention predicate；冻结 induction code、source split、artifact
   hashes 与复杂度上限。hand-authored canonical controller 只作为 ceiling，不作为新的 authentic arm。
3. **加入 provenance controls。** 至少冻结 `neural-only`、`source-induced authentic`、
   `source-effect shuffled/permuted`、`generic hand-authored scaffold`、`target-native isomorphic ceiling`
   五个 matched arms；额外做 leave-one-source-lineage-out，检查结果是否只是 relineage。
4. **校准 selective applicability。** 在 source/development evidence 上冻结 coverage、precision、selective
   risk 与 abstention threshold；禁止根据 formal target losses 再改 rule。
5. **一次性跑 untouched DiscoveryWorld reserve。** task IDs、eligibility、primary paired endpoint、exact test、
   negative-transfer upper bound、failure gate 和所有 controls 必须先 commit/push；ineligible task 留在
   denominator 中作为 tie，不能换 seed 或延长 horizon。
6. **成功后再做第二 target replication。** 优先 ALFWorld multiplicity 或 TIR 非 maze interface；不要一开始
   同时烧四域成本。第二域应复用完全相同的 induced symbolic IR 和 applicability API，只更换
   target-native grounder。
7. **最后才做 source-specific heterogeneity。** 为六个 lineage 增加样本量，区分 shared structural prior
   与真正 source-specific program effect；当前每 lineage 5--6 tasks 的表不能承担该结论。

### Phase 3 最小成功条件

- formal target outcomes 前冻结全部 artifacts、thresholds、controls 与 gates；
- source-induced authentic 显著优于 matched neural-only；
- authentic 显著或按预注册 margin 优于 shuffled/permuted source；
- abstention coverage 与 discordant negative-transfer risk 同时通过冻结界限；
- generic scaffold 不能解释全部增益；
- independent audit 能从 raw receipts 重建每个 induced rule、online route、outcome 与 paired statistic；
- 失败、ineligible 和 infrastructure restart 均 fail closed，不生成替代 cohort。

如果 Phase 3 通过，claim 才能从“source-qualified symbolic mechanism 可迁移且有 utility”升级为
“symbolic program 与 selective applicability 可由 source interventions 学习并跨域产生 causal utility”。
