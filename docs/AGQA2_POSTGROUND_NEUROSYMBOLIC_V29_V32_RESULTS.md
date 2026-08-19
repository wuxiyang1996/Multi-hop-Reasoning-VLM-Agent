# AGQA 2.0 post-ground neural-symbolic transfer：V29–V32

> **V33–V40 temporal replication update（2026-08-18）：** 第二个 Candy Crush
> arity-2 recurrent temporal program 在 fresh AGQA `BEFORE/AFTER` 上为
> `68/100 vs 65/100`、5W/2L、`p=0.2265625`，未通过预注册 gates。因此本文的 V32
> `QUERY_OBJECT` 结论仍成立，但不能扩展为 multi-operator 或 broad AGQA transfer。见
> [`AGQA2_MULTI_OPERATOR_NEUROSYMBOLIC_V33_V40_RESULTS.md`](AGQA2_MULTI_OPERATOR_NEUROSYMBOLIC_V33_V40_RESULTS.md)。

## 结论

AGQA atomic `QUERY_OBJECT` 的 game-to-video transfer 现在通过了 fresh、video-disjoint、
预注册的 success-rate 验证：

| V32 fresh formal arm | correct / 120 | 相对 target-native |
|---|---:|---:|
| target-native neural-only | 34 | — |
| source-induced unified harness | **40** | **+6** |
| source effect-shuffled | 34 | 0 |
| handwritten generic 3-view ceiling | 40 | +6 |
| target-written equivalent ceiling | 40 | +6 |

primary paired endpoint 为 **6 wins / 0 losses / 114 ties**，exact one-sided
`p = 0.015625`。source program 在 56/120 个 task 上授权，其余 64 个 task fail closed
并保留 target-native prediction。全部预注册 gates 通过；OpenRouter reported cost 为
`$1.056773768`，低于 `$1.30` cap。

机器可复核摘要见
[`results/agqa2_postground_neurosymbolic_v29_v32_summary.json`](results/agqa2_postground_neurosymbolic_v29_v32_summary.json)，
完整 formal report 为
`runs/agqa2_postground_v32_formal/report.json`（report SHA-256
`d186266bf91f5d80b1774774725c11b7d8098a0bf5ca034e36f76cfb3a0dfc54`）。

## 原来的 negative 为什么出现

V28 的 `14 vs 36` 不是一个干净的 source-transfer endpoint，而是三个错误叠加后的结果：

1. **batch gate 和 state applicability 混在一起。** `minimum_decisive_accuracy`
   失败后，一个 dataset-level boolean 关闭全部 120 个 source executions；它没有允许
   source program 对每个 state 单独 select/abstain。
2. **abstention fallback 错了。** source abstain 后旧 evaluator 回到较弱的 direct-only
   prediction，而不是保留同一个 target-native two-ontology baseline。因此 base gate 一失败，
   paired comparison 自动退化成 direct `14` 对 target-native `36`。
3. **symbolic binding 层次错了。** V29 strict adapter 一度把三个 raw neural votes 当成三个
   symbolic candidate bindings。实际上 raw votes 是 perception evidence；target-native neural
   grounder 应先把它们解析为 zero/one/many bindings，source-induced symbolic program 再执行
   transition、terminal 和 abstention rules。

修复没有降低旧的 `0.75` threshold，也没有从 V28 formal 选择 relation 白名单。新 runtime
将四个 authority 明确分开：

```text
source (state, action, effect, next_state)
  -> induced recurrent ENTITY_GOAL_RELATION program
  -> target-native neural evidence and binding resolution
  -> source transition / terminal / abstention execution
  -> calibrated unified authorization
  -> target-native object executor
```

source program 仍然是 Sokoban source-only artifact：

- authority：`SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY`；
- operator：typed `UPDATE(ENTITY_GOAL_RELATION, arity=2)`；
- transition：positive relation-coverage update，`ONE_OR_MORE`；
- terminal：`entity_goal_relation_coverage == 1.0`；
- abstention：zero binding、multiple bindings、non-positive delta、unobservable terminal；
- held-out source confirmation：63/63 unique source-success selections，93 authentic effect
  bindings，0 shuffled-effect bindings；
- 没有 `EXPLORE/BACKTRACK/COMMIT` named template。

## V29–V32 顺序与污染边界

### V29：修复旧 composition，development only

V25 + V28 的 150 个已消费、video-disjoint rows 被明确退役成 adaptation data。执行
source-derived strict binding rule，并把 abstention fallback 改为 target-native 后，结果从旧报告的
负值变成 `52 vs 48`，4 wins / 0 losses。它只用于未来 route calibration，不是 confirmatory
claim。

### V30：outcome-unread legacy prospective audit

V27 在读取任何 gold 前因单个 JSON transport failure 停在 119/120。V30 在打开这 119 个
official answers 前冻结 evaluator、hashes 和 gates，然后零 provider-call 评估。strict raw-vote
adapter 为 `34 vs 31`，3 wins / 0 losses，但 `p=0.125`，因此按原 gate正确标为
`FORMAL_NOT_QUALIFIED`。它没有被重标成成功。

V30 同时显示 matched 3-view consensus 为 39/119，而 strict adapter 为 34/119。这定位了最后一个
层次错误：sensor disagreement 不能直接等价于 multiple symbolic entity bindings。

### V31：post-ground adapter development

V25、V28 和已失败的 V30 共 269 个 rows 全部作为 consumed adaptation data。修正后由
target-native grounder 先解析 candidate set，source program 再检查 binding：

- source-induced：93/269；
- target-native：79/269；
- 15 wins / 1 loss，net `+14`；
- utility/authenticity posterior lower bound：`0.7498755481`；
- source 与 handwritten/target-written ceiling：269/269 predictions 相同。

V31 只冻结未来 route calibration，不提供 formal claim。

### V32：fresh formal

在任何 V32 provider/gold call 前：

- 从 official AGQA balanced test metadata 固定 hash-ranked pool；
- 排除所有 prior-config 和已存在 local videos；
- 下载、hash、decode 120 个新 Charades videos；
- 三个 relation groups 各 40；
- 冻结 source、adapter、executor、calibration、controls、gates 和 `$1.30` cost cap；
- 一次性运行，无 post-outcome rule、threshold、sample 或 model change。

结果为 `AGQA2_POSTGROUND_V32_FORMAL_QUALIFIED`，所有 gates 为 true。

## unified harness 到底做了什么

V32 不是把 game action 输出到视频。它迁移的是 intervention-grounded symbolic structure：

1. source-only learner 给出 content-addressed recurrent relation program；
2. Qwen isolated-relation view 与两个 Gemini ontology views 是 target-native neural grounding；
3. target grounder 将 perception evidence 解析为候选 relation-object binding；
4. source program 只在 effect、terminal、binding cardinality 和 calibration 同时成立时授权；
5. `AGQAObjectExecutor` 是唯一能输出 target-native object label 的组件；
6. source abstain 时保留 target-native arm，而不是回退到更弱 comparator。

旧 base report 仍显示 `grounder_qualified=false`，因为旧的全批次 absolute gate 要求至少 60 个
decisions 且 decisive accuracy ≥0.75；V32 实际为 56 decisions、36 correct（0.643）。V32 没有偷偷
修改该历史 gate，而是将 **receipt integrity、row applicability、directional utility** 分开评估。
正式 claim 来自冻结的 paired source-vs-target endpoint，而不是旧 base status。

## 可以和不可以声称什么

现在可以声称：

> 在 fresh、balanced、atomic AGQA `QUERY_OBJECT` 子集上，从 Sokoban interventions 自动归纳的
> recurrent relation program，与 target-native neural grounder 和统一 fail-closed harness 组合后，
> 相对 matched target-native baseline 将 correct 从 34/120 提高到 40/120；effect-shuffled source
> 不产生增益。

不能声称：

- source program 优于 handwritten generic controller；二者在 V32 是 120/120 相同；
- Sokoban domain-specific object semantics 被迁移；迁移的是 anonymous relation dynamics；
- full AGQA、compound temporal questions 或任意 natural-video reasoning 已解决；
- 旧 V28 negative report 可以删除或改写成成功；它保留为发现 harness bug 的失败记录。

## 关键实现与结果

- strict diagnosis adapter：`src/motif_transfer/agqa_goal_relation_transfer.py`
- post-ground adapter：`src/motif_transfer/agqa_postground_relation_transfer.py`
- outcome-blind prediction composition：
  `src/motif_transfer/agqa_postground_relation_evaluation.py`
- V31 development audit：`scripts/audit_agqa2_postground_v31_development.py`
- V32 freezer：`scripts/freeze_agqa2_postground_v32_formal.py`
- V32 collector/evaluator：`scripts/collect_agqa2_postground_v32_formal.py`
- V32 frozen config：`configs/agqa2_postground_v32_formal.json`
- V32 preregistration：`configs/agqa2_postground_v32_formal_preregistration.json`
- V32 formal report：`runs/agqa2_postground_v32_formal/report.json`
