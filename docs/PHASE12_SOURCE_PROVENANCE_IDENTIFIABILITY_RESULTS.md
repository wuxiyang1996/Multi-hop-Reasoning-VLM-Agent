# Phase 12：source provenance 与 program content 的可识别性

## 最终答案

现在可以明确回答：

> **在 execution 阶段有效的是 symbolic program 的结构内容，而不是它来自 source game 这一 provenance。**
> 如果人工在 target 侧写出 extensionally isomorphic controller，并给它完全相同的 frozen
> target-native neural grounding，它可以得到完全相同的行为。

这不等于 source intervention 没有价值。当前被实验证明的 source value 是上游的
**program acquisition / information value**：source-only intervention rollouts 自动归纳出该结构；在
DiscoveryWorld 的 matched induction curve 中，source program 使用 0 条完整 target 成功轨迹，而
target-only induction 在 `K=0` abstain、到 `K=1` 才恢复同构 program。因此目前量到的 source
information value 是 **1 条完整、有序、成功的 target trajectory**。

正确 claim 应该写成：

> Source interventions provide an automatic upstream mechanism for acquiring and selecting a
> reusable symbolic program. Once identical program content is supplied, its source provenance is
> neither necessary nor identifiable from target behavior.

## 为什么必须做新对照

V13/V14 的 `target_native_recurrent_relation_ceiling` 与 authentic arm 虽然结果相同，但实现仍经过
同一个 source artifact interface，并共享 source-program runtime。它只能算 oracle ceiling，不能作为
“完全独立 target-written controller”的充分证据。

V15 新增了 `target_written_isomorphic_multiplicity_controller`：

- 直接从 ALFWorld target interface 写出 recurrence：首个 relation 后继续 acquisition、保持同一个
  receptacle handle、完成剩余 relation；
- concrete action 仍只由相同 frozen target-native neural grounder 和 causal-effect head 排序；
- 不读取 source artifact path、hash、operator ID、source identity 或 source confirmation；
- runner 传入一个 tripwire mapping，任何 lookup、iteration 或 length inspection 都会立即中止；
- 不复用 unified source authorization route，也不会产生 source admission。

因此它是一个真正的 source-blind target-written specification control，而不是给 authentic arm 换名。

## ALFWorld 全量结果

V15 在已经消耗的完整 V13+V14 ALFWorld population 上复跑 target-written controller。这个诊断不是新的
prospective success experiment；prospective success 证据仍来自原 V13/V14。V15 的问题是更窄的：
给定完全相同 program content，source provenance 是否还改变 behavior？

| Population | Tasks | Raw | Source-induced | Target-written | Exact action/state traces |
|---|---:|---:|---:|---:|---:|
| V13 formal reserve | 24 | 13 | 20 | 20 | 24/24 |
| V14 remaining population | 21 | 11 | 18 | 18 | 21/21 |
| Combined | **45** | **24** | **38** | **38** | **45/45** |

不只是 success count 相同：

- 45/45 完整 action traces 逐步一致；
- 45/45 before/after state hashes、effect receipts、步数与 official outcome 一致；
- source artifact read attempts 为 0；
- target-written source admissions 为 0；
- source-induced 与 target-written 都比 raw 多成功 14 个任务。

失败轨迹也逐步一致，所以这不是两个不同 policy 偶然获得相同 aggregate success rate，而是 frozen
target interface 上的 extensional policy equivalence。

完整报告位于 `runs/alfworld_target_written_provenance_v15/report.json`；独立组合审计位于
[`results/source_provenance_identifiability_v15.json`](results/source_provenance_identifiability_v15.json)。

## source intervention 仍然贡献了什么

### 1. Source-only acquisition 是真实的

使用的 acquisition program 满足：

- induction authority 为 `SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY`；
- `target_data_read=false`；
- `named_controller_template_used=false`，没有手工 `EXPLORE/BACKTRACK/COMMIT` 模板；
- held-out source trajectory conformance 通过；
- shuffled-effect binding 被拒绝；
- source artifact 与 fresh confirmation hashes 一致。

所以 target-written 等价对照并没有推翻“source intervention 学到了结构”；它推翻的是更强、但不合理的
“结构一旦给定后，其 source 出身仍有额外执行魔力”。

### 2. Matched target information budget 有正值

DiscoveryWorld V26 两臂共享完全相同的 185 个 grounding transitions 和 frozen MLP，仅改变 global
program 的来源：

| Program acquisition | Complete ordered target demonstrations | Result |
|---|---:|---|
| source-induced | 0 | program available |
| target-only | 0 | abstain |
| target-only | 1 | recovers isomorphic program |

这量到了 1 条 target demonstration 的替代值。它没有测量人工编写 controller 的小时数，也没有证明
source 优于所有可能的 target-side prior、LLM synthesis 或 human specification。

### 3. 不是任意 generic controller 都能工作

Phase 9 的 catalog 有 11 个 source-induced programs；四个 target interface 匿名选中了 3 个不同
program bodies，wrong-family 在 4/4 route 上都 abstain。四条 route 中破坏 selected program 的
effect/terminal binding 都显著降回 control。因而结论不是“随便写个 target controller 都行”，而是：

> **写出正确的同构 controller 当然可以；困难和可迁移价值在于如何从 source interventions 自动得到并
> 选择正确 content。**

## 统计与可识别性边界

如果两个 controller 对所有 target states 实现同一个函数，它们的 provenance 不可能通过 target
behavior 被识别。继续增加 untouched target tasks 也无法解决这个逻辑上的 non-identifiability；它只会
再次确认 extensional equivalence。

因此后续要估计的不是：

```text
same program + source label  vs  same program + target label
```

而是：

```text
source intervention budget + automatic induction
vs
matched target evidence / synthesis / human specification budget
```

本轮仍保留以下边界：

- V15 是 consumed-population post-hoc diagnostic，不新增 prospective performance claim；
- 当前 target-information value 的量化来自 DiscoveryWorld，并且只量到 1 条完整 trajectory；
- ALFWorld 已证明 structure 对 success 有 +14/45 的 policy utility，但尚未在 ALFWorld 内直接测量
  target-side program acquisition sample complexity；
- human authoring cost、LLM synthesis cost 和更强 target priors 尚未比较；
- 不能声称 source provenance 本身有 causal effect。

## 复现

ALFWorld/TextWorld 需使用兼容的 Python 3.11 环境：

```bash
PY=/fs/gamma-projects/vlm-robot/conda/envs/browsergym/bin/python

$PY scripts/run_alfworld_target_written_provenance_v15.py
PYTHONPATH=src python scripts/audit_source_provenance_identifiability_v15.py
pytest -q \
  tests/test_alfworld_target_written_equivalent.py \
  tests/test_source_provenance_identifiability_v15.py
```

最终判定 gates 全部通过：

1. source-blind target-written exact trace equivalence：通过；
2. program content 对 ALFWorld success 的非零 utility：通过；
3. source-only induction 与 held-out/shuffled controls：通过；
4. matched target `K=0` abstention、`K=1` recovery：通过；
5. multi-program content specificity 与 wrong-family abstention：通过。
