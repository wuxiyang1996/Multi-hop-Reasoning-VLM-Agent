# 从旧 Phase-1 skill 提取内部 reasoning backbone

## 当前结论

旧 skill、checkpoint 和 rollout 可以复用，但三者的证据权限不同：

- `selected_skill_id` 只证明某段 transition 在执行哪个旧 skill；
- skill bank 的 protocol、description 和 execution hint 只是 **不可信假设**；
- observation、native actions、executed action、official transition 和 replay receipt 才是可验证证据。

旧实现按 `selected_skill_id` 的变化切 maximal run，再让 Agent 组合完整 run。该产物现统一解释为 **SkillUsageGraph**：它描述 skill 使用与切换，不能证明单个 skill 内部的 reasoning backbone。

新实现位于 `src/motif_transfer/skill_internal.py`。它不会重新发现或重命名旧 skill，而是按每个真实 skill 聚合全部执行：

```text
historical skill-bank entry (untrusted)
                  +
all receipt-grounded executions of the same skill
                  ↓
GroundedSkillExecutionSet
                  ↓ Motif Agent proposes internal spans/nodes/edges
SkillInternalGraph candidate
                  ↓ deterministic receipt/topology/control audit
backbone-eligible candidate or fail-closed rejection
```

`selected_skill_id` 只决定 execution membership，不决定内部 node boundary。Motif Agent 可以在同一 skill span 中提议任意连续 node occurrence，但不能创造 transition。Harness 将 occurrence 重新绑定到精确 receipt，并拒绝越界、重叠、无相邻证据的 edge 和通用线性图。

## 防止“动作序列 = reasoning backbone”

存在 branch、cycle 或 self-loop 仍不够。Strider `COMMIT/ATTACK` 的真实 pilot 产生了一个 receipt-valid 的 `RIGHT → RIGHT/B` 图；它具有 self-loop，却完全由 source action identity 解释。因此新增 action-identity control：

```text
若每个 node 都对应唯一 source action，且不同 node 仅由 action identity 分开
→ ACTION_IDENTITY_EXPLAINS_NODE_PARTITION
→ backbone_eligible = false
```

这不是手工语义 ontology；它是对“模型是否只重新编码具体动作”的信息控制。后续还必须加入 receipt-only、alpha-renamed action、shuffled topology 和 skill-off controls。只有 authentic graph 在 qualification/held-out 上超出这些 matched controls，才可能进入 `SOURCE_SUPPORTED`。

## 固定数据切分

在任何 Agent proposal 前，以 episode 为单位固定：

- discovery：只用于提出 node、edge 和 evidence request；
- qualification：用于冻结候选和补充 intervention receipt；
- held-out：只用于最终 recurrence/prediction/control 检验。

六游戏长轨迹目前机械导入得到：13 个真实执行 skill、131 个 maximal executions、1769 个 transition；13 个 skill 都能找到旧 bank sidecar，11 个具有完整三分数据。摘要见 `docs/results/phase1_skill_execution_sets_v1_summary.json`。

## Agent 选择证据，而不是人写 predicate

当现有轨迹无法区分候选图时，Motif Agent 可以输出：

```json
{
  "execution_index": 0,
  "source_offset": 3,
  "alternative_action_ordinal": 2,
  "question": "untrusted hypothesis to distinguish"
}
```

Harness 只允许它引用 discovery execution 中真实存在的 offset 和 native alternative action，并把请求记录为 `REQUESTED_NOT_OBSERVED`。只有环境完成 snapshot/prefix replay 后，它才能升级成 observed intervention receipt。Agent 的 question 不具有证据权限。

## 真实 pilot 结果

使用冻结的 GPT-5-mini 做了三个 source-only proposal smoke：

1. Candy Crush `COMMIT/CLEAR`：模型提出三个语言节点，但没有 receipt-supported edge，拒绝。
2. Strider `COMMIT/POSITION`：模型返回空 motif；这比编造图更符合 fail-closed 协议。
3. Strider `COMMIT/ATTACK`：得到 receipt-valid 非线性图，但节点完全等价于 `RIGHT` 和 `B` 动作类别，被 action-identity control 排除。

因此当前状态仍是：**0 个 backbone-eligible motif，0 个 `SOURCE_SUPPORTED` motif**。这不是负迁移结论，而是说明现有旧日志尚未证明 skill 内部 reasoning structure。

## 下一验收门槛

下一阶段不应直接扩大 ALFWorld 数量，而应依次完成：

1. 对候选运行同预算 receipt-only、alpha-renamed、action-only、shuffled 和 skill-off proposal/control；
2. 让 Agent 对无法区分的 graph hypotheses 提出最小 replay request；
3. 在 qualification 上冻结 graph，不再修改 schema/prompt；
4. 在 held-out 上测 topology recurrence、path prediction 和 termination/recovery prediction；
5. 只把 source-specific grounding 删除后的 graph projection 交给 target one-shot partial binding；
6. 最终用 matched target-only、generic graph、shuffled graph 和 other-game graph 证明 target adaptation cost 的增量下降。

在第 4 步通过前，任何图只能叫 `CANDIDATE`，不能称为已提取的 transferable reasoning backbone。
