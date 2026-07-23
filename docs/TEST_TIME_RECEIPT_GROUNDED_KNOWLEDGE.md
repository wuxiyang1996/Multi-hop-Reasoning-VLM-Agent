# Test-Time Reasoning with Receipt-Grounded Source Knowledge

## 1. Primary claim

本项目不再把主问题表述为 `skill transfer`。主问题冻结为：

> We study whether test-time reasoning in a target MDP can benefit from
> source-derived, receipt-grounded knowledge extracted from heterogeneous
> source MDPs.

中文：

> 我们研究目标 MDP 中的测试时推理，是否能从异构 source MDP 中提取出的、由 receipt
> 支撑的知识中受益。

Source 不迁移 policy 或 skill。它只提供关于控制规律、失败信号、验证流程和适用边界的
不可信知识。目标 Agent 可以在测试时推理中试用这些知识，但每一次使用都必须被 live target
receipt 验证、拒绝或标记为 inconclusive。

推荐标题是：

> **Can Test-Time Reasoning Benefit from Receipt-Grounded Knowledge across
> Heterogeneous MDPs?**

更短版本是：

> **Test-Time Reasoning with Transferred Receipt-Grounded Knowledge**

## 2. Transfer object

实验明确区分三个不同强度的 source object：

```text
source MDP rollouts
    ├── raw source receipts
    ├── weak receipt-grounded control knowledge       <- primary object
    └── exact source graph topology                   <- strong secondary probe
```

### 2.1 Raw source receipts

只提供冻结 source evidence，不提供 Agent 总结出的控制陈述。它用于排除“更多上下文或
trajectory retrieval 本身就有效”的解释。

### 2.2 Weak receipt-grounded control knowledge

每条知识包含：

- `CONTROL_REGULARITY`：source 中重复出现的控制规律假设；
- `FAILURE_SIGNATURE`：什么 observable evidence 曾与失败或假设反驳共同出现；
- `VERIFICATION_ROUTINE`：什么 evidence 必须在继续或提交前获得；
- `APPLICABILITY_BOUNDARY`：什么 evidence 应关闭 source intervention。

这些 role 是 Harness 的权限/schema，不是 source→target predicate ontology。每条自然语言
hypothesis 都是 untrusted；它只有 source receipt provenance 和跨 episode recurrence，
没有语义真值权限。

Target one-shot binding 不包含：

- source node 到 target span 的完整 partition；
- source edge 到 target boundary 的映射；
- target action；
- `COLLECT→TAKE` 一类人工映射；
- embedding/名称相似度产生的 alignment。

它只能产生 target-native、可在线检验的：

- information need；
- expected evidence；
- verification test；
- recovery/source-off condition；
- termination test。

### 2.3 Exact topology prior

旧协议要求 target transitions 覆盖全部 source nodes，并对齐全部 source edges。这个条件继续
保留，但降为强假设和 ablation，不再代表项目的主 claim。如果 weak prior 有效而 exact topology
无效，结论是 source knowledge 有用，但 source graph 本身不能迁移。

## 3. Online authority

```text
Decision Agent selects a target-native proposal
        ↓
Harness tests an untrusted source-derived hypothesis
        ↓
ADMIT / REPLAN / ABSTAIN
        ↓
environment executes only a Decision-Agent action
        ↓
live target receipt
        ↓
SUPPORTED / REFUTED / INCONCLUSIVE
        ↓
retain / revise / source-off / target-only fallback
```

Harness 永远不能生成、替换、排序或执行 target action。`REFUTED` 立即从 version space 删除
对应 hypothesis；候选分歧、schema failure、receipt mismatch 或证据不足都 fail closed。

## 4. Source qualification

Source knowledge 不能只由 Agent 写出来。每条 clause 必须：

1. 引用真实 source transition/replay receipts；
2. 至少跨两个 discovery episode 重现；
3. 候选选择不能读取 qualification/held-out；
4. 在 qualification 与 held-out 上有 frozen predictive support；
5. authentic source 必须与 skill-off/random、shuffled evidence 和 other-game controls 分离；
6. 自然语言解释、模型 confidence 和 action 名称不能作为 qualification truth。

当前代码中的 `ReceiptGroundedKnowledge` 与 `audit_receipt_grounded_knowledge()` 只负责 hash、
lineage、receipt existence 和 recurrence。它们不会判断自然语言 hypothesis 是否“听起来合理”。

## 5. Frozen target conditions

主实验至少包含：

| Condition | 检验内容 |
|---|---|
| `target_only` | 没有 source context |
| `generic_reasoning` | matched token budget 的通用验证建议 |
| `source_receipts_only` | 原始 source evidence，无 induced knowledge |
| `authentic_weak_control_prior` | 主 treatment |
| `shuffled_evidence_prior` | 保留形式，破坏 statement/evidence 对应 |
| `other_game_control_prior` | 排除任意 source context 效应 |
| `exact_topology_prior` | 检验完整 source graph 是否提供额外价值 |

所有条件共享 Decision model、task IDs、seed、environment/tool budget、adaptation examples 和
Harness token budget。主指标是 examples-to-success AUC、official score、environment/tool
steps、invalid/repeated actions、verification precision、negative-transfer recovery latency
和总成本。

## 6. Attribution rules

只有同时满足以下条件，才能称为 source-derived knowledge benefit：

```text
authentic weak prior > target-only
authentic weak prior > generic reasoning
authentic weak prior > source receipts only
authentic weak prior > shuffled evidence
authentic weak prior > other-game prior
```

解释边界：

- authentic 与 generic 相同：只能说明通用 test-time reasoning prompt 有效；
- authentic 与 receipts-only 相同：只能说明 retrieval/context 有效；
- authentic 与 shuffled 相同：不能证明 receipt grounding 有因果价值；
- weak prior 有效、exact topology 无效：支持 knowledge benefit，不支持 graph transfer；
- 全部 source 条件无效或有害：报告负迁移与 source-off latency，停止正迁移主张。

## 7. Game-trained Harness / LoRA 的角色

Harness LoRA 不是主张成立的前提。它检验另一个独立问题：

> 游戏 receipts 训练是否让 Harness 更擅长发现可检验假设、校准不确定性并及时 source-off，
> 从而降低新领域的 adaptation cost？

因此先用 GPT-5-mini 等强 frozen Harness 做 feasibility oracle。若强模型在冻结 schema 下仍无法
产生通过 source/target gate 的 knowledge use，就不应先训练更小 Harness 来掩盖机制失败。
只有强 oracle 显示任务可行后，才比较 base、game-receipt-trained 和 verified-target-LoRA。

## 8. Current evidence boundary

截至 2026-07-23：

- fresh Strider/Thunder no-human-hints collection 已完成并通过 receipt integrity audit；
- 这只证明 source evidence 可用于后续 induction，不证明存在有用知识；
- GPT-5-mini 的 ALFWorld exact-topology proposals 两次未通过 span/recurrence gate；
- 该失败现在解释为强 topology hypothesis 失败，不能直接否定更弱的 control-knowledge hypothesis；
- Qwen3.5-9B Harness LoRA 继续保持暂停，直到 frozen weak-prior oracle smoke 给出可行证据。

下一轮应先从 discovery-only source receipts 提议 `ReceiptGroundedKnowledge`，通过 source gates
后在 ALFWorld adaptation split 生成不含 topology alignment 的 target hypothesis，再执行上述
七条件 matched evaluation。
