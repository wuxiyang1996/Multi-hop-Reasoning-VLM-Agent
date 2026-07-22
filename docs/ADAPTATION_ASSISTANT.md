# Reasoning Motif + Harness 作为快速适应助手

## 一句话定义

从 Phase-1 游戏经验中提取 receipt-grounded control motifs，用一个 target example 初始化可证伪 binding，并在后续真实交互中持续验证，帮助一个固定的 Decision Agent 更快适应新任务或新领域。

## 它不是什么

- 不是把游戏动作映射到 Web、OS、视觉或视频动作；
- 不是替 Decision Agent 生成 action；
- 不是声称模型本身不会规划、验证或 replan；
- 不是把 `observe → act → verify` 通用提示包装成 source skill；
- 不是在 test episode 上继续训练模型并称之为 transfer。

## 在线协议

```text
Phase-1 receipt-grounded motifs
              ↓
one target adaptation example
              ↓
provisional binding version space
              ↓
Motif Agent: information need / expected delta / failure route
              ↓ advisory only
Decision Agent: target-native reasoning and action
              ↓
official transition receipt
              ↓
Harness: retain / eliminate / abstain / target-only fallback
```

一个建议可以描述需要验证的结构，但不能给出 target action。例如：

```json
{
  "current_role": "evidence_acquisition",
  "open_hypotheses": ["candidate_1", "candidate_2"],
  "information_need": "obtain evidence that separates the candidates",
  "expected_transition": "at least one candidate becomes unsupported",
  "failure_route": "replan if the observation does not update the version space",
  "termination_test": "one viable candidate remains",
  "verdict": "ADMIT"
}
```

其中所有语义字段均由 Agent 提议并保持 untrusted；Harness 只检查它们引用的 observation、proposal 和 transition receipt 是否存在，以及后续 official outcome 是否支持该 treatment。

## Motif 何时有意义

模型本身可能已经知道“验证后再提交”。因此 motif 必须提供比 generic reasoning 更具体、可干预的控制结构，例如验证顺序、branch topology、失败后回到哪个决策节点以及停止条件。

只有满足下列证据链时，才称为 source-derived adaptation value：

1. motif 在 source skill-on/off matched intervention 中可归因；
2. target authentic treatment 超过 target-only；
3. authentic 超过等预算 generic protocol；
4. shuffled topology 和 other-source motif 不能解释提升；
5. 改善体现在 success、sample efficiency、交互效率或成本中的至少一个预注册指标。

## 可能的结论

- **Positive adaptation**：更少 examples/steps/cost 达到相同或更高成功率；
- **Generic-only**：通用提示与 authentic motif 同样有效；
- **Redundant**：Decision Agent 自己已经恢复了相同结构；
- **Negative transfer**：motif 限制或误导 Decision Agent；
- **Not applicable**：target interaction 不提供 motif 所需的可观察反馈；
- **Inconclusive**：样本、controls 或 identity 不完整。

研究价值不要求所有 motif、模型和领域都正迁移。可靠地选择何时介入、何时退出，以及快速检测 negative transfer，本身就是 adaptation assistant 的核心能力。
