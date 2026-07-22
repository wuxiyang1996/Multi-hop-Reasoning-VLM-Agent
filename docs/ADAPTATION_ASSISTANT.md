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

## 无启发式边界

这里的“无 heuristic”是可审计的代码约束，不是要求 Agent 没有先验：

- Agent 可以从 observation/transition 提议 segment、binding、prediction 和 verdict；这些都是
  `UNTRUSTED_AGENT_CLAIM`，不会自动成为事实。
- Harness 不包含游戏动作名称、target predicate ontology、source→target 映射、embedding 相似度、
  手写 task decomposition 或“合理性”打分。
- Harness 只处理整数 index、exact native-action membership、receipt/hash、图引用、重复运行的集合交集、
  official outcome 和预先固定的调用/replan budget。
- raw example 上跨重复运行稳定是 binding 准入条件。full-action alpha rename 只是 attribution control：
  同样稳定则标记 `GENERIC_STRUCTURAL`；raw 稳定但 alpha 不稳定则标记
  `TARGET_GROUNDED_PROVISIONAL`，仍须在线验证；raw 自身不稳定才拒绝。
- one-shot adaptation 与 evaluation 分开。adaptation 产出带 hash 的 frozen binding artifact；evaluation
  默认只加载它，禁止利用 held-out episode 重新生成 binding。
- 多个 viable bindings 不按 hash、confidence 或“合理性”选一个。Harness 只接受 unanimous verdict；传给
  Decision Agent 的语义字段只保留各 candidate response 的 exact intersection，其余置空。

这避免两个相反错误：既不把 target action 名泄漏伪装成领域无关 backbone，也不把所有依赖 one-shot
target evidence 的合法 grounding 一律删除。

## Source reasoning observer

旧 `source_agent_v2` 曾把 planning Agent 的 proposal 放回 Decision prompt，并在 proposal 有效时直接执行
它的 action；这种数据不能证明“从原 skill-loaded policy 提取 backbone”。live shadow observer 虽然闭环
可用，但与 source policy 共享生成服务时会改变请求调度和后续 policy context。实测加入 exact-request seed
后，Columns/Thunder 的 matched action/reward sequence 仍不完全相同；因此不能用 non-inferiority tolerance
把它当成非干扰数据。

正式 source evidence 改用两阶段 offline shadow annotation：

1. 原 9B + game LoRA Decision Agent 在没有 observer 的条件下完成 rollout，整条 trajectory 先冻结为
   immutable source transition receipts；
2. shadow prediction 再逐 receipt 运行，只读 before observation、已冻结 action ordinal 和原 Decision
   reasoning；该调用的代码路径不提供 after observation；
3. prediction 写入后，verifier 才读取 after observation、official reward 和同一 transition receipt；
4. annotation 只能写入独立 artifact，不能回写 source trajectory、skill bank、Decision prompt 或环境；
5. Harness 只验证 exact schema、action ordinal、source lineage 和可重算 hash。Agent abstain/schema gap 原样
   保留，不做 parser repair。

live shadow 的权限边界仍用于在线 target Harness：

1. 原 9B + game LoRA Decision Agent 始终独立选择 native action；
2. action 已冻结但尚未执行时，Motif/Harness Agent 只能预测该 action 的 observable delta 或 abstain；
3. Harness 要求 prediction 中的 action index 与已冻结 Decision action 精确相同，任何改写都作为 gap；
4. 环境执行后才运行 post-transition verifier，并要求完整 before/action/after receipt；
5. matched seed/initial-observation 审计只报告 exact action/reward equality 与 coverage，不在 Harness 内写入
   人工 non-inferiority threshold。

Source overlay 还会同时移除旧 runner 的 `GAME_CRITICAL_ACTIONS` 提示和 Tetris “best-first/action 1”提示；
这两者是人工游戏知识，不能成为 backbone evidence。Tetris macro action 本身仍保留，因为它是 wrapper
公开的 native action interface，而不是 Harness 的语义规则。reference 与 observer 两边必须使用同一个
no-human-hints overlay receipt，否则等价审计拒绝运行。

早期 “observer 从完整 native list 独立提出 1–3 actions” 协议在 Columns/Thunder 的 48 个 step 上得到
0 个 schema-valid cycle：4K context 被长 action interface 占满，9B 只续写了 action list。禁止用裁剪 action
list 或 parser heuristic 修补；现改为上述 post-decision/pre-transition shadow prediction。

实际调试结果进一步限定了协议：completion-mode 即使缩短输入仍会续写 source observation；改用同一
Qwen3.5-9B 的 chat template 后，pre-transition prediction 在 Columns/Thunder 均达到 3/3 schema-valid。
post verifier 也切到相同的 raw-observation + chat JSON 协议后，两游戏均得到 3/3 完整 closed-loop cycles。
Columns 的三个 verdict 为 1 `SUPPORTED` + 2 `REFUTED`，Thunder 为 3 `REFUTED`；这些负证据全部保留。
chat template 只约束通用消息边界/JSON 接口，不加入游戏 ontology 或 source→target mapping。
冻结机械汇总见 [`results/source_shadow_prediction_v7_smoke.json`](results/source_shadow_prediction_v7_smoke.json)。

完整 matched run 随后证明 live observer 不能作为主路径：Columns 和 Thunder 虽都有 24/24 schema-valid
closed-loop cycle，但 reference/observer 的 action 与 reward sequence 均非 exact。改用 immutable-trajectory
offline annotation 后，v5 对两个游戏都覆盖 24/24 source step，所有非排除 receipt 审计通过；Columns 得到
5 valid、17 abstain、2 malformed，Thunder 得到 8 valid、14 abstain、2 malformed。valid verdict 分别为
Columns 3 `SUPPORTED` + 2 `REFUTED`，Thunder 7 `SUPPORTED` + 1 `REFUTED`。这证明协议可运行，但也表明
9B shadow coverage 仍低，尚不能称为 transferable motif evidence。冻结汇总见
[`results/source_offline_shadow_v5_summary.json`](results/source_offline_shadow_v5_summary.json)。

因此后续只能从“冻结 source receipt + valid offline shadow annotation”提议 motif，而不是从第二个 Agent
自己生成的一套游戏轨迹提取。若 coverage 太低，正确结论是 source evidence 不足，而不是放宽规则或
手写语义补齐。

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
