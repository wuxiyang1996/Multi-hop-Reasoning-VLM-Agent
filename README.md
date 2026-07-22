# Two-Agent Motif Transfer Harness

当前 ALFWorld one-shot online smoke、失败审计和下一阶段 gates 见
[`docs/ALFWORLD_ONE_SHOT_STATUS.md`](docs/ALFWORLD_ONE_SHOT_STATUS.md)。

这是从历史仓库拆出的最小研究核心。仓库只保留两个有模型判断能力的角色：

1. **Decision Agent**：读取目标环境的原生 observation/action space，并且是唯一可以选择环境动作的 Agent。
2. **Motif/Harness Agent**：从真实 rollout receipt 提议行为 motif，在一个 adaptation example 上初始化跨域 binding，并在在线交互中给出 `ADMIT / REPLAN / ABSTAIN` 建议。它不能生成、替换或执行目标动作。

确定性 Harness 不是第三个 Agent。它只验证结构、receipt、原生动作成员关系、实验身份和 official outcome。Agent 的自然语言解释从不被当作真值。

## 研究目标

本项目不假设模型缺少推理能力，也不试图把一套游戏 policy 直接搬到新领域。我们的目标是：从六种游戏的 skill-conditioned / skill-disabled rollout 中提取可复现的控制结构，把它作为一个 **online adaptation assistant**，帮助固定的 Decision Agent 用更少 target examples 和交互快速适应异构任务。

这里迁移的是 receipt-grounded 的行为图及其可检验 adaptation hypothesis，而不是游戏 action 名称、手写 ontology、完整 policy 或通用提示词。Decision Agent 始终保留自己的推理能力和目标领域 action authority；Motif/Harness Agent 只帮助它判断当前缺少什么信息、哪个假设值得验证、何时继续、replan 或 abstain。

更准确的研究问题是：

> 在模型已经具备通用推理能力时，来自 source experience 的可验证 motif，能否降低新领域的 adaptation cost？

因此主要 claim 不是“恢复并迁移模型内部 reasoning backbone”，而是：

```text
source experience
  → adaptation hypotheses
  → online verification
  → lower target adaptation cost
```

一个 target example 只会创建 `TARGET_PROVISIONAL` binding。后续每一步都在线验证；证据不足时 abstain 或退回 target-only。只有 matched controls 的 official outcome 能把候选更新为正迁移、负迁移、generic-only 或 inconclusive。

“快速适应”必须用 examples-to-success、environment steps、无效/重复动作、token/tool cost 和 negative-transfer recovery latency 衡量，而不能只报告最终成功率。

## 快速运行

```bash
python -m pip install -e '.[test]'
pytest -q
python examples/smoke_two_agent.py
```

## 目录

- `src/motif_transfer/decision_agent.py`：唯一的动作 authority。
- `src/motif_transfer/motif_harness_agent.py`：motif、binding 和在线审查接口。
- `src/motif_transfer/harness.py`：fail-closed 验证与 matched evaluation。
- `src/motif_transfer/runtime.py`：双 Agent 交互循环。
- `src/motif_transfer/legacy_import.py`：把旧 mega-skill 读成只读 lineage/baseline。
- `docs/ARCHITECTURE.md`：权限边界和数据流。
- `docs/ADAPTATION_ASSISTANT.md`：快速适应助手的完整定位与证据标准。
- `docs/EXPERIMENT_PROTOCOL.md`：六游戏到远域的实验设计。
- `docs/OLD_RESULTS.md`：旧 checkpoint、rollout 和 mega-skill 的保留方式。
- `docs/IMPLEMENTATION_STATUS.md`：当前机械审计、正在运行的采集和尚未授权的 claim。

## 旧游戏模型如何进入新 Harness

旧 Decision Agent 不改接口：skill-on 仍执行 `skill_selection LoRA → selected game skill → action_taking LoRA → native action`，skill-off 只移除 skill context。Motif/Harness Agent 不参与 source action。

rollout 完成后，确定性代码在精确记录的 `selected_skill_id` 变化点切出连续片段。不能使用 `selected_skill_sha256` 做边界，因为旧系统的该字段包含随 observation 变化的动态 guidance，同一 skill 也会产生不同哈希。旧 `segment` LoRA 只做它原来训练过的 all-candidate skill ranking；它不生成 action proposal，也不直接生成 motif 图。独立 Motif/Harness Agent 只能组合完整的机械片段，并从 hash-bound ranking receipts、transition receipts 和 replay receipts 中提出候选图；matched controls 再判断是否存在 source-derived 增量价值。

历史实现仍完整保存在 Git parent `948f64a` 以及原始工作副本中；本分支不复制 checkpoint、rollout、Slurm log 或生成结果。
