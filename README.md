# Two-Agent Motif Transfer Harness

当前 ALFWorld one-shot online smoke、失败审计和下一阶段 gates 见
[`docs/ALFWORLD_ONE_SHOT_STATUS.md`](docs/ALFWORLD_ONE_SHOT_STATUS.md)。
旧 Phase-1 skill 的内部图提取、真实 GPT-5-mini pilot 和当前零通过结论见
[`docs/SKILL_INTERNAL_BACKBONE.md`](docs/SKILL_INTERNAL_BACKBONE.md)。
冻结的 88-call source control matrix 摘要见
[`docs/results/skill_internal_matrix_v1_summary.json`](docs/results/skill_internal_matrix_v1_summary.json)。
game training 的 weight-level reasoning transfer、skill-context effect 与 Harness interaction 计划见
[`docs/GAME_TRAINING_REASONING_TRANSFER.md`](docs/GAME_TRAINING_REASONING_TRANSFER.md)。

这是从历史仓库拆出的最小研究核心。仓库只保留两个有模型判断能力的角色：

1. **Decision Agent**：读取目标环境的原生 observation/action space，并且是唯一可以选择环境动作的 Agent。
2. **Motif/Harness Agent**：从真实 rollout receipt 提议行为 motif，在一个 adaptation example 上初始化跨域 binding，并在在线交互中给出 `ADMIT / REPLAN / ABSTAIN` 建议。它不能生成、替换或执行目标动作。

确定性 Harness 不是第三个 Agent。它只验证结构、receipt、原生动作成员关系、实验身份和 official outcome。Agent 的自然语言解释从不被当作真值。

## 研究目标

本项目不假设模型缺少推理能力，也不试图把一套游戏 policy 直接搬到新领域。我们的目标有两层：
先检验六游戏训练是否让 game-trained Decision Agent 相对 base model 获得可跨域复用的程序性推理能力；
再从 matched training/context interventions 中提取可复现控制结构，把它作为一个
**online adaptation assistant**，帮助对应的 Decision Agent 用更少 target examples 和交互适应异构任务。

这里迁移的是 receipt-grounded 的行为图及其可检验 adaptation hypothesis，而不是游戏 action 名称、手写 ontology、完整 policy 或通用提示词。Decision Agent 始终保留自己的推理能力和目标领域 action authority；Motif/Harness Agent 只帮助它判断当前缺少什么信息、哪个假设值得验证、何时继续、replan 或 abstain。

更准确的研究问题是：

> 游戏训练是否把可复用的程序性推理能力写入模型权重，以及 source-derived Harness 能否在远域中
> 更快、安全地调用这些能力？

当前新增的 source-induction 目标是从旧 skill 的多次真实执行中发现可验证的
skill-internal reasoning motif。这里的 backbone 指外显 rollout 中重复出现、经过
matched controls 与 intervention 支持的控制图；它不声称读取模型不可观测的内部思维。
通过 source qualification 后，它才可作为 adaptation hypothesis 进入下面的在线流程：

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
- `src/motif_transfer/skill_internal.py`：按旧 skill 聚合执行、接收内部图 proposal 并做 fail-closed audit。
- `docs/ARCHITECTURE.md`：权限边界和数据流。
- `docs/ADAPTATION_ASSISTANT.md`：快速适应助手的完整定位与证据标准。
- `docs/EXPERIMENT_PROTOCOL.md`：六游戏到远域的实验设计。
- `docs/OLD_RESULTS.md`：旧 checkpoint、rollout 和 mega-skill 的保留方式。
- `docs/SKILL_INTERNAL_BACKBONE.md`：旧 skill 内部 backbone 的表示、controls 与真实 pilot。
- `docs/IMPLEMENTATION_STATUS.md`：当前机械审计、正在运行的采集和尚未授权的 claim。

## 旧游戏模型如何进入新 Harness

旧 Decision Agent 不改接口：skill-on 仍执行 `skill_selection LoRA → selected game skill → action_taking LoRA → native action`，skill-off 只移除 skill context。Motif/Harness Agent 不参与 source action。

rollout 完成后，确定性代码在精确记录的 `selected_skill_id` 变化点切出连续 execution。不能使用 `selected_skill_sha256` 做边界，因为旧系统的该字段包含随 observation 变化的动态 guidance，同一 skill 也会产生不同哈希。`selected_skill_id` 只决定哪些 transition 属于同一 skill，不能决定内部 motif node。独立 Motif/Harness Agent 可以在 execution 内部提出 receipt-bound spans 和 graph；旧 bank 文本只作为不可信 hypothesis。matched controls、qualification、intervention 和 held-out 共同判断是否存在 source-derived 增量价值。

历史实现仍完整保存在 Git parent `948f64a` 以及原始工作副本中；本分支不复制 checkpoint、rollout、Slurm log 或生成结果。
