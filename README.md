# Two-Agent Test-Time Reasoning Harness

本项目不迁移 skill 或 policy；我们检验 target agent 的 test-time reasoning 是否能从 source
MDP 中抽取的 motif、receipts 和 experiential knowledge 获得可归因收益。新的主协议见
[`docs/TEST_TIME_RECEIPT_GROUNDED_KNOWLEDGE.md`](docs/TEST_TIME_RECEIPT_GROUNDED_KNOWLEDGE.md)。
当前实验账本网页见 [`docs/EXPERIMENT_LEDGER.html`](docs/EXPERIMENT_LEDGER.html)。
跨五个 repo/worktree 的完整项目总账本见
[`docs/PROJECT_WIDE_EXPERIMENT_LEDGER.html`](docs/PROJECT_WIDE_EXPERIMENT_LEDGER.html)。
旧 exact-topology motif binding 继续保留为强假设和 ablation，不再代表主 claim。

## Source-only 9B neural-symbolic transfer（当前结果）

独立于上面的 test-time motif 主协议，我们已经完成 source-intervention symbolic program
transfer 的 9B controller substitution：Qwen3.5-9B LoRA 只使用 source 数据训练，在 WebShop、
ALFWorld、DiscoveryWorld、TIRBench、CLEVRER、AGQA2 六个 benchmark 的冻结 replay 上达到
`2,246/2,246` route exact，并保持 `1,864/1,864` success-critical native actions 等价。

这覆盖六个 benchmark、五个 semantic domain（CLEVRER 与 AGQA2 同属 video understanding）。
正式 success gain 继承自原 content-addressed formal runs，不是新的 live 9B rerun；target-native
grounder/executor 仍然是 domain-specific。论文表、claim boundary 与 hashes 见
[`docs/HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_V3_RESULTS.md`](docs/HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_V3_RESULTS.md)，
机器可读 evidence map 见
[`docs/results/harness_controller_qwen35_9b_six_benchmark_v3.json`](docs/results/harness_controller_qwen35_9b_six_benchmark_v3.json)，
中英文实现网页见
[`docs/neurosymbolic-skill-transfer-implementation-zh.html`](docs/neurosymbolic-skill-transfer-implementation-zh.html) 与
[`docs/neurosymbolic-skill-transfer-implementation-en.html`](docs/neurosymbolic-skill-transfer-implementation-en.html)。

当前 ALFWorld one-shot online smoke、失败审计和下一阶段 gates 见
[`docs/ALFWORLD_ONE_SHOT_STATUS.md`](docs/ALFWORLD_ONE_SHOT_STATUS.md)。
最新 GPT-5-mini weak-knowledge Harness canonical pilot、六条件结果、接口修复和多 seed
复现门槛见
[`docs/ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md`](docs/ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md)。
旧 Phase-1 skill 的内部图提取、真实 GPT-5-mini pilot 和当前零通过结论见
[`docs/SKILL_INTERNAL_BACKBONE.md`](docs/SKILL_INTERNAL_BACKBONE.md)。
冻结的 88-call source control matrix 摘要见
[`docs/results/skill_internal_matrix_v1_summary.json`](docs/results/skill_internal_matrix_v1_summary.json)。
game training 的 weight-level reasoning transfer、skill-context effect 与 Harness interaction 计划见
[`docs/GAME_TRAINING_REASONING_TRANSFER.md`](docs/GAME_TRAINING_REASONING_TRANSFER.md)。
当前 Strider source gate 的 measurement、selection leakage、旧 human-hint contamination 与下一轮停止规则见
[`docs/SOURCE_GATE_FAILURE_DIAGNOSIS.md`](docs/SOURCE_GATE_FAILURE_DIAGNOSIS.md)。
移除 SIV-Bench 后的四域七 Cell 资产、executor、污染与可运行性诊断见
[`docs/TARGET_FEASIBILITY_DIAGNOSIS.md`](docs/TARGET_FEASIBILITY_DIAGNOSIS.md)。
VisualToolBench 官方 rubric judge、single-turn v2 修正和游戏 Motif matched-transfer 协议见
[`docs/VTB_OFFICIAL_MOTIF_TRANSFER_PROTOCOL.md`](docs/VTB_OFFICIAL_MOTIF_TRANSFER_PROTOCOL.md)。
测试时 reasoning motif 的主张边界、相关工作、强 baselines、source gate、target factorial 与停止规则见
[`docs/TEST_TIME_REASONING_MOTIF_PLAN.md`](docs/TEST_TIME_REASONING_MOTIF_PLAN.md)。
新冻结的 ALFWorld/VTB `8/8/24` target rollout splits、数据权限与当前采集状态见
[`docs/TARGET_ROLLOUT_COLLECTION.md`](docs/TARGET_ROLLOUT_COLLECTION.md)。

这是从历史仓库拆出的最小研究核心。仓库只保留两个有模型判断能力的角色：

1. **Decision Agent**：读取目标环境的原生 observation/action space，并且是唯一可以选择环境动作的 Agent。
2. **Motif/Harness Agent**：从真实 rollout receipt 提议行为 motif，在一个 adaptation example 上初始化跨域 binding，并在在线交互中给出 `ADMIT / REPLAN / ABSTAIN` 建议。它不能生成、替换或执行目标动作。

确定性 Harness 不是第三个 Agent。它只验证结构、receipt、原生动作成员关系、实验身份和 official outcome。Agent 的自然语言解释从不被当作真值。

## 研究目标

本项目不假设模型缺少推理能力，也不试图把一套游戏 policy 或 skill 搬到新领域。主目标是从
matched source interventions 中提取 receipt-grounded knowledge，并检验它能否作为测试时
**online reasoning assistant**，帮助冻结的 Decision Agent 用更少 target examples 和交互适应异构任务。
六游戏训练是否让 game-trained Decision Agent 相对 base model 获得可跨域复用的程序性推理能力，
作为独立的 weight-level factorial 检验，不能替代 source-derived knowledge 对 test-time
reasoning 产生增量收益的证据。

Source 只提供关于 control regularities、failure signatures、verification routines 和
applicability boundaries 的不可信假设，而不是游戏 action 名称、手写 ontology、完整 policy
或 target action mapping。Decision Agent 始终保留自己的推理能力和目标领域 action authority；
Harness Agent 只帮助它判断当前缺少什么信息、哪个假设值得验证、何时继续、replan、source-off
或 abstain。每次使用必须由 live target receipt 验证或拒绝。

主研究问题是：

> 目标 MDP 中的 test-time reasoning，能否从异构 source MDP 中抽取出的 receipt-grounded
> knowledge 获得相对 generic prompting、raw receipt retrieval 和 shuffled controls 的可归因收益？

当前 source-induction 目标是从旧 skill-conditioned rollout 的多次真实执行中发现可验证的
控制规律、失败信号与验证流程。Source skill 只是数据分层与 intervention 条件，不是 transfer
object。旧 graph topology 路径仍单独测试，但只有在提供超过 weak prior 的增量时才支持结构迁移。
通过 source qualification 的知识才能进入下面的在线流程：

```text
source experience
  → adaptation hypotheses
  → online verification
  → lower target adaptation cost
```

一个 target example 只会创建 `TARGET_PROVISIONAL` binding。后续每一步都在线验证；证据不足时 abstain 或退回 target-only。只有 matched controls 的 official outcome 能把候选更新为正迁移、负迁移、generic-only 或 inconclusive。

“快速适应”必须用 examples-to-success、environment steps、无效/重复动作、token/tool cost 和 negative-transfer recovery latency 衡量，而不能只报告最终成功率。

截至 2026-07-23，四任务 canonical pilot 曾得到一个 task-0 正例，但冻结 hypothesis、环境、
prompt 与预算后的 10 个 matched actor-seed 复制没有复现 source-specific benefit：
target-only `0/10`、generic `2/10`、raw receipts `0/10`、authentic `1/10`、shuffled `0/10`、
empty/abstain `0/10`。authentic 对 target-only 仅 1 个 discordant win（单侧 exact
`p=0.5`），对 generic 为 0 win / 1 loss；唯一 authentic 成功的 seed 上 generic 也成功。
因此 task 0 已从 `REPLICATION_CANDIDATE` 降为 prompt/sampling-sensitive variance。当前不扩到
20 seeds、不做 one-shot claim，也不训练 Harness LoRA；下一步必须先在 disjoint target
adaptation split 上验证新的、可预测 future official progress 的 Harness 机制。

## 快速运行

```bash
python -m pip install -e '.[test]'
pytest -q
python examples/smoke_two_agent.py
```

VisualToolBench 官方协议复现实验另装：

```bash
python -m pip install -e '.[test,vtb]'
```

## 目录

- `src/motif_transfer/decision_agent.py`：唯一的动作 authority。
- `src/motif_transfer/motif_harness_agent.py`：motif、binding 和在线审查接口。
- `src/motif_transfer/harness.py`：fail-closed 验证与 matched evaluation。
- `src/motif_transfer/runtime.py`：双 Agent 交互循环。
- `src/motif_transfer/legacy_import.py`：把旧 mega-skill 读成只读 lineage/baseline。
- `src/motif_transfer/skill_internal.py`：按旧 skill 聚合执行、接收内部图 proposal 并做 fail-closed audit。
- `src/motif_transfer/source_execution_motifs.py`：把旧 skill sub-episode 作为原子 evidence，
  在完整 episode 上发现并严格审计跨 execution motif。
- `src/motif_transfer/control_priors.py`：定义不含 target action/topology alignment 的
  receipt-grounded knowledge、弱 control prior、机械 provenance audit 与 matched controls。
- `src/motif_transfer/execution_graph_scoring.py`：只用 discovery 拟合冻结图，并在
  qualification/held-out 上做 blind null/shuffled scoring。
- `src/motif_transfer/multihorizon_runner.py`：执行 matched h=1/2/4/8 closed-loop source forks。
- `docs/ARCHITECTURE.md`：权限边界和数据流。
- `docs/ADAPTATION_ASSISTANT.md`：快速适应助手的完整定位与证据标准。
- `docs/EXPERIMENT_PROTOCOL.md`：六游戏到远域的实验设计。
- `docs/OLD_RESULTS.md`：旧 checkpoint、rollout 和 mega-skill 的保留方式。
- `docs/SKILL_INTERNAL_BACKBONE.md`：旧 skill 内部 backbone 的表示、controls 与真实 pilot。
- `docs/IMPLEMENTATION_STATUS.md`：当前机械审计、正在运行的采集和尚未授权的 claim。
- `docs/ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md`：最新 weak-knowledge oracle pilot、
  canonical 结果、失败修复和复现 gate。

## 旧游戏模型如何进入新 Harness

旧 Decision Agent 不改接口：skill-on 仍执行 `skill_selection LoRA → selected game skill → action_taking LoRA → native action`，skill-off 只移除 skill context。Motif/Harness Agent 不参与 source action。

rollout 完成后，确定性代码在精确记录的 `selected_skill_id` 变化点切出连续 execution。不能使用 `selected_skill_sha256` 做边界，因为旧系统的该字段包含随 observation 变化的动态 guidance，同一 skill 也会产生不同哈希。`selected_skill_id` 只决定哪些 transition 属于同一 skill，不能决定内部 motif node。独立 Motif/Harness Agent 可以在 execution 内部提出 receipt-bound spans 和 graph；旧 bank 文本只作为不可信 hypothesis。matched controls、qualification、intervention 和 held-out 共同判断是否存在 source-derived 增量价值。

历史实现仍完整保存在 Git parent `948f64a` 以及原始工作副本中；本分支不复制 checkpoint、rollout、Slurm log 或生成结果。
