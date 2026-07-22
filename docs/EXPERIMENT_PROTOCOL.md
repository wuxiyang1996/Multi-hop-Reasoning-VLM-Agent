# 六游戏到远域的实验协议

## 核心假设

本实验不检验“模型是否会推理”，而检验：

> 对一个固定、已经具有通用推理能力的 Decision Agent，source-derived reasoning motif 加 online Harness 是否能用更少 target supervision、更少环境试错和更低推理成本完成新领域适应。

Motif 是 adaptation hypothesis generator，而不是 target policy。若强模型已经能自行恢复相同结构，motif 可以是冗余的；若 motif 限制了模型，它也可能产生负迁移。这两种结果都必须被实验显式保留。

## Phase A：source motif discovery

对六个游戏分别采集多个 episode，并做 matched skill-on / skill-off intervention：相同 checkpoint、seed、初始状态、step/token budget。source Decision Agent 保持旧系统的原生调用链：`skill_selection → selected skill → action_taking → native action`；skill-off 只移除 skill bank/context。记录完整 observation、native actions、selected-skill receipt、原生 action response、执行动作、transition、replay fork 和 official outcome。旧 policy 没有 proposal-set/post-verdict 接口，因此不得事后伪造这些字段。

source discovery 分成两个权限不同的步骤：

1. 确定性代码仅在精确记录的 `selected_skill_id` 改变处形成 maximal contiguous run；这不是文本 clustering，也不解释 skill 语义。`selected_skill_sha256` 含动态 guidance，不能作为稳定 skill identity。
2. 旧 `segment` LoRA 只执行其训练时的原生任务：对该游戏 skill bank 的全部候选做严格排序。它不生成 motif JSON。每个完整排序与 segment receipt、candidate-bank hash、adapter hash 和 raw-response hash 绑定。
3. 独立 Motif/Harness Agent 才能根据多个 segment ranking receipts 与 replay receipts 提议图结构。Agent 只能引用完整 run，不能移动 step、重新分段或创造边界。所有 role 名和自然语言解释均为 untrusted；Harness 只验证 lineage、拓扑、replay 支持和 matched intervention，不能认可语义。

候选需要在 held-out source episodes 上重现，并显示相对 skill-off/renamed/randomized controls 的稳定差异，才成为 `SOURCE_SUPPORTED`。不按 skill 文本、action 名称或 embedding 聚类；“合并”仅表示 verified structural fingerprint 与干预效应均一致。

## Phase B：one-shot online grounding

每个 target task family 只给一个 adaptation example：

1. Motif Agent 为 source motif 提议 target binding hypothesis，以及 information need、expected transition、failure route 和 termination test。
2. Harness 标记为 `TARGET_PROVISIONAL`，不认可其语义。
3. Decision Agent 仍在 target-native interface 中独立选 action。
4. Motif Agent 每一步只能建议 admit/replan/abstain。
5. live receipt 与 official feedback 持续淘汰不一致 binding；这一步更新的是外部 version space，不在正式测试 episode 中更新模型权重。
6. 无可行 binding 时回退 target-only；若需要区分多个 binding，则请求额外 example，并记录额外监督成本。

这里的 one-shot 表示“用一个 example 启动适应”，不表示一个 example 能唯一确定 binding。后续交互仍持续验证；如果无法区分候选，可以请求额外 example，并把它计入 adaptation cost。

如果 motif 没帮助，应区分：`NOT_APPLICABLE`、`NEGATIVE_TRANSFER`、`GENERIC_ONLY` 和 `INCONCLUSIVE`，而不是强制映射或继续消耗 rollout。

## Phase C：matched target evaluation

每个样本至少运行：

- `authentic`：真实 source motif；
- `target_only`：不提供 source；
- `generic_protocol`：等长但无 source 内容的通用控制提示；
- `shuffled_topology`：receipt 数量相同、图结构破坏；
- `other_source`：来自不匹配游戏/episode 的 motif。

五组必须共享 initial state、prefix、policy、model、sampling、tool/token/step budget。主结论只使用环境 official success/score；Agent judge 只用于诊断。

## 最小判据

- `authentic > target_only` 且优于 generic、shuffled、other-source：正迁移 pilot evidence。
- generic 与 authentic 相当：只能称 generic scaffold effect。
- target-only 优于 authentic：负迁移。
- 缺少 identity 或完整 controls：inconclusive。

先在两个 target benchmark 做小规模可证伪 pilot；只有观察到 attributable separation 才扩到四领域、八 benchmark。Game SFT 可以作为 source policy 初始化或统一 base-policy control，但不能替代 target grounding，也不能默认免除 SFT；“可以跳过 target SFT”必须由 target-only matched sample-efficiency 曲线证明。

## 快速适应指标

最终 success 之外必须报告：

- `examples_to_success`：达到预注册 success 水平需要多少 target examples；
- `steps_to_adaptation`：binding 达到稳定可用状态前需要多少真实 transition；
- invalid、repeated 和 no-progress actions；
- token、model call 和 environment/tool call cost；
- 首次失败到正确 replan 的步数；
- negative motif 被拒绝或回退 target-only 的延迟；
- 在相同 supervision budget 下的 success/score 曲线。

如果 authentic motif 与 generic protocol 最终成功率相同，但显著减少 examples、steps 或 tool cost，仍可支持“快速适应助手”的 claim。如果它只增加上下文、没有改善任何 adaptation metric，则应判为无增量价值。

## 模型能力交互

至少比较 base 9B、game-co-evolved 9B 和一个更强模型。预期 motif 可能主要帮助较小或不稳定的模型；若增益随模型能力增强而消失，结论应限定为“外部经验图补偿小模型的在线适应”，而不是普遍 reasoning transfer。
