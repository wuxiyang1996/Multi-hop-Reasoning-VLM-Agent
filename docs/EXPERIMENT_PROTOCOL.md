# 六游戏到远域的实验协议

## 核心假设

本实验不检验“模型是否会推理”，而检验两个相互关联但必须分开的假设：

> 六游戏 skill learning / SFT / RL 是否把可跨领域复用的程序性推理先验写入模型权重；
> source-derived motif 加 online Harness 是否能在新领域中更快、安全地调用这些能力。

因此必须分别测量 `game-trained − base` 的 weight-level effect，以及同一 Decision model 内
`authentic Harness − generic Harness` 的 source effect。Motif 是 adaptation hypothesis generator，
不是 target policy。若 game-trained model 已经内化能力，motif 可以冗余；若 base 与 game-trained 都被
同一个 generic Harness 改善，则不能归因于游戏训练；若 motif 限制模型，也可能产生负迁移。
完整 factorial 和归因规则见 [`GAME_TRAINING_REASONING_TRANSFER.md`](GAME_TRAINING_REASONING_TRANSFER.md)。

## Phase 0：game-training weight attribution

在进入 motif transfer 前，先在相同 source snapshot 上运行：

```text
B       base model
G−S     game-trained model + masked skill context
G+S     game-trained model + authentic skill context
G+Rand  game-trained model + unrelated/random skill context
```

checkpoint、observation、native actions、prompt template、request seed 和预算必须 hash-bound。只有
预注册 treatment 可以推进主环境；其他条件是 shadow query，或从相同 snapshot 建立独立 replay fork。
比较 `G−S−B`、`G+S−G−S` 和 `G+S−G+Rand`，分别识别 weight training、online skill context 和
authentic content effect。旧 trajectory 的 post-hoc graph 不能替代该实验。

## Phase A：source motif discovery

对六个游戏分别采集多个 episode，并做 matched skill-on / skill-off intervention：相同 checkpoint、seed、初始状态、step/token budget。source Decision Agent 保持旧系统的原生调用链：`skill_selection → selected skill → action_taking → native action`；skill-off 只移除 skill bank/context。记录完整 observation、native actions、selected-skill receipt、原生 action response、执行动作、transition、replay fork 和 official outcome。旧 policy 没有 proposal-set/post-verdict 接口，因此不得事后伪造这些字段。

旧版正式 source reasoning evidence 使用 offline shadow annotation：先在没有 observer 的条件下冻结完整 source
trajectory；随后 prediction 调用只读取 before observation、冻结 action ordinal 和原 Decision reasoning，
verifier 调用才读取 after observation 与 official reward。annotation 只能旁路写新 artifact，不能回写 policy、
environment 或 source receipt。live observer 仅作协议诊断：实测即使使用 request seed，共享推理服务仍不能
保证 Columns/Thunder 的 exact action/reward equality，因此不得用容差阈值把它升级为主证据。

该 offline annotation 仍可作辅助 evidence，但它无法区分 base 与 game-trained weight effect，也不能证明
shadow Agent 提议的 rationale 是原 policy 的内部 reasoning。新的主证据是 Phase 0 的 matched snapshot
counterfactual receipts，以及随后隐藏 after-state 的 blind qualification。

reference collection 移除旧 runner 中人工 `critical action` / `prefer action 1` 提示，并保存 patched-overlay
hash。任何 schema 失败、action index 改写或 post-verdict 缺失的 annotation 都显式排除；不允许 parser
heuristic。

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
5. live receipt 与 official feedback 持续淘汰不一致 binding；多候选的 advisory 必须 unanimous，不能按 hash
   或 Agent confidence 挑一个；这一步只更新外部 version space，不在正式测试 episode 中更新模型权重。
6. one-shot adaptation 单独生成带 hash 的 frozen binding artifact；held-out evaluation 默认禁止重新 binding。
7. 无可行 binding 时回退 target-only；若需要区分多个 binding，则请求额外 example，并记录额外监督成本。

这里的 one-shot 表示“用一个 example 启动适应”，不表示一个 example 能唯一确定 binding。后续交互仍持续验证；如果无法区分候选，可以请求额外 example，并把它计入 adaptation cost。

如果 motif 没帮助，应区分：`NOT_APPLICABLE`、`NEGATIVE_TRANSFER`、`GENERIC_ONLY` 和 `INCONCLUSIVE`，而不是强制映射或继续消耗 rollout。

### Phase B0：strong-teacher feasibility upper bound

正式 one-shot 之前，允许使用多个 frozen adaptation artifacts 做一次明确标注为
`FEASIBILITY_UPPER_BOUND` 的强 teacher pilot。它只回答机制是否可能工作，不能计入 one-shot
结果，也不能用来选择 held-out 样本。adaptation 文件必须按 canonical exact filename 选择，
逐个记录 hash、receipt count 和 collection error；不得由宽泛 glob 混入 failed/debug artifact。

2026-07-23 的 ALFWorld canonical pilot 使用四个 canonical adaptation artifacts 和四个
qualification tasks。它在 task 0 得到 authentic-only success，但冻结 10-actor-seed
replication 未复现 source-specific benefit：target-only `0/10`、generic `2/10`、authentic
`1/10`。因此不把 adaptation budget 降为一个 clean example，也不扩 unchanged 20 seeds。
任何后续机制必须在 disjoint target adaptation split 上重新预注册，不能沿用该 task 调参。

## Phase C：matched target evaluation

每个样本至少运行：

- `authentic`：真实 source motif；
- `target_only`：不提供 source；
- `generic_protocol`：等长但无 source 内容的通用控制提示；
- `shuffled_topology`：receipt 数量相同、图结构破坏；
- `other_source`：来自不匹配游戏/episode 的 motif。

五组必须共享 initial state、prefix、policy、model、sampling、tool/token/step budget。远程 provider 对相同 seed
仍可能非确定，因此同一 matched run 对完全相同的 Decision request 使用 exact-request memoization；只有
history/prompt 真正分叉才重新采样。主结论只使用环境 official success/score；Agent judge 只用于诊断。

## 最小判据

- `authentic > target_only` 且优于 generic、shuffled、other-source：正迁移 pilot evidence。
- 单个 task、单个 seed 满足该关系：只标为 `REPLICATION_CANDIDATE`，不能升级整体 claim。
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

至少比较 base 9B、game-co-evolved 9B 和一个更强模型。首先测 `game-co-evolved/off − base/off`，
再在每个模型内部比较 authentic/generic/shuffled Harness，最后报告 training × Harness interaction。
预期 motif 可能主要帮助较小或不稳定的模型；若增益随模型能力增强而消失，结论应限定为“外部经验图
补偿小模型的在线适应”，而不是普遍 reasoning transfer。
