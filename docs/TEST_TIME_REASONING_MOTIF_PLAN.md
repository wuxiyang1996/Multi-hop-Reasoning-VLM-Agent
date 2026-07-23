# 游戏到远域的测试时 Reasoning Motif 迁移计划

## 1. 主张边界

本项目的主问题固定为：

> 能否只从六种游戏的真实 rollout 与 skill intervention 中发现一个非平凡、可复现、
> evidence-carrying 的控制图，并在不更新目标模型权重、不预定义 source→target
> 语义映射的前提下，通过在线 Harness 将它实例化到异构目标领域？

这里的迁移发生在 test time：

```text
game rollout + matched intervention receipts
  → frozen opaque motif graph
  → one adaptation example 初始化 provisional binding
  → Harness 在线提出/验证控制建议
  → Decision Agent 独占 target-native action
  → official target outcome
```

迁移对象不是游戏动作、游戏名词、自然语言经验总结，也不是不可观测的模型内部思维。
它是一个从外显行为中识别出的、带 source receipt 的随机控制图。图至少包含两个不同
node、一个有向 edge，以及可观察的 branch、recovery 或 termination 之一；单步规则、
固定的 `observe → act` 循环和“先思考再回答”不算非平凡 motif。

游戏 SFT/LoRA 是否把推理能力写进模型权重是独立问题。它可以作为第二个 factorial，
但不能替代 test-time motif transfer 的证据。

## 2. Related work 与真正的近邻

截至 2026-07-22，最相关的工作可分为五组。

### 2.1 同环境技能复用

[Voyager](https://arxiv.org/abs/2305.16291) 从 Minecraft 交互中生成、验证和检索可执行
代码技能，并能在新的 Minecraft world 中复用。它证明了 environment-feedback、
self-verification 和 compositional skill library 的价值，但 observation、action API 和
动力学仍属于同一个游戏。我们的实验不能只证明“skill library 有用”；必须跨越不共享
action/observation interface 的领域。

### 2.2 轨迹经验、workflow 与 reasoning memory

[ExpeL](https://arxiv.org/abs/2308.10144) 和
[Reflexion](https://arxiv.org/abs/2303.11366) 在测试时把成功/失败经验转成语言 insight 或
reflection。[Agent Workflow Memory](https://arxiv.org/abs/2409.07429) 从 web 轨迹中归纳并
复用 workflow；其 cross-domain 仍发生在 web navigation 家族。
[ReasoningBank](https://arxiv.org/abs/2509.25140) 从成功和失败经验中蒸馏 reasoning memory，
通过 embedding retrieval 注入测试时上下文，并与 test-time scaling 结合。
[Agent KB](https://arxiv.org/abs/2507.06229) 更直接地研究跨 agent framework 的经验共享：
检索 workflow 用于 planning，并用 disagreement gate 降低干扰。

这些工作是必须实现的强 baseline，而不是只在文字中比较。若 game motif 不能超过
等 token 的自然语言 insight、raw trajectory retrieval、generic workflow 和 ReasoningBank-style
memory，就不能说 graph 或 source provenance 带来增量价值。

### 2.3 从轨迹自动生成 declarative skill

[Trace2Skill](https://arxiv.org/abs/2603.25158) 用多个分析 Agent 从成功/失败轨迹提出 patch，
再合并为 declarative skill，并报告跨模型规模和 OOD 数据的收益。它与我们的“Agent 从轨迹
提取 skill”非常接近。主要差别必须由实验建立：我们的 source 与 target interface 极远，
motif 的 node/edge 必须引用真实 transition receipt，binding 在 live target transition 上逐步
验证，而且 source 可以被 fail-closed 地关闭。只把多条游戏轨迹总结成一份 SOP 已不够新。

### 2.4 游戏训练带来的权重级推理迁移

[STRATAGEM](https://aclanthology.org/2026.acl-long.897/) 是最直接的 novelty threat。它在三个
文本游戏上 self-play，用 LLM 评估轨迹的 abstraction、deepening、adaptation 和 coherence，
调制训练 advantage，并在数学、通用推理和代码任务上测试权重级迁移。

因此“游戏训练可以改善远域推理”本身不能作为我们的核心 novelty。我们的可区分部分是：

- 不用人工命名的 abstraction rubric 选择 motif；
- 不依赖权重更新来承载主迁移效应；
- 从 official transition 和 matched skill-on/off intervention 识别控制图；
- 在完全不同的交互接口中做 one-shot 初始化和逐步 online falsification；
- Decision Agent 与 Harness action authority 分离；
- 显式测量 source-off、negative transfer 和 recovery latency。

### 2.5 相近接口的跨站点 grounding

[Transferable Interaction Patterns](https://arxiv.org/abs/2606.17645) 把 web 操作模板绑定到
新页面控件，展示跨网站复用。它仍共享网页控件和 primitive action 家族，而且使用 live layout
similarity 检索。我们的目标更远：游戏到 embodied household、视觉工具、长视频/视觉推理等
不共享 primitive interface 的领域；同时不允许 embedding similarity 决定真实性。

## 3. 不能退化成什么

以下结果均不足以支持主张：

1. authentic 只优于 target-only，但不优于 generic/insight/raw-trajectory controls；
2. motif 文本含有 `collect → take` 一类人工跨域映射；
3. Motif Agent 直接选择 target tool、参数或最终答案；
4. source graph 只由自然语言 rationale 支持，没有 transition receipt；
5. 用 target held-out outcome 修改 graph、prompt、schema 或 source candidate；
6. renamed motif 失效而 authentic 有效，说明收益依赖游戏词汇或身份；
7. shuffled graph 与 authentic 同样有效，说明图拓扑没有增量；
8. 只比较 game-trained 与 base 权重，却称为 test-time skill transfer。

## 4. Source motif 的发现与冻结

### 4.1 数据

六游戏各自重新采集 fresh no-human-hints 的 matched rollout：

```text
B       base model, no game skill context
G−S     game-trained model, skill context masked
G+S     game-trained model, authentic selected skill
G+Rand  game-trained model, mechanically sampled non-authentic skill
```

每个 snapshot 固定 observation、native-action set、prefix、checkpoint、prompt、seed 和预算。
主环境只执行预注册的 authentic action；其他 condition 从相同 snapshot replay fork。历史数据只进入
discovery，不进入 confirmatory qualification。

episode 在运行前按 seed hash 固定为：

```text
discovery / qualification / held-out
```

### 4.2 Motif Agent 的输出

Motif Agent 只看 discovery receipts，输出：

```text
opaque node IDs
directed edges
branch/loop/termination structure
每个 node/edge 的 source receipt references
对下一 source event、transition 或 multi-horizon value 的 blind prediction
```

它可以归纳结构，但它的自然语言解释没有证据效力。Harness 只机械检查 receipt lineage、时间连续性、
intervention identity、图拓扑和 prediction scoring。图节点不使用动作名称、skill 描述、人工
predicate ontology、embedding cluster 或游戏类别定义。

### 4.3 不再要求脆弱的 exact signature recurrence

当前 Phase 7 的 exact node/edge recurrence 会把随机 policy 中结构相同但表面 signature 不同的
episode 全部判负。下一版用冻结的 predictive graph qualification 替代“完整 signature 必须逐项相同”：

- Discovery 后冻结 graph、emission schema 和全部评分函数；
- 在 qualification/held-out 上，Motif Agent 只能做 blind node filtering 和 next-event prediction；
- 比较 motif graph 与 size-matched null graph、shuffled edges、single-state loop 的 held-out
  predictive log score；
- 对 1/2/4/8-step matched replay 测 authentic 相对 `G−S/G+Rand` 的 official value；
- 图复杂度用预注册 MDL penalty 控制，不能靠增加 node 数拟合 held-out。

这不是降低 source gate。`SOURCE_SUPPORTED` 仍必须同时满足：

```text
receipt_valid
nontrivial_graph
heldout_predictive_gain_over_null
heldout_authentic_value_gain
alpha_rename_invariance
shuffled_graph_separation
```

阈值、置信区间和最多候选数必须在读取 qualification outcome 前写入 config。若六游戏没有任何
candidate 通过，停止“显式 transferable motif”主张，不从失败数据上继续调定义。

## 5. Test-time online binding

一个冻结 source motif 进入目标领域时，Harness 只得到：

- opaque graph 与 source receipts；
- 一个 target adaptation example 的真实 observation/proposal/action/outcome receipts；
- 当前 live target history；
- 当前 Decision proposal，但无 target action authority。

Motif Agent 可以提出多个 provisional graph-to-history alignment，不需要也不允许构造永久的
`source object/action → target object/action` 字典。每一步输出仅为：

```text
candidate node distribution
expected observable target delta or termination test
ADMIT / REPLAN / ABSTAIN / SOURCE_OFF
supporting source + live receipt IDs
```

Decision Agent 自行产生 target-native proposal。执行后，Harness 用真实 target receipt 更新候选；
若 prediction 被反驳、候选全部消失或干预成本达到上限，立即 `SOURCE_OFF`，从同一 matched prefix
回到 target-only。在线修正的是 provisional binding，不是 source motif 的文本或拓扑。

## 6. Target 实验矩阵

### 6.1 Test-time 主矩阵

同一个冻结 Decision model 至少比较：

```text
target_only
generic_reasoning                    # 等结构、等 token，无 source receipt
raw_source_trajectory                # trajectory retrieval baseline
natural_language_insight             # ExpeL/ReasoningBank-style baseline
authentic_game_motif
renamed_authentic_motif              # 预期与 authentic 等价
shuffled_game_motif
other_game_supported_motif
```

所有条件匹配初始资产、Decision 请求 cache、模型、tool schema、interaction/token/tool-call budget 和
judge。Harness compute 也必须匹配；target-only 可用无信息 payload/影子调用匹配延迟和 token 成本，
但不能让影子输出影响 Decision。

主 estimand：

```text
source structural value = authentic − max(generic, insight, raw trajectory, shuffled, other-game)
alpha invariance        = authentic ≈ renamed
online safety           = negative-transfer recovery latency and regret after SOURCE_OFF
adaptation efficiency   = area under examples-to-success curve for k ∈ {0,1,2,4}
```

首个闭环使用已实现 official tool loop 的 VTB single-turn subset；第二个使用 ALFWorld。只有机制和
归因在这两个异构交互域成立后，才扩展到剩余可运行的视觉/视频目标。不能为了凑“四域”把缺 official
evaluator 的 benchmark 写入主结果。

### 6.2 权重迁移附加矩阵

为回答 game SFT/LoRA 是否已经改善远域推理，另做：

| Decision weights | Harness off | Generic | Authentic |
|---|---:|---:|---:|
| base 9B | ✓ | ✓ | ✓ |
| game-trained 9B | ✓ | ✓ | ✓ |

它分别估计 weight effect、test-time motif effect 及 interaction。当前 35B target Decision 与主要
9B game LoRA 不同构，不能用于证明 weight-level transfer；35B 可以验证 Harness 机制，但最终归因
实验必须使用架构匹配的 base/game-trained checkpoint。

## 7. 执行顺序与停止规则

### Stage A：冻结协议

- 冻结 source event schema、candidate budget、MDL/null graph 和 source gate；
- 冻结 target conditions、token/compute matching、official metrics 和 statistical plan；
- 此后 adaptation/held-out 失败不能触发 prompt/schema 修改，只能产生新版本协议。

### Stage B：六游戏 source qualification

- fresh matched collection；
- discovery-only motif proposals；
- qualification 选择或拒绝；
- held-out 只做一次 confirmatory predictive/value test；
- 至少得到两个独立 `SOURCE_SUPPORTED` motifs，分别充当 authentic 和 other-game control。

### Stage C：VTB adaptation 与 smoke

- 补齐官方 full-tool 外部 capability；
- 只用 adaptation example 初始化 binding；
- 在两个预冻结 held-out smoke 上跑八条件；
- 先验证 identity、authority、receipt、cache 和 official judge，不根据 outcome 改 protocol。

### Stage D：ALFWorld 与扩域

- 用同一 frozen motif 和 Harness protocol 跑 ALFWorld；
- 预注册各域 target-native metric 与共同效率指标；
- 只有跨至少两个远域复现后，才使用“cross-domain reasoning motif transfer”。

### Stage E：最终判定

- `authentic > strong text/retrieval/graph controls` 且 renamed 等价：支持结构化 test-time transfer；
- generic/insight 同样好：只支持通用 Harness 或经验提示；
- authentic 频繁 source-off 但能快速恢复：支持安全拒绝，不支持正迁移；
- authentic 造成持续 regret：记录 negative transfer；
- source gate 无 motif 通过：停止显式 motif 主张，转向 weight-level game reasoning transfer；
- base/game-trained 均无增益：结论是当前六游戏训练与表示不足以产生可测远域迁移，不再事后
  发明 ontology 或挑选成功案例。

## 8. 当前距离

工程上，VTB 的双 Agent interposition、receipt、exact-request cache、official judge 和 treatment
compiler 已基本就绪。科学证据上仍处于 source identification 阶段：

```text
online Harness mechanism       已验证
matched generic negative effect 已记录
SOURCE_SUPPORTED motif         0
far-domain positive transfer   0
weight-level transfer evidence 0
```

因此下一项真正改变研究状态的工作不是继续调 target prompt，而是完成 Stage A/B，得到或明确拒绝
第一个 held-out source-supported nontrivial graph。
## 9. Skill sub-episode 不是 motif 边界

旧 Phase-1 数据已经包含 6 个游戏、13 个 skill、131 次 skill execution 和
1769 个 transition receipts。这里的 execution/sub-episode 按
`MAXIMAL_RECORDED_SKILL_ID_RUN_V2` 机械切分：只要 selected skill 改变，旧
sub-episode 就终止。

这些旧结果继续作为 motif 的原子证据，但不再作为 motif 的搜索边界：

```text
transition receipts
  -> maximal recorded skill execution (旧 sub-episode，不能再拆成动作 motif)
  -> complete episode decision cycle（允许连接多个 sub-episode）
  -> Agent-proposed execution graph
  -> mechanical recurrence/shortcut audit
  -> blind qualification + held-out prediction
  -> matched h=1/2/4/8 official-return intervention
```

因此我们同时避免两个错误：

1. 在单个 skill 内按 `LEFT/RIGHT` 或宏动作聚类，把 motor pattern 称为 reasoning；
2. 只看完整 action stream，却丢掉已经获得的 skill-conditioned execution evidence。

### 9.1 2026-07-22 extraction diagnosis

六游戏旧轨迹的 execution-level affordance 为：

- Thunder：discovery/qualification/held-out 有 14/16/14 个 executions，
  分别有 12/14/12 条跨 execution 边；
- Strider：19/18/21 个 executions，分别有 17/16/19 条边；
- Tetris、Candy Crush、Streets of Rage：每个旧 episode 只有一个固定-skill
  execution，当前数据不能支持跨 skill graph；
- Columns：跨 execution 结构只落在 qualification/held-out，冻结 discovery 没有
  边，不能根据后验结果重分 split。

GPT-5-mini 的 discovery 诊断得到：

- Thunder authentic-reasoning：Agent abstain；
- Thunder receipt-only：提出一个 5-node branching graph，但严格审计发现 node/edge
  不跨 discovery episode 重现，并且 action sequence 单字段即可恢复 node partition；
- Strider authentic-reasoning：提出
  `attempt -> verify/recover -> retry` hypothesis；但 recovery 只有一次，若干 edge
  只有一个 episode 支持，且 length/action/reward sequence 单字段能够恢复 partition；
- Strider receipt-only：Agent abstain。

因此 Strider 图只能登记为 fresh-data hypothesis，不能标记 `SOURCE_SUPPORTED`。
official reward 可以作为执行后的在线 success/failure evidence；它不能单独成为
“reasoning backbone 已被发现”的证明。

对两个非空 proposal 做了额外的、非 confirmatory 的机械 blind scoring：

- Thunder receipt-only 在 qualification/held-out 相对 global null 的平均 log-score gain
  为 `-1.148/-1.102`，相对 shuffled topology 也为负；
- Strider authentic 在 qualification/held-out 相对 global null 为
  `-0.896/-0.899`，相对 shuffled topology 也为负。

因此当前两个旧候选不只是“样本太少暂缓接受”，而是已经在未参与 discovery 的 execution
边界上表现为无预测增益。它们应标记 rejected；fresh collection 检验的是新数据能否产生新的
稳定候选，不能继续调这两个旧图。

### 9.2 Frozen fresh-source gate

下一轮只在 Thunder 与 Strider 首先收集 fresh no-human-hints 数据。每个游戏至少
12 个 episode，预先按 seed 分为 4 discovery / 4 qualification / 4 held-out。若希望
每条分支都获得更稳定的重复支持，应扩到 24 个 episode（8/8/8）。

候选只能由 discovery 提出，并必须满足：

- 每个 node 至少由两个独立 discovery episode 支持；
- 每条 edge 至少由两个独立 discovery episode 支持；
- node partition 不能被 skill token、单一动作序列、execution length、return sign
  或 reward-sign sequence 的单字段 lookup 精确恢复；
- qualification 与 held-out 只用于冻结后的 blind next-execution prediction；
- 最终 source gate 还要求 authentic 在 common-continuation 和 full-policy-regime
  两种 estimand 上，于 h=8 同时优于 `G_MINUS_S` 与 `G_PLUS_RANDOM`；
- h=1/2/4 仅作诊断，不能替代 h=8。

如果 fresh Thunder/Strider 仍无候选通过以上 gate，就停止“显式 transferable
reasoning motif”主张。仍可报告 game training 对 target performance 的整体影响，
但不能把差异归因于已识别 motif。

## 10. Trained Harness 的长期角色

项目保留两条互补路径：

```text
Path A: 显式 source motif
game receipts -> source-supported motif -> one-shot target binding

Path B: learned adaptation Harness
game receipts -> train Harness adaptation ability
              -> few-shot target interaction
              -> discover target-native motif/skill
              -> verify whether source structure helps
```

Path B 不是在新领域强制寻找 source motif 的同义版本。Trained Harness 的首要目标是帮助
Decision Agent 快速适应新领域，并从少量 target adaptation interactions 中发现该领域自己的
可执行结构：

- 哪些 proposal 在什么可观察上下文中有效；
- 哪些 transition 表明需要继续、replan、abstain 或终止；
- 哪些多步 execution 在不同 target episodes 中复现；
- 哪些 target-native motif/skill 能降低后续任务的 adaptation cost；
- source motif 是有用 prior、generic-only context、负迁移，还是完全不适用。

Harness 的训练监督只来自可验证 receipts，而不是人工 predicate 或 source→target ontology：

```text
(history, proposal, predicted delta, real transition, official outcome)
    -> verification / uncertainty / replan / abstain / source-off
```

可使用的 source 自监督任务包括 next-transition prediction、proposal/outcome consistency、
replay-fork contrast、skill-on/off contrast、failure recovery 和 calibrated abstention。
Harness 不训练成游戏 action policy，也不能直接执行 target-native action；Decision Agent 始终
保留 action authority。

进入新领域时：

1. `k=0`：Harness 只携带游戏训练得到的验证与适应能力；
2. `k=1/2/4`：只读取 adaptation split 的真实 target receipts，提出 target-native motif/skill；
3. 每个 target motif 必须跨 adaptation executions 重现，并在 held-out test 前冻结；
4. source motif 只作为可拒绝 hypothesis，与 target-only discovery 并行；
5. 可选的小 LoRA 只能用 verified adaptation receipts 训练，不能读取 test outcome；
6. test 时在线验证，失败立即 abstain 或回退 target-only。

最终 factorial 至少包含：

| Harness weights | Source motif | Target motif discovery |
|---|---|---|
| base | off | off |
| base | off | on |
| game-trained | off | on |
| base | authentic | on |
| game-trained | authentic | on |

由此分别识别：

- 游戏训练是否提高 Harness 的 domain-adaptation ability；
- target-native motif discovery 是否本身有用；
- 显式 source motif 是否提供额外增量；
- game-trained Harness 与 source motif 是否存在 interaction。

如果 Path A 失败但 Path B 降低 examples-to-success、environment steps 或 recovery latency，
项目可以支持“game-trained selective adaptation Harness”主张，但不能继续声称已迁移一个显式
source reasoning motif。

### 10.1 当前可执行状态（2026-07-23）

Path B 的第一版 source-only 训练集已机械构造完成：

- 六游戏、36 个 episode、10,614 条训练样本；
- train / validation / source-held-out 为 3,552 / 3,534 / 3,528；
- 监督任务为 next transition、official outcome、transition membership、
  recorded adjacency 和 missing-evidence abstention；
- label 只来自 source transition receipt，不使用 target 数据、Agent 自评、人工 predicate
  或 source→target mapping；
- `NOT_OBSERVED_FOR_THIS_RECEIPT` 只表示该 after-state 不属于当前 receipt，不被解释为
  环境中“不可能发生”。

Qwen3.5-9B 的 400-step LoRA 作业为 Slurm `7122386`，依赖 fresh source array
`7122289` 成功完成后启动。该模型目前只能称为 **game-receipt-trained Harness
candidate**，不能预先称为 transferable Harness。其后继 job `7122389` 会在冻结、
按 objective 平衡的 source-held-out subset 上同时计算 base 与 LoRA 的 completion-token
NLL；这只是一项训练 sanity gate，不是跨域迁移证据。

目标域审计器也已经实现。Motif/Harness Agent 可以在 adaptation split 提议 target-native
span、node 和 edge，但 verifier 会机械拒绝：

- 引用 test episode、receipt 不匹配、越界或重叠的 span；
- 未跨至少两个 adaptation episode 重现的 node/edge；
- 没有 branch/cycle 的平凡图；
- 可由 length、reward/terminal sequence、proposal count 或 selected ordinal
  单字段恢复的 partition。

冻结的比较不是“trained 对 base”一个数字，而是六个 matched contrast：weight-only、
target-native discovery、game training given discovery、explicit motif given base、
explicit motif given game training，以及 authentic-vs-shuffled topology。所有比较共享
Decision Agent、target examples、工具/环境预算与随机种子。

第一版训练仍有明确边界：它主要训练 receipt verification、lineage recognition 和
calibrated abstention，并且仍可看见原始游戏 interface，因此没有直接证明学到了非平凡多步
reasoning motif，也没有排除 interface memorization。后续 structural/alpha-renamed view
必须作为 matched ablation，而不能在看到 target outcome 后再设计。只有它在
ALFWorld 与 VTB 的冻结 `k=0/1/2/4` adaptation 曲线上优于 base，且不是 shuffled source
或额外 token 带来的效果，才能升级为 transferable-Harness 证据。
