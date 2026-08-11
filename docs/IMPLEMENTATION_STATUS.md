# Implementation Status

## Harness retargeting 跨版本审计（2026-08-11）

五个历史工作目录的 docs、raw reports 与关键 logs 已完成只读审计。审计结论、负结果、
受限正结果及 frozen-skill/target-harness 权限边界见
[`HARNESS_RETARGETING_BITTER_LESSONS.md`](HARNESS_RETARGETING_BITTER_LESSONS.md)。后续实验必须
固定同一个 target Harness，对比 null、shuffled/wrong、authentic frozen source skill 与
target-written oracle；仅 `skill+Harness > raw target-only` 不足以归因 source skill transfer。
权限矩阵、fail-closed invariants、paired gates 与实现入口见
[`HARNESS_RETARGETING_PROTOCOL.md`](HARNESS_RETARGETING_PROTOCOL.md)。
Controlled cross-semantic reference run 已通过完整五条件矩阵；结果与严格 claim boundary 见
[`HARNESS_RETARGETING_SMOKE_V1_RESULTS.md`](HARNESS_RETARGETING_SMOKE_V1_RESULTS.md)。

## 主 claim 与 weak-prior contract 更新（2026-07-23）

主问题已从 `skill/motif transfer` 改为：target MDP 的 test-time reasoning 能否从异构 source
MDP 提取的 receipt-grounded knowledge 获得可归因收益。Source skill 继续作为 rollout
分层和 intervention 条件，但不再是 transfer object。

新增 `ReceiptGroundedKnowledge`、四类 control knowledge role、hash/receipt/跨 episode
recurrence audit、artifact round-trip 和 matched weak-prior controls。`BindingHypothesis`
现在显式区分 `EXACT_TOPOLOGY` 与 `WEAK_CONTROL_PRIOR`；后者一旦携带 node/edge target
alignment，version-space 会 fail closed。冻结 target matrix 新增 raw receipts、generic、
authentic weak prior、shuffled evidence、other-game 与 exact-topology 条件，从而区分 induced
knowledge、额外上下文、通用 reasoning、receipt grounding 和强 graph hypothesis。

新协议见
[`TEST_TIME_RECEIPT_GROUNDED_KNOWLEDGE.md`](TEST_TIME_RECEIPT_GROUNDED_KNOWLEDGE.md)，
机器可读配置为 `configs/test_time_receipt_knowledge_v1.json`。旧 motif 文档和代码继续作为
强假设诊断，不再代表主 claim。

## 已完成

- 两个 Agent 的权限边界；
- structured adaptation advisory；
- deterministic binding version space；
- 完整 Decision Agent proposal-set → selection → transition → post-verdict cycle；
- hash-bound episode artifact；
- exhaustive native replay-fork receipt；
- motif anti-collapse 和 receipt-type validation；
- multi-pair target evaluator；
- 五条件 source qualifier；
- frozen OpenAI-compatible Motif Agent；
- base/receipt-only prompt condition；
- 六游戏旧 checkpoint、adapter 和 evidence audit；
- A5000 fresh source collection与严格 readiness audit。
- VisualToolBench 官方 commit/tool-contract preflight、single-turn v2 manifest、rubric APR/ARS scorer；
- VTB 六条件结构迁移矩阵（alpha-renaming 不变性、shuffled/other-game/generic destructive controls）；
- VTB `row:105` 官方 target-only adaptation trace：20 calls、0 final answer、cap exhausted。
- VTB online two-Agent interposition：Decision 独占工具参数，Harness 只 review/verify，live receipt fail-closed；
- VTB shared exact-request Decision cache 与 SOURCE_SUPPORTED 五-treatment compiler；
- VTB 3-round matched mechanism smoke：首 proposal exact-cache matched；generic replan 后 2 calls 均失败，0 final answer。

## 2026-07-21 旧资产审计

正式六游戏旧 instrumented 数据包含：

- 6 games、12 episodes、144 steps；
- 137 个 Agent-origin executions；
- 166 个有效 replay-fork receipts；
- 12/12 episodes 有环境返回的 total reward；
- 0 action proposal sets；
- 0 post-transition Agent verdicts；
- 0 explicit replan/abstain observations。

所以旧数据的 `motif_ready_games=0`。它仍可用于 observation/action trace baseline，不能用于声称提取到了 closed-loop reasoning motif。

六游戏 best checkpoint 全部通过 hash 验证，并且 6/6 都具备完整的 `segment / contract / curator` LoRA。代码复核后不再把这些 adapter 名称直接等同于新架构角色：

```text
old segment  → native source-segment skill ranking only
old contract → 暂不授权；需先复原其训练接口
old curator  → 暂不授权；需先复原其训练接口
new Motif/Harness Agent → graph/binding/advisory proposals
```

旧 LoRA 可以提供 source evidence，但不会因为名称相似自动成为 Motif/Harness Agent。所有模型输出保持 untrusted。

## 2026-07-22 协议 smoke 与正式重跑

首轮 array `7119160` 在 `gammagpu13` 触发节点级 `No CUDA GPUs are available`，已停止且保留日志。重提 array `7119165` 显式排除 `gammagpu10,gammagpu13`，使用独立 `phase1_fresh_v2` 输出目录，不覆盖首轮 artifact。

`7119165` 随后落到 `gammagpu15`，该节点 `/scratch0` 已满，模型尚未启动即在 cache rsync 阶段失败。A6000 重提 `7119184` 因资源等待被替代，没有产生实验结果。

A5000 容量 smoke `7119166` 成功加载 18.65 GiB 模型权重，但 `.82` utilization 下没有剩余 KV cache，按预期 fail-closed；复测 `7119170` 只把 vLLM utilization 提高到 `.96`，成功启动模型、LoRA 和真实 rollout 请求。

首个 4-way array 在两个 tasks 同落一台节点时暴露了 vLLM API/engine 相邻端口可能碰撞的问题，已在产生正式 evidence 前停止。修复后每个 array task 保留 20 个端口的隔离区间，并使用新的输出目录重提。每个 task 的 seed、checkpoint、episode 数和 step budget 不变，因此只改变运行可靠性和 wall-clock throughput，不改变实验 treatment。

首轮完整采集 `7119194` 暴露了一个科学上不可接受但 fail-closed 的问题：

- 24/24 planning attempts 和 24/24 post-transition attempts 都有日志；
- 但 Qwen3.5-9B 被当作 raw completion 调用，24/24 输出均未通过 exact-JSON schema；
- 所以 0 cycle 被 Harness 接纳，动作全部退回原 Decision Agent。

该批已停止并保留为 protocol-failure evidence。clean repo 现在在每个 job 的临时 overlay 中把两次结构化调用切换为模型原生 chat template；旧 source repo 保持只读，parser 和证据门槛没有放宽。

修复后的 2-step smoke `7119269` 通过：

```text
2/2 valid action proposal sets
2/2 valid post-transition verdicts
2 imported complete decision cycles
19 linked exhaustive replay forks
0 audit/import gaps
motif_ready = true
```

其中观察到了 `REFUTED → REPLAN` 和 `SUPPORTED → CONTINUE` 两种不同闭环结果，证明日志不是固定填充同一个 verdict。

`7119271` 随后通过了 schema/readiness smoke，但代码复核发现其 planning prompt 未包含 active skill，并且合法 planner action 会绕过 `action_taking` LoRA。因此该批已停止：已完成部分只作为 unconditioned-planner calibration，不能算 skill-loaded policy evidence。

进一步复核后，没有把 `action_taking` LoRA 强制改造成 JSON proposal generator。那会改变旧 policy 的训练接口，并把 JSON instruction-following 混入所谓 reasoning backbone；`7119296` 只作为该失败设计的诊断，不进入结果。

最终 source-policy 协议为：

```text
skill-on: skill-selection LoRA → selected skill → action_taking LoRA → native action
skill-off: same action_taking LoRA → native action, no skill bank/context
Motif Agent: post-hoc extraction only; never controls source action
```

每步保存 selected skill ID/hash/context hash、原生 action response hash、action adapter、执行动作、transition、reward、replay fork 和 official outcome。不会为 source policy 伪造 proposal set 或 post-verdict。

matched native-policy smoke `7119327` 已通过：skill-on/off 各 2/2 `action_taking` steps，skill-on 有 2/2 selected-skill receipts，skill-off 为 0/2；两边均为 0 import gaps、`treatment_integrity=true`。native PEFT Motif extraction smoke 为 `7119331`。

正式 native-policy collection 为 `7119332`：6 games × skill-on/off × 2 episodes × 12 max steps。此前依赖任务 `7119337/7119339` 会错误要求旧 segment LoRA 直接生成新 motif schema，已在启动前取消，没有结果进入实验。

替代流程已经实现：在 recorded selected-skill hash 的变化点机械切段；旧 segment LoRA 只运行训练时的 all-candidate ranking prompt；输出必须覆盖 skill bank 的每个候选且无重复，并生成 hash-bound `SkillRankingReceipt`。

`7119332` 已完成全部 12 个 batch，共 6 games × skill-on/off × 2 episodes × 12 steps。readiness 重新导入 288/288 transitions、0 import gaps，所有 batch 的 content hash、event chain、official outcome、replay receipt 和 treatment identity 均通过，`source_qualification_can_start=true`。其中 274 步直接来自 Agent action，13 步是旧 policy 的 anti-repetition `POLICY_POSTPROCESSOR`，1 步是输出截断后的 parser `FALLBACK`；后两类保留真实 origin，不能算作 Agent reasoning evidence，但也不能从执行轨迹中删除。

首个 ranking array `7119346` 在上述 origin importer 修复时已停止，未完成结果不进入实验。干净重提的 matched base/adapter native-ranking array `7119359` 已完成，输出使用新的 `phase1_native_rankings_v2` 目录。

重新采集矩阵为：

```text
6 source games
× {authentic_skill_loaded, skill_disabled}
× 2 episodes
× 12 max steps
```

每一步保持旧 policy 的原生接口。采集代码不会根据 reward、target 结果或 skill 名称筛选 episode。Motif Agent 事后读取 `SourcePolicyStepRecord`；authentic 条件可见 untrusted source reasoning 与 selected skill ID，receipt-only 只可见哈希、动作序号、reward sign、terminal 和 replay topology。

完整采集成功后，先对每个游戏运行固定两组 native ranking：

```text
base 9B + native segment-ranking prompt
co-evolved segment LoRA + identical native segment-ranking prompt
```

这一步只检验旧 segment head 是否仍能对真实新 rollout 做它原来学过的识别任务，不等于 motif transfer。下一步由独立 Motif/Harness Agent 读取 ranking receipts 与 replay forks，再做 authentic/receipt-only/renamed/randomized graph proposal 和 source qualification；模型不能自行晋升 motif。

`7119359` 的严格结果为：

```text
base 9B:              11/25 valid all-candidate rankings
co-evolved segment:    0/25 valid all-candidate rankings
base comparable top1:  0/6 matches selected source skill
co-evolved errors:    24 candidate-set violations + 1 non-JSON
```

旧 segment LoRA 会输出当前游戏 bank 中不存在的旧/混合 skill catalog。Harness 没有做 alias matching、补齐或 parser 放宽。因此该 LoRA 不能作为可靠 backbone extractor，只保留为失败 control。

随后使用 `openai/gpt-5-mini`（OpenRouter）作为独立、无 action authority 的 Motif/Harness proposal Agent，在真实 native receipts 上完成 smoke。期间修复了四个会制造伪阳性的 Harness 问题：无效单节点 schema example、edge fork 与 source node 未绑定、允许 cherry-pick replay forks、跨条件 fingerprint 错误包含 treatment label；同时 source transition receipt 新增 episode/step identity，避免跨 episode content collision。

最终 Columns receipt-only 对照在 skill-on 与 skill-off 各 2/2 episode 得到完全相同的 canonical fingerprint：

```text
c1414e116ec82e46538e32404df1f881a7eaea074e8661a93b03863176038f79
```

其结构只是 `AGENT → POLICY_POSTPROCESSOR`，并携带同源 exhaustive replay forks。它是可复现的 generic executor motif，不是 game-skill content 或 reasoning-backbone 的增量证据，不能晋升为 `SOURCE_SUPPORTED`。

旧 `segment` LoRA 的 vLLM loader smoke `7119247` 已证明当前服务路径不兼容：adapter safetensors 本身完整（352 tensors），但 vLLM 在 Qwen3.5 packed projection 激活时触发 `NoneType.shape`。因此 co-evolved extraction 改用 matched PEFT/Transformers 加载；不能静默回退 base 9B，也不能把 loader failure 记作模型效果。

PEFT smoke `7119281` 根据 adapter 的真实 tensor namespace fail-closed 选择训练时一致的 text-only loader。结果为：co-evolved authentic 与 receipt-only 均成功执行且无 model error；base 在同一 backend 下返回非 JSON并被 strict parser 拒绝。旧依赖 job `7119288` 已随无效 source collection 取消；后续只对 native-policy receipts 运行 matched extraction。

## 2026-07-22 Tetris/Candy 长轨迹配对审计

为了区分“best action LoRA 本身有效”和“skill context 有增量”，定向重跑了 Tetris/Candy：

```text
2 games × {skill-on, skill-off} × 6 episodes × 50 max steps
sequential episode reset
exact initial-state-hash pairing
same checkpoint file hashes in each on/off pair
```

Tetris 顶层接口经 manifest 证明为 `TetrisMacroActionWrapper`。所有 576 个 on/off 顶层动作都属于当步 recorded native macro list；日志中的 rotate/move/hard-drop 是 wrapper 内部展开，不是 Decision Agent 的 primitive action。Candy 的 600 个动作全部属于当步 native adjacent-swap list。旧 Gym adapter 会先对空的通用 mapping 打 warning，随后 Candy 环境使用自己的 coordinate mapping 解析；receipt membership 和真实 state transition 再次验证执行。

两游戏的 action LoRA 在 50-step 设置下均复现出较高原域回报：

```text
Tetris skill-off mean:       195.83  (checkpoint historical mean 245.96)
Candy skill-off mean:        565.17  (checkpoint historical mean 620.00)
```

严格配对的 skill-context 差值为：

```text
Tetris: skill-on 187.50 - skill-off 195.83 =  -8.33
        1 positive / 2 zero / 3 negative pairs
        95% descriptive t interval [-49.58, 32.92]

Candy:  skill-on 632.50 - skill-off 565.17 = +67.33
        5 positive / 0 zero / 1 negative pairs
        95% descriptive t interval [-110.22, 244.89]
```

样本只有 6 对且两个区间都跨 0，因此不声称统计显著正/负迁移。Candy 是值得扩大样本的正向 pilot；Tetris 没有显示增量。更重要的是，Tetris skill-on 的 292 步全部选择 `COMMIT/POSITION`，Candy 的 300 步全部选择 `COMMIT/CLEAR`，每个 episode 都是 0 skill switch。因此本批证明了 checkpoint policy 和静态 skill context 的原域行为，但没有产生多阶段 reasoning motif，也不能证明跨域 reasoning-backbone transfer。

完整机器可读结果保存在本地生成目录 `runs/phase1_tetris_candy_long_v1/paired_report.json`；该目录不进入 Git。定向批次的 receipt、import、initial-state pairing、checkpoint identity、wrapper identity 和 native-action gates 全部通过。六游戏总门保持为 false，因为该批有意只覆盖两游戏。

## Streets/Strider 长程筛查与切换 replay

第一次长程提交 `7119530` 暴露出 source collector 显式使用 `use_query_engine=False`，使 skill-on 退化为 skill bank 前三项的非排序 fallback。该批 skill-on 被中止并隔离到 `runs/phase1_unranked_control_7119530`，没有进入 semantic-skill 结论；skill-off 不受该问题影响。修复后要求 Qwen embedding query engine 成功加载，否则 fail closed。正式 semantic skill-on 重跑 `7119540` 与保留的 matched skill-off 共同形成：

```text
2 games × {skill-on, skill-off} × 6 episodes × 50 max steps
1200/1200 imported transitions, 0 gaps
exact (requested seed, initial-state hash) pairing
```

配对结果为：

```text
Streets: skill-on 121.67 - skill-off 93.33 = +28.33
         selected skill: COMMIT/ATTACK 300/300; 0 switches

Strider: skill-on 183.33 - skill-off 150.00 = +33.33
         paired deltas: +50, 0, +50, +50, +50, 0
         selected skills: POSITION 212, EXPLORE 59, ATTACK 29
         exact selected-skill-ID switches: 52
```

Gym-V 对 requested seed 的 reset 实际产生相同 initial-state hash。旧分析只比较 hash multiset，会把重复初态误称为 seed pairing；现已改为每对同时要求 requested seed 与 initial hash 相同。另一个问题是 `selected_skill_sha256` 包含动态 observation guidance，同一 skill 每步也会变化；用它分段会把 Strider 的 52 次真实切换膨胀到约 261 个伪切换。现在所有边界只使用 exact recorded `selected_skill_id`，得到 58 个 maximal runs 和 52 个边界。

CPU 作业 `7119593` 在每个边界从相同 requested seed 重放原 prefix，并穷举当步除原动作外的全部 11 个 native alternatives：

```text
52/52 switch boundaries
572/572 replay-fork receipts
status = INTERVENTION_OBSERVED for every receipt
Slurm exit 0; elapsed 39:09
```

Supplemental bundle 与原 manifest/events/episodes 三个文件的 SHA-256、receipt 文件 hash 和逐条 content hash 绑定；importer 对任一 mismatch、非 observed status 或 count mismatch fail closed。

Replay collector 随后加入边界内 fork 并行；每个 alternative 仍创建独立 adapter、reset 相同 seed 并重放完整 prefix，`executor.map` 保持 receipt 顺序。8-worker 全量校验作业 `7119635` 用时 `8:03`，相对顺序版 `39:09` 加速 `4.86×`。两版 572 行 receipt JSONL 的 SHA-256 均为 `184e9c05ff4d794c2badf2c3df24ee07225d37f0c642ffd12c6cdfa2e013bdb7`，证明吞吐优化没有改变证据内容。

## Strider motif proposal matched controls

GPT-5-mini 只作为无 action authority 的 untrusted Motif proposal Agent。Agent 只能组合 exact skill-ID 形成的完整 maximal runs，不能切 step、移动 step 或创造边界；Harness 只接受携带 source-node 全部 observed forks 的 edge。初次 smoke 同时输入旧 step-1 forks，模型 6/6 选择了与真实切换无关的前两个 run，Harness 以 `no observable control variation` 全部拒绝。修复后 proposal 只输入 572 个 switch-boundary forks。

Harness 使用 skill ID 的匿名等价类，而不是名称：先按 episode 首次出现编号，再在候选图内做 alpha-renaming；任意 skill rename 不改变 fingerprint。连续重复相同 signature 使用 run-length collapse，避免把领域特定的持续 3 步或 7 步误当成不同 backbone；原始 transitions 仍全部是强制 evidence。

四个等预算条件的结果为：

```text
condition            accepted    dominant fingerprint recurrence
authentic              6/6       3/6
receipt_only           6/6       4/6
renamed                5/6       4/5 accepted
shuffled_topology      3/6       2/3 accepted
```

authentic 与 receipt-only 在 5/6 episode 的 fingerprint 完全相同；renamed 的前五个 accepted 中也与对应 receipt-only 相同。真实 run/fork topology 有助于构造合法图，但 skill 名和自然语言 reasoning 没有显示增量，且 authentic dominant fingerprint 不优于 controls。因此当前候选保持 `GENERIC/BEHAVIORAL STRUCTURE ONLY`，不得晋升 `SOURCE_SUPPORTED`，也尚不进入远域 positive-transfer claim。

Columns/Thunder 的 50-step semantic skill-on/off 长程筛查提交为 `7119628`。skill-on 已完成：Thunder 为 300 transitions、38 次切换、0 gaps；Columns 因部分 episode 提前结束为 277 transitions、5 次切换、0 gaps。8-worker supplemental replay 分别得到 Thunder 418/418、Columns 55/55 observed receipts。

Columns 只有 3/6 episode 有切换。四个条件都在这三个 episode 生成 3/3 accepted candidates，但 authentic 的三个 fingerprint 各只出现一次，没有 recurrence，因此为 `INCONCLUSIVE/SPARSE`。

Thunder 的四条件离线重审为：

```text
authentic           5/6 accepted; dominant fingerprint 3/6
receipt_only        6/6 accepted; dominant fingerprint 3/6
renamed             4/6 accepted; dominant fingerprint 3/6
shuffled_topology   5/6 accepted; dominant fingerprint 3/6
```

最后三个 Thunder episode 的 dominant fingerprint 在四个条件中逐一完全相同。它是可复现的行为控制结构，但不依赖 skill 名、reasoning 文本或原时间顺序，故标为 `GENERIC BEHAVIORAL MOTIF`，不能作为 game reasoning-backbone 的 attributable evidence。

`7119628` 四个 task 最终全部 exit 0。联合 paired report 的 action membership、import gaps、requested-seed + initial-hash pairing、checkpoint identity 和 wrapper identity gates 全部通过：

```text
Columns: skill-on 95.67 - skill-off 80.17 = +15.50
         5 positive / 1 zero / 0 negative
         descriptive 95% t interval [0.12, 30.88]

Thunder: skill-on 533.33 - skill-off 250.00 = +283.33
         4 positive / 1 zero / 1 negative
         descriptive 95% t interval [-138.62, 705.29]
```

Columns 的小样本区间略高于 0，Thunder 的均值增益大但区间跨 0；两者都只作为原域 skill-context pilot。它们不能覆盖 matched generic/renamed/shuffled outcome controls，也不能推导跨域 transfer。Readiness 的 batch integrity gates 全部为 true；`six_games_have_skill_on_off=false` 仅因为该 report root 有意只包含 Columns/Thunder，其他四游戏位于各自的定向长程 root。

## 尚未获得授权的 claim

- 尚无 `SOURCE_SUPPORTED` motif；
- 尚未证明 old co-evolved 9B 优于 base 9B；
- 尚未证明 authentic motif 优于 generic/shuffled/other-source；
- 尚未授权 ALFWorld large-scale experiment；
- 尚未证明可以跳过 target SFT。

## ALFWorld one-shot online vertical slice（2026-07-22）

真实 ALFWorld OOD 五条件 one-shot smoke 已完成，五组 initial-state hash 一致，official success 都为
1。target-only / authentic / generic / shuffled / other-source 分别使用 5 / 6 / 7 / 5 / 5 步。
因此 online mechanism 已接通，但 authentic 没有超过 controls，并有相对 target-only 的 +1 step
负效率信号。当前 source status 仍为 `GENERIC_ONLY`，不得称为 source reasoning-backbone transfer。

完整错误审计、运行协议和下一阶段 gates 见
[`ALFWORLD_ONE_SHOT_STATUS.md`](ALFWORLD_ONE_SHOT_STATUS.md)。

后续修复已加入 multi-hypothesis structural binding、full-action alpha-renaming、两次 induction
intersection、post-transition binding evidence、version-space elimination，以及最多一次 source-induced
replan 后的 target-only fallback。真实默认复跑因两次 induction 的稳定交集为空，从 step 0 回退，
仍以 baseline 相同五步 official success；这是正确的 fail-closed 结果，不是 positive transfer。

只有 fresh source evidence 通过审计、frozen 9B comparison 和 source qualification 后，才把后续
ALFWorld run 解释为 positive-transfer pilot；未过 gate 时只允许像本轮一样作为机制/负迁移诊断。

## Skill-internal frozen control matrix（2026-07-22）

旧 skill bank 与六游戏长轨迹已按真实 `selected_skill_id` 聚合成 13 个
`GroundedSkillExecutionSet`；其中 11 个具有 episode-level discovery / qualification / held-out
三分数据。`selected_skill_id` 只定义 membership，不再作为内部 motif node boundary。

GPT-5-mini 在七个同源条件加 matched skill-off 上完成 88 次冻结调用。总使用量为
4,734,617 prompt tokens、120,044 completion tokens。authentic 的 11 次调用只产生两个候选，
均因 trivial/overlap/action-identity gate 被拒绝，故 authentic backbone-eligible 为 0。六个
discovery-eligible graph 全部来自 bank-masked、reasoning-masked、receipt-only、shuffled 或
skill-off controls。

模型输出中有 19 次无 receipt-supported adjacent edge、2 次 offset 越界，全部 fail-closed。
Control graph 在 post-hoc qualification/held-out alignment 上也可达到高 topology fit，证明该指标
不是 blind predictive evidence。当前仍为 0 `SOURCE_SUPPORTED` graph，不进入 ALFWorld transfer。
紧凑审计见 `results/skill_internal_matrix_v1_summary.json`。

## Game-training matched source gate（2026-07-22）

已实现同 snapshot 的 `B / G−S / G+S / G+Rand` collector、authentic-first authority ordering、prompt/
checkpoint/adapter receipts、同 seed+prefix one-step replay、重放 startup retry、pre-replay model-record
持久化和纯 lineage 的多步候选排名。最终 Strider POSITION job `7121659` 完成 155 snapshots 与
620/620 observed replays；61 项单元测试与 overlay patch static checks 通过。

结果只有 causal influence：三 split 均出现 weight、skill-context 与 authentic-vs-random action/state
separation。Discovery graph 为 6 nodes / 16 edges，但 qualification/held-out exact recurrence 分别只有
nodes 42/50、46/51，edges 20/36、19/37；所有 matched one-step rewards 均为 0。严格 Phase 7 为 false，
不授权远域 positive-transfer claim。完整报告见 `results/strider_position_matched_v2_phase7.json`，ATTACK
负诊断和结构排名分别见 `results/strider_matched_policy_v3_phase7.json` 与
`results/strider_v3_structural_skill_ranking.json`。

## Source gate 失败分解（2026-07-22）

新增可复跑诊断确认：当前失败发生在 transfer 之前的 measurement 与 representation 层。155 个
`POSITION` snapshot 在 live authentic path 上的 h1/h2/h4/h8 正 reward support 分别为 0/0/1/4，
而 treatment replay 只观察一步，因此 delayed value 尚未识别。当前 effect-signature graph 也不是
reasoning program；qualification/held-out edge recurrence 仍为 20/36、19/37。

同时发现旧结构排名用三个 split 选择候选，构成 held-out selection leakage；旧六游戏数据均无
no-human-hints exclusion receipt，且 Streets/Strider/Thunder response 明确出现 critical-action 语言。
现有 Phase 7 因此保留为诊断而非 transferable-backbone 证据。完整结论、fresh no-hint 候选统计与
停止规则见 `SOURCE_GATE_FAILURE_DIAGNOSIS.md` 和
`results/source_gate_failure_diagnosis_v1.json`。

## 四域七 Cell feasibility audit（2026-07-22）

SIV-Bench 已从 target matrix 移除。冻结矩阵现在是 VisualToolBench、TIR-Bench、
Video-Holmes、MiniWoB、WebShop、ALFWorld valid-seen/valid-unseen，共四域七个 cell。

原始资产并不缺失：VTB 1204 rows，TIR 1215 rows 且 1255/1255 image refs 可用，
Video-Holmes 3388 questions 且 503/503 unique videos 可用，MiniWoB/WebShop 历史
episode 与 official reward 可读。七个 cell 的 adaptation/test manifest 已冻结；fail-closed
target-only smoke 已在 TIR、Video-Holmes、MiniWoB 和 ALFWorld 上真实执行。MiniWoB 的冻结样本
获得 official reward=1；TIR 暴露伪造 receipt/坐标 grounding，Video-Holmes 暴露搜索耗尽而不提交，
VTB 暴露 typed tool-argument 与官方 rubric evaluator 尚未接入的缺口；官方 judge 代码已在
`xi1ngang/VisualToolBench@d4f200a` 核实。WebShop full 1k-product server 已恢复，
冻结 live smoke 4 步获得 official reward=0.6667。所有这些结果都与旧 dispatcher 的 deterministic
fallback 隔离。

Source audit 同时确认 `reasoning_aligned_mega_skills.json` 含 target-domain members，原文件禁止
作为 source treatment；all-stage mega bank 的 180 members 中只有 79 个属于冻结六游戏 allowlist。
完整结果和后续顺序见 [`TARGET_FEASIBILITY_DIAGNOSIS.md`](TARGET_FEASIBILITY_DIAGNOSIS.md)。

## Execution-level motif failure diagnosis（2026-07-22）

旧 Phase-1 的 13 个 skills、131 次 executions、1769 个 transitions 已重新表示为两层证据：
旧 maximal selected-skill sub-episode 是不可再拆的 execution evidence，完整 episode 才是跨
execution motif 的搜索边界。这样阻断了 action-level Agent 把 `RIGHT/LEFT` 或 macro action
partition 冒充 reasoning graph，同时没有丢弃旧 skill-conditioned rollout。

六游戏机械审计显示，当前冻结 discovery split 中只有 Thunder 与 Strider 有足够跨 execution
边界。GPT-5-mini 的四个 discovery 条件产生两个非空 proposal：

- Thunder receipt-only 的 5-node graph 不跨 discovery episode 重现，并可由 action sequence
  单字段恢复 node partition；
- Strider authentic 的 `attempt -> verify/recover -> retry` hypothesis 含 singleton recovery，
  多条 edge 只出现于一个 episode，且 length/action/reward sequence 可恢复 partition。

严格 re-audit 为 `0/4 accepted`。额外 blind scorer 只用 discovery node receipts 拟合 emission/
transition model；Thunder 在 qualification/held-out 相对 global null 的 log-score gain 为
`-1.148/-1.102`，Strider 为 `-0.896/-0.899`，两者相对 shuffled topology 也为负。因此旧候选已
拒绝，不进入 target Harness。

已新增 frozen fresh-source config、execution-level proposal/audit、单字段 shortcut 检验、
qualification/held-out blind null/shuffled scorer，以及真正 closed-loop 的 h=1/2/4/8 runner
contract。后者区分 first-action + common `G−S` continuation 与 full treatment regime，并在
snapshot hash mismatch 或非法动作时 fail closed；不会用录制动作冒充 policy continuation。
当前 104 项测试通过。下一步是用 best checkpoint 对 Thunder/Strider 收集至少 12 episodes/game
的 fresh no-human-hints matched trajectories；若仍无候选通过 discovery、blind 和 h8 value 三门，
停止显式 transferable-backbone 主张。

该 fresh collection 已于 2026-07-23 通过 Slurm array `7122289` 完成。Strider 与 Thunder 分别在
`gammagpu17/gammagpu14` 使用一张 RTX A6000、best Qwen3.5-9B checkpoint、12 episodes ×
100-step budget；两项均以 exit 0 完成真实 rollout，共各 12 episodes、1200 transitions。运行代码位于隔离 worktree
`Multi-hop-Reasoning-VLM-Agent-source-fresh-v1`，每个 manifest 将记录 runtime file hashes、
no-human-hints profile、seed/split contract 和 matched-treatment receipts。启动 receipt 见
`results/fresh_source_execution_motif_v1_launch.json`。

依赖 audit job `7122387` 最初因 Slurm 环境未加入本仓库 `src/` 而失败，不是数据或科学 gate
失败。现已为全部新 Slurm entrypoint 显式设置 `PYTHONPATH`，并对相同产物重跑机械审计；
Strider 与 Thunder 均为 `accepted=true`。Thunder 包含 428 个 matched replay receipts（四个
treatment 各 107），Strider 包含 8 个（各 2）；两个游戏的 execution traces 均有效。完整
receipt 见 `results/fresh_source_execution_motif_v1_integrity.json`。这只确认数据完整并有
execution-level 搜索空间，尚未确认存在通过 blind recurrence 与 h8 value gate 的 motif。

## 双分支 transfer pipeline（2026-07-23）

显式 motif 与 learned Harness 现在作为两条独立、可组合但不互相替代的路径执行：

1. Path A 继续由 fresh no-human-hints source receipts 搜索 source-supported graph，并要求
   discovery recurrence、shortcut exclusion、qualification/held-out blind prediction 和
   matched h=8 official value 全部通过；
2. Path B 用六游戏 36 个旧 episode 机械生成 10,614 条 source-only receipt supervision，
   训练 Qwen3.5-9B LoRA，使 Harness 学习 transition/outcome prediction、recorded lineage
   recognition 和证据不足时 abstain；
3. target adaptation 只允许 Harness 提议 target-native span/node/edge，Decision Agent
   始终独占 native action authority；机械审计拒绝 test leakage、非复现图、平凡图和
   single-field shortcut。

Path B 训练作业 `7122386` 已提交，依赖 Path A array `7122289` 成功后启动。数据 manifest
为 `results/harness_pretraining_v1_manifest.json`，launch receipt 为
`results/transferable_harness_v1_launch.json`。冻结 factorial 与 matched contrasts 位于
`../configs/two_path_transfer_v1.json`。后继 job `7122389` 将执行 source-held-out
base-vs-LoRA completion-NLL sanity audit；训练 collator 已确保长 prompt 不会吞掉全部
completion labels。

当前证据边界不变：数据构造与训练启动只证明 pipeline 可执行；尚未证明 trained Harness
在 ALFWorld/VTB 上降低 adaptation cost，也尚未发现通过 source gate 的显式 transferable motif。

## Target adaptation rollout collection（2026-07-23）

新的 target manifest 已在任何新 outcome 可见前冻结。ALFWorld-valid-unseen 与
VTB-single-turn 各包含 8 adaptation、8 qualification 和 24 held-out tasks；选择只使用
benchmark-native ID 的 salted hash，并排除全部旧 smoke 样本。

ALFWorld 的 8 个新 adaptation target-only rollouts 已作为 Slurm array `7122411` 完成：
Qwen3.5-35B-A3B Decision Agent、30-step cap、无 Harness、无 source motif，逐步保留完整
Decision cycle 与 transition receipt。最终 4/8 official success、152 transitions、平均
19 steps。机械审计全部通过。task 2 的初次 256-token JSON 截断 artifact 被保留，扩大到
512 token 后单独重跑；重跑在第 8 步产生真实越界 action，因此作为 invalid-Decision
负样本保留，而不是继续采样到成功。

VTB 的 ID 已冻结但未运行。当前 keys 文件仍没有官方工具要求的 `SERP_API_KEY` 与
`OPENWEATHER_API_KEY`，所以 paper-faithful full-tool preflight 失败；不会用 degraded
adaptation 冒充正式 VTB collection。详细协议见
[`TARGET_ROLLOUT_COLLECTION.md`](TARGET_ROLLOUT_COLLECTION.md)。

GPT-5-mini 已作为强 Harness oracle 对 8 个 adaptation traces 做两次冻结诊断。v1 因越界
offset、span 重复归属和 edge 不复现被拒绝；使用 `E0...E7` alpha-renamed episode IDs 与
显式 record count 后，v2 消除了越界引用，但仍因 span overlap、重复归属和 edge 不复现被拒绝。
v2 的 search/observe/move/take/deliver 图主要还是 target action phases，尚非跨域 reasoning
backbone。由于强 oracle 尚未通过，Qwen3.5-9B Harness LoRA job `7122386` 已在分配 GPU 前
置为 user hold；不会用较小模型训练掩盖机制本身尚未成立。

## ALFWorld weak-knowledge canonical pilot（2026-07-23）

exact-topology oracle 失败后，已实现不要求 source→target node alignment 的 weak
receipt-grounded knowledge 路径。GPT-5-mini 只初始化和审查可证伪 control hypothesis，
Qwen3.5-35B-A3B 是唯一 target action authority。六条件为 target-only、generic、raw source
receipts、authentic weak prior、shuffled clause-to-receipt evidence 和 empty/abstain。
所有 source hash 在进入模型前改成短 alias，再由确定性代码映射回真实 receipt。

运行中修复了四类接口错误：长 hash 导致伪 citation、teacher 越权建议 ALFWorld action、
verification 在 precondition 未成立时错误记 supported，以及 `PASS/ACCEPT` 等 schema 外
verdict 触发整段 fallback。未知 verdict 现在只安全降级为本步 `ADMIT` 并保留 violation
receipt，不做 synonym heuristic。

adaptation loader 还发现宽泛 `task_*.json` 会混入
`task_2.failed_schema_256tokens.json`。canonical v5 只接受 exact
`task_<integer>.json`，固定 task 0/1/2/3 的 82 个唯一 receipts，并保存每个 artifact 的路径、
SHA-256、record count 与 collection error。task 2 是 8 个有效 transition 后发生 actor
越界的 partial example；保留但显式披露。

最终 Slurm array `7122688` 四项均 `COMPLETED 0:0`，六条件 initial-state hash 逐任务一致，
无 runtime condition error：

| condition | official success | mean steps | repeated actions |
|---|---:|---:|---:|
| target-only | 2/4 | 20.75 | 13.00 |
| generic | 2/4 | 18.50 | 13.50 |
| raw receipts | 2/4 | 18.50 | 12.75 |
| authentic weak prior | **3/4** | **14.50** | **7.50** |
| shuffled evidence | 2/4 | 18.50 | 13.25 |
| empty/abstain | 2/4 | 21.25 | 14.00 |

task 0 是 clean replication candidate：authentic 在 16 步 official success，其他五条件均
30 步失败；该 authentic run 无 protocol violation、fallback 或 Harness-forced replan。
task 2 中 raw receipts 7 步、authentic 8 步，故只支持 source context 可能有用，不支持
induced knowledge 的额外价值。task 1 为 ceiling case；task 3 全失败且 authentic initializer
abstain。

这仍是四样本 adaptation feasibility upper bound，不是 one-shot；每个 condition 也只有一个
远程 actor rollout。canonical 状态因此只是 `REPLICATION_CANDIDATE`，不是
`SOURCE_KNOWLEDGE_SUPPORTED`。final canonical run 估算成本 `$1.31`。完整报告见
[`ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md`](ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md)。

冻结 task-0 复制随后已完成 10 个 actor seeds（`92000`--`92009`），固定 environment seed
`81000`、initial-state hash、source/adaptation hashes、六个 hypothesis、prompt 与预算。结果为：

| condition | official success | mean steps |
|---|---:|---:|
| target-only | 0/10 | 30.0 |
| generic | **2/10** | **26.5** |
| raw receipts | 0/10 | 30.0 |
| authentic weak prior | 1/10 | 27.6 |
| shuffled | 0/10 | 30.0 |
| empty/abstain | 0/10 | 30.0 |

authentic 对 target-only 是 1 win / 0 loss / 9 ties，单侧 exact `p=0.5`；对 generic 是
0 win / 1 loss / 9 ties。唯一 authentic 成功 seed 同时也是 generic 成功 seed。故 task 0
现已降级为 `PROMPT_OR_SAMPLING_SENSITIVE_VARIANCE`，不能支持 source-derived knowledge
benefit。复制的选择后有效 artifacts 估算 `$5.38`，不含 smoke、失败 raw 调用和 debug。

seed 4 raw condition 暴露了同 payload memoized schema retry 无效的问题；retry 现在携带 parse
error 与 schema-repair instruction。只对该预先声明的 seed/condition 用相同冻结输入重跑，
最终 10 个 replicate 均通过完整性检查。当前 115 项测试、ruff、py_compile 全通过。

不继续相同协议的 20-seed 扩展，不做 one-shot，也不训练 Harness LoRA。下一步若继续，必须
预注册新的 target-side predictive Harness：先在 disjoint adaptation split 学到可由 future
official progress 检验的 failure detector / information need，再把 source knowledge 作为独立
冻结 prior 做增量 ablation。若在 ALFWorld 和第二远域都不能降低 adaptation cost，则删除
cross-domain source benefit claim，只保留 target-trained adaptive Harness 方向。
