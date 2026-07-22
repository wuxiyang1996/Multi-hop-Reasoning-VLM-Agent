# Implementation Status

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

## 尚未获得授权的 claim

- 尚无 `SOURCE_SUPPORTED` motif；
- 尚未证明 old co-evolved 9B 优于 base 9B；
- 尚未证明 authentic motif 优于 generic/shuffled/other-source；
- 尚未授权 ALFWorld large-scale experiment；
- 尚未证明可以跳过 target SFT。

只有 fresh source evidence 通过审计、frozen 9B comparison 和 source qualification 后，才进入 ALFWorld one-shot vertical slice。
