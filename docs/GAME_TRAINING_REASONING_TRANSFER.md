# 游戏训练能否提升远域推理：可归因的 Harness 计划

## 核心研究假设

我们的首要假设不是“游戏动作可以迁移”，而是：

> 在六种游戏上的 skill learning、SFT 或 RL 可能把程序性推理先验写入模型权重；
> 一个在线 Harness 可以在异构目标领域中识别、实例化、验证并安全调用这些能力，
> 从而降低 one-shot / few-shot adaptation cost。

候选能力包括形成可区分假设、获取证据、预测 transition、检查结果、失败恢复和停止。
这些名称只是研究描述，不是 Harness 内置 ontology。是否真的学到，必须由 matched interventions
和 official outcomes 决定。

这一假设不能预先当作事实。当前 88-call post-hoc matrix 中 authentic backbone-eligible 为 0，
只否定“从旧动作轨迹自由分段即可恢复 backbone”这一具体方法；它既没有证明也没有否定
game training 的 weight-level reasoning transfer。

## 必须分开的三个效应

在相同 source snapshot、observation、native actions、decoding seed 和预算上比较：

```text
B       base model，无 game SFT、无 skill context
G−S     game-trained model，skill context masked
G+S     game-trained model，authentic skill context
G+Rand  game-trained model，随机或不相关 skill context
```

由此定义三个可观测 contrast：

```text
weight-level training effect = G−S − B
online skill-context effect  = G+S − G−S
authenticity effect          = G+S − G+Rand
```

`G−S − B` 回答训练是否改变了模型本身；`G+S − G−S` 回答旧 skill bank 是否仍提供在线增量；
`G+S − G+Rand` 排除“只是多给上下文”的解释。不能用其中一个 contrast 代替另一个。

如果可用，还应加入同 token 数的 shuffled-game-SFT / action-only-SFT checkpoint，排除训练量、
格式学习或领域词汇暴露本身的影响。缺少这些 checkpoint 时必须把结论限制为 checkpoint-level
association，而不是训练数据内容的完整因果归因。

## Matched snapshot counterfactual receipt

下一份 source evidence 不再是 post-hoc trajectory segmentation，而是在同一个冻结 snapshot 上调用
多个模型/context treatment。建议 receipt 至少包含：

```text
snapshot hash
observation/native-action hash
base checkpoint hash
game-trained checkpoint + LoRA hash
condition/context hash
exact prompt/template hash
decoding configuration + request seed
raw response hash
parsed reasoning/action/prediction
native-action membership
official transition（仅被执行 treatment）
```

主轨迹只能由预注册 treatment 执行。其他 treatment 默认是 shadow query，没有环境 action authority；
若需要比较真实 outcome，必须从相同 snapshot/prefix 建立独立 replay fork，不能串行污染主轨迹。

实现上还必须限制 inference observer-effect：authentic `G+S` 要先以普通单请求生成并冻结，三个 shadow
controls 只能随后查询，不能把 authentic 放进新的 batch shape。同 seed 的 cross-run identity 只作为
repeatability diagnostic；baseline-repeat 本身也会分叉时，它不能用来识别 shadow effect。仅有 request seed
不足以保证 vLLM 在不同进程/GPU 上逐 token 可复现。

第一轮使用历史 evidence 已经机械记录的完整 `COMMIT/ATTACK` spans，原因不是我们认为 ATTACK
具有某种可迁移语义，而是它在预先存在的 source bank 中有 25 个连续 span / 29 个 transition，足以构造
多步 graph。不能再按 reward、具体 action token 或研究者对画面的判断二次选点。后续扩展仍应覆盖全部
skill IDs，或使用预注册 stable-hash sampling。

当前 collector 在 live rollout 的同一 snapshot 上并行生成四个 treatment；`G+Rand` 只从当步已经由旧系统
检索出的非 authentic candidates 中按 `hash(seed, step, source_skill_id)` 选择，不使用名称、embedding 或人工
语义。主轨迹只执行 `G+S`，四个 shadow action 的 outcome 则由相同 seed、相同真实 prefix 的独立 replay
产生。每个 run 只额外生成两个 JSONL 文件，而不是每帧一个文件。

## 从 contrast 中提取，而不是从动作名称提取

Motif Agent 可以在多个 matched receipts 上提出图，但 node 必须对应重复出现的 treatment difference，
例如某种条件下 proposal set、prediction、replan 或 action distribution 稳定改变。Harness 不解释改变的
语义，只检查：

- snapshot/checkpoint/seed 是否精确匹配；
- authentic 与 controls 的 response/action/prediction 是否真的分离；
- separation 是否跨 episode 重现；
- edge 是否引用连续 lineage 或 observed replay fork；
- action alpha-renaming、bank masking 和 other-skill 后是否仍能解释结果。

单个模型生成的自然语言 rationale、动作序列 self-loop、reward 后验分段和完整轨迹上的自由 alignment
都不能证明 training-induced reasoning。

## Blind source qualification

Discovery 后冻结 graph 和 prediction schema。Qualification / held-out 的每个测试点只暴露：

```text
before-state
native actions
history prefix
checkpoint/condition identity
frozen graph
```

Agent 必须在看不到当前 official after-state、reward 和未来 observation 的条件下输出 current/next-node、
treatment separation prediction 和可机械评分的 transition prediction。随后才揭示 matched model responses
与 official transition，由 Harness 评分。

当前 `FrozenGraphAlignment` 是看完整 execution 后的 topology-fit diagnostic，只用于发现 schema/generic-fit
问题，不再作为 source qualification evidence。

### Phase 1–7 的冻结 gate

本轮先使用完全机械的 effect signature，避免 Agent 给图命名后把通用语言模板混入证据：

```text
(weight_action, weight_state,
 skill_action,  skill_state,
 auth_action,   auth_state,
 weight_reward_cmp, skill_reward_cmp, auth_reward_cmp)
```

node 是 signature 的内容 hash；edge 只连接同一 episode 内相邻的 source-skill snapshots。episode seeds 按
升序 round-robin 固定拆为 discovery / qualification / held-out。只用 discovery 构图，然后原样检查后两组
的 exact node/edge recurrence。这里没有 predicate、动作 ontology 或自然语言 clustering。

Phase 7 分开报告三个 gate，禁止合并叙述：

- `SOURCE_AUTHORITY_ORDER_SAFE`：authentic-first；当前 action 在 shadow 之前已冻结。它不声称未来轨迹
  跨独立 vLLM run 完全相同；
- `SOURCE_CAUSAL_SUPPORTED`：skill context 与 authentic-vs-random separation 在三组都出现；
- `SOURCE_GRAPH_SUPPORTED`：discovery graph 的 node/edge 在两个 blind split 精确重现；
- `SOURCE_VALUE_SUPPORTED`：held-out 的 authentic treatment 有更高 official one-step reward；
- `PHASE7_PASS`：上述 gate 同时成立。

动作或 after-state 不同只能证明 influence，不能证明 reasoning/value。即使 Phase 7 通过，也只允许进入远域
factorial，不能直接声称 reasoning backbone 已迁移。

## 远域 factorial

对每个目标 benchmark，至少运行：

| Decision model | Harness treatment |
|---|---|
| base | off |
| base | generic |
| base | authentic game-derived motif |
| game-trained | off |
| game-trained | generic |
| game-trained | authentic game-derived motif |
| game-trained | shuffled / other-game motif |

三个关键量分别是：

```text
weight transfer       = game-trained/off − base/off
Harness source value  = authentic − generic（同一 Decision model 内）
interaction           = [(G+auth)−(G+generic)] − [(B+auth)−(B+generic)]
```

interaction 为正，才支持“Harness 特别调用了游戏训练所得能力”。如果 base 与 game-trained 都被同样改善，
它更可能是 generic scaffold；如果 game-trained/off 已经改善但 Harness 无增量，则支持 weight-level transfer，
但不支持 Harness 必要性。

## 结果解释

1. `game-trained/off > base/off`，Harness 无增量：存在 weight-level transfer。
2. 两个模型都被 generic/authentic 同样改善：通用 Harness effect，不能归因于游戏训练。
3. `game-trained + authentic` 超过所有 matched controls：最强的 training × Harness evidence。
4. game-trained 更差或 motif 被快速拒绝：negative transfer / over-specialization。
5. identity、blind prediction 或 controls 不完整：inconclusive。

无论哪种结果，都不能默认跳过 target SFT。只有 supervision-matched adaptation curve 证明在相同成功率下
减少 target examples，才可以声称 minimal-SFT 或 no-target-SFT advantage。

## 调整后的执行顺序

1. 冻结并保留 `SKILL_INTERNAL_V1` 的 88-call 负基线；
2. 已定义并实现 matched policy/replay receipts 与 checkpoint/context treatment registry；
3. 已实现 exact live-prompt matched snapshot collector，shadow treatments 无 action authority；
4. 已诊断 ATTACK 的非连续性，并按冻结的结构排名选择 POSITION；
5. 已用 causal signature 构图并测量 `G−S−B`、`G+S−G−S`、`G+S−G+Rand`；
6. 已冻结 discovery graph，在 qualification / held-out 做 blind exact recurrence；
7. source gate 未全部通过，因此不运行远域 positive-transfer factorial；
8. 只有出现 attributable separation，才扩展到四领域八 benchmark。

## Strider Phase 1–7 结果（2026-07-22）

ATTACK 第一轮有 60 个 matched snapshots，但 discovery / held-out 的连续 edge 都为 0；它只有 causal
influence，没有多步 graph，且 held-out 唯一 one-step reward 来自 `G−S`。随后只按三 split 中最小连续
edge 数排序旧系统实际选择的 skill ID：POSITION 为 48/27/47，EXPLORE 为 11/22/12，ATTACK 为
0/1/0，因此机械选择 POSITION。排名不读取 ID 的自然语言含义、动作或 reward。

正式 POSITION job `7121659` 在一张 L40S 上用时 37:37，得到：

```text
6 episodes × 50 steps; protocol failures = 0
155 matched snapshots
620 policy records = 155 × {B, G−S, G+S, G+Rand}
620/620 replay status = INTERVENTION_OBSERVED
parser fallback = 0; action_taking adapter fallback = 0
617 replay startup first-attempt success; 3 second-attempt success
sampling order = AUTHENTIC_FIRST_SHADOW_AFTER_V1 for 620/620
```

三 split 的 `(snapshots, weight effects, skill effects, authenticity effects)` 分别为：

```text
discovery      (54, 36, 38, 44)
qualification  (50, 24, 33, 37)
heldout        (51, 28, 38, 37)
```

因此 checkpoint weights 与 authentic context 都会稳定改变 action/after-state，`SOURCE_CAUSAL_SUPPORTED`
通过。但四 treatment 的所有 one-step official reward 都为 0，`SOURCE_VALUE_SUPPORTED` 失败。authentic
prompt 平均约 981 tokens，random control 约 956–964，并非逐 snapshot exact-token match，因此
authentic-vs-random 仍有小的 context-length confound，不能把 separation 全部解释为语义。

Discovery effect graph 有 6 nodes / 16 edges。Blind recurrence 为：

```text
qualification nodes 42/50; edges 20/36
heldout       nodes 46/51; edges 19/37
```

因为冻结协议要求 exact full recurrence，`SOURCE_GRAPH_SUPPORTED` 失败。最终：

```text
SOURCE_AUTHORITY_ORDER_SAFE = true
SOURCE_CAUSAL_SUPPORTED     = true
SOURCE_GRAPH_SUPPORTED      = false
SOURCE_VALUE_SUPPORTED      = false
PHASE7_PASS                 = false
```

另外，同 seed 的 baseline-repeat 与 baseline-vs-shadow 20-step 诊断都产生 41 个字段 mismatch，说明独立
vLLM 进程/GPU 的 exact trajectory identity 本身不可复现，不能用它识别 observer effect。正式 collector
只作更窄、可保证的声明：当前 authentic action 先冻结，shadow 随后查询；每个 outcome 再由相同 prefix
独立 replay。结论是当前存在 source policy influence，但尚无“有价值且可盲重现的 reasoning backbone”证据，
所以不授权 ALFWorld/远域正迁移实验。
