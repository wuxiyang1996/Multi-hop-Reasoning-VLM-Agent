# 视频 neural-symbolic transfer：Dynamics / Tools / MDP 设计

## 决策

当前 CLEVRER V1 不能继续沿着 `sample_frames → 重新问一次答案` 调 grounder。它已经是一个有用的
失败诊断，但不是正确的视频 neural-symbolic MDP。V1 adaptation 的结果是：

- baseline `11/18`；
- oracle candidate `12/18`，说明只有很小的单窗口 headroom；
- authentic source `11/18`，与 target-only 相同；
- marginal source `12/18`，反而高于 authentic；
- authentic 只在 `1/18` 个样本上改变 target action。

因此失败不能先解释为“semantic grounder 不够强”。更根本的问题是：V1 没有显式 dynamics state、
typed test outcome、belief transition 或 causal executor。原计划中的 semantic-only V2 不运行。

正确的最小闭环是：

```text
question + low-bandwidth video scout
    -> target-native query graph
    -> target-native world/dynamics particles
    -> typed predicate TEST candidates
    -> source sees only anonymous TEST/COMMIT value features
    -> target-native tool returns a calibrated predicate receipt
    -> Bayesian world-particle update
    -> dynamics / counterfactual simulation
    -> symbolic program execution
    -> native answer belief
    -> repeat TEST or COMMIT
```

迁移对象仍然可以是 intervention-grounded symbolic structure，但 target grounding 必须包括
**perception、dynamics 和 executor**，而不只是把多看几帧后的 VLM answer 当作新 belief。

## TIR 与视频的真实同构和非同构部分

| 项目 | TIR | CLEVRER / structured video |
|---|---|---|
| 环境 | 静态图像，隐藏答案可由局部像素/文字直接揭示 | 固定视频，但答案由隐藏轨迹、事件图和 dynamics 派生 |
| 隐藏假设 | answer slot 本身通常足够 | world/dynamics/event-graph particle；多个 worlds 可导出同一答案 |
| TEST | `zoom/read_text/describe(region)` | `verify predicate(entities, window)`；window 只是底层取帧参数 |
| observation | crop/OCR，通常直接改变答案证据 | object state、track、collision、entry/exit、event order 及可靠度 |
| transition | 直接更新 answer belief | 先更新 world posterior，再运行 dynamics/executor，最后聚合 answer belief |
| COMMIT | 单个 MCQ slot | STAR/NExT-QA 是 MCQ slot；CLEVRER 应提交整题 choice-label vector |
| 规划难度 | 一层 active perception 常可近似 | perception + temporal association + model-based prediction/counterfactual |
| 当前证据 | adaptation preflight `7/16`，baseline/controls `6/16`；非 formal | V1 adaptation fail；尚无 transfer 证据 |

两者真正共享的是：

```text
uncertainty / evidence reliability / budget
    -> 哪个 TEST 的 expected value 最大
    -> 继续 TEST 还是 COMMIT
```

两者不共享 raw tool、时间/空间坐标、object vocabulary、query program 或 dynamics model。

## 为什么视频仍然是 belief MDP，而不是把帧时间当环境 transition

CLEVRER 视频在 episode 开始时已经固定；agent 选择时间窗不会改变视频中的物理世界。因此对 agent
而言，它首先是一个 active-perception POMDP：潜在 world `W` 不变，TEST 改变 epistemic belief。
但 `W` 本身必须包含时间演化：object attributes、trajectories、collision graph、dynamics parameters
和 counterfactual worlds。predictive / counterfactual answer 通过 target-native generative transition
产生，而不是存在于某个尚未看的窗口中。

这可以避免两个概念错误：

1. “看晚一点”不能观察 CLEVRER 的 unseen future；官方 predictive target 需要 dynamics rollout；
2. “移除物体”不是可在 factual video 中执行的 sensing action；它是 simulator 中的 intervention。

## 状态

最小状态定义为：

```text
s_t = (q, {W_i, w_i}_i, H_t, b_t(y), n_t, c_t)
```

- `q`：由问题文本解析出的 target-native query graph；
- `W_i`：world particle，包含 object trajectories、event graph、dynamics/counterfactual prediction；
- `w_i`：该 particle 的 posterior weight；
- `H_t`：已绑定的 typed evidence receipts；
- `b_t(y)`：executor 在每个 `W_i` 上运行后按 `w_i` 聚合出的 native answer belief；
- `n_t`：remaining TEST budget；
- `c_t`：各 probe 的重复次数。

对 CLEVRER，一道题的若干 candidate 不是标准 one-of-K MCQ。正式 unit 应是整道 question：每个
`W_i` 经 executor 得到一个 bit vector，例如 `1010`；`COMMIT(1010)` 的成功条件是整题 exact
match。官方 per-choice accuracy 只作为诊断指标。V1 把每个 candidate 单独变成 A/B，会丢失整题
约束并放大 label-prior baseline，因此不能作为最终协议。

## 动作

顶层保持 source-compatible 的两类 action：

```text
TEST(probe_id)
COMMIT(native_answer)
```

但一个合法 `TEST` 必须已经由 target candidate generator 编译为：

```text
probe = (
    predicate_kind,
    entity_refs,
    time_window,
    target_tool,
    expected_sensor_reliability,
    P(predicate=true | W_i) for every world particle
)
```

首版只允许可观测、可校准的 predicate：

- `OBJECT_PRESENT / OBJECT_ATTRIBUTE`；
- `OBJECT_MOTION / OBJECT_TRACK`；
- `COLLISION`；
- `ENTRY / EXIT`；
- `EVENT_ORDER`；
- `CAUSAL_ANCESTOR` 只在其组成事件都有 evidence 后使用。

`sample_frames` 仍可作为 wrapper 的底层 transport primitive，但它不是 agent 的 symbolic TEST。
`find_moment("something useful")` 或自由文本 window hypothesis 也不够，因为它们没有有限 outcome
space，无法定义 `P(o | W, TEST)` 或可靠 belief transition。

首版不把 `SIMULATE` 暴露给 source controller。每次 TEST 后，target transition 固定执行一次
dynamics/counterfactual executor。以后若研究 compute allocation，可增加 target-only 的
`REFINE_DYNAMICS` option，但不能把它与 sensing TEST 混在同一个 outcome model 中。

## Tool receipt

视频工具必须返回 typed measurement，而不是只返回 frame indices：

```json
{
  "probe_id": "collision:red-sphere:blue-cube@1.0-2.0",
  "predicate_kind": "COLLISION",
  "entity_refs": ["red sphere", "blue cube"],
  "window": [1.0, 2.0],
  "observed_true": true,
  "sensor_reliability": 0.87,
  "evidence_sha256": ["...", "..."],
  "target_native_measurement": {
    "tracks": [],
    "event_time_distribution": []
  }
}
```

硬约束：

1. receipt 的 probe、entities 和 window 必须与 selected action 完全绑定，禁止重绑；
2. reliability 必须从 target adaptation/calibration 得到，不能由回答器自报；
3. gold answer、official functional program、oracle event graph 不得进入 runtime receipt；
4. frame hashes、model/config hash 和 tool arguments 必须保存；
5. 感知失败要返回低可靠度或 abstain，不能生成自由文本“symbolic update”后强行更新。

现有 wrapper 的 `track_object/find_moment/detect_objects_at_frame/sample_frames` 可以提供取帧、
检测和导航基础，但还缺 CLEVRER-native identity association、velocity/collision measurement、typed
reliability 以及 counterfactual executor。不能仅通过把已有 tool 名换成 symbolic label 来绕过这些缺口。

## Observation 与 transition

对 probe `a`，target dynamics ensemble 给出每个 particle 下的 predicate likelihood：

```text
L_i(a) = P(o=1 | W_i, a)
```

收到 typed observation 后：

```text
w'_i ∝ w_i P(o | W_i, a)
b'(y) = Σ_i 1[Executor(W_i, q)=y] w'_i
```

然后更新 event graph、重新 rollout predictive/counterfactual particles，再生成下一轮 probes。source
controller只读取从 `b(y)` 和 `L_i(a)` 计算出的 9 个匿名特征：

```text
is_test
expected_information_gain
expected_map_confidence_gain
predicted_outcome_balance
current_map_confidence
current_entropy
remaining_test_fraction
candidate_hypothesis_probability
action_repeat_fraction
```

这里 information gain 必须在 **derived answer belief** 上计算，而 likelihood 和 posterior 在
**world particles** 上更新。这正是视频与 TIR 之间需要的 compilation layer。

## Reward 和 budget

主要结论应先回答“是否提高 success rate”，因此第一阶段：

- 所有 conditions 使用相同 scout、相同最大 TEST 数和相同底层视觉带宽；
- primary reward 是 official whole-question exact match；
- per-choice accuracy、tool latency 和 token/frame cost 是诊断；
- source 先只负责“同预算下选择哪个 TEST”，避免 test cost 让少看帧伪装成成功；
- target-native headroom 和 action selection 通过后，再冻结一个小 test cost，评估 net return 和
  adaptive TEST/COMMIT timing。

如果不同 condition 实际得到不同总帧数，需要同时报告 fixed-one-TEST matched fork；否则无法区分
“source 选择更好 evidence”与“某个 condition 单纯看得更多”。

## 分阶段实验与停止规则

### Gate 0：source legitimacy

当前 TIR 使用的 frozen source 来自 48 个 synthetic hidden-rule `game` surfaces、9,268 个 matched
value examples。它验证的是 controlled abstract mechanism，不是旧 Thunder/Sokoban rollouts 已学出
该结构。真实游戏日志的结论仍是：one-step value 未通过，delayed value 未识别，explicit recurring
program 未通过。

因此要分开报告：

- **Track A**：controlled source → video，用于验证 MDP compilation mechanism；
- **Track B**：fresh no-hint real-game rollouts → source gate → video，只有 source held-out value、
  recurrence 和 shuffled/marginal controls 先通过后，才允许声称 game-derived transfer。

### Gate 1：target-native perception（不做 transfer claim）

先验证 object binding、tracking、motion 和 observed collision：

- object identity / attribute accuracy；
- track association、position/velocity error；
- collision / entry / exit event F1 和 calibration；
- receipt binding 与 fail-closed invariants。

官方 tracks/event annotations可以作为 evaluator；不能作为 runtime input。

### Gate 2：target-native dynamics / executor headroom

按难度分开，不能把三类题先混在一个 18-sample number 中：

1. explanatory：observed event graph + causal ancestor；
2. predictive：factual dynamics rollout；
3. counterfactual：remove-object intervention + alternate rollout。

做 perception-oracle、dynamics-oracle、program-oracle 三个诊断 ablation，定位瓶颈。只有完整
target-native pipeline 明显优于 overview answer baseline，且 oracle probe 显示足够 action headroom，
才进入 transfer preflight。

### Gate 3：adaptation transfer preflight

冻结 question/video-disjoint split 和全部 gates，再比较：

- target-only learned VoI selector；
- authentic source + target-native grounding；
- within-state shuffled source；
- source marginal；
- random typed probe；
- oracle probe（只用于 headroom，不参与 policy）。

必须同时满足：typed evidence response、authentic action contrast、authentic 优于 baseline、authentic
优于合理 target-only、authentic 优于 shuffled/marginal。否则不读取 qualification/held-out。

### Gate 4：benchmark 顺序

1. **CLEVRER**：最适合验证 perception → event graph → dynamics → executor 机制；
2. **STAR**：有 natural video、situation hypergraph 和 functional programs，最适合验证同一接口能否
   从 synthetic video 迁到 natural video；
3. **NExT-QA**：用最终 causal/temporal QA success rate 验证，但缺少同等级的显式 state supervision；
4. **Video-Holmes**：更依赖 audio、narrative latent causes 和 commonsense，现有日志已说明它不适合做
   第一个机制 benchmark。

## 已落地的最小状态机

`src/motif_transfer/video_dynamics_mdp.py` 已实现：

- world particles 到 native answer/bit-vector belief 的聚合；
- typed predicate probe 及逐 particle likelihood；
- expected answer information gain；
- calibrated receipt 的 Bayesian update；
- 与 frozen source 完全对齐的 9D TEST/COMMIT features；
- budget、repeat count、receipt/action binding 和 evidence hash invariants。

这只是正确 harness 的 symbolic spine，不等于 target perception/dynamics 已经完成。下一项工程工作应是
CLEVRER-native grounder 和 executor adapter，而不是再训练 semantic candidate-uplift MLP。

## 2026-08 三 benchmark structured-video adaptation 结果

实现 `structured_video_transfer.py` 后，在 video-disjoint frozen adaptation splits 上完成了三个真实
benchmark 的 matched probe collection。共同协议为：4 个 neural world/event hypotheses、3 个 typed
predicate probes；probe grounder 只看 predicate、entities、normalized temporal window 和 focused
frames，完全不看 question、options 或 gold。每个 condition 在相同 TEST budget 下用同一 receipt 做
Bayesian world transition，再 COMMIT native answer。STAR 使用 16 个官方 Charades clips，NExT-QA
使用 12 个官方 NExTVideo clips，CLEVRER 使用 12 个官方 validation videos。

### One-TEST 结果

| benchmark | baseline | target IG | target expected accuracy | authentic | shuffled | marginal | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLEVRER | 2/12 | 2/12 | 2/12 | 2/12 | 0/12 | 1/12 | 3/12 |
| STAR | 9/16 | 9/16 | 10/16 | 10/16 | 7/16 | 10/16 | 11/16 |
| NExT-QA | 6/12 | 7/12 | 7/12 | 6/12 | 6/12 | 5/12 | 8/12 |

三者均为 `ADAPTATION_PREFLIGHT_FAIL`，qualification/held-out 均未读取：

- CLEVRER 的 gold whole-question answer 仅被 world particles 覆盖 `6/12`。这是 dynamics/world
  generation failure；selector 不能在不存在正确 world 的情况下修复答案。
- STAR authentic 比 baseline/IG/shuffled 高，但与一个更强的 target expected-accuracy selector和
  source marginal 完全相同，不能归因于 authentic source state-action correspondence。
- NExT-QA target-native typed probing 从 `6/12` 提到 `7/12`，oracle 为 `8/12`；但 authentic 只有
  `6/12`。这里 target mechanism 有正 evidence response，而 source 是 negative transfer。

### Two-TEST sequential reanalysis

同一批 matched receipts 支持不调用模型的 `TEST → update → replan → TEST → COMMIT` 重放：

| benchmark | target IG | target expected accuracy | authentic | shuffled | marginal | oracle |
|---|---:|---:|---:|---:|---:|---:|
| CLEVRER | 1/12 | 0/12 | 1/12 | 1/12 | 1/12 | 4/12 |
| STAR | 9/16 | 9/16 | 9/16 | 8/16 | 9/16 | 9/16 |
| NExT-QA | 7/12 | 7/12 | 7/12 | 5/12 | 7/12 | 9/12 |

增加 TEST 数不是单调有益：感知 receipt 仍有噪声，错误 measurement 会使 posterior 变差。因此正式
MDP 最终需要 reliability-calibrated adaptive TEST/COMMIT，而不是硬编码“多看总会更好”。不过两步
结果仍没有 authentic 严格超过 target expected accuracy 与 marginal，说明 blocker 不只是 one-step
budget。

### 当前可支持与不可支持的结论

可支持：

1. 三个 benchmark 已共享同一个 typed-probe/world-particle neural-symbolic MDP，并在真实视频上执行；
2. STAR/NExT-QA 的 world coverage、typed evidence 和 Bayesian transition 可产生 answer headroom；
3. authentic source 通常优于 shuffled，说明 source correspondence 并非完全任意；
4. target-native expected accuracy 是必须加入的强 control，不能只与 information gain 比。

不可支持：

1. 当前 controlled synthetic-game value prior 在任一视频 benchmark 上产生独有 success-rate gain；
2. 更多 TEST 自动改善视频 QA；
3. CLEVRER 在没有 target-native dynamics generator/executor 时能靠 generic VLM world particles解决；
4. 真实 Thunder/Sokoban rollout skills 已迁移到视频。

下一项原则性工作不是调 threshold/seed/MLP，而是：CLEVRER 接入 target-native dynamics predictor；
STAR 利用 adaptation situation graphs监督一个 oracle-free graph grounder；NExT-QA 使用 relation
annotations校准 event receipts；source 端则需要先产生 transition-aware multi-step game value，而不是
继续扩大静态 active-identification synthetic domains。

## 2026-08 严格三 benchmark 结论：执行完成，transfer 未通过

后续实验把 source 换成真实游戏 source gate 已通过的 typed IR：

```text
BIND --[CARRIER_BOUND]--> RELATE
  source: MiniGrid PutNear + MiniWorld PutNext

BIND --[CARRIER_BOUND]--> MUTATE
  source: MiniGrid DoorKey + UnlockPickup
```

source gate 为 `24/24` cells passed、`174` matched forks、`0` replay mismatch；冻结 summary SHA256
为 `1386d2023dbf1ebac88992dd49cad1aa1e323299a0937c4e2dd3bf4505613969`。target 侧实现了：

- candidate-factorized compiler；
- separate-frame neural BIND tracker；
- 独立 identity audit；
- matched global+zoom RELATE / before→after MUTATE grounder；
- exact target-only、reversed、wrong-guard、node-only、source-marginal、shuffled controls；
- outcome-blind family transfer-utility calibration和 prospective frozen qualification。

因此本轮不再是旧的“高层 skill 名塞进 prompt”。source edge 真实、target grounding neural、symbolic
guard 可执行；但严格 attribution 仍失败。

### 最终结果

| benchmark / phase | baseline | target-only | authentic source | strongest causal control | 结论 |
|---|---:|---:|---:|---:|---|
| CLEVRER NS-DR adaptation | 0/12 | 8/12 with-edge | 8/12 | 8/12 same architecture | target headroom；source contribution 未识别 |
| STAR BIND→RELATE adaptation | 6/16 | 10/16 | 9/16 | 10/16 reversed/wrong | authentic 低于 target/control |
| STAR BIND→MUTATE adaptation | 6/16 | 6/16 | 7/16 | 7/16 wrong/node/marginal/shuffled | +1 不能归因于正确 edge |
| NExT-QA selective adaptation | 8/12 | 7/12 | 9/12 | 7/12 | adaptation gate 通过 |
| NExT-QA frozen qualification | 18/18 | 16/18 | 16/18 | 16/18 | prospective negative transfer |

CLEVRER 使用官方 neural dynamics predictions 与 symbolic executor；官方 functional program 是明确
披露的 program-oracle runtime input，answer labels 仅供 evaluator。without-edge 为 `7/12`，with-edge
为 `8/12`，证明 target-native dynamics/executor 有真实 headroom。但 authentic compilation 与完整
same-architecture target control 是同一个 pipeline，所以不能把 `8/12` 归因于 source。

STAR 的 RELATE verifier 本身把 baseline `6/16` 提到 `10/16`，但 authentic 为 `9/16`。改用 source
中另一个真实 edge `BIND→MUTATE` 后，显式测量 candidate-specific `before_state→after_state`；只有
identity audit 通过且 bound/unbound 变化至少 `0.05` 才允许执行，否则 fail closed。authentic 得到
`7/16`，确实高于 target-unbound `6/16`，但 wrong-guard、node-only、marginal 与 shuffled 全部也是
`7/16`。这说明多出的一个 correct answer 来自 bound observation/target neural evidence，而不是
正确的 source edge 或 correspondence。

NExT-QA adaptation 上，用 leave-one-video-out、held row 完全排除的 family utility 得到 authentic
`9/12`，baseline `8/12`，其余 edge controls最多 `8/12`，所以在读取 qualification 前冻结了 policy：
只在 Causal family 使用 intervention。policy SHA256 为
`893415c58936b32bda87e336975dfe5b7b52f2674ff6677ee7de888f0fa19062`。随后恢复并逐个解码验证
`18/18` 个冻结 qualification videos，运行全部 matched forks。结果 baseline `18/18`；六个 Causal
interventions 使 authentic、target、reversed、wrong、marginal 和 shuffled 都变为 `16/18`。因此
adaptation gain 没有 prospective generalize，而且所有 edge conditions 完全相同。

qualification 的 base prior 使用了更小、但 outcome-blind 的 scout collector，而不是 adaptation
world-model+probe 的完整 prompt contract。这会阻止把 baseline 从 `8/12` 到 `18/18` 解释为同一模型
的纯 split generalization；它不会挽救 transfer claim：这个更强的 target-only baseline 已饱和，且
authentic 与 target/错误边 controls 仍完全相同。qualification policy 未作任何修改；held-out 始终
未读取。

### Bitter lesson

当前 transferable artifact 仍然太弱。它只提供拓扑
`BIND→RELATE/MUTATE` 与 `CARRIER_BOUND` guard；一旦 target-native pipeline 已实现同一拓扑，source
版本和 same-architecture target control必然相同。反过来，当 neural BIND 有噪声时，graph topology
又不足以校准“何时相信这个 handle”，所以 marginal/shuffled controls 会复制 authentic 的行为。

因此不能靠增加 prompt、family rule、post-hoc threshold 或更多同类 target forks把当前结果调成
transfer。下一版 source 必须迁移更低层且会改变决策的量：

1. 从更大游戏 rollouts 学出的 calibrated abstract transition operator，而不只是 edge 名；
2. source-trained guard/option-value 参数，在 target adaptation 只学习 neural grounding map；
3. source 参数必须在 prospective target rows 上产生与 marginal/shuffled 不同的 action；
4. 仍要求 authentic 严格超过同预算 target learner和全部 causal controls。

在得到这种 source artifact 前，三个 video benchmark 的正确状态是
`TRANSFER_NOT_VALIDATED`。这里“worked out”指机制、对照和 prospective gate 已经跑通并给出可复现的
否证，不是 success-rate transfer 已成立。

## V7/V8 video-only repair：旧 qualification 结论被更严格 adaptation 取代

对旧日志复查后确认了三个会夸大或掩盖效果的问题：

1. candidate compiler 虽然输出 `window_fraction`，collector 实际仍对整段视频均匀抽帧；90 秒左右的
   NExT-QA 视频会漏掉短动作；
2. wrong/shuffled correspondence 经常复用同一个 carrier，因而 control 在像素和 action 上是 no-op；
3. failed BIND guard 仍回退执行 unbound RELATE/MUTATE，等价于给错误 control 一次额外 target-native
   observation，而不是 MDP 中正确的 no-op transition。

V7/V8 只修改 video 路径，未修改 TIR、WebShop 或 ALFWorld。新执行契约是：

```text
question text without options + whole-clip scout
    -> broad temporal localization
    -> dense decode of 48 frames inside that window
    -> independent carrier BIND + identity audit
    -> independent visually distinct decoy BIND + identity audit
    -> matched unbound / authentic / wrong / shuffled RELATE or MUTATE

failed symbolic guard -> NOOP_TO_BASELINE
```

localizer 从未看到 options、gold、official program 或 question family；所有条件使用同一 localization、
frame count、resolution 和 label layout。wrong control 的 decoy panel hash 必须与 authentic 不同；严格
gate 要求每个 sample 都产生 action contrast，且至少 90% candidates 的 observation 不同。实际
NExT-QA、STAR、CLEVRER 的 distinct fractions 分别为 `1.000`、`0.9375`、`1.000`。

### 修复后 adaptation 结果

| benchmark / typed edge | baseline | target-unbound | authentic | wrong | shuffled | strict gate |
|---|---:|---:|---:|---:|---:|---|
| NExT-QA BIND→RELATE | 8/12 | 8/12 | 8/12 | 8/12 | 1/12 | fail |
| STAR BIND→RELATE | 6/16 | 11/16 | 6/16 | 7/16 | 2/16 | fail |
| STAR BIND→MUTATE | 6/16 | 12/16 | 7/16 | 7/16 | 3/16 | fail |
| CLEVRER BIND→RELATE | 0/12 | 4/12 | 4/12 | 1/12 | 2/12 | fail |
| CLEVRER BIND→MUTATE | 0/12 | 1/12 | 0/12 | 1/12 | 1/12 | fail |

这些结果解决了“benchmark 为什么不 work”的工程问题，也把科学结论变得更清楚：target-native video
program 确实有 headroom，例如 STAR unbound MUTATE 为 `12/16`、CLEVRER RELATE 为 `4/12`；但
source-transferred BIND routing 没有严格超过同预算的 target-native control。STAR 的旧 `7/16 > 6/16`
不再构成 transfer evidence，因为强 target control 是 `12/16`。CLEVRER authentic 与 target 都是
`4/12`，只能证明 neural-symbolic target program 有效，不能识别 source contribution。

因此没有冻结任何 post-hoc family/threshold selector，也没有打开新的 confirmation。新的 6-per-family
confirmation 和 4-per-family reserve 已在读取 outcome 前冻结；完整 baseline collector 也已改为复用
adaptation 的同一个 `_propose_world_model` prompt contract，但 adaptation gate 未通过，所以没有运行。
confirmation/reserve 仍 sealed。机器可读结论见
`docs/results/video_v7_v8_adaptation_summary.json`。

当前 blocker 已不是 video wrapper、sampling 或 control execution，而是 source artifact 本身只有
`execution_authority=SYMBOLIC_ROUTING_ONLY`。下一次值得运行的实验必须先从 source games 学到会改变
decision 的低层参数，例如 calibrated guard reliability、transition likelihood 或 option value；只继续
迁移 `BIND→RELATE/MUTATE` 的拓扑，无法与 target-native 同构 executor 做非平凡区分。

## 2026-08 V37 matched natural-video formal 与 cost stop

后续 V37 已经完成此前缺少的 matched-model、matched-frame、video-disjoint 正式测试，不应再被描述为
“尚未运行 fresh natural video”。STAR + NExT-QA 共 `201` questions、`28` video clusters；direct 与
typed proof 都使用 Gemini 3.1 Pro 和同一组 24 frames，与 V36 adaptation 的 sample/video overlap 都为 0。

结果为 direct `150/201`、raw typed proof `151/201`（8W/7L）、authentic Sokoban source CATE
`149/201`（3W/4L）；inverted source control 为 `152/201`。全部 13 个 formal gates 失败。402 条
provider-call usage records 合计约 `$11.63`。这说明 blocker 不是帧数、样本数或 generic candidate
grounding，而是 raw target evidence operation 只有 `+1/201` headroom，且 authentic source 没有超过
inverted/marginal controls 的独特 applicability value。

因此 natural-video 新调用暂停，Video-Holmes 降级；CLEVRER `511/720 vs 489/720` 仍保留为 structured
event-graph positive boundary。完整结论与机器审计见
[`WEBSHOP_V14_AND_VIDEO_COST_BOUNDARY.md`](WEBSHOP_V14_AND_VIDEO_COST_BOUNDARY.md) 和
[`results/natural_video_cost_boundary_v38_audit.json`](results/natural_video_cost_boundary_v38_audit.json)。
