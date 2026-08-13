# WebShop neural-symbolic transfer V9–V11

## Decision

在限定的 claim boundary 内，V11 独立 fresh-goal replication 通过冻结 gate：

```text
REAL_WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_INDEPENDENTLY_VALIDATED
```

这次验证的是 controlled hidden-rule game source 中学到的、逐状态
`TEST versus COMMIT` intervention-value structure，经过 WebShop-native neural
grounding 后，能提高 local WebShop 的 strict success rate。它不是 arbitrary-domain
transfer，也还不是 Sokoban/Thunder raw rollout 直接迁移。

## What transferred

迁移对象不是 source action token 或高层 skill 文本，而是：

```text
target state and budget
    -> target MLP predicts candidate causal outcomes
    -> map target-native candidates to TEST / COMMIT features
    -> frozen source value ensembles compare TEST against COMMIT
    -> select or abstain
```

Source contract 绑定 controlled V3 formal config，迁移
`STATE_DEPENDENT_TEST_VS_COMMIT_MATCHED_INTERVENTION_VALUES`。WebShop target
grounder 是 12-hidden-unit MLP，输入 13 个 target-native action/state features，预测：

- state change；
- termination；
- immediate reward；
- prerequisite progress。

它只使用 WebShop adaptation 的 18 个 counterfactual rows 训练，并在 7 个 calibration
rows 上检查。critical calibration 中，constraint TEST 的 progress prediction 为
`0.903`，satisfied-state COMMIT 的 reward/termination predictions 均约 `0.972`。

这里仍有人工设计：TEST/COMMIT feature correspondence、all-visible-constraints
applicability guard 和 abstention rule 是显式定义的；neural 部分负责 target causal
grounding，source ensemble 负责 state-dependent symbolic value comparison。因此这是
neural-symbolic transfer 的受控验证，不是 end-to-end 自动发现全部 abstraction。

## Experiment chain

- V8 找到旧方法的反事实 grounding failure，并以 fail-closed guard 消除了 harmful
  early commit，但零 intervention、没有收益。
- V9 用真实 target intervention rows 训练 grounder；单个 confirmation goal 上
  authentic reward 为 `.833`，但因只把“任一 constraint satisfied”当作 ready，未达
  strict success。
- V10 改为“所有可见 goal constraints 都满足”才 ready。原始 24 held-out goals 上
  authentic 相对 target 与 source controls 都是 `5–0`，效果方向为正，但双侧 exact
  `p=.0625`，所以诚实状态仍是 `NOT_VALIDATED`。
- V11 在读取 goal text/outcome 前冻结下一段连续 goal IDs `50–81`，不合并 V10
  outcome，并使用同一 runner、grounder、source models 和四个 controls 做独立复现。

## V11 formal result

每个 condition 有 32 个 goals；总计 160 份 receipts。receipt matrix、自身 hash、冻结
runner/grounder hash 和 matched initial-state hashes 均通过完整性检查。

| condition | strict success | rate | mean reward |
|---|---:|---:|---:|
| target-only | 7/32 | 21.9% | 0.5234 |
| target-native myopic | 2/32 | 6.3% | 0.5964 |
| authentic source + target | **16/32** | **50.0%** | **0.6641** |
| shuffled source + target | 9/32 | 28.1% | 0.4453 |
| source marginal + target | 9/32 | 28.1% | 0.4453 |

冻结的 paired gates 全部通过：

| authentic versus | strict W–L–T | success delta | reward delta | two-sided exact p | action-contrast goals |
|---|---:|---:|---:|---:|---:|
| target-only | 9–0–23 | **+28.1pp** | **+0.1406** | **.003906** | 13 |
| target-native myopic | 14–0–18 | **+43.8pp** | **+0.0677** | **.000122** | 25 |
| shuffled source | 7–0–25 | **+21.9pp** | **+0.2188** | **.015625** | 7 |
| source marginal | 7–0–25 | **+21.9pp** | **+0.2188** | **.015625** | 7 |

authentic 对任何 control 都没有 strict loss。它显著超过 shuffled 和 marginal source，
排除了“任意 source prior 都有用”以及“只要 TEST/COMMIT 边际频率即可”的解释；它也
超过只最大化 target MLP immediate reward 的 myopic policy，说明收益不是 neural
grounder 单独造成的。

## Operational retry disclosure

`webshop.73` 首次运行时，target planner 在同一个 cached decision request 上连续返回
不可解析 completion，五个 matched conditions 因共享 target cache 而全部得到同一
schema failure。处理方式是：保留原五份失败 receipt，以空 cache 对完整五条件组做一次
对称 retry；没有选择性重跑某个 condition，也没有改 runner、grounder、model、seed、task
或 gate。

retry 的五个条件初始状态 hash 与原运行相同，全部 strict-fail 且 reward 都为 `.667`。
因此 `webshop.73` 对四个 paired effects 都严格贡献 tie；剔除它、保留首次共同 failure
作为 tie、或采用成功 retry，W/L 和 exact p 完全不变。成功 retry 只使 frozen
`zero_failures` operational gate 可计算为 true。

这个 operational retry rule 没有写在最初的 V11 frozen config 中，所以最严格的表述是：
**treatment-effect evidence 独立复现且对该 retry 不敏感；完整 protocol pass 带一次透明的
对称 provider-retry caveat。** 后续 harness 应在 freeze 前声明 whole-group retry 上限和
provider failure taxonomy。

## What is established

可以支持：

1. 在这个 WebShop setup 中，正确的 state-dependent source TEST/COMMIT relation 能把
   strict success 从 `7/32` 提高到 `16/32`；
2. target-native counterfactual neural grounding 可以把 source structure 落到 WebShop
   constraint click、navigation 和 Buy Now 等完全不同的 action space；
3. 效果依赖正确 source relation，并非只来自 prompt、source marginal 或 target-only
   immediate-outcome model；
4. V10 的 `p=.0625` 趋势在不复用 V10 outcomes 的 V11 fresh goals 上得到独立支持。

仍不能支持：

1. real Sokoban/Thunder rollouts 已经能提供同样有效的 source value ensemble；
2. Video-Holmes/TIR/VLM transfer 已成立；它们的 adaptation preflight 仍失败；
3. 换 WebShop implementation、planner model 或真实网站后效果仍成立；
4. abstraction 和 applicability 是全自动学出的。

## Bitter lessons and next experiment

1. 先验证 target intervention 是否真的改变 observation/reward；observational MSE 不能替代
   off-policy counterfactual calibration。
2. partial purchase reward 不等于 constraints ready；必须把 visible `checked=false` 与
   reward confidence 分开。
3. source controls 必不可少。若只比 target-only，多个“所有 source variants 都成功”的
   goals 会被错误解释为 transfer。
4. 五个 authentic-only source-control wins 的双侧 p 仍是 `.0625`；不能改成单侧或把 V10
   和 V11 临时拼接。V11 的第六、第七个 win 才使冻结双侧 gate 通过。
5. 下一步最有价值的实验不是再扩 synthetic source，而是把真实 Sokoban/Thunder rollouts
   通过 matched interventions 蒸馏到同一个 state-dependent schema；冻结当前 WebShop
   grounder 和 evaluation tasks，只替换 source ensemble，并保留 authentic/shuffled/marginal
   controls。这才能回答“真实 game skill 是否迁移”，而不只是“机制是否可行”。

## Authoritative artifacts

- V9 frozen target grounder：
  `docs/results/webshop_neural_symbolic_v9_frozen_grounder.json`
- V9 adaptation/calibration intervention rows：
  `docs/results/webshop_neural_symbolic_v9_adaptation_grounding_rows.json` and
  `docs/results/webshop_neural_symbolic_v9_calibration_grounding_rows.json`
- V10 frozen protocol：`configs/webshop_neural_symbolic_v10_frozen.json`
- V10 final report：`docs/results/webshop_neural_symbolic_v10_final_report.json`
- V11 frozen protocol：`configs/webshop_neural_symbolic_v11_fresh_replication.json`
- V11 receipt integrity summary：
  `docs/results/webshop_neural_symbolic_v11_receipt_integrity.json`
- V11 operational retry audit：
  `docs/results/webshop_neural_symbolic_v11_operational_retry.json`
- V11 final report：`docs/results/webshop_neural_symbolic_v11_final_report.json`

Frozen V11 summarizer SHA-256：
`35897355ce402d5a5cc174b080348867f75f4a46bcf767a29671f174bdc2fdb2`。

Final report content hash：
`eb186dc1036f86b44788e9ddcf0e2ee8a0f35db443cb8a835de759b033aeaf37`。
