# AGQA2 Layer B：Raw-video end-to-end transfer

## 研究目标

Layer A 使用 official STSG 隔离并验证了 symbolic mechanism；它不能单独支持
“game-acquired skills transfer to raw-video understanding”。Layer B 的目标是验证下面这条
完整、可审计的路径：

```text
raw video
  -> frozen off-the-shelf VLM/tool grounder
  -> typed event graph
  -> target-native semantic slots
  -> game-induced Harness
  -> shared executor
  -> final answer
```

只有在真实视频输入和 matched grounding 条件下，`source_induced` 相对预注册基线在
最终 QA accuracy 上产生显著、可复现且 negative transfer 有界的增益，才允许写：

> Game-acquired skills transfer to video understanding.

否则只能报告 Layer A 的 grounding-isolated mechanism transfer，以及 Layer B 的负结果或
selective development signal。

## 五个 matched arms

正式比较固定为：

1. `neural_only`
2. `generic_scaffold`
3. `source_permuted`
4. `source_induced`
5. `target_written_isomorphic`

五臂必须逐 task 共享完全相同的：

- 原始视频及其精确 frame hashes；
- frame-selection policy 和总 frame-presentation budget；
- frozen VLM/tool grounder 及其输出的 typed event graph；
- question parser 和 target-native semantic slots；
- executor；
- frozen fallback actor；
- answer normalization 和 evaluator。

唯一可替换变量是 symbolic Harness。`target_written_isomorphic` 用于测量相同结构的
target-written ceiling，并要求与 `source_induced` action/prediction equivalence；它不是
独立的 source-provenance 证据。`generic_scaffold` 拥有完整 VM primitive，不得人为削弱。
`source_permuted` 必须破坏 source intervention/effect lineage，而不能只改 source 名称。

## Authority boundaries

在 runtime artifacts 全部冻结前，grounder、parser、Harness、executor 和 fallback 均不得
读取 official answer、official STSG、official functional program 或任何当前 reserve outcome。
问题文本只能用于产生 perceptual obligations 和 operator-free semantic slots；grounder 不得
选择答案或输出 controller action。

Target-native engineering 是允许且必须披露的：frame sampling、visual ontology binding、
typed-event schema、question parser 和 executor adapter 都属于 target-native interface。
可迁移变量必须仍然是从 source game interventions 归纳出的 operators、compositions 和
abstention/commit structure。

## 分阶段协议

1. **Development acquisition**：只在 development videos 上选择 off-the-shelf grounder、
   frame budget 和 outcome-blind evidence verification protocol。
2. **Freeze**：冻结 grounder、parser、executor、fallback、五臂定义、统计检验和全部 gates。
3. **Fresh qualification**：使用 task-、question-、video-disjoint 的新 reserve，只运行一次。
4. **Untouched formal**：仅在 qualification 全部通过后冻结并运行；formal 结果不用于修复。

Fresh qualification 还必须与 semantic-parser 的 train/validation supervision task IDs 严格
disjoint；否则即使 runtime 不读取 functional program，parser 仍可能记忆该问题的 target
semantics。为此已冻结 `parser_disjoint_pool_v1`（1,024,483 rows），后续 v3 只能从该 pool
并排除所有历史 Layer-B cohorts 后抽样。

一个 qualification/formal reserve 一旦读取 outcome，就永久标记为 consumed。不得在同一
reserve 上修改 threshold、router、prompt、frame count、fallback 或 source policy 后重试。

## 预注册 gates

至少要求：

- `source_induced` final QA accuracy 高于 `neural_only` 和 `source_permuted`；
- paired exact McNemar/binomial test（`source_induced` vs `neural_only`）达到 `p < .05`；
- negative-transfer losses 不超过预注册上限；
- 对 **unconstrained accuracy superiority** claim，`source_induced` 必须高于非削弱的
  `generic_scaffold`；
- 对 **risk-constrained transfer** claim，各 symbolic arm 必须使用相同的预注册 loss budget，
  `source_induced` 必须是所有满足该 budget 的 symbolic arms 中 accuracy 最优者；不满足风险
  budget 的 generic arm 仍完整报告，但不能作为 safe policy；
- `target_written_isomorphic` 与 `source_induced` action/prediction equivalence 为 100%；
- 所有 matched-input、authority-boundary 和 provenance audits 通过。

Outcome-blind execution coverage（过去部分日志简称 `intrinsic`）只统计 typed executor 能否
commit；它**不是 grounding accuracy，也不测量 event recall/precision**。即使 coverage 提高，
只要 final QA paired gain 没有通过，Layer B 仍然失败。真正的感知质量必须由独立的
grounding annotation audit 或冻结后的 final-QA evaluation 测量。

## 当前状态（2026-09-01）

- Layer B contracts、operator-free semantic parser、五臂 Harness、shared executor、shared
  fallback 和 provenance tests 已实现。
- parser held-out exact accuracy 为 `99.586%`，validity 为 `99.7285%`。
- 已消费的 `qualification_v2`：outcome-blind execution coverage `78/128 = 60.94%`；最终
  `source_induced 68/128`，`neural_only 63/128`，但只有 `16W/11L`、`p=.4421`，且
  negative-transfer gate 失败。该 cohort 还与 parser SFT task IDs 重叠，因此只能作为
  封存的负诊断，不能重试、转成成功或用于 confirmatory claim。
- `qualification_v3` 已按一次性协议运行并封存为负结果：closed-world source Harness 为
  `112/256`，neural-only 为 `109/256`，但 `27W/24L, p=.77977`，negative-transfer gate
  失败。parser 对该 reserve 的 operator-free semantic target 为 `256/256` exact，因此主要
  失败不是 parser，而是 positive-only event graph 把“未观察到”错误地当成“明确为假”。
- 已实现 source-induced open-world guard：共享视觉工具额外输出 `SUPPORTED / REFUTED /
  UNKNOWN` 原子命题 receipt；只有 source Harness 使用 source interventions 中学到的
  satisfied/failed/unknown guard 结构决定 commit，generic 仍是非削弱 closed-world VM。
- 在永久消费的 v3 上，冻结后的三值 diagnostic 得到 source `118/256`、neural `109/256`、
  generic `113/256`；source 对 neural 为 `20W/11L, p=.14961`。它首次同时高于 neural 与
  generic，并把 losses 从 24 降到 11（通过 5% 安全门），但仍不显著，不能作为 Layer B
  成功证据。
- 已在读取任何新 outcome 前冻结 parser-disjoint、task/video-disjoint 的 `qualification_v4`：
  512 tasks / 512 videos（8 个 semantic roots 各 64）。v4 使用完全冻结的 Qwen32 scoped
  three-value claim grounder、无 confidence threshold 的 source open-world policy 和原五臂
  matched protocol。
- v4 已运行并永久封存。source 为 `258/512`，neural/source-permuted 为 `238/512`，即
  `44W/24L, p=.02053`，且 losses `4.69%` 通过 5% 门；但 unrestricted generic 为
  `264/512`，因此原 all-gates protocol 失败，untouched formal **未授权**。generic 自身相对
  neural 为 `67W/41L`，虽有更高 raw accuracy，但 losses `8.01%` 不满足同一安全约束。
  完整解释与 hashes 见
  [`AGQA2_LAYER_B_RAW_VIDEO_V4_RESULTS.md`](AGQA2_LAYER_B_RAW_VIDEO_V4_RESULTS.md)。
- 已在任何新 runtime/provider call/outcome 前冻结 `risk_replication_v1`：384 tasks / 384
  videos（8 roots × 48），与 acquisition、development、qualification V1–V4 的 task 和 video
  overlap 均为 0，semantic-parser supervision overlap 也为 0。该 replication 使用
  capacity-matched cohort freezer，避免按 root 贪心选择造成的 video collision。
- 新 control 不再把 `source_permuted` 简化成全 fallback：它保留 source 的 15 个 operator、
  32 条 composition edge 和相同 root-aware guard，只对 operator/effect lineage 做固定
  derangement。预注册 primary gates 要求 source 同时显著胜过 neural-only 与这个
  matched-permuted control，loss 不超过 5%，并在所有满足相同 loss budget 的 symbolic arms
  中 accuracy 最优。配置见
  [`agqa2_layer_b_risk_replication_v1_preregistration.json`](../configs/agqa2_layer_b_risk_replication_v1_preregistration.json)。
- 新作业链 Slurm `7419419`–`7419425` 已完成并永久封存。source 为 `174/384`，neural 与
  matched-permuted 均为 `168/384`，方向为正且 loss `18/384=4.69%` 通过安全门，但
  `24W/18L, p=.44080` 未通过两个 significance gates。因此 broad risk replication 失败，
  不能写成 confirmatory pass。完整结果见
  [`AGQA2_LAYER_B_RISK_REPLICATION_V1_RESULTS.md`](AGQA2_LAYER_B_RISK_REPLICATION_V1_RESULTS.md)。
- 已在任何新 runtime/outcome 前冻结 source-compatible typed-temporal replication：256 个
  新 task / 256 个新 video，`duration_choice` 与 `duration_extremum` 各 128；它与 1,408 个
  历史 Layer-B task/video 以及 parser supervision 的 overlap 均为 0。选择依据仅为 source
  artifact 中的 `EFFECT_RANKING + ORDERED_ENDPOINTS`（duration choice 另需
  `GUARDED_BRANCH`），不使用 target answer 或 correctness。
- 新 replication 已完成。五臂共享 Qwen3-VL-32B 48/96-frame receipts、SlowFast、parser、
  router、executor 和 Qwen3.5-9B fallback。`source_induced` 为 `135/256 = 52.73%`，
  neural-only 与 matched source-permuted 均为 `106/256 = 41.41%`；paired `34W/5L`，
  exact `p=2.43e-6`，loss `1.95%`。因此预注册的 selective claim 得到 confirmatory support。
- 额外的 `source_is_best_feasible_symbolic_arm` 门未通过：非削弱 generic target-VM scaffold
  为 `149/256 = 58.20%`。所以允许的结论是 source-acquired ordered-effect structure
  **确实迁移并显著改善 raw-video final QA**，但不能声称它优于更宽的 target-engineered
  controller，也不能外推为 full-distribution AGQA2。详见
  [`AGQA2_LAYER_B_TYPED_TEMPORAL_REPLICATION_V1_RESULTS.md`](AGQA2_LAYER_B_TYPED_TEMPORAL_REPLICATION_V1_RESULTS.md)。

这份协议是 Layer B claim 的约束，不会覆盖
[`AGQA2_FULL_TYPED_OPERATOR_TRANSFER_V1_RESULTS.md`](AGQA2_FULL_TYPED_OPERATOR_TRANSFER_V1_RESULTS.md)
中已经成立但仅限 official-STSG 的 Layer A 结果。
