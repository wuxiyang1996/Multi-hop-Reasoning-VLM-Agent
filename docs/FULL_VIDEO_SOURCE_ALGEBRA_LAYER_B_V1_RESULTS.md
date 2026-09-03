# Full source-algebra video transfer：AGQA2 + CLEVRER Layer B

## 结论

现在有两类彼此独立、边界清楚的视频证据：

1. **CLEVRER fresh full-benchmark Layer B 已通过。** 在 400 个从未进入历史 runtime 的
   validation videos 上，每个视频固定抽取 descriptive、explanatory、predictive、
   counterfactual 各一题，共 1,600 题。source-induced Harness 相对 frozen Qwen3.5-9B
   graph actor 提升 27.56 percentage points，且 matched-permuted、negative-transfer、
   isomorphic 与 shared-state gates 全通过。
2. **AGQA2 fresh powered broad Layer B 已通过。** 从官方 balanced-train archive 中先按历史
   raw-runtime exposure 排除视频，再哈希冻结并 range-download 180 个新视频；question-only
   parser gate 后整体剔除一个 invalid video block，最终为 179 个视频、1,790 题。相同匿名
   game controller 相对 neural-only / matched-permuted 提升 29 个正确答案，`69W/40L`，
   exact two-sided `p=.00704`；所有 negative-transfer、coverage、shared-grounding 与
   isomorphic gates 通过。

这支持的是：**game intervention rollouts 归纳出的 typed symbolic structure，在共享的
target-native neural grounding 下，可以选择性提高 final video-QA accuracy。** 它不证明
source provenance 本身优于一个 extensionally identical target-written controller，也不声称
raw-video QA SOTA。

### Query Grounder V2 后续资格审查（不替换主表）

为降低 AGQA `query-object` 的 grounding bottleneck，又对一个完全本地、answer-blind 的
target-native grounder 做了独立 powered qualification。它使用 public ontology、stable entity
tracks、typed semantic roles、跨帧 event deduplication、64-frame SGDET 与 48-frame SlowFast；
五臂未来若使用该 grounder，必须共享完全相同的 receipts。

在冻结的 320 个 balanced-train videos / 640 个 query-object tasks 上，固定 threshold `0.60`
得到：entity inventory recall `90.16%`、169 个 supported predictions、其中 116 正确，precision
`68.64%`、95% Wilson lower bound `61.30%`、coverage `26.41%`；role、dedup、authority 与 contract
均为 100%，全部预注册 gates 通过。运行不调用 API。

一次 outcome 打开前的数值合约故障被显式记录：两个 float32 softmax probability 为
`1.0000001192092896`。修复只把 `1e-6` 浮点误差范围内的 probability 规范到 `[0,1]`，更大越界仍
fail closed；样本、模型、checkpoint、帧预算、ontology、ranking 和 threshold 均未改变。

这仍只是 **grounder qualification，不是新的 transfer formal**。资格审查通过后，严格 inventory
audit 发现 official balanced-test 的 1,814/1,814 videos 都已有历史 raw-frame exposure，因而无法再
分配满足 video-disjoint 的 untouched test reserve。不能把复用 test videos 的结果包装成 fresh。
所以论文主表继续使用下文已冻结的 1,790-task five-strata AGQA formal；V2 只作为“grounding 可被
更可靠地局部解决”的补充证据。机器摘要见
[`results/agqa_query_grounder_v2_powered_qualification_v1.json`](results/agqa_query_grounder_v2_powered_qualification_v1.json)。

## Anonymous source controller（新增）

旧聚合 artifact 的 29 个 `PRESENCE`、`FIRST_EVENT` 等名称不再被当作 source learner 的
输出。新的 `runs/anonymous_video_harness_v1/controller.json` 直接汇总六个 game lineage 中
由 `(state, action, effect, next_state)` 学出的 content-addressed state deltas：

- 6/6 source lineages 通过，held-out closed-loop 为 `58/58`；
- effect-shuffled programs 为 `0/58`；
- 自动汇总为 3 个匿名 `OP_*` 和 3 条在 held-out route 中实际观察到的 transition；
- operator ID 只由 preconditions 与 state delta 的内容 hash 得到，不信任输入名称；
- 任意 alpha-renaming 输入 operator labels 后，operator inventory、transition graph 与 control
  metrics 完全不变；
- 未知、零匹配或多匹配统一 fail closed。

在 target 上，adapter 只把一个 target-native symbolic candidate 的 outcome-blind qualification
编码为 `HIGH/LOW`。匿名 source controller 执行 `attempt → commit` 或
`attempt → release/fallback`；它看不到 benchmark identity、question answer 或 source identity。
同一个 controller artifact（SHA
`cd48690e64392f8702e1a239afa1739c25863ce4bd6ae2efea355ac70467bfe6`）未经修改用于两个
video benchmarks。

这也修正了方法边界：**source-induced 的是 operator instance、observed transition 与
abstention behavior；universal typed video VM 以及 target-native grounding/binding 仍是
designer-specified。** 因而不再把 29 个 VM opcode 误称为完全 ontology-free source induction。

## Source-only typed VM capability catalog

`runs/full_video_source_algebra_v1/source_algebra.json` 只读取 source artifacts、held-out
source rollouts 和 source shuffled/permuted controls；没有读取 AGQA/CLEVRER outcome 或 DSL
token。最终 artifact 包含：

- 29 个 typed primitives；
- 73 条 exact output-to-input composition edges；
- 六个 evidence families：typed spatial preconditions、relational goal binding、ordered
  topology、counterfactual search、cyclic identity、multihorizon effects；
- held-out repeat equality `576/576`，固定 candidate derangement 为 `312/576`；
- Streets of Rage 2 与 Strider 未通过 source qualification，作为显式 abstention 保留；
- artifact SHA-256：`c442412f9e4739bfb885d33337e0c8c31ab4bfe94ed495e7ec830d821d2246c3`。

必须区分“归纳 controller program”和“提供 universal VM”。各 source program、support、transition
evidence、control outcome 与 abstention 是从 source intervention artifacts 自动读取/验证的；但
`PRESENCE`、`ORDERED_PATH`、`LOGICAL_AND` 等 canonical type 名及六类 source artifact 到这些
type 的 canonicalization 规则是 designer-specified、target-independent VM capability catalog。
实际 routing 已改用上述匿名 state-delta controller；不得把 VM catalog 表述为完全无 ontology
先验的 program synthesis。

V2 target signature binding 为每个 AGQA family 加入至少一条 source-lineage composition
anchor。测试保证 authentic algebra 对 AGQA/CLEVRER 全部已声明 family 可授权，而保持相同
29 primitives / 73 edges 的 semantic-label derangement 对全部 family 都 fail closed。

## CLEVRER：fresh raw-video-grounded formal

### Pipeline

```text
raw MP4
  -> 32-frame content hashes
  -> frozen cached off-the-shelf NS-DR predictions
  -> compact typed event graph
  -> frozen Qwen3.5-9B fallback OR source-authorized target-native executor
  -> final answer
```

所有五臂共享 raw-video IDs、32-frame budget、NS-DR receipt、public-text operator-free parser、
executor code 和 Qwen9B fallback。只有 Harness 不同：neural-only、generic eager symbolic
ceiling、matched source-permuted、source-induced、target-written isomorphic。

Grounder qualification 在 disjoint development videos 上通过：object inventory exact 92.5%、
attribute micro accuracy 98.52%、observed collision precision 94.19%、recall 92.05%。由于该
qualification 没有验证 future/counterfactual event fidelity，source 在 formal 前固定只对
descriptive 与 explanatory commit，对 predictive/counterfactual fail closed 到同一 9B
fallback。四类题仍全部进入 formal 统计。

### Formal result

| Arm | Correct | Accuracy |
|---|---:|---:|
| neural-only Qwen3.5-9B | 556/1600 | 34.75% |
| source-permuted | 556/1600 | 34.75% |
| source-induced | **997/1600** | **62.31%** |
| target-written isomorphic | 997/1600 | 62.31% |
| generic target-native ceiling | 1123/1600 | 70.19% |

Source vs neural/permuted 均为 `464W/23L/1113T`，net +441，one-sided exact
`p=3.91e-108`；negative-transfer loss fraction 1.44%；source commits 800/1600；isomorphic
prediction equivalence 100%。

逐 family：

| Family | Neural | Source | Generic ceiling | Source action |
|---|---:|---:|---:|---|
| descriptive | 48.75% | **85.75%** | 85.75% | commit |
| explanatory | 10.75% | **84.00%** | 84.00% | commit |
| predictive | 64.25% | 64.25% | 67.75% | abstain |
| counterfactual | 15.25% | 15.25% | 43.25% | abstain |

正式报告 SHA-256：`30ddf5d1a2031b5ed3d7349768b585688bc8c2a1e743ce407a0d550088dbeeed`。
Qwen actor 共 1,600 calls，2,345,134 input tokens、34,643 output tokens，按运行时
OpenRouter 标价约 `$0.24`。USD 数字是历史实验估算，不是 provider receipt 中冻结的 charge；
可审计原始成本量是 1,600 calls 与 2,379,777 total tokens。

匿名 controller substitution 在不读 formal answer 的 frozen predictions 上为 1,600/1,600
exact：800 commit、800 fallback，source-induced、isomorphic 与 fail-closed permuted 行为全部
不变。报告 SHA 为
`b859bc17c26d04668d8914f235d2035e7eac866e3ec78b95c7a3c10ae5d3a937`。

### Observable failure taxonomy

| Outcome class | Tasks | Meaning |
|---|---:|---|
| symbolic recovery | 464 | source correct、neural wrong |
| negative transfer | 23 | source wrong、neural correct |
| committed both correct | 215 | 两臂都正确 |
| committed residual error | 98 | commit 后两臂都错 |
| abstained fallback correct | 318 | fail closed 后 shared actor 正确 |
| abstained generic headroom | 209 | source abstain 且 neural 错，但 generic ceiling 正确 |
| abstained shared failure | 273 | source abstain 且 neural/generic 均错 |

最后两类主要来自 predictive/counterfactual。`committed residual error` 仅是 observable category；
没有额外 intervention 时不能把其中 98 题强行归因给 grounder、parser 或 executor。

V1 consumed diagnostic 必须保留：旧实现把 symbolic explicit executor 错标为
`neural_only`，并让 source 无条件切换到更弱的 trajectory reconstruction，结果 67.0% vs
69.06%、104W/137L。V2 没有在同一 reserve 重试，而是修正 arm semantics 后冻结新的
video-disjoint cohort。

## AGQA2：fresh powered broad raw-video formal

### 为什么是 powered replication

第一份 24-video / 240-task fresh diagnostic 保持完全相同的方法，得到 neural 105/240、source
108/240（6W/3L，`p=.508`）。方向为正且 loss gate 通过，但 discordant sample 只有 9，不能称为
validated。该 cohort 随即封存，未用于调 controller、grounder、threshold 或 gate；只用观察到的
effect size 决定增加独立视频数。

随后从 official balanced-train 的 7,786 个视频 ID 中排除所有历史 raw-runtime exposure 和 exact
parser-supervision tasks，在 1,573 个合格候选视频中哈希选择 180 个。通过 HTTP range 仅下载选中
视频（313MB），全部 content-bound。Question-only parser 为 1,799/1,800 valid；唯一 invalid 的
`OEE36` 在任何 pixel inference 前连同整个 10-task block 一起排除，保留 179 videos × 5 public
structural strata × 2 tasks = 1,790 tasks。

### 共享 Layer-B pipeline

五臂共享：48-frame Qwen3-VL-32B raw event graph、96-frame single-proposition Qwen3-VL-32B
atomic evidence、operator-free Flan-T5 parser、typed executor 和 thinking-disabled Qwen3.5-9B
fallback。Grounder、atomic verifier 与 fallback 都看不到 source controller、official scene graph、
functional program 或 answer。只有 Harness 在 neural-only、generic、source-permuted、
game source-induced、target-written isomorphic 之间变化。

Base grounding 为 1,790/1,790 receipts，1,670 个非空 event graphs、7,026 events；15 个 provider
structured-output failures（0.84%）按冻结 contract 产生空图并 fail closed。Grounding receipts 成本
`$0.8834`，atomic claims `$0.5224`，总 receipt-reported OpenRouter cost `$1.4058`；本地 9B actor
无 API cost。

### Formal result

| Arm | Correct | Accuracy | Symbolic commits |
|---|---:|---:|---:|
| neural-only Qwen3.5-9B | 803/1790 | 44.86% | 0 |
| source-permuted | 803/1790 | 44.86% | 0 |
| **source-induced** | **832/1790** | **46.48%** | 542 |
| target-written isomorphic | 832/1790 | 46.48% | 542 |
| generic target-native ceiling | 833/1790 | 46.54% | 874 |

Source 相对 neural/permuted 为 `69W/40L`，net +29、`+1.62 pp`，exact two-sided
`p=.00704075`；negative-transfer loss fraction 2.23%；source commit coverage 30.28%；
source-permuted 与 neural prediction equivalence 100%，target-written isomorphic 与 source
prediction equivalence 100%。Generic 是 ceiling、不是 pass gate；它只比 source 多 1 个正确答案，
source-vs-generic 为 66W/67L、`p=1.0`。

逐 structural stratum：

| Stratum | Neural | Source | Generic | W/L |
|---|---:|---:|---:|---:|
| choose | 155/358 | 158/358 | 170/358 | 4/1 |
| compare | 198/358 | **219/358** | 218/358 | 39/18 |
| logic | 184/358 | 184/358 | 189/358 | 17/17 |
| query | 60/358 | 60/358 | 60/358 | 0/0 |
| verify | 206/358 | **211/358** | 196/358 | 9/4 |

因此 AGQA claim 是 **broad-distribution selective transfer**，不是每个 family 都改善；主要净收益
来自 compare、verify 与 choose，logic 持平，query 全部 fallback-equivalent。

机器 formal SHA：`89fa9c68ed99ee6ab122a0fd306224a4bfb68e06289c2e71c8012585c5c7e417`；
pre-outcome receipt SHA：`d7420a51df4a77bab1f5b81b7feb88909d58d59f267fa5fc01b0e3d31c7d6827`。
旧 512-task broad 与 256-video temporal results 仍作为独立 replication evidence 保留，但新的
powered result 才是当前 fresh authoritative AGQA Layer-B result。

## Validation verdict 与复现

机器审计状态为 `BOTH_VIDEO_BENCHMARKS_FRESH_LAYER_B_VALIDATED`（bundle SHA
`0d6808e1f7c4a69f73ee438a1eed5bbf05d4edccc318b1492c14a45d2f193d11`）：

- CLEVRER：fresh full-benchmark selective Layer-B formal validated；
- AGQA2：fresh powered balanced-train broad raw-video formal validated，另有两个早期独立
  selective replications。

从 repo 根目录复现 source compilation、controller substitution、CLEVRER evaluator、failure
taxonomy、bundle hash audit 与相关测试：

```bash
bash scripts/reproduce_two_video_transfer_v2.sh "$PWD"
```

该命令可在 fresh checkout 中运行：若本地不存在被 `.gitignore` 排除的完整 AGQA run
目录，它会校验并解压 tracked 的 4.9 MiB portable audit archive，精确重建 bundle 后运行
targeted tests。它不重新调用 provider，也不重新采集 raw-video grounding；portable archive
包含冻结 JSON receipts/predictions，但不包含 raw videos 或 checkpoints。完整 raw collection
仍依赖已记录的 off-the-shelf grounder outputs 与历史 provider calls。

## Paper-safe claim

可以写：

> A source-only anonymous controller induced from game interventions yields significant,
> low-negative-transfer final-QA improvements on fresh raw-video-grounded CLEVRER (1,600
> questions) and AGQA2 broad (1,790 questions) reserves under matched grounding, source-
> permutation, and isomorphic-controller controls.

不能写：

- AGQA2 official-test transfer（fresh result 使用 official balanced-train，不是 test）；
- every CLEVRER family receives symbolic transfer（predictive/counterfactual 当前 abstain）；
- source provenance is necessary（target-written isomorphic 与 source 完全相同）；
- generic target-native symbolic engineering is weaker（两个视频 benchmark 的 generic ceiling
  都可能更强）；
- live raw-pixel NS-DR inference was rerun（CLEVRER 使用 content-bound cached off-the-shelf
  predictions）。
- Query Grounder V2 已在新的 official-test transfer reserve 上验证（所有 1,814 个 test videos
  都已有历史 raw-frame exposure；当前仅完成独立 qualification）。

## Canonical artifacts

- Source algebra: `runs/full_video_source_algebra_v1/source_algebra.json`
- CLEVRER cohort: `configs/clevrer_full_raw_video_v2_public_cohorts.json`
- CLEVRER shared runtime: `runs/clevrer_full_raw_video_v2/shared_runtime.json`
- CLEVRER neural actor: `runs/clevrer_full_raw_video_v2/qwen9b_graph_actor.json`
- CLEVRER frozen predictions: `runs/clevrer_full_raw_video_v2/five_arm_predictions.json`
- CLEVRER formal: `runs/clevrer_full_raw_video_v2/formal_report.json`
- AGQA broad compatibility: `runs/agqa2_layer_b_raw_video_v1/qualification_v4/full_source_algebra_compatibility_v1.json`
- AGQA temporal compatibility: `runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/full_source_algebra_compatibility_v1.json`
- AGQA powered cohort: `runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/public_cohort.json`
- AGQA powered grounding: `runs/agqa2_full_train_broad_powered_v4/qwen32_grounding_full1790.json`
- AGQA powered atomic claims: `runs/agqa2_full_train_broad_powered_v4/atomic_claims_full1790.json`
- AGQA powered fallback: `runs/agqa2_full_train_broad_powered_v4/shared_fallback_full1790.json`
- AGQA powered pre-outcome: `runs/agqa2_full_train_broad_powered_v4/preoutcome_receipt.json`
- AGQA powered formal: `runs/agqa2_full_train_broad_powered_v4/formal_evaluation.json`
- Anonymous source controller: `runs/anonymous_video_harness_v1/controller.json`
- CLEVRER anonymous substitution: `runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json`
- CLEVRER failure taxonomy: `runs/clevrer_full_raw_video_v2/failure_taxonomy_v1.json`
- AGQA anonymous substitution: `runs/agqa2_layer_b_raw_video_v1/anonymous_harness_substitution_v1.json`
- Two-video paper bundle: `docs/results/two_video_transfer_bundle_v2.json`
