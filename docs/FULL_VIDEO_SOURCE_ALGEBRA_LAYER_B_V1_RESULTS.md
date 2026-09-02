# Full source-algebra video transfer：AGQA2 + CLEVRER Layer B

## 结论

现在有两类彼此独立、边界清楚的视频证据：

1. **CLEVRER fresh full-benchmark Layer B 已通过。** 在 400 个从未进入历史 runtime 的
   validation videos 上，每个视频固定抽取 descriptive、explanatory、predictive、
   counterfactual 各一题，共 1,600 题。source-induced Harness 相对 frozen Qwen3.5-9B
   graph actor 提升 27.56 percentage points，且 matched-permuted、negative-transfer、
   isomorphic 与 shared-state gates 全通过。
2. **AGQA2 raw-video Layer B 已有两份既有 efficacy evidence，并已接上新的 full source
   algebra provenance。** 512-task broad qualification 与 256-video temporal replication
   都显著优于 neural-only / matched-permuted；新的 outcome-blind compatibility audit 在
   768/768 semantic receipts 上 authentic 全授权、matched-permuted 全 abstain。但 AGQA2
   official test 已没有 untouched video，不能再制造一份新的 video-disjoint official-test
   formal。

这支持的是：**game intervention rollouts 归纳出的 typed symbolic structure，在共享的
target-native neural grounding 下，可以选择性提高 final video-QA accuracy。** 它不证明
source provenance 本身优于一个 extensionally identical target-written controller，也不声称
raw-video QA SOTA。

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

## AGQA2：现有 Layer B evidence 与 full-algebra bridge

### Raw-video evidence

| Cohort | Neural | Source | Generic | Source vs neural | Status under current claim |
|---|---:|---:|---:|---:|---|
| broad, 512 tasks/videos | 238/512 | 258/512 | 264/512 | 44W/24L, `p=.0205` | selective transfer signal |
| temporal, 256 tasks/videos | 106/256 | 135/256 | 149/256 | 34W/5L, `p=2.43e-6` | independent selective replication |

两个实验都使用 raw video，禁止 official STSG/runtime functional program，五臂共享 frames、
grounder、parser、executor 与 fallback；isomorphic equivalence 为 100%，loss fraction 分别为
4.69% 和 1.95%。旧 preregistration 要求 source 必须胜过 generic，因此 artifact status 写作
`GATES_FAILED`；当前目标明确把 generic 当 target-native ceiling 报告，不能篡改旧 status，
也不能把它重写成新 formal。

新的 full-algebra compatibility audit 完全不读 outcome：

- broad 512：8 个 structural families，authentic `512/512 AUTHORIZED`，permuted
  `512/512 ABSTAINED`，report SHA `1d83a025...ddb5928`；
- temporal 256：authentic `256/256 AUTHORIZED`，permuted `256/256 ABSTAINED`，report SHA
  `5bac5dce...e1e51`。

另有 grounding-isolated Layer A 2,400-task result：source 96.67%、Qwen9B neural 81.04%，
379W/4L，覆盖 choose/compare/logic/query/verify；但它使用 official STSG，不能当 Layer B
raw-video evidence。

### 为什么不能再跑 fresh official AGQA formal

最新 outcome-blind inventory 显示：official test 1,814/1,814 videos 已进入历史 runtime，
untouched 为 0；train formal-holdout 777 个中也已有 775 个被消费，只剩 `99XHN`、`EYABM`，
且本地 raw MP4 均缺失。即使取得两段视频，两个样本也无法构成 broad statistical reserve。
因此当前不能满足“新的 video-disjoint official AGQA broad reserve”这一更强要求。

匿名 controller 已对两个既有 raw-video results 完成 substitution replay：broad 为 232 commit /
280 fallback，temporal 为 107 commit / 149 fallback；两者 source/isomorphic、fallback 与
matched-permuted action equivalence 均为 100%。报告 SHA 为
`bd4e607b226f0569e5aff3d598b4383a5063a2f35f8e22a9af4a6a3dba2c076e`。这增强了“同一
source-induced controller 跨 benchmark”证据，但它是 consumed-result audit，不产生新的 AGQA
fresh efficacy claim。

## Validation verdict 与复现

机器审计状态为 `TWO_VIDEO_TRANSFER_EVIDENCE_BUNDLE_VALIDATED`（bundle SHA
`f86998c288a487af83454ee4f551f91fbe24e76d12313ef8dd56ba51294ffc38`）：

- CLEVRER：fresh full-benchmark selective Layer-B formal validated；
- AGQA2：两个相互独立的既有 raw-video selective replications validated，且匿名 controller
  substitution validated；没有新的 untouched official-test formal。

从 repo 根目录复现 source compilation、controller substitution、CLEVRER evaluator、failure
taxonomy、bundle hash audit 与相关测试：

```bash
bash scripts/reproduce_two_video_transfer_v1.sh "$PWD"
```

该命令不重新调用 provider，也不重新采集 raw-video grounding；它审核冻结 artifacts。完整的
raw collection 依赖已记录的 off-the-shelf grounder outputs 与历史 provider calls。

## Paper-safe claim

可以写：

> A source-only typed operator algebra induced from game interventions yields significant,
> low-negative-transfer improvements on a fresh 1,600-question CLEVRER raw-video-grounded
> reserve, and is compatible with independently validated selective AGQA2 raw-video transfer
> results under matched grounding and permutation controls.

不能写：

- full AGQA2 official-test raw-video transfer has a new untouched replication；
- every CLEVRER family receives symbolic transfer（predictive/counterfactual 当前 abstain）；
- source provenance is necessary（target-written isomorphic 与 source 完全相同）；
- generic target-native symbolic engineering is weaker（两个视频 benchmark 的 generic ceiling
  都可能更强）；
- live raw-pixel NS-DR inference was rerun（CLEVRER 使用 content-bound cached off-the-shelf
  predictions）。

## Canonical artifacts

- Source algebra: `runs/full_video_source_algebra_v1/source_algebra.json`
- CLEVRER cohort: `configs/clevrer_full_raw_video_v2_public_cohorts.json`
- CLEVRER shared runtime: `runs/clevrer_full_raw_video_v2/shared_runtime.json`
- CLEVRER neural actor: `runs/clevrer_full_raw_video_v2/qwen9b_graph_actor.json`
- CLEVRER frozen predictions: `runs/clevrer_full_raw_video_v2/five_arm_predictions.json`
- CLEVRER formal: `runs/clevrer_full_raw_video_v2/formal_report.json`
- AGQA broad compatibility: `runs/agqa2_layer_b_raw_video_v1/qualification_v4/full_source_algebra_compatibility_v1.json`
- AGQA temporal compatibility: `runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/full_source_algebra_compatibility_v1.json`
- AGQA current inventory: `docs/results/agqa2_untouched_inventory_v16_20260902.json`
- Anonymous source controller: `runs/anonymous_video_harness_v1/controller.json`
- CLEVRER anonymous substitution: `runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json`
- CLEVRER failure taxonomy: `runs/clevrer_full_raw_video_v2/failure_taxonomy_v1.json`
- AGQA anonymous substitution: `runs/agqa2_layer_b_raw_video_v1/anonymous_harness_substitution_v1.json`
- Two-video paper bundle: `docs/results/two_video_transfer_bundle_v1.json`
