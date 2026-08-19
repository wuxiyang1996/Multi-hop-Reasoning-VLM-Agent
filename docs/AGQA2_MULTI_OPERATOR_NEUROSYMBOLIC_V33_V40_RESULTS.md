# AGQA 2.0 第二类 symbolic program：V33–V56

## 最终结论（V55/V56）

Candy Crush source-induced temporal-function program 与 target-native neural video
grounding 的组合，现已在 AGQA 2.0 atomic `BEFORE/AFTER` 子集上通过 fresh qualification
和一次 video-disjoint official-test formal。相对完全相同的 neural-only direct baseline：

| endpoint | rows | 授权 | source-induced / neural-only | paired result | status |
|---|---:|---:|---:|---:|---|
| V55 fresh train qualification | 300 | 61 | **213 / 203** | **11W / 1L**, `p=.0031738` | QUALIFIED |
| V56 untouched-video test formal | 300 | 47 | **190 / 176** | **14W / 0L**, `p=.0000610` | **FORMAL QUALIFIED** |

V55 与 V56 的 video-ID overlap 为 0；两轮使用完全相同的 V54 artifact、rule、source program、
neural acquisition、target grounder、executor 和 success gates。V56 的全部 14 个 gate 均通过，
reported provider cost `$2.0884 < $2.70`。formal report SHA 为
`3937f0ccb8bf15287e56359cdeb5f4addc6fd3bb9f414630cc8b9bed9abd6369`。

因此，现在可以支持的窄 claim 是：

> 从 game interventions 归纳出的 typed recurrent temporal program，在不把 target answer、scene
> graph 或 source identity 暴露给 target-native neural grounder 的条件下，通过统一 harness
> selectively intervenes，并在新 AGQA2 atomic `BEFORE/AFTER` videos 上把 paired success 从
> 176/300 提高到 190/300。

这仍然不是 full AGQA2、通用 video MDP 或 source provenance necessity 的证明。generic scaffold
和 target-written equivalent 与 source-induced arm 的 300/300 predictions 完全一致；因此结果验证
的是 transferable symbolic structure + target-native neural grounding，而不是“只有 Candy 身份才能
产生该策略”。effect-shuffled source 300/300 abstain、wrong-type source 300/300 abstain，说明 typed
effect/authenticity boundary 是必要的 control。

机器可复核摘要是
[`results/agqa2_asymmetric_support_v55_v56_summary.json`](results/agqa2_asymmetric_support_v55_v56_summary.json)。

## V33–V40 原始 formal 失败

Candy Crush source-induced temporal-function program 到 AGQA atomic
`BEFORE/AFTER` 的第二类迁移**没有通过 formal success-rate 验证**。

| endpoint | source-induced | matched target-native | paired result | status |
|---|---:|---:|---:|---|
| V37 strict development | 73/100 | 67/100 | 6W / 0L，`p=0.015625` | 未过 `minimum_wins=7` |
| V38 aggregate-recurrence method selection | 76/100 | 67/100 | 10W / 1L，`p=0.005859` | development only |
| V40 fresh formal | **68/100** | **65/100** | **5W / 2L，`p=0.2265625`** | **NOT_QUALIFIED** |

V40 在 100 个新、video-disjoint 的 atomic temporal rows 上授权 33 次，净增益为
`+3`。它通过了 receipt integrity、current-outcome blindness、cost、wrong-source、
effect-shuffled、generic-scaffold 和 target-written-equivalent controls，但失败于四个预注册
efficacy gates：最少 wins、最多 losses、最小净增益和最大 p-value。因此不能写成“第二个
source program family 已验证”，也不能和 V32 合并成 broad AGQA claim。

机器可复核摘要是
[`results/agqa2_aggregate_temporal_v40_formal_summary.json`](results/agqa2_aggregate_temporal_v40_formal_summary.json)。
完整报告位于 `runs/agqa2_aggregate_temporal_v41_completion/report.json`，report SHA 为
`45015d0bdf7c1157044c4786af4437f8b526bfa80215aa741580e0b83437b03f`。

## 迁移的结构是什么

source 不是手工写的 `EXPLORE/BACKTRACK/COMMIT` 模板。Candy Crush artifact 来自 source
`(state, action, effect, next_state)` tuples，只使用 source interventions 归纳出：

- `SPARSE_TEMPORAL_EFFECT_FUNCTION`；
- typed `SCORE(TEMPORAL_EFFECT_VECTOR, arity=2)` operator；
- `recurrent=True`；
- 由 H1/H8 effect endpoints 组成的 sparse convex score；
- source qualification 和 shuffled-effect controls；
- program SHA
  `cc5f177fd54a2f17ccaa688ea1ecf6d0ae2395554a97a921563e70c32721601d`。

target 侧仍是 target-native neural grounding：Qwen primary view，加 Gemini rescan/tiebreak，
每次 visual call 只看一个 operand、frames 和 timestamps，不读取 answer、functional program、
scene graph、competing operand 或 source identity。symbolic binder 只在 typed arity、recurrence、
effect authentication 和 interval ordering 同时满足时调用 target-native `before/after` executor；
否则保留 matched direct prediction。

错误的 Columns arity-1/non-recurrent contract 在 100/100 rows 上被 type checker 拒绝；
effect-shuffled source 在 100/100 rows 上 abstain。这说明统一 harness 的 type/effect boundary
工作正常，但这些 controls 不能替代 success-rate endpoint。

## 为什么 V37 差一个 win，V38 不能直接当结果

V33/V37 把 source contract 的 `recurrent=True` 实现为“两个 operand 各自至少有两个 neural
views”，也就是最少 `2+2` evidence。V37 得到 27 次授权、6W/0L，但预注册要求至少 7 wins，
所以正确状态是 `NOT_QUALIFIED`，不能事后降低门槛。

类型审计发现 recurrence 属于整个 arity-2 operator，而不是分别属于两个参数。V38 因此采用
最小 operator-level recurrence：两个参数都 grounded，总 view 数至少 3，所有 cross-view
interval pairs 严格分离且关系一致。这条修改在已打开的 development outcomes 上提出，所以
V38 的 10W/1L 只能用于 method selection；真正检验必须是冻结后的 V40。

## V40 为什么没有复制 V38

post-hoc failure localization 显示：

- V40 中原来的 `2+2` strict subset 只有 16 次授权，得到 3W/0L；方向稳定但 coverage 和
  power 不够；
- aggregate `2+1/1+2` 新增 17 次授权，只贡献 2W/2L，净增益为 0；
- 两个 formal losses 都来自这个新放宽的 asymmetric evidence 区域；
- 因此 V38 上从 6W/0L 到 10W/1L 的增益没有在 fresh videos 上稳定复制。

这不是 effect binding 或 unified harness plumbing 失败；它是 target-native interval grounding
在单侧只有一个有效 view 时 calibration 不稳定。继续调 confidence、挑 event phrase 或重采样
同类 formal pool 都会变成追 outcome，不允许用来修复 V40。

## V34–V41 的运行边界

- V34：98/100 receipts 后遇到两个 schema failures；没有进入 evaluator。
- V35：修正 JSON retry 和 interval envelope，99/100 后发现 evidence chronology failure；没有
  evaluator result。
- V36：排序/去重已有 evidence IDs 后完成 100/100 grounding，但 postground evaluator 缺少旧
  evidence-lineage key；没有 prediction report。
- V37：只补 bookkeeping key，在同一 development pool 上得到完整 strict result。
- V38：outcome-informed operator-level recurrence method selection，不是 formal。
- V39：在 key load/provider call 前因缺少 development dependency field 中止；100 个视频未曝光，
  原样转入 V40。
- V40：100 个 runtime receipts 全部冻结后，legacy base evaluator 读取 gold；report assembly 因
  dependency 缺少 `report_sha256` alias 中止，未输出任何 formal metric，也未进入 source
  prediction loop。
- V41 completion：只增加
  `report_sha256 == development_report_sha256` schema alias，复用 V40 的 sample、adapter、gates、
  predictions 和 evaluator hashes；没有方法或 threshold 修改。最终结果仍记为 V40 formal
  endpoint，而不是新的 resample。

V40/V41 的 assembly caveat 已写入 machine summary。因为最终 endpoint 本身未通过，这个 caveat
更不能被用来弱化失败结论。

## V40 时点可以声称什么

在 V54/V55/V56 发生前，AGQA 正向 claim 只有 V32：

> 在 fresh、atomic、open-answer AGQA `QUERY_OBJECT` 子集上，Sokoban interventions 归纳出的
> recurrent relation program 与 target-native neural grounder 组合后，将 matched baseline 从
> 34/120 提高到 40/120，6W/0L，`p=0.015625`。

V33–V40 当时新增的是一个有价值的边界结果：相同 unified harness 可以执行另一类 game-induced
typed program，且 controls 正常，但其 temporal success gain 未通过 fresh replication。

不能声称：

- source 优于 handwritten generic/target-written equivalent；三者 predictions 完全一致；
- source provenance 是必要条件；
- full AGQA 或 general video MDP transfer 已解决。

## V40 后的合法下一步（已执行）

按预注册 failure policy，V40 method 没有 resample，也没有修改或重解释 V40。实际执行的是第一条
合法路径：仅在新的 AGQA train/development videos 上用有限规则类训练 abstention-only target-native
interval applicability model，经过多个失败 qualification 后冻结 V54，再依次执行 V55 fresh
qualification 和 V56 one-shot formal。

关键文件：

- source/target binder：`src/motif_transfer/agqa_aggregate_temporal_transfer.py`
- syntax normalization：`src/motif_transfer/agqa_operand_normalization_v2.py`
- V38 method evaluator：`scripts/evaluate_agqa2_aggregate_temporal_v38.py`
- V40 collector：`scripts/collect_agqa2_aggregate_temporal_v40_formal.py`
- V40 preregistration：`configs/agqa2_aggregate_temporal_v40_formal_preregistration.json`
- V41 schema-only completion config：`configs/agqa2_aggregate_temporal_v41_completion.json`

## V42–V54：target-native applicability induction（2026-08-19）

V40 失败后没有重采样 formal。后续只在 official train metadata 的新、跨实验 video-disjoint
development pools 上学习一个 **abstention-only target-native applicability model**；Candy source
program、typed operator、target executor 和 neural acquisition grounder 均保持不变。calibrator
只能撤销已有 binding，不能生成/移动 interval、改变 relation 或读取当前 outcome。

| 阶段 | rows | 授权 | source / target | paired | 状态 |
|---|---:|---:|---:|---:|---|
| V44 view-only qualification | 150 | 41 | 104 / 100 | 7W / 3L, p=.171875 | NOT_QUALIFIED |
| V47 interval qualification | 150 | 34 | 110 / 105 | 6W / 1L, p=.0625 | NOT_QUALIFIED |
| V49 temporal-support qualification | 200 | 46 | 142 / 134 | 10W / 2L, p=.019287 | NOT_QUALIFIED（loss gate） |
| V52 directional-support qualification | 250 | 24 | 175 / 168 | 7W / 0L, p=.0078125 | NOT_QUALIFIED（coverage/win gates 各差 1） |

V46 在 150/150 base receipts 冻结后，calibrated evaluator 因缺少旧 core 要求的
`qualified_v33_development_report_sha256` alias 中止。V47 只加入等于既有 V45 artifact hash 的
兼容 alias，未改 rule、gate、sample 或 receipt；V46 不能被重新标成成功。对应 abort receipt 在
`docs/results/agqa2_interval_reliability_v46_runtime_abort.json`。

四轮 model development 是有限类穷举 induction，而不是手写 task-id/词汇过滤：

- V42：8 个 singleton-view subsets；
- V45：192 个 view × gap × endpoint-spread rules；
- V48：768 个规则，增加 48-frame proxy 上的 temporal support span，risk-first 选择；
- V51：2,304 个规则，增加 typed resolved-relation applicability；
- V54：3,072 个规则，允许 `before/after` 使用不同 temporal-support threshold。

V51 在 700 个 consumed rows 上自动选择 `before`、`rescan/tiebreak`、gap≥1、spread≤32、
span≥6，训练为 28W/0L；V52 在新 250 rows 上复制出 7W/0L 和显著 p-value，但预注册还要求
至少 25 次授权和 8 wins，实际为 24 和 7，所以仍不得进入 formal。

V52 之后冻结的 V54 artifact 在 950 个 consumed rows 上从 3,072 条规则中
自动选择 `before span≥6 / after span≥12`，得到 38W/0L。artifact SHA 为
`61679d425b7b221f056e668087fc622aaeccfe6305c863dd24536d424297d241`，rule SHA 为
`ccc436199f314b6e02815862ab7c9bbcfcda192e63c7a4ab406a7c5034962a81`。在该时点它还不能支持
transfer claim；后续 V55/V56 按上述顺序完成了独立验证。

## V55–V56：fresh qualification 与 one-shot formal（2026-08-19）

V55 在任何新 outcome call 前冻结 300 个 official-train rows，预注册 gate 为：300 valid/unique、
authorization≥35、wins≥9、losses≤1、net gain≥8、one-sided exact `p≤.05`、四类 control 全通过、
cost≤$2.70。结果为 61 authorizations、213/300 vs 203/300、11W/1L、净增 +10、
`p=.003173828125`，全部 gate 通过，report SHA 为
`8e81b8b754ca7a174d9fe09e8202e171ce384aecdb2b999868fe32358388e27a`。

V55 通过后才冻结 V56：300 个 official-test atomic rows，排除全部已配置或已下载 video IDs，
selection SHA 为 `d17c907ab4dea25d29fc30cb40763eb44fa804fd125050f73f7b64a4ffd67ab5`。
V56 沿用 V55 的每个 gate，不按 test outcome 改 threshold。结果为 47 authorizations、190/300 vs
176/300、14W/0L、净增 +14、`p=0.00006103515625`，全部 gate 通过。

### V56 assembly caveat

首次 formal collection 有 3 个 provider JSON transport failures；collector 保存 297 个完整
receipts 并 fail closed。相同 frozen calls 只补齐 3 个缺项，没有换 sample。300/300 receipts 固定后，
legacy V34 report path 依次暴露两个行政 schema alias 缺失：qualification summary 缺
`report_sha256`，preregistration 缺 `qualified_v33_development_report_sha256`。原始 frozen config
`83e766ff547a233b80659caeec6aebdda4cc6a03460154154de88cb6a2e9d76e` 保留不动；两个独立
completion audit 证明只改变 dependency/prereg 路径与 alias，grounder、base evaluation protocol、
post-ground protocol、samples、prompts、models、predictions 和 gates 均未改变。第二个 alias 的值
严格等于已冻结 V54 artifact SHA。最终 evaluator 100% 复用原 300 receipts，没有新增 provider call。

这项 caveat 必须随 formal claim 一起报告：第一次失败发生在 outcome loop 已启动、persisted formal
report 尚未生成之后。它是可审计的 report-assembly compatibility repair，不是 outcome-guided method
repair；对应 audit SHA 为 `8fab4eb74c8e944af6fcdc61194a723aaf4df40cdf5dc7d2d9a7a231b30565af`
和 `2914f0c9ab200901342cf13496f8cb3ecd05056f32720f11406c1ca73b454328`。

## 现在的边界与下一步

当前可确认两类 game-induced program 到 AGQA 子域的正向 transfer：V32 `QUERY_OBJECT` 和 V56
atomic `BEFORE/AFTER`。下一步不应再次采样同一 V56 claim，而应做以下之一：

- 用另一 target-native video grounder/model 做锁定规则 replication；
- 在 AGQA 的新 operator family 上重新做 source-only induction → qualification → formal；
- 设计不会与 generic scaffold 完全等价的 source-specific intervention，检验 source provenance；
- 若扩张到“video MDP”，必须加入 multi-step action/state update benchmark，V56 本身不支持该 claim。

V55/V56 关键文件：

- V54 calibrator：`src/motif_transfer/agqa_asymmetric_support_calibrator.py`
- V54 artifact：`configs/agqa2_asymmetric_support_v54/training_artifact.json`
- V55 preregistration：`configs/agqa2_asymmetric_support_v55_qualification_preregistration.json`
- V56 original preregistration：`configs/agqa2_asymmetric_support_v56_formal_preregistration.json`
- V56 original frozen config：`configs/agqa2_asymmetric_support_v56_formal.json`
- V56 evaluator-compatible completion config：`configs/agqa2_asymmetric_support_v56_formal_completion_v2.json`
- V56 report：`runs/agqa2_asymmetric_support_v56_formal/report.json`
