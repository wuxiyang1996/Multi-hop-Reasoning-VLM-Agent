# AGQA 2.0 第二类 symbolic program：V33–V40

## 结论

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

## 当前可以声称什么

仍然成立的 AGQA 正向 claim 只有 V32：

> 在 fresh、atomic、open-answer AGQA `QUERY_OBJECT` 子集上，Sokoban interventions 归纳出的
> recurrent relation program 与 target-native neural grounder 组合后，将 matched baseline 从
> 34/120 提高到 40/120，6W/0L，`p=0.015625`。

V33–V40 新增的是一个有价值的边界结果：相同 unified harness 可以执行另一类 game-induced
typed program，且 controls 正常，但其 temporal success gain 未通过 fresh replication。

不能声称：

- Candy temporal program 已提高 AGQA `BEFORE/AFTER` success rate；
- 两种 source-induced program families 都得到 confirmatory validation；
- source 优于 handwritten generic/target-written equivalent；三者 predictions 完全一致；
- source provenance 是必要条件；
- full AGQA 或 general video MDP transfer 已解决。

## 下一步

按预注册 failure policy，当前 V40 method 不再 resample。若未来重开 temporal track，应先在新的
AGQA train/development videos 上训练 target-native interval uncertainty model，使 asymmetric
`2+1/1+2` evidence 能预测自身错误；或者从 source rollouts 归纳与 Allen interval relation 更直接
同构的 program。完成后必须冻结新的 method identity，再开一次独立 formal；不能修改或重解释
V40。

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

V52 之后的 V54 artifact 只能算下一候选方法：它在 950 个 consumed rows 上从 3,072 条规则中
自动选择 `before span≥6 / after span≥12`，得到 38W/0L。artifact SHA 为
`61679d425b7b221f056e668087fc622aaeccfe6305c863dd24536d424297d241`，rule SHA 为
`ccc436199f314b6e02815862ab7c9bbcfcda192e63c7a4ab406a7c5034962a81`。它尚未经过新的
video-disjoint qualification，因此 **不能宣称 temporal transfer gate 已通过**，更不能开 test
formal。下一合法步骤是冻结 V54 的独立 train qualification；只有所有 gate 同时通过，才能原样
冻结 untouched official-test formal。
