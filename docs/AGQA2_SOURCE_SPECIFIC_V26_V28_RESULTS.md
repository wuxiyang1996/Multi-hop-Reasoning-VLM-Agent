# AGQA 2.0 source-specific transfer：V26–V28

## 结论

这轮 powered、预注册实验给出了一个有价值的负结果：

> V25 的 atomic `QUERY_OBJECT` structural mechanism transfer 仍然成立；但当前
> source-controlled third view 相对强 target-only ontology ensemble 的
> **source-specific provenance 没有得到验证**。

V28 使用 120 个全新、video-disjoint test videos，每个 relation group 40 个。grounder 在
development qualification 后冻结为
`19ff586a6f28da233b89401775a266f27174f06fe753f3ebdb7fa267fbf8eb42`；V26 原先定义的
source-specific gates、paired endpoint 和成本上限均未修改。

最终 acquisition `120/120` 完成，但 base mechanism gate 失败：

- decisive executions：`72/120`；
- decisive correct：`38/72 = 52.8%`，低于冻结的 `75%`；
- direct：`14/120`；
- preauthorization typed fallback：`40/120`；
- target-only two-ontology fallback：`36/120`；
- reported provider cost：`$1.0594 < $1.20`。

因为 base mechanism 没有 qualification，统一 harness 正确地 fail closed：executor
authorization 为 `0`，正式 source harness 回退到 direct，因此 formal source-vs-target-only 是
`14/120` vs `36/120`，所有 source-specific primary gates 未通过。

机器可读汇总见
[`agqa2_query_object_v26_v28_summary.json`](results/agqa2_query_object_v26_v28_summary.json)。

## 预注册问题

V25 的 post-hoc audit 中，source-controlled 2-of-3 相对 two-ontology target-only baseline 是
2 wins、0 losses。V26 因此固定 120 rows，并预注册：

- source-vs-target-only wins 至少 5；
- losses 至多 2；
- 净正确数至少 `+5`；
- discordant correctness 的单侧 exact paired `p <= .05`；
- target-only decisive 至少 36；
- base mechanism 自身仍需通过 accuracy、coverage、negative-transfer 和 cost gates。

120 rows 在 V25 的 `2/30` pilot win rate 下，观察至少 5 wins 的名义概率约 90.8%。这不是看到
V28 outcome 后补设的门槛。

## V26/V27 为什么中止

V26 使用未修改的 V24 grounder，完成 `119/120` runtime receipts 后在 `QMIKJ-29239` 的 primary
ontology JSON 上失败。formal report 未创建，gold evaluator 未启动。post-abort、gold-blind envelope
diagnostic 显示 `finish_reason=length`，completion 正好触及 300-token cap。

V27 只把 cap 从 300 提高到 500，在 development 上重新通过，再对完全相同、未评测的 V26 pool
进行 replay；同一 task 仍产生不完整 JSON，V27 也在 `119/120` fail closed，没有 formal outcome。

因此 V28 不继续放宽 parser，也不再重复同一 pool。它只给 `visual_description` 和 `uncertainty`
各加 160-character 上限，保持 decision、confidence、evidence、模型、frames 和 thresholds 不变。
该 grounder 在 development 上得到：

- unified harness `12/18` vs direct `6/18`；
- 6 wins、0 losses；
- decisive `11/12 = 91.7%`；
- 所有 qualification gates 通过。

随后 V28 冻结了另一个 120-video pool，并显式排除 V26/V27 的全部视频。V28 runtime `120/120`
完成，说明 bounded schema 解决了 acquisition failure，但没有解决 semantic calibration。

## Post-hoc candidate audit

为了区分“source signal 完全不存在”与“signal 不足以安全授权”，可在 formal failure 后查看
preauthorization candidate。这个诊断不改变正式结论：

| comparator | correct |
|---|---:|
| matched direct | 14/120 |
| target-only two-ontology fallback | 36/120 |
| source-controlled typed candidate | 40/120 |
| actually authorized unified harness | 14/120 |

typed candidate 相对 target-only 是 5 wins、1 loss、净 `+4`，单侧 exact `p=0.109375`。它没有达到
预注册的净 `+5` 或 `p<=.05`，并出现一次 negative transfer。

relation-group 分解进一步显示它不是广泛优势：

| group | source candidate | target-only | direct |
|---|---:|---:|---:|
| PERCEPTION | 18/40 | 14/40 | 6/40 |
| MANIPULATION_CONTACT | 14/40 | 14/40 | 2/40 |
| SPATIAL_SUPPORT | 8/40 | 8/40 | 6/40 |

净增益全部集中在 PERCEPTION，主要是 `watching -> laptop/mirror`；唯一 loss 是
`lying on -> sofa`，而 target-only/direct 得到 `bed`。所以目前更像一个窄 relation-family signal，
不是统一、domain-general 的 source advantage。

## 正确的 claim boundary

仍可声称：

- game-induced typed structure + target-native grounding 可在 atomic AGQA 子集上改善 direct；
- V25 已通过 30-video structural mechanism qualification；
- bounded output protocol 可以稳定完成 120-video open-answer acquisition。

不可声称：

- source provenance 已被 powered experiment 验证；
- source controller 优于预注册的强 target-only ensemble；
- full AGQA 或 compound temporal reasoning 已解决；
- 多跑一个 reserve seed 就可以修复本次失败。

## 下一步

不应立即运行 V29 reserve。下一步必须先形成新的 development-only hypothesis：学习一个
source-applicability/calibration function，预测 source third view 在 ontology disagreement 时何时可信，
并在多个 held-out development splits 上同时满足 accuracy、coverage 和 negative-transfer gates。
只有该 calibration 在看不到新 test outcomes 时稳定，才值得再冻结一次最终 replication。
