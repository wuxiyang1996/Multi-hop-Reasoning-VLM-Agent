# AGQA 2.0 full-distribution selective neural-symbolic transfer：V57–V62

## 结论

现在已经验证的是 **operator-unfiltered AGQA2 distribution 上的 selective
cross-domain transfer**：从游戏 rollouts 归纳出的 typed temporal/recurrent primitives，经过
AGQA-native neural grounding 和 symbolic temporal-window/relation/object composition，在不适用时
保持完全相同的 direct prediction，在适用且通过 abstention checks 时提高 QA success。

最终 fresh cross-split V62 formal replication 的全部冻结 gates 均通过：

| endpoint | split / sampling | rows / videos | applicable / authorized | source / direct | paired rows | video clusters | status |
|---|---|---:|---:|---:|---:|---:|---|
| V60 qualification | official train，exact composite family | 240 / 240 | 240 / 83 | **55 / 39** | **16W / 0L**, `p=1.53e-5` | — | QUALIFIED |
| V61 broad formal | official test，operator-unfiltered | 1,769 / 590 | 760 / 271 | **554 / 490** | **67W / 3L**, `p=4.85e-17` | **61+ / 3−**, `p=2.37e-15` | NOT_QUALIFIED：绝对 loss gate |
| V62 fresh replication | official train，operator-unfiltered | 900 / 300 | 418 / 165 | **290 / 249** | **44W / 3L**, `p=1.23e-10` | **41+ / 3−**, `p=8.09e-10` | **FORMAL QUALIFIED** |

V62 row-level 净增益为 `+41`；negative authorization rate 为 `3/165 = 1.82%`
（冻结上限 `2%`），row win/loss ratio 为 `14.67`（下限 `5`）。video-level negative
cluster rate 为 `3/300 = 1%`（上限 `1%`），cluster win/loss ratio 为 `13.67`
（下限 `5`）。所有未授权行都逐行满足 `source_prediction == direct_response`，所有 runtime
rows 都确认未读取 answer、functional program、scene graph 或 source identity。V62 reported
provider-receipt cost 为 `$4.02023 < $4.50`。

机器可读摘要见
[`results/agqa2_full_distribution_v57_v62_summary.json`](results/agqa2_full_distribution_v57_v62_summary.json)。

## “full” 的准确含义

V57 审计发现此前已验证的 selective routes 只覆盖 official train 的 `45,137 / 1,600,894`
行，即 `2.8195%`。V61/V62 因此没有再按 gold operator、functional-program signature、question
family 或 taxonomy 选择样本：

- V61 覆盖当时全部 590 个仍未使用的 official-test 视频；每视频取最多 3 个最低哈希问题，共
  1,769 行，其中一个官方视频只有 2 题；
- V62 从剩余 official-train 视频中最低哈希选择 300 个全新视频，每视频 3 个最低哈希问题；
- public question parser 只在 runtime 决定 program 是否 applicable；unsupported 或 abstained
  rows 保持 identical direct；
- V61/V62 的主要统计单位是 video cluster，而不是把同一视频的 2–3 个问题视为完全独立。

因此这里的 “full-distribution” 是 **不筛 operator 的确定性视频分层抽样**，不是对官方全部
669,207 个 test questions 逐题运行，也不是声称每一种 AGQA question family 都已由 symbolic
program 解出。V61 applicability 包括 `BEFORE=183 / AFTER=165 / WHILE=363 / BETWEEN=49`；
V62 为 `107 / 106 / 167 / 38`，说明执行路径不是单一 before/after 表面模板。

## 迁移的 neural-symbolic structure

source side 不使用手写 `EXPLORE/BACKTRACK/COMMIT` controller。统一 source-only induction 从
游戏 `(state, action, effect, next_state)` rollouts 产生 typed operators、effect authentication、
recurrent transition 和 abstention contract。V59/V60 为 compound AGQA query 组合两个精确 typed
source programs：

- Candy temporal source 提供 temporal effect/interval primitive；
- Sokoban recurrent relation source 提供 recurrent relation primitive；
- target side 只更换为 AGQA-native neural grounder：anchor interval consensus、symbolic
  interval-to-window executor、isolated relation grounding、window object ontology consensus；
- symbolic executor 只有在 source type/effect contract、cross-view interval consensus、window
  relation/object consensus 和 frozen applicability rule 同时满足时才能覆盖 direct answer。

V60、V61、V62 使用同一个 grounder SHA：
`a5199eab3d4a80431d76a7480258d73bff25298a989880834eabf5460f927e5a`。
V62 明确冻结 `grounder_change_after_v61=false`，所以 V61 outcomes 没有用于修 prompt、模型、
parser、abstention 或 executor。

这验证的是 **source-induced primitives + target-native composition/grounding**，不是 source-only
归纳出完整 AGQA program。generic scaffold 与 target-written equivalent 尚未被正式击败，source
provenance necessity 也未建立。

## 为什么 V61 没过而 V62 合法通过

V61 在任何 outcome call 前预注册 `row losses <= 2` 和 `negative video clusters <= 2`。实际虽然
得到 67W/3L、净 `+64` 和 61/3 positive/negative clusters，两个绝对 gate 都差 1，因此原始
V61 status 必须保留为 `FORMAL_NOT_QUALIFIED`，不能事后改阈值。

问题在于绝对 loss count 不随样本规模扩展：从 240 行扩到 1,769 行仍要求最多 2 个 loss，不能
稳定表达 selective transfer 的 negative-transfer risk。V62 在新数据的任何 provider/outcome call
前公开披露 V61 failure，并把 endpoint 改为可扩展的 rate/ratio gates：

- negative authorization rate `<=2%`；
- negative video-cluster rate `<=1%`；
- row 和 cluster win/loss ratio 都 `>=5`；
- 同时保留最少 applicable、authorization、wins、net gain、row/cluster exact p-value、blindness、
  identical fallback、grounder hash 和 reported-cost gates。

V62 使用 300 个新、与全部已配置/下载视频 disjoint 的 official-train videos。它不是对 V61 同一
数据的重解释，而是 unchanged grounder 在 fresh cross-split data 上的 confirmatory replication。
V61 failed record 与 V62 passed record 必须一起报告。

## 运行与恢复审计

所有 runtime receipt 在 gold evaluator 打开前原子写入。V61 首轮有 15 个 malformed/schema
failures；第二轮恢复 8 个，剩余 7 个确认是失败 provider response 被 task-local call cache 重放。
这些坏 cache 被移动到可恢复隔离目录后，相同 prompt/model/input 重新请求并补齐 1,769/1,769。
V62 同样在 892/900 后只隔离 8 个失败 response cache；6 个首轮恢复，剩余 2 个第二轮恢复，最终
900/900。没有修改 frozen collector、grounder、sample 或 gates。

报告中的 cost 是成功 runtime receipts 汇总的 **reported provider cost**；被隔离的失败 provider
responses 不在最终 report 汇总中，因此不能把该数字解释为账单总成本。

核心回归测试：

```text
pytest -q tests/test_agqa_temporal_localized_query.py \
  tests/test_agqa_program_transfer.py \
  tests/test_agqa_query_object_grounder.py
36 passed
```

## Claim boundary

现在成立：

> 在 operator-unfiltered、fresh、video-clustered AGQA2 distribution samples 上，游戏 source-only
> rollouts 归纳出的 typed temporal/recurrent symbolic primitives，配合 target-native neural
> grounding、symbolic composition 和 fail-closed abstention，相对 identical neural-only direct
> baseline 得到可重复、显著且 negative-transfer rate 受控的 success-rate 增益。

仍不成立：

- 所有 AGQA question families 已解决，或整个 669k test set 已运行；
- full AGQA distribution 的绝对准确率已经足够高；V62 harness accuracy 是 `290/900`；
- source provenance 必要，或 source-induced 明确优于 generic/target-written equivalent；
- video MDP 的多步 action/state-update claim；这里仍是选择性 video QA program execution；
- neural perception 可以被 source symbols 直接迁移；迁移的是 intervention-grounded symbolic
  structure，perception 必须使用 target-native neural grounding。

## 关键文件

- full coverage audit：`scripts/audit_agqa2_full_distribution_v57.py`
- compound IR/grounding：`src/motif_transfer/agqa_temporal_localized_query.py`
- frozen collector：`scripts/collect_agqa2_temporal_localized_query_v59.py`
- V60 qualification freeze：`scripts/freeze_agqa2_temporal_localized_query_v60_qualification.py`
- V61 selection/prereg：`scripts/freeze_agqa2_full_distribution_v61_formal.py`
- V61 evaluator：`scripts/evaluate_agqa2_full_distribution_v61.py`
- V62 fresh replication freeze：`scripts/freeze_agqa2_full_distribution_v62_replication.py`
- V62 rate-gated evaluator：`scripts/evaluate_agqa2_full_distribution_v62.py`
- V61 report：`runs/agqa2_full_distribution_v61_formal/report.json`
- V62 report：`runs/agqa2_full_distribution_v62/report.json`
