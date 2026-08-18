# AGQA 2.0 active neural-symbolic transfer：V3–V13 结果

## 结论

AGQA 2.0 已在 V13 的新视频 reserve 上通过预注册的 neural-symbolic success gate。12 个候选
来自 12 个新 Charades videos；排除 93 个既往 manifest 或 aborted preflight 的 video IDs，按
calibrated target evidence 在每个 source type 选择 3 个后才读取 9 个 gold。最终 matched direct
是 `6/9`，统一 harness 是 `7/9`，即 **1 win、0 loss、净 +1**；6 个 authorized decisions 中
5 个正确，decisive accuracy 为 `83.3%`。全部 route、negative-transfer、leakage 和 `$0.10`
成本 gate 均通过，实际累计成本 `$0.084566216`。

这个结果验证的是：匿名 game-induced typed IR 能控制 target-native acquisition 和 selective
authorization，并在一个冻结、video-disjoint 的 AGQA 子集上提高 success rate。它**不验证 source
provenance 的不可替代性**：target-written equivalent 仍是 9/9 dynamics match；因此正确表述是
`structural mechanism transfer + target-native neural grounding`，不是“游戏语义本身被迁移”或
“广义 AGQA 已解决”。

此前 V8 fresh reserve 使用 12 个从未做过 V3–V7 model call 的 Charades videos，先在完全不读
direct/gold/program/scene graph 的条件下按 typed evidence 选择每类 3 个，再读取 9 个 gold。
结果是 3 个 typed decisions、0/3 正确、0 win/2 loss。全部预注册 gate 没有通过，因此
unified harness 授权数为 0，最终保持 matched direct 的 4/9，而不是执行有害的 source program。

## 实现的 neural-symbolic boundary

运行时数据流为：

1. public question 经显式 operand grammar 变成 typed query；无法从问题本身获得 operands 的
   open-ended question 直接 abstain，不读取 official functional program。
2. source-only IR 只决定 acquisition dynamics：relation 和 event-order 可 recurrent rescan，
   duration 是 non-recurrent dual-operand scan。
3. visual grounder 每次只看到一个 operand、时间戳和 RGB frames；看不到完整问题、另一 operand、
   answer、program、scene graph 或 source identity。
4. recurrent source operator 最多允许两次 rescan。primary/rescan 冲突时才触发第三个 isolated
   low-cost view；只有 2/3 observability majority 才能恢复为 OBSERVED/UNOBSERVED，否则 abstain。
5. calibrated executor 在 typed/direct 一致时授权；冲突时只有 local-object corroboration、recurrent
   majority、one-vs-multiple、boundary-aligned nesting 或 repeated-interval dominance 才允许 typed
   override。所有规则在 evaluator 读取 answer/program 之前冻结。
6. qualification 之前 source executor 始终关闭；gate 任一失败后统一 harness 回退到同帧 direct。

QUERY_OBJECT 另加入了本地 OpenCV Zoo YOLOX corroboration。它只看 VLM 定位的 frames；在
development 的 MXATD 上，VLM `pan` 与 COCO `bowl` 独立一致后映射到 AGQA 宽类 `dish`，产生
了一个真实的 direct→typed win。这个映射和模型哈希都进入 grounder semantic lineage。

## 逐版结果

| 版本 | split | 结果 | typed / direct | 关键发现 |
|---|---|---|---:|---|
| V3 | development 9 | qualified | 5 / 4，1W/0L | 本地 object corroboration 产生正增益 |
| V3 | reserve | runtime incomplete | 未评测 | inflection validator 与 provider null envelope |
| V4 | reserve | runtime incomplete | 未评测 | open-ended duration question 没有 public operands |
| V5 | reserve 9 | not qualified | 3 / 2，1W/0L | 只有 5 次 decisive，3/5 正确 |
| V6 | reserve 9 | not qualified | 5 / 6，1W/2L | lexicographic rescan chooser 导致 negative transfer |
| V8 | development 12→9 | qualified | 9 / 6，3W/0L | outcome-used adaptation；不具 confirmatory 意义 |
| V8 | reserve 12→9 | **not qualified** | 2 / 4，0W/2L | evidence rank 不能预测真正 correctness |
| V9 | development 12→9 | qualified | 9 / 6，3W/0L | answer-space fix + calibrated executor |
| V9 | reserve 12→9 | **not qualified** | 5 / 5，0W/0L | 只有 3 次 calibrated decisive；无 gain |
| V10 | development 12→9 | **not qualified** | 7 / 4，3W/0L | 机制 gates 通过，但 model routing 错接使成本 `$0.1078` |
| V11 | development 12→9 | qualified | 7 / 4，3W/0L | rescan/tiebreak 接线修复；成本 `$0.0835` |
| V11/V12 | reserve preflight | aborted before calls | 未评测 | typed arity 与 OR→CHOOSE grammar 漏洞；0 calls |
| V13 | development 12→9 | qualified | 7 / 4，3W/0L | relation-independent OR→CHOOSE requalification |
| V13 | reserve 12→9 | **qualified** | **7 / 6，1W/0L** | fresh video-disjoint success-rate gain |

V13 reserve 的最终哈希摘要在
[`agqa2_active_grounding_v13_reserve_result.json`](results/agqa2_active_grounding_v13_reserve_result.json)。
V8 reserve 的摘要在
[`agqa2_active_grounding_v8_reserve_result.json`](results/agqa2_active_grounding_v8_reserve_result.json)。
V3/V4 runtime failures、V5/V6/V8/V9 reserve 负结果、V10 cost failure 与 V11/V12 preflight
aborts 均保存在 `docs/results/`，没有被覆盖或重新命名为成功。

## Bitter lessons

1. **route/type correctness 不是 semantic grounding correctness。** V8 仍是 9/9 route
   correct，但 `shoe or window` 这类 object-choice surface form 被 parser 当作 EXISTS，导致
   typed executor 输出 `yes`，而正确答案空间是 object noun。
2. **高 confidence 和大 temporal margin 都不可靠。** V8 evidence selector 选出的三个
   decisions 全错；Gemini 对 `holding paper`、`tidying floor` 等状态持续时间明显过延伸。
3. **更多 rescan 不能自动带来更好证据。** primary/rescan 冲突如果被强行择优会产生 loss；
   改成 conjunction 能消除旧 development loss，却显著降低 coverage。
4. **不能用不断换 fresh seed 来制造成功。** V5、V6、V8 三个完整 reserve 均失败；继续迭代
   reserve 会成为 optional stopping，而不是验证。
5. **真正的 blocker 是 target grounder，不是 symbolic IR。** source-permuted 与
   target-written controls 始终通过，但 IR 没有可靠的 target-native predicates/durations 可执行。
6. **recurrent 应表示 consensus dynamics，而不是“再看一次”。** V13 的唯一净增益
   `MYBRN-1024` 来自三视图 majority：typed `carrying bag=yes` 修正 direct `no`，且两次 neural
   views 对 object=`bag` 一致。错误 source type 不能实例化这个 controller。
7. **preflight 是实验的一部分。** V11/V12 在下载/调用前分别发现非 atomic relation operand 与
   OR answer-space 漏洞并终止；没有把不合法 pool 当作新的失败 seed 消费。

## Claim boundary 与下一步

V13 可以进入“小样本、typed-applicability-closed 的 natural-video transfer validated”统计，但
不能外推为 full AGQA、任意 natural video 或 source-specific game knowledge transfer。尤其：

- runtime selector 可读取冻结的 direct response 做 calibration，但不读 answer/program/scene graph；
- freeze 阶段用 functional program 只校验 route/type，故不是 untouched-metadata evaluation；
- target-written equivalent 9/9，说明收益来自可迁移结构，而非游戏 provenance 本身；
- 样本只有 9 个，净增益只有 1 题，需要更大的、一次性冻结 replication 才能估计效应大小。

若继续，下一实验应固定 V13 不再调参，扩大到至少 30 个新 video-disjoint evaluation rows，并把
`QUERY_OBJECT` 作为单独的 grounder qualification 任务，而不是混回已通过的 atomic route。

## V14–V19 replication 更新

上述 30+ row replication 已完成，而且推翻了仅依据 V13 小样本作出的乐观判断。V15、V16、V17、
V19 四次 fresh replication 合计 165 rows；反事实 typed fallback 为 `110/165`，高于 direct 的
`102/165`，但共有 8 次 negative transfer，所有 formal replication 都未通过冻结 gate。由于
unified harness fail closed，实际授权为 0，部署分数没有提高。`QUERY_OBJECT` 按预注册 stop policy
未继续。

完整结论、逐版结果、V14 勘误与下一步见
[`AGQA2_ACTIVE_NEUROSYMBOLIC_TRANSFER_V14_V19_RESULTS.md`](AGQA2_ACTIVE_NEUROSYMBOLIC_TRANSFER_V14_V19_RESULTS.md)。
