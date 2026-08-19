# AGQA 2.0 active neural-symbolic transfer：V14–V19 最终审计

## 最终结论

AGQA 2.0 pipeline 已经能够端到端运行，但严格的 game-to-AGQA selective transfer **没有通过验证**。
最终冻结的 V19 fresh reserve 在 60 个新、跨实验 video-disjoint 样本上完成 `60/60` typed routing，
39 次 decisive execution 中 26 次正确；若事后无条件采用 typed fallback，会将 matched direct 从
`35/60` 提高到 `38/60`，即 6 wins、3 losses、净 `+3`。由于预注册要求 0 loss，V19 未通过；
fail-closed unified harness 因而授权 0 次，最终仍为 `35/60`。

所以应区分两个结论：

- **工程结论成立**：同一 typed IR、target-native visual grounding、symbolic execution、controls、
  qualification 和 fail-closed deployment 能在 AGQA 跑通。
- **科学结论尚不成立**：现有 grounder 无法可靠预测何时 symbolic override 优于 direct，不能声称
  已提高部署后的 AGQA success rate。

完整、带哈希的机器可读汇总见
[`agqa2_active_grounding_v14_v19_summary.json`](results/agqa2_active_grounding_v14_v19_summary.json)。

## V14–V19 发生了什么

| 版本 | 数据与角色 | decisive | typed / direct | wins / losses | gate |
|---|---|---:|---:|---:|---|
| V14 | 36-candidate replication preflight | — | — | — | 调用前中止 |
| V15 dev | 复用 V13 development receipts | 6/9，6 correct | 7 / 4 | 3 / 0 | qualified |
| V15 fresh | 30 个 V14 未调用样本 | 23/30，18 correct | 24 / 21 | 5 / 2 | **not qualified** |
| V16 dev | V15 fresh 仅作 development | 21/30，18 correct | 26 / 21 | 5 / 0 | qualified |
| V16 fresh | 30 个新 videos | 19/30，15 correct | 20 / 19 | 1 / 0 | **not qualified**：coverage/wins 不足 |
| V17 fresh | powered 45 个新 videos | 27/45，18 correct | 28 / 27 | 4 / 3 | **not qualified** |
| V18 dev | Claude 独立 adjudicator，7 个 V17 overrides | 0/7 judge correct | 1 / 3 final | 0 / 2 retained | **not qualified** |
| V19 dev | temporal-only selective rule，复用 V17 receipts | 23/45，17 correct | 30 / 27 | 3 / 0 | qualified |
| V19 fresh | 最终 60 个新 videos | 39/60，26 correct | 38 / 35 | 6 / 3 | **not qualified** |

四次完整 fresh replication（V15、V16、V17、V19）合计 165 rows：route `165/165`，direct
`102/165`，反事实 typed fallback `110/165`，共 16 wins、8 losses，表面净增益 `+8`。但是所有
formal replication 都没有通过各自冻结 gate；统一 harness 总授权为 0，实际部署分数仍是
`102/165`。这些 fresh runs 的累计 reported provider cost 为 `$1.39020758`。V18 development
adjudicator 另花费 `$0.072858`，也未通过。

## 关键修复

### 1. 修正 grounder identity 边界

V14 在任何 provider/runtime call 之前中止，因为旧 `grounder_sha256` 错误包含了 dataset-level
candidate/evaluation quotas。V15 起将哈希拆成：

- `grounder_sha256`：只包含逐样本 acquisition、grounding、calibration 与 execution semantics；
- `evaluation_protocol_sha256`：包含 sample count、selection 和 qualification gates。

因此扩大 9→30→45→60 rows 不再伪装成 grounder drift，而任何 grounder 参数变化仍会改变
semantic identity。对应回归测试已加入。

V14 abort 记录中的 `raw_video_decode_or_grounder_inspection_started=false` 表述不够精确：下载校验
确实做过本地 first-frame integrity decode，但没有 neural/model inspection、provider call、runtime
receipt 或 outcome read。为保持冻结哈希不变，原文件未修改，勘误见
[`agqa2_active_grounding_v14_preflight_abort_erratum.json`](results/agqa2_active_grounding_v14_preflight_abort_erratum.json)。

### 2. 收紧 symbolic override

V15 暴露两个 failure mode：低置信度 EXISTS false positive，以及重复/重叠事件下的 earliest-start
顺序歧义。V16 加入：

- EXISTS override 最低置信度；
- BEFORE/AFTER 必须是 globally separated intervals。

V17 仍出现 3 个 loss。V19 因此进一步冻结为 **temporal-operator-only transfer**：

- relation/EXISTS acquisition 可以保留用于审计，但不能产生 source override；
- BEFORE/AFTER 仅允许每个 operand 恰有一个事件，且两个 interval 全局分离；
- duration operator 保留 interval topology execution。

这在 V19 development 清除了 loss，但没有在 final fresh reserve 复现：最终三个 loss 是两个
duration comparison（`AOAY0-9753`、`FOMJM-2441`）和一个 temporal order
（`UETQS-32151`）。

### 3. 尝试独立强模型 adjudication

V18 用 `anthropic/claude-sonnet-4.6` 对 V17 的 7 个 source overrides 看完整 48-frame timeline，
同时隐藏 typed/direct/gold/program/source。它没有解决问题：按 AGQA gold 为 `0/7`，授权的两次都
是 negative transfer。人工可解释的冲突包括“看向/透过 window”是否等价于“watching window”、
开门时是否算“touching door”等。这表明瓶颈不是简单换一个更强 VLM，而是 target ontology 和
event-boundary calibration。

## Neural-symbolic transfer 到底迁移了什么

运行时确实复用了匿名 game-induced typed operators 来控制 target acquisition dynamics：

- recurrent relation/event scans；
- non-recurrent dual-operand duration scans；
- typed transition/abstention；
- wrong-type source permutation 必须 fail closed。

四次 fresh replication 的 source-permuted control 和 target-written-equivalent control 都是
`165/165`。前者说明 type safety 有效；后者也明确限制了 claim：当前证据支持的是
**structural mechanism transfer + target-native neural grounding**，没有证明游戏 source
provenance 不可替代，更没有证明 source-specific game semantics 已迁移到 AGQA。

此外，freeze 阶段使用 question 和 functional-program root 校验 route/type compatibility，因此这不是
“完全 untouched benchmark metadata”评测；运行时 grounder 本身仍不读取 answer、functional program、
scene graph 或 source identity。

## 为什么 V19 当时停止 QUERY_OBJECT 和继续采样

预注册顺序要求先通过 atomic temporal/relation base grounder，再单独 qualification open-vocabulary
`QUERY_OBJECT`。V19 final 没通过 base gate，因此继续 QUERY_OBJECT 会违反 stop policy，也会把
一个更难的 perception/ontology 问题混入尚未稳定的 base claim。

同样，不应继续更换 fresh seeds 直到出现 0 loss。V15、V17、V19 已重复显示“平均增益为正、但
negative transfer 不为零”；更多无冻结机制变化的采样会变成 optional stopping。

## 合法的下一步

下一步不是继续手调 symbolic heuristic，而是单独解决 target-native grounding：

1. 在 AGQA train/development 的 video-disjoint split 上训练或校准 ontology-aware event grounder，
   明确学习 action boundaries、object relation semantics 和 repeated-event segmentation。
2. 与相同帧预算、相同强模型的 target-only baseline 比较；不能只把廉价 Qwen direct 当唯一对照。
3. 在 development 冻结 applicability threshold，保留现有 typed IR、source-permuted、
   target-written-equivalent 和 0-loss negative-transfer gate。
4. 只做一次新的 fresh confirmation；通过后再启动隔离的 `QUERY_OBJECT` qualification。

在这之前，准确状态是：
`PIPELINE_EXECUTION_VALIDATED_TRANSFER_UTILITY_NOT_QUALIFIED`。

## 后续隔离的 QUERY_OBJECT track

V19 formal 结论和 atomic route artifacts 保持不变。之后没有把 `QUERY_OBJECT` 混回 V19，而是另建
train-development → frozen grounder → fresh test reserve 的独立 lineage。V25 在 30 个新 videos 上
通过全部 gate：unified harness `14/30` 对 matched direct `4/30`，10 wins/0 loss。详细设计、
runtime-incomplete V23、target-only post-hoc control 和 claim boundary 见
[`AGQA2_QUERY_OBJECT_V20_V25_RESULTS.md`](AGQA2_QUERY_OBJECT_V20_V25_RESULTS.md)。
