# AGQA 2.0 V2：frame-only target-native grounder

## 结论

我们已经把 V1 的 official-functional-program compiler 降级为 **evaluator-side symbolic
oracle**，并实现了真正的 frame-only AGQA grounder。新 grounder 只接收 public question、16
个 chronological proxy frames 和 timestamps，完全不接收 answer、functional program、
`sg_grounding`、official reasoning labels、source identity 或 answer candidates。

实现与 transport 均工作，但 grounder **没有通过 qualification**：

`AGQA2_FRAME_GROUNDER_DEVELOPMENT_NOT_QUALIFIED`

因此 unified harness 对 9/9 tasks 都拒绝授权 source-induced executor，并保留 target-native
direct fallback。没有 neural-symbolic transfer success claim。

## 数据与冻结

- 本地已有 104 个 STAR/Charades videos；其中 19 个与 AGQA balanced metadata video IDs 重合。
- 从三个 V1 exact-route families 各冻结 3 题，共 9 题、9 个互不重叠 videos。
- 选择规则是 route 内 SHA-256 ranking，再做全局 video-disjoint；freeze 阶段没有读取 answer 或
  scene graph。
- 所有视频此前已经下载，本轮新增视频下载为 0 bytes。
- 这是 consumed-development pilot，不是 untouched formal split。

## Grounder 与统一 IR

模型为 OpenRouter `qwen/qwen3-vl-32b-instruct`。每个视频均匀采样 16 帧，组成四个 chronological
panels。strict JSON Schema 限制 frame bounds、typed enums 和字段；本地 validator 继续检查：

- answer/program/scene-graph/source-identity leakage；
- event role、interval、evidence 与 confidence；
- question-native obligation：relation、before/after、duration score；
- fail-closed abstention。

两个纯表示冲突使用显式、可审计的 canonicalization：

- reversed endpoints → `SWAPPED_REVERSED_INTERVAL_ENDPOINTS`；
- cited evidence 超出声明区间 → 把区间扩展为 endpoints 与 cited evidence 的最小包络，并记录
  `EXPANDED_INTERVAL_TO_COVER_EVIDENCE`。

两者都不新增、clamp 或猜测视觉证据。

frame receipt 预测的匿名 type 被送入与 WebShop、DiscoveryWorld、TIR、ALFWorld、CLEVRER 相同的
`select_source_contract`。grounder 未通过 qualification 时，即使 type exact match，也必须返回
`TARGET_GROUNDER_NOT_QUALIFIED`。

## 结果

| 指标 | 结果 | Gate |
|---|---:|---|
| Strict-schema valid receipts | 9/9 | pass |
| Program-family route accuracy | 9/9 | pass |
| Decisive typed execution coverage | 7/9 | pass（要求 ≥6） |
| Decisive typed accuracy | 3/7 = 42.9% | **fail**（要求 ≥50%） |
| Qwen direct | 5/9 | baseline |
| Unqualified typed fallback | 3/9 | 0 wins / **2 losses** |
| No-negative-transfer | 2 losses | **fail** |
| Actual unified-harness authorizations | 0/9 | correct fail-closed behavior |
| Actual unified-harness accuracy | 5/9 | 与 direct 相同 |

主要错误不是 route classification，而是 neural grounding：

- object relation：把 `chair` 看成 `bed`；
- event order：把 laptop opening 与 standing→sitting 顺序倒置；
- 两个 duration questions 的 coarse interval 导致 longer/shorter 判断错误；
- 另外两个 relation questions 正确 abstain，没有编造 object decision。

所以 16-frame uniform grounder 已能稳定输出 typed receipts，却还不能可靠定位细粒度 object/action
interval。事后降低 confidence 或 accuracy threshold 会使 claim 无效，因此没有这样做。

## 成本边界

被最终报告接受的 receipts：19 provider calls，OpenRouter reported cost `$0.002717`。这不等于
整个 development 过程的精确累计成本：早期 8B schema failures 和 aborted calls 没有全部持久化
usage。config 保留了完整失败顺序，但累计 spend 标记为不可精确重建。

## 下一步（如果继续）

下一版不应只换更大 LLM。应更换 grounding dynamics：

1. 16 帧只做 scout；对每个 operand 单独做 action/object-conditioned temporal resampling；
2. duration 使用 dense interval head 或 target-native action segmentation，而不是让 VLM 从均匀帧
   直接估计区间；
3. relation object query 使用独立 object grounding receipt；
4. 在新的 consumed-development videos 冻结 threshold，再做 video-disjoint reserve；
5. source authentic/permuted 22/27 tie 仍需先修复，否则即使 grounder 通过也不能证明 source
   provenance。

在这些条件满足前，AGQA source executor 保持关闭。

## Artifacts

- Grounder contract: `src/motif_transfer/agqa_frame_grounder.py`
- Frozen manifest: `configs/agqa2_frame_grounding_v2_manifest.json`
- Development config: `configs/agqa2_frame_grounding_v2_development.json`
- Collector: `scripts/collect_agqa2_frame_grounding_v2.py`
- Compact result: `docs/results/agqa2_frame_grounding_v2_summary.json`
- Tests: `tests/test_agqa_frame_grounder.py`、
  `tests/test_collect_agqa2_frame_grounding_v2.py`

本地完整 raw report 位于 ignored `runs/agqa2_frame_grounding_v2_development/report.json`；compact
result 记录其 content hash、逐题 receipt hash 与所有 gates。
