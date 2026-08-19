# AGQA 2.0 `QUERY_OBJECT` neural-symbolic transfer：V20–V25

## 最终结论

隔离的 open-answer `QUERY_OBJECT` route 已通过一次预注册、30-row、全新 video-disjoint 的
formal confirmation。V25 使用与 V24 development 完全相同的 grounder
`2a84c1c9f170206bd216e171072c70d8faf73bd6d7c89d562fff808bf7141a3e`，最终：

- route：`30/30`；
- decisive：`16/30`，其中 `12/16` 正确，恰好达到冻结的 `75%` threshold；
- matched direct：`4/30`；
- unified harness：`14/30`；
- 10 wins、0 losses，净 `+10`；
- source-permuted wrong-type abstention：`30/30`；
- target-written-equivalent dynamics：`30/30`；
- reported provider cost：`$0.264821712 < $0.35`。

所有预注册 gate 均通过，因此这不是 counterfactual typed 分数：qualification 后 unified harness
实际授权了 16 次，并取得 `14/30`。

机器可读、带哈希的汇总见
[`agqa2_query_object_v20_v25_summary.json`](results/agqa2_query_object_v20_v25_summary.json)。

## 它解决了什么

原有 public parser 会把 `before/after/while/between` 等 temporal subquery 一并塞进
`QUERY_OBJECT` relation，导致 object grounder 实际承担多跳 temporal parsing。V20 起只保留：

- 一个显式 relation；
- 无 temporal/nested clause；
- 无每题 answer candidates；
- functional-program root 只在 freeze/evaluator 侧确认 answer space 为 `Query(...)`；
- runtime grounder 不读取 gold、program、scene graph、direct response 或 source identity。

target-native acquisition 使用三个 neural views：

1. game-induced recurrent relation controller 驱动的 isolated-relation grounder；
2. Gemini 2.5 Flash Lite 的固定全局 object-ontology view；
3. Gemini 3 Flash 的相同 ontology、独立模型 view。

symbolic executor 只接受 neural-only 2-of-3 canonical consensus；matched direct 不参与投票。
全局 ontology 来自公开的 STAR/Charades dataset taxonomy，不是每题候选列表。

## V20–V25 过程

| 版本 | 数据/变化 | decisive | harness / direct | wins/losses | 结果 |
|---|---|---:|---:|---:|---|
| V20 dev | 18 个 AGQA train videos；base + Gemini ontology | 5，4 correct | 9 / 6 | 3 / 0 | coverage 未过 |
| V21 dev | 加 Claude，冻结 neural 2-of-3 | 10，8 correct | 10 / 6 | 4 / 0 | 仅成本失败：`$0.4035` |
| V22 dev | 不改 gate/consensus，Claude→Gemini 3 | 10，9 correct | 10 / 6 | 4 / 0 | **qualified**，`$0.1587` |
| V23 reserve | 30 个新 test videos | 29 runtime receipts | 未评测 | 未评测 | schema runtime-incomplete |
| V24 dev | deterministic interval-envelope repair；83 calls 全缓存 | 10，9 correct | 10 / 6 | 4 / 0 | **qualified**，0 新调用 |
| V25 reserve | 另 30 个新 test videos，V23 videos 全排除 | 16，12 correct | **14 / 4** | **10 / 0** | **qualified** |

V23 没有被偷偷算作失败 seed 后继续调阈值。其唯一失败是模型自报 `start_frame=11`，却引用
F8–F11 evidence；两次 schema retry 都重复该格式错误。collector 在 evaluator 开始前终止，未生成
formal report。V24 只加入：

```text
start_frame = min(start_frame, existing_evidence_frames)
end_frame   = max(end_frame, existing_evidence_frames)
```

它不能创建 observation、label、confidence 或 evidence。V24 在原 development receipts 上 83/83
provider attempts 全缓存重放并重新通过；之后 V25 使用另一个完全新 pool。V23 的 immutable abort
记录见 [`agqa2_query_object_v23_runtime_abort.json`](results/agqa2_query_object_v23_runtime_abort.json)。

## 非平凡性审计

V25 的 raw direct `4/30` 确实受输出格式影响，例如 `a laptop screen` 对 gold `laptop`。
因此又做了两个不增加模型调用的诊断：

| baseline | correct | 与 source harness 比较 |
|---|---:|---:|
| raw matched direct | 4/30 | harness +10 |
| post-hoc lexical ontology normalization | 7/30 | harness +7 |
| post-hoc two-ontology-view target-only fallback | 12/30 | harness +2 |
| source-controlled 2-of-3 unified harness | **14/30** | — |

source-controlled third view 相对 two-ontology target-only control 是 2 wins、0 losses：

- `7ZL8E-15871`：`leaning on → chair`；
- `QLAS7-4748`：`closing → closet`。

所以 +10 并非全部可归因于格式 canonicalization；在这个 pool 上，source-controlled third view
还有净 `+2`。但这个 target-only comparison 是 formal run 完成后的 post-hoc diagnostic，不是 V25
预注册 gate，不能包装成独立 confirmatory claim。

## Claim boundary

成立的是：

> 在 atomic、无每题候选的 AGQA `QUERY_OBJECT` 子集上，game-induced recurrent typed structure、
> target-native cross-model grounding 和 fail-closed symbolic consensus 可以通过一次 30-video
> formal gate，并提高实际 unified-harness success rate。

仍不成立的是：

- full AGQA 或 compound temporal `QUERY_OBJECT` 已解决；
- 游戏语义或 source provenance 不可替代；
- target-written controller 不能获得相同 dynamics；
- 相对一个预注册的最强 target-only ontology ensemble 已确认显著优势。

特别是 target-written-equivalent 为 `30/30`，所以正确措辞仍是
**structural mechanism transfer + target-native neural grounding**。若要升级 source-specific claim，
下一次实验必须在冻结前把 two-ontology target-only 设为正式 baseline，并要求 source-induced
controller 对它也有正增益和 0 negative transfer；不能继续复用 V25 outcomes。

## V26–V28 follow-up

上述 source-specific follow-up 已执行。powered V28 confirmation 没有通过：正式 unified harness
因 base decisive accuracy gate 失败而全量 abstain；post-hoc preauthorization candidate 虽为
`40/120` vs target-only `36/120`，但只有 5 wins、1 loss、净 `+4`、单侧 exact `p=0.109375`。
因此 V25 的 structural claim 保持不变，source provenance 仍未建立。详见
[`AGQA2_SOURCE_SPECIFIC_V26_V28_RESULTS.md`](AGQA2_SOURCE_SPECIFIC_V26_V28_RESULTS.md)。
