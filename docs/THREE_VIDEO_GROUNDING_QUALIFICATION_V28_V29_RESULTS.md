# 三视频 Grounding Qualification：V28–V29

## 结论

当前 `visual_reasoning_wrapper` grounding protocol **没有通过 transfer 前置验证**，因此不能接到 source/game symbolic skill 上，也不能宣称能提高 CLEVRER、STAR 或 NExT-QA success rate。

这不是 transport 或工具未执行的问题。两轮实验均完成全部 12 个 frozen、already-consumed development 样本，真实调用了 wrapper 的 `detect_scene_changes` 与 `compare_frames`，随后通过 OpenRouter 产生 target-native semantic receipts。失败发生在语义 grounding：自由文本 receipt 经常把物体身份、动作类别或动作方向判断错，并以高置信度误导下游 answer model。

## 实验边界

- 数据：CLEVRER、STAR、NExT-QA 各 4 个已经消费的 adaptation 样本。
- selection 在 V28 runtime call 前冻结；没有打开 fresh confirmation split。
- source skill、source graph、game token 与 source controller 全部关闭。
- runtime 不见 gold、CLEVRER functional program 或 STAR situation graph。
- 每个 answer branch 固定使用 8 帧。
- 五个 matched conditions：
  - `uniform_direct`
  - `uniform_receipt`
  - `localized_direct`
  - `localized_receipt`
  - `shifted_receipt`

因此这两轮只回答一个问题：**grounding tool 本身是否已经足够可靠，可以进入 transfer 实验？**

## V28：candidate-conditioned 单事件 receipt

模型：`openai/gpt-4.1-mini`，72 calls，OpenRouter reported cost `$0.045256`。

| Condition | Correct | Accuracy |
|---|---:|---:|
| uniform direct | 8/12 | 66.7% |
| uniform + receipt | 6/12 | 50.0% |
| localized direct | 8/12 | 66.7% |
| localized + receipt | 5/12 | 41.7% |
| shifted + receipt | 6/12 | 50.0% |

`localized_receipt - uniform_direct = -3/12 = -25.0pp`。`localized_receipt - shifted_receipt = -1/12`，连 evidence necessity 都没有出现。

主要失败是 option hallucination。grounder 虽然不输出 answer slot，却看到 candidate concepts，随后生成了看似确定但错误的事件：

- STAR clothes 被写成 broom；
- STAR dish 被写成 sandwich；
- NExT-QA guitar 被写成 bottle；
- STAR sat on sofa 被写成 laid on sofa。

这些错误 receipt 的 confidence 通常仍是 `0.9–0.95`，所以 confidence threshold 不能修复它们。

## V29：candidate-blind 多事件 ledger

V29 根据 V28 failure 做了两个 repair：

1. grounder 完全看不到 answer candidates；
2. 单事件改成 1–6 个事件的 typed ledger，每个事件只能引用 1–3 个稀疏 evidence frames。

模型升级为 `openai/gpt-4.1`。12/12 样本、72 calls 全部完成，OpenRouter reported cost `$0.270794`。

| Condition | Correct | Accuracy |
|---|---:|---:|
| uniform direct | 7/12 | 58.3% |
| uniform + ledger | 4/12 | 33.3% |
| localized direct | 7/12 | 58.3% |
| localized + ledger | 4/12 | 33.3% |
| shifted + ledger | 3/12 | 25.0% |

`localized_ledger - uniform_direct = -3/12 = -25.0pp`。V29 首次得到微弱的 evidence necessity：`localized_ledger - shifted_ledger = +1/12`，但 ledger 自身造成的损害远大于这项信号。

分 benchmark：

| Benchmark | Uniform direct | Localized + ledger | Delta |
|---|---:|---:|---:|
| CLEVRER | 0/4 | 0/4 | 0 |
| STAR | 4/4 | 2/4 | -2 |
| NExT-QA | 3/4 | 2/4 | -1 |

STAR situation graph 只在离线 evaluator 中使用。严格把 question verb 与 gold object（或 question object 与 gold verb）同时绑定到 official action ID 后，V29 ledger 只有 `1/4` 样本命中正确 action semantics；timestamp overlap 为 `4/4`，说明工具通常找到了“有人在动”的区间，却没有识别出正确的动作与实体。平均 maximum interval IoU 约 `0.242`。

典型方向错误：

- `sit on sofa` 被 ledger 写成 `lies down on sofa`；
- `put down bag` 被写成 `picks up bag / moves bag`；
- `tidy clothes` 只被写成移动一个不明物体；
- guitar interaction 被写成 manipulates lamp-like object。

## Bitter lesson

当前 wrapper tools 可作为 **temporal proposal/navigation tools**，不能作为 semantic event grounder：

```text
pixel change / frame comparison
        ↓ useful for proposal
candidate time windows
        ↓ not sufficient for
entity identity + action class + temporal direction
```

增加模型大小、去掉 candidate leakage、加入 sparse citations，都没有使自由文本 event ledger 可靠。直接把 ledger 注入 answer prompt 会把 perception error 放大为 symbolic error；这正是 neural-symbolic pipeline 最危险的 failure mode。

## 下一步应做什么

下一版不应继续 prompt-refinement 或扩大样本，而应把 semantic grounding 改成 benchmark-native typed measurement：

1. **先 STAR**：利用 target-native verb/object ontology 定义 `BIND(entity) -> BEFORE/AFTER -> ACTION(verb, object) -> ABSTAIN`，但 official situation graph 仅用于 evaluator。每个 action predicate 必须独立引用 before/after frames。
2. **再 CLEVRER**：使用颜色/材质/形状实体绑定与 `ENTER/EXIT/COLLIDE/MOVE` typed predicates；predictive/counterfactual answer 与 observed event ledger 明确分层。
3. **最后 NExT-QA**：它没有逐帧 gold，只能在 STAR/CLEVRER grounder 通过 intrinsic gate 后，用 evidence deletion/time shift 测 downstream utility。
4. 只有当 STAR semantic action accuracy、CLEVRER typed-event accuracy、以及 `localized > shifted` 都通过，才重新接入 intervention-grounded source symbolic structure。

这三步是 **target-native neural grounding + intervention-grounded symbolic transfer**；不是把游戏 token、游戏动作或自由文本 heuristic 搬到视频域。

## Artifacts

- V28 manifest: `configs/three_video_grounding_qualification_v28_manifest.json`
- V28 config: `configs/three_video_grounding_qualification_v28_development.json`
- V28 report: `runs/three_video_grounding_qualification_v28_development/formal_report.json`
- V29 config: `configs/three_video_grounding_qualification_v29_candidate_blind_development.json`
- V29 report: `runs/three_video_grounding_qualification_v29_candidate_blind_development/formal_report.json`

V28 receipts SHA256: `bfe2447a2cc107e01569305fc4fcc79a9076a60861294dfc42eeaa29b487711a`
V28 report SHA256: `5682ab4f1d671a0eab167ee269017fcea7378347b4f02d7f09d4a228a19a14c3`
V29 receipts SHA256: `63c4dbb3eff12ded5b0907a86b794d477f03d9d1978d0acfdfd2c23dbe84d8a4`
V29 report SHA256: `15ec79cffcaa03e42631c638cf7ff1b0d99490736a8181cecb9ebb418bfa6259`
