# 三视频 typed grounding V30–V33：独立 cascade 有信号，strong-target gate 未通过

## 结论

V30–V33 修复了此前自由文本 event ledger 的主要工程问题：每个 neural call 只看到一个无 slot
候选，输出固定的 `ENTITY_BINDING → PRECONDITION → POSTCONDITION →
DIRECTIONAL_OR_CAUSAL_LINK → CLAIM_ENTAILMENT` receipt；symbolic executor 仅在证据闭合时改答，
binding rotation 是破坏性控制。48 帧、panel bytes、视频 bytes 和模型身份都被记录并校验。

但是当前自然视频路径**还不能进入 fresh confirmation**：

- Gemini same-model direct 为 `9/12`，typed executor 为 `8/12`，paired `0 wins / 1 loss`；
- 独立 Claude direct 为 `6/12`，同一 Gemini typed receipts + symbolic executor 为 `8/12`，
  paired `2 wins / 0 losses`；
- 后者在 STAR 和 NExT-QA 各产生一个 recovery，binding rotation 分别只有 `1/4` 和 `2/4`；
- 但弱/独立 baseline 上的 `+2` 不能覆盖最强 matched target-only 为 `9/12` 的事实。

因此准确状态是：

> `CROSS_MODEL_CASCADE_SIGNAL_STRONG_TARGET_GATE_FAILED`

Gemini 与 Claude 在这里是诊断角色，不是最终 architecture 的两个必需组件。Gemini 是 V29 后语义
错误最少的 target-native claim grounder；Claude 用来做独立 baseline/反证，避免同一模型同时生成
proof 与答案导致自洽偏差。结果说明 cascade 可以修复较弱独立 baseline，但没有说明多模型堆叠优于
最强 target-only。

## 逐基准结果

| Benchmark | Gemini direct | typed on Gemini | Claude direct | typed on Claude | binding control on Claude |
|---|---:|---:|---:|---:|---:|
| CLEVRER | 2/4 | 1/4 | 1/4 | 1/4 | 0/4 |
| STAR | 3/4 | 3/4 | 2/4 | 3/4 | 1/4 |
| NExT-QA | 4/4 | 4/4 | 3/4 | 4/4 | 2/4 |
| pooled | 9/12 | 8/12 | 6/12 | 8/12 | 3/12 |

V30 自身用旧 GPT-4.1 V29 baseline 时从 `7/12` 到 `8/12`，但该提升存在 cross-model confound；
V31 的 same-model comparison 已将它否定。因此 V30 的 `+1` 不作为正证据。

## Grounder intrinsic audit

自然视频的候选级 receipt 已明显比 V28/V29 ledger 可靠：STAR + NExT-QA 共 36 个候选，
covered accuracy 为 `33/35 = 94.3%`，coverage 为 `35/36 = 97.2%`。STAR 的 `sit` 与 `lie`
混淆被修复；NExT-QA 为 `19/20` 候选正确。

剩余 STAR 失败 `Interaction_T3_2531` 不是简单的低分辨率问题。Gemini 在高分辨率二次验证中仍同时
支持 `put down bag` 和 `open bag`；Claude 对两者均 abstain。人工像素检查与 official situation graph
表明 clip 的纸袋朝向/开口可造成 `open` 幻觉，而 official ontology 只记录 `put bag somewhere`。
这需要 adaptation-supervised primary-action/ontology grounder，而不是继续改 prompt。

CLEVRER 的 generic VLM receipt 在 predictive/counterfactual 上只有 `5/10` 候选正确，所以 V30 executor
不再承担 CLEVRER 正式路径。该 benchmark 已有独立的 validated route：V14 使用 neural dynamics
predictions、symbolic executor 和 learned proof-grounded recovery，在 720 个 prospective formal 样本上
从 `489/720` 提高到 `511/720`，并通过 frozen lineage、binding/action/inverted/marginal controls。
它与本 12-row grounding diagnostic 不做 pooled claim。

## 为什么现在不读取 fresh

formal gate 中仍有两个失败项：

1. typed executor 没有超过 strongest matched Gemini direct；
2. 本地 NExT-QA 的 31 个视频全部在旧实验中出现过，虽然仍有 58 个 question-disjoint 样本，但没有
   video-disjoint confirmation。STAR 尚有 74 个完全未见视频。

下一步应先在 disjoint adaptation videos 上训练 STAR primary-action grounder，并取得/下载未见的
NExTVideo 或明确降级为 question-disjoint replication；之后重新运行 strong matched target gate。
不能用更多 Gemini/Claude calls、post-hoc family rule 或单个 ambiguous sample heuristic 来授权 fresh。

## Artifacts

- V30 receipts: `runs/three_video_typed_claims_v30_development/receipts.json`
- V31 matched Gemini: `runs/three_video_typed_claims_v31_matched_development/receipts.json`
- V33 independent Claude: `runs/three_video_matched_claude_direct_v33_development/receipts.json`
- formal report: `runs/three_video_typed_claims_v30_v33_development/formal_report.json`
- machine summary: `docs/results/three_video_typed_grounding_v30_v33_summary.json`
- separate CLEVRER V14: `runs/sokoban_clevrer_proof_v14_formal/formal_report.json`
