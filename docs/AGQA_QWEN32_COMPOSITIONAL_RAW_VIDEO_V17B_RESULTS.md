# AGQA2 Qwen32 compositional raw-video transfer：V15–V18 / V17b

## 结论

新的 one-shot、video/task-disjoint AGQA2 duration-compositional formal 已通过全部
primary transfer gates：

| Arm | Correct | Accuracy | Symbolic commits |
|---|---:|---:|---:|
| neural-only Qwen3.5-9B | 107/256 | 41.80% | 0 |
| source-permuted | 107/256 | 41.80% | 0 |
| **game source-induced** | **123/256** | **48.05%** | 63 |
| target-written isomorphic | 123/256 | 48.05% | 63 |
| generic target-native ceiling | 132/256 | 51.56% | 148 |

Source 相对 neural-only 与 matched-permuted 都是 `20W/4L/232T`，净增 16 个正确答案、
`+6.25 pp`，exact two-sided McNemar `p=0.00154388`。Negative-transfer loss fraction
为 `4/256 = 1.56%`，target-written isomorphic prediction equivalence 为 100%。因此本轮
验证的是：

> 在完全共享、answer-blind 的 raw-video grounding、question parser、typed executor 和
> frozen neural fallback 下，从 game interventions 归纳的 Harness structure 可以显著提高
> fresh natural-video compositional QA success。

它不是 AGQA official-test/full-distribution 或 raw-video QA SOTA 结果；`>55%` secondary
target 没有达到。

## 实际 pipeline

```text
raw Charades MP4
  -> uniform 48-frame Qwen3-VL-32B event graph
  -> uniform 96-frame Qwen3-VL-32B event graph
  -> answer/source-blind generic-executability router
  -> operator-free Flan-T5 semantic slots
  -> shared typed event graph
  -> [only the symbolic Harness arm changes]
  -> source-authorized typed executor or shared Qwen3.5-9B fallback
  -> final answer
```

Qwen grounder 可以读取 question 和 operator-free semantic slots 以定位感知证据，所以它是
**query-conditioned、answer-blind**，不是 question-blind。旧 preoutcome JSON 中
`question_blind_multi_event_graph` 是沿用的字段名；实际 gate 检查的是 multi-event graph，
不能据该字段声称 grounder 看不到 question。Grounder 不读取 answer、official STSG、functional
program、source controller 或 target outcome，也不得直接输出 selected choice/final answer。

五臂使用相同的：

- 256 个 raw videos 与每题 144-frame 总 presentation budget；
- 两个冻结 Qwen3-VL-32B grounding views 与 deterministic router；
- question-only semantic parser；
- V3 typed target adapter 与同一个 executor；
- thinking-disabled Qwen3.5-9B fallback。

唯一变化是 symbolic Harness：neural-only、generic scaffold、matched source-permuted、
game source-induced、target-written isomorphic。

## 为什么旧版本失败，以及修了什么

### V15：SlowFast 三窗口不够

在已经消费的 development videos 上，public Charades SlowFast action logits 能产生多事件图，
但三窗口 interval 太粗。结果为 source `78/256`、neural `74/256`，仅 `14W/10L`，
`p=.541`。这不是可确认的 transfer signal。

### V16：Qwen perception 正常，typed lexicalizer 错

Qwen32 48/96-frame receipts 有 232/256 routed multi-event graphs，但旧 V2 adapter 只得到
11/256 source commits，因此在打开 outcomes 前被 pre-gate 拦截。例如原始 Qwen events 已正确
区分：

```text
hold + bag              -> S3: holding a bag
hold + cup of something -> S8: holding a cup of something
```

旧 lexicalizer 因两个 action 共用 `hold` verb，把 S3/S8 同时绑定到每个 event，最后把两个
action 合并成一个 `hold`，触发 `ARGMAX_EMPTY_SET`。V3 adapter 只根据 operator-free typed tree
识别 `semantic_tuple(actions, action_literal)`，再用 grounded object 解歧 full action phrase。
它不读取 target answer/outcome，也不改变 source controller、VM operators 或 applicability rule。

修复后的 outcome-blind V18 gate 为：60/256 source commits、0 permuted commits、36/256
source-neural disagreements、232/256 multi-event、isomorphic 100%。Consumed development
随后得到 source 117/256、neural/permuted 103/256，`19W/5L`，`p=.00661`。只有该
development qualification 通过后，才冻结 final protocol 与 fresh cohort。

### V17a：安全中止，不是一次 formal trial

第一次 protocol freeze 后，cohort freezer 错把“视频已在本地存在”当作 selection 前提。
由于最后 351 个 fresh videos 尚未下载，本轮在 cohort/runtime/provider/outcome 全部开始前中止。
审计确认 duration-choice 和 duration-extremum 仍分别有 210/172 个 fresh candidate videos。
V17b 改为先按 exclusion ledger 和 task hash 冻结 IDs，再从 official Charades archive 下载并
content-verify。V17a 没有创建 cohort，也没有产生或读取 formal predictions/outcomes。

## Formal freshness 与冻结顺序

1. 全部 34 个历史 AGQA cohort JSON 被汇总为 immutable exclusion ledger：7,056 个历史
   exposed videos、20,347 个 task IDs。
2. 从 parser-disjoint semantic pool 中按固定 salt 做 deterministic bipartite capacity matching；
   duration-choice / duration-extremum 各 128，且 256 个 video IDs 全部不同。
3. 选中 videos 在冻结时本地均不存在；之后从 official Charades archive range-extract，256/256
   通过 size、CRC32、SHA-256 和 decode 检查，共 418,691,313 bytes。
4. Semantic parser 为 256/256 valid；grounding、fallback、五臂 decisions 全部在 outcome 前冻结。
5. Preoutcome gate 通过后 evaluator 才首次读取 `train_balanced.txt` 中这 256 个 answers。

Formal 使用 official balanced-train 的 fresh controlled reserve，不是 official test。

## Grounding 与成本

- 48-frame view：256 receipts；96-frame view：256 receipts；
- receipt-reported OpenRouter cost：`$0.290463264`；
- routed candidate counts：226 / 30；
- routed non-empty graphs：250/256；multi-event graphs：241/256；
- 两视图总共 2,107 events；
- 10 次 provider exception 和 27 个 empty view graphs 全部 fail closed；没有人工修复 payload；
- shared fallback 在本地单张 RTX A6000 上运行，无 API cost。

## Structural breakdown 与 failure taxonomy

| Root | Tasks | Neural | Source | Generic | Source commits | W/L |
|---|---:|---:|---:|---:|---:|---:|
| duration-choice | 128 | 74 | 74 | 74 | 4 | 0/0 |
| duration-extremum | 128 | 33 | **49** | 58 | 59 | 20/4 |

所以这是 selective transfer：净收益来自 duration-extremum；duration-choice 基本 fail closed，
不能写成两个 family 都被改善。

| Observable outcome class | Tasks |
|---|---:|
| symbolic recovery | 20 |
| negative transfer | 4 |
| committed shared failure | 22 |
| abstained fallback correct | 86 |
| abstained shared failure | 107 |

Generic ceiling 比 source 高 9 个正确答案。它说明 target-native unrestricted symbolic execution
仍有 headroom，不是 source 必须击败的 baseline。Target-written isomorphic 与 source 完全相同，
所以结果证明 program content 的 transfer，不证明 source provenance 对 extensionally identical
controller 是必要的。

## 可用于主表的行

| Benchmark | Scope | Tasks | Neural | Source | Generic ceiling | Gain | W/L | p | Neg. transfer | Iso. eq. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AGQA2 | fresh duration-compositional balanced-train | 256 | 41.80% | **48.05%** | 51.56% | **+6.25 pp** | 20/4 | .00154 | 1.56% | 100% |

这行可以与 CLEVRER fresh Layer-B `34.75% -> 62.31%` 一起进入六域 controlled-transfer
主表，但必须保留 scope 列；不能把 256-task row 标成 “full AGQA2 test”。旧 1,790-task
broad result（44.86% -> 46.48%，`p=.00704`）仍可作为独立 broad-distribution replication，
不与本行合并计数。

## 复现与 canonical artifacts

不重新调用 provider，只审核冻结 artifacts、重建 AGQA/two-video bundles 并运行相关测试：

```bash
bash scripts/reproduce_agqa_qwen32_compositional_v17b.sh "$PWD"
```

Canonical paper artifacts：

- AGQA V17b paper bundle：`docs/results/agqa_qwen32_compositional_formal_v17b.json`
- two-video V3 bundle：`docs/results/two_video_transfer_bundle_v3.json`
- formal evaluation：`runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_formal_evaluation_v17b.json`
- preoutcome：`runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_five_arm_preoutcome_v17b.json`
- protocol：`runs/agqa2_offtheshelf_qwen32_formal_protocol_v17b/protocol.json`
- cohort manifest：`runs/agqa2_offtheshelf_qwen32_formal_v17b/manifest.json`
- anonymous game controller：`runs/anonymous_video_harness_v1/controller.json`

AGQA bundle SHA-256：`29b568b4b7a313bd0ee573535e0dd40a6bcebdfbc360d8336ac6903cee60102f`。
Two-video bundle SHA-256：`1b86db2ea9d9605cf51c1d208e2d64b42b7d44359e17d2e24a8342be08f81066`。
