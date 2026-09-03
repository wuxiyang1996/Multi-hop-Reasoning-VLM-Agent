# Game → four-domain neural-symbolic transfer (V16)

## 结论

同一个从 Sokoban 学到并冻结的三状态搜索 automaton，已经通过统一 harness 接到 WebShop、ALFWorld、DiscoveryWorld 和 TIRBench。四域的机制门均通过，四域 success delta 都为正，paired strict loss 都为 0。

但是证据等级不同：**WebShop 是本轮唯一 prospectively frozen、source-artifact-specific fresh formal confirmation**；ALFWorld 是已消费 development tasks 的重新执行，TIRBench 是已消费 formal receipts 的重分析，DiscoveryWorld 是 V16 对先前 fresh replication 的 retrospective equivalence。因此当前最准确的状态是：

> `FOUR_DOMAIN_MECHANISM_SUPPORTED_EVIDENCE_TIERS_MIXED`

这足以说明 neural-symbolic transfer 机制可工作，但不能写成“四域都完成了新的 V16 prospective confirmation”。机器可审计的统一结果在 `docs/results/search_automaton_transfer_v16_summary.json`。

## 实际迁移的是什么

source artifact 只包含三个 intervention-grounded event→action edge：

| target-grounded event | transferred symbolic action |
|---|---|
| 没有 active candidate，且仍有未尝试候选 | `EXPLORE_UNTRIED` |
| active candidate 被 target-native effect evidence 否定 | `BACKTRACK_REPLAN` |
| target-native commit predicate 已验证 | `COMMIT_VERIFY` |

同时迁移一个闭环 attempt ledger：候选只有在执行后观察到 target-native effect 才能推进状态；未知事件、低置信 grounding、冲突 evidence 或缺失 native binding 都会 abstain。Sokoban 的坐标、动作 token、路径长度、候选顺序和 solver 都不会跨过 transfer boundary。

因此这不是 prompt 中写一句“先探索、失败再回退”的高层 skill，而是：

```text
frozen source event automaton
        + target-native neural event grounding
        + target-native action realization
        + effect-updated symbolic ledger
        + fail-closed applicability gate
```

source gate 来自 85 个 fresh Sokoban states。Authentic policy 为 85/85；event-permuted、commit-only 和 always-backtrack 为 0/85，ledger-blind 为 18/85。三个 edge 都有 matched intervention advantage 1.0。重要边界是这些是 real Sokoban states 上由 solver 构造的 candidate corruptions，不是自然在线 gameplay rollout；abstract-action counterfactual 的成功标签来自显式 automaton continuation semantics，因此 1.0 是结构一致性检查，不是无偏的经验 treatment-effect estimate。真正非平凡的 empirical evidence 来自 target 端未见 WebShop formal tasks 上 11 个 strict rescue，以及错误 binding/无 ledger controls 的失败。

## 四域结果

| target | evidence tier | raw | authentic | paired vs raw | 核心解释 |
|---|---:|---:|---:|---:|---|
| WebShop | fresh formal | 7/32 | **18/32** | 11W / 0L / 21T, p=0.00098 | source routes + target neural outcome/option grounding |
| ALFWorld | consumed dev reexecution | 4/8 | **7/8** | 3W / 0L / 5T, p=0.25 | source routes + hierarchical target event grounder |
| DiscoveryWorld | retrospective equivalence | 10/16 | **13/16** | 3W / 0L / 13T, p=0.25 | V16 reproduces prior target-native recovery actions |
| TIRBench | consumed formal reanalysis | 14/24 | **23/24** | 9W / 0L / 15T, p=0.00391 | source routes over frozen target neural path bindings |

这里的 p 值是 paired strict wins/losses 的 exact two-sided binomial test。ALFWorld 和 DiscoveryWorld 样本小，方向为正但单域并不显著。

### WebShop fresh formal

Formal protocol 在观察任何 formal outcome 前冻结了 32 个任务、五个 condition、source artifact、target grounder、goal manifest、controller/runner hash、模型 `openai/gpt-4.1-mini`、12-step budget 和开发 qualification report。160 个 task-condition cells 一次执行完成，无 provider/schema failure，且每题五条件 initial-state hash 相同。

| condition | strict | pass | mean reward |
|---|---:|---:|---:|
| raw target | 7/32 | 15/32 | 0.4650 |
| authentic source + target | **18/32** | **19/32** | **0.5901** |
| event-permuted | 7/32 | 15/32 | 0.4650 |
| ledger-blind | 4/32 | 18/32 | 0.4706 |
| isomorphic target-native ceiling | 18/32 | 19/32 | 0.5901 |

Authentic 对 raw 的 graded reward 为 13W/5L/14T，净 +8，exact p=0.0963。也就是说 aggregate reward 提升，但仍有 5 个部分奖励负迁移任务，不能声称逐任务 dominance。新增的 anytime gate 只在动作图最短路径证明 strict completion 已不可能时让 source abstain，并由 target-native reward predictor 尝试收尾；它把开发集 mean reward 从 0.636 提升到 0.733，但尚没有 incumbent memory，因此不能消除所有 formal 个例损失。

Authentic 与 isomorphic target-native ceiling 在 32 个任务上完全一致。这是重要的正、反两面证据：source symbolic structure 确实可移植并能被 target grounder 正确执行；但它没有优于把同一个算法直接写在 target 端，因而不能声称 source-specific algorithmic advantage。

### ALFWorld

八个已消费 `valid_unseen` qualification tasks 上，raw 4/8，authentic 7/8，permuted 4/8，ledger-blind 1/8，target ceiling 7/8。五条件实际 task path 与首状态 hash 逐任务一致。三个 source action 都被执行；authentic 改变 417 个 action positions，平均步数从 52.6 降到 27.5。

这证明 V16 runtime 与 target-native hierarchical event grounder 的机制兼容，但不是新的 held-out confirmation。

### DiscoveryWorld

对先前 seeds 11–20 的 independent replication receipts 做 hash-locked V16 relineage。16 个 eligible forks 中，历史 authentic 13/16、target-native myopic 10/16，3W/0L。V16 三个 route 都被覆盖，每个历史 recovery action 都被复现或明确记录为 source abstention；policy 未读取 oracle scorecard。

它只证明新 automaton 与先前成功程序在这些 receipts 上等价。仍需一个在运行前绑定 V16 artifact 的 prospective DiscoveryWorld reserve。

### TIRBench

在 24 个已消费 TIR maze formal receipts 上，用冻结 target neural color/direction bindings 重新执行 V16 route。raw 14/24，authentic 23/24，permuted、ledger-blind、commit-only 均为 14/24，target exhaustive ceiling 23/24；9W/0L，p=0.00391。三个 source edge 全部被使用。

这是很强的 mechanism reproduction，但不是新的 fresh TIR reserve。

## 这次真正修掉的错误

1. **Observation change 不是 progress。** ALFWorld 初版把页面/文本变化当作 effect，导致 6/8 tie 且不产生 backtrack。现在只有 target workflow 的 required-option change 或官方 terminal success 才更新 ledger。
2. **Abstention 不能终止 target episode。** 候选耗尽或 source binding 低置信时，source 交还控制权给 target policy，而不是把 episode 判失败。
3. **WebShop option evidence 必须 product-local。** 从商品 A 选择的颜色/尺寸不能泄漏到商品 B；换商品时 verified/pending ledger 会重置。
4. **Finite-horizon search 必须 anytime。** 若动作图下界证明剩余步数不可能完成 strict goal，source abstains，target 选择即时奖励最大的 terminal action，而不是继续搜索到 0 分。
5. **空 option 集合的 audit 不能误报。** 无规格前置条件时，option predicate 是 vacuously ready；audit 只把 source-authorized、实际违反非空 prerequisites 的 commit 算 unsafe。
6. **必须同时报告 ceiling 和破坏性 controls。** Permuted/ledger-blind 的差距支持正确 event binding 与 memory 的必要性；ceiling tie 限制了 source-specific novelty claim。

## 可复核入口

```bash
PYTHONPATH=src:. python scripts/summarize_search_automaton_transfer_v16.py

PYTHONPATH=src:. pytest -q \
  tests/test_search_automaton_transfer_v16.py \
  tests/test_sokoban_search_automaton_v16.py \
  tests/test_webshop_search_automaton_v16.py \
  tests/test_webshop_coverage_transfer_v14.py \
  tests/test_alfworld_search_automaton_v16.py \
  tests/test_discoveryworld_search_automaton_v16.py \
  tests/test_tir_search_automaton_v16.py
```

WebShop formal reserve 已消费，不应重跑或用于继续调参。接下来的合法确认顺序是：冻结新的 ALFWorld V16 reserve、DiscoveryWorld V16 reserve 和 TIRBench V16 reserve，然后各自只运行一次。
