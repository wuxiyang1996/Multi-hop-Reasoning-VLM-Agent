# Phase 2：Game → WebShop neural-symbolic transfer utility

## 结论

截至 2026-08-16，fresh V4 confirmatory run 的状态为：

> **PHASE2_WEBSHOP_CAUSAL_UTILITY_VALIDATED — 17/17 preregistered gates passed.**

在 32 个全新、ASIN 与 goal semantics 均不重叠的 WebShop synthetic goals 上：

| Condition | Strict success | Pass success | Mean reward | Mean steps | Failures |
|---|---:|---:|---:|---:|---:|
| Raw target-only neural policy | 9/32 (28.1%) | 20/32 | 0.604 | 5.72 | 0 |
| **Authentic game-derived symbolic controller + target neural grounding** | **19/32 (59.4%)** | **24/32** | **0.725** | 6.03 | **0** |
| Event-binding permuted control | 9/32 (28.1%) | 20/32 | 0.604 | 5.72 | 0 |
| Ledger-blind control | 7/32 (21.9%) | 23/32 | 0.648 | 8.31 | 0 |
| Target-native isomorphic ceiling | 19/32 (59.4%) | 24/32 | 0.725 | 6.03 | 0 |

Authentic vs raw 的 matched comparison 为：

```text
strict wins / losses / ties = 10 / 0 / 22
paired exact two-sided p     = 0.001953125
reward wins / losses / ties = 14 / 1 / 17
reward exact p               = 0.0009765625
absolute success gain        = +10/32 = +31.25 percentage points
```

因此，Phase 1 的 “24/24 mechanism executes” 已经在 WebShop 上升级为一个更强但仍有边界的结论：

> 在 frozen、matched、fresh 32-task WebShop experiment 中，正确的 source-derived symbolic search structure 将 strict success 从 9/32 提高到 19/32；打乱 event binding 或移除 stateful ledger 都不能复现该提升。

机器可复核文件：

- frozen manifest：`configs/phase2_webshop_utility_v4/manifest.json`
- formal report：`runs/phase2_webshop_utility_v4/report.json`
- independent audit：`docs/results/phase2_webshop_utility_v4_audit.json`
- live preflight：`docs/results/phase2_webshop_utility_v4_preflight.json`
- 160 个逐 cell receipts：`runs/phase2_webshop_utility_v4/receipts/`

关键 hashes：

```text
manifest_sha256 = 16a900de39fa78bfbd489c6d95111d65f485fb0405e797d8129f4ea53ca18e2f
report_sha256   = 56d03ada15c5c779d8f750ae110d9373558a17470f22caea1388c6846f7b5666
audit_sha256    = 1eb63e955e85eef85ab1642f039490b47a84de0fb745d58e433bb3f272e8aac3
```

## 这为什么是 neural-symbolic transfer

迁移的不是游戏按键、画面 token、商品名称或 raw trajectory，而是 intervention-grounded symbolic controller：

```text
UNBOUND  -> EXPLORE_UNTRIED
REFUTED -> BACKTRACK_REPLAN
VERIFIED -> COMMIT_VERIFY
```

正式执行链是：

```text
six independent game intervention lineages
  -> frozen typed search-automaton artifacts
  -> target-native neural WebShop candidates
  -> symbolic event binding + attempt ledger
  -> explore / backtrack / commit decision
  -> BrowserGym-native action
  -> observed target effect updates the ledger
```

神经部分只在 WebShop 当前 accessibility tree 内生成和验证 native actions；symbolic 部分从游戏 evidence 迁移 domain-invariant 的 search state、event transition 和 intervention choice。source artifact 不提供 ASIN、BID、query、candidate order 或 WebShop action。

V4 authentic route 实际执行了：

```text
EXPLORE_UNTRIED  = 143
BACKTRACK_REPLAN = 19
COMMIT_VERIFY    = 26
total decisions  = 188
unsafe commits   = 0
```

这不是只附加一个高层 skill label：controller 的 transition 在 online trajectory 中改变了 target-native candidate choice，21/32 个任务的 authentic action sequence 与 raw 不同。

## 为什么 controls 是非平凡的

五个条件共用相同目标、初始 state、模型、temperature、candidate budget、grounder 和最大步数。只改变 symbolic structure：

1. **Raw target-only**：直接选 neural rank-zero action。
2. **Authentic**：正确绑定 `UNBOUND/REFUTED/VERIFIED`，并维护 attempt ledger。
3. **Event-permuted**：保留 source artifact 和 target candidates，但破坏 event→action binding；结果精确退回 9/32。
4. **Ledger-blind**：保留局部 symbolic rules，但不维护正确的跨步 attempt state；结果为 7/32。
5. **Target-native ceiling**：由 target 侧写出同构 automaton，测试 source artifact 能否精确实例化该 mechanism；它与 authentic 在 32/32 任务的 success、reward 和 action sequence 上完全一致。

因此提升不能用“多了一段 prompt”“多调用模型”或“任何 symbolic heuristic 都有帮助”解释。正确的 event binding 和 stateful ledger 都是必要部分。

## 六个 source lineages

32 个目标在 outcome 前按固定 round-robin 分给六个独立 game artifacts，数量为 6/6/5/5/5/5：

| Source lineage | Tasks | Authentic success | Raw success | Strict W/L |
|---|---:|---:|---:|---:|
| Tetris | 6 | 5 | 2 | 3/0 |
| Candy Crush | 6 | 2 | 1 | 1/0 |
| Columns | 5 | 3 | 1 | 2/0 |
| Streets of Rage 2 | 5 | 4 | 2 | 2/0 |
| Thunder Force III | 5 | 2 | 1 | 1/0 |
| Strider | 5 | 3 | 2 | 1/0 |

每个 lineage 都有 source decisions，六个 artifact hash 均不同，且每个 lineage 对 raw 都是 0 strict losses。不过每个 lineage 只有 5–6 个目标，因此这些行是 replication strata，不是六个 powered per-game effect estimates。

## Prospective isolation

V4 manifest 在任何 V4 target action、provider call 或 outcome 前冻结：

- 32 个目标各有唯一 goal hash、ASIN 和 normalized instruction；
- 与 V14 的 56 个 goals、Phase 1 WebShop tasks、Phase 2 V1/V2/V3 cohorts 和 human goals 全部 disjoint；
- source assignment 不读取 V4 target semantics 或 outcome；
- 五个 matched conditions 和 17 个 gates 在执行前固定；
- live preflight 32/32 goal hashes 一致，0 action、0 provider call、0 outcome read；
- 每个正式 receipt 精确记录一次 target reset，historical outcome reuse 为 false。

独立 audit 从 160 个 receipts 重新调用 aggregator，得到与 formal report byte-equivalent 的对象和相同 report hash；17/17 gates 全部重现。

## 失败历史与 bitter lessons

失败没有删除或重写：

1. **V1：错误 Python 环境。** Python 3.13 缺 `gymnasium`，在第一个 target reset 前失败。start marker 被保留，V1 标为 consumed-incomplete，0 reset/action/provider/outcome。
2. **V2：effect 很强但 16/17 gates。** Authentic 19/32、raw 5/32、14W/0L，但一个目标在 authentic、ledger-blind、ceiling 的第 5 步连续生成 invalid target action，三个 receipts 带 failure；预注册的 `all_receipts_complete` gate 正确拒绝结果。
3. **V3：preflight 捕捉 server race。** Threaded Flask lazy initialization 并发执行多次 goal generation，造成 1/32 live hash mismatch。V3 formal 从未启动，0 action/provider/outcome。
4. **V4：只做通用 runtime 修复。** Server 固定 `threaded=False`；所有 schema retries 都无合法 action 时，只能返回 task/source/condition-blind 的安全 `noop()` 并审计。它不捕获 HTTP/transport error，也不能创建有利 action。V4 formal 中该 fallback 实际触发 0 次。

这些失败说明：fresh cohort 和强 controls 不够；还必须把 interpreter、server initialization、candidate validity 和 one-shot semantics 全部写进 frozen protocol。否则 infrastructure variance 会被误当成 transfer variance。

## Claim boundary

现在可以准确写：

> **Game-derived neural-symbolic search structure causally improved success on a fresh 32-task WebShop cohort under matched neural grounding and negative controls.**

但不能写得更强：

- **不是**六个游戏分别有统计显著的 WebShop transfer estimate；统计功效属于 aggregate common policy。
- **不是**source controller 超过 target-native algorithm；authentic 与同构 target-native ceiling 精确相等。证明的是可迁移、可实例化，而不是比 target-written 同构规则更优。
- **不是**迁移了 WebShop semantic grounding；目标仍需要 target-native neural candidate generator、BID/action validator 和 frozen outcome grounder。
- **不是**任意 source skill 都有用；六个 artifacts 共享同一 canonical search policy。当前 evidence 支持 transferable mechanism，不支持 source-specific semantic skill diversity。
- **不是**所有 metric 都改善；authentic mean steps 6.03 略高于 raw 5.72，因为它把更多任务推进到成功。成功率和 reward 提升成立，aggregate step reduction 不成立。
- **尚未**证明同一 success-rate effect 跨 ALFWorld、DiscoveryWorld、TIRBench 泛化；这三个领域目前仍停留在 Phase 1 mechanism-transfer evidence。

## 下一步

最有价值的下一步不是再换 WebShop seed，而是把同一 utility protocol 移到 ALFWorld：

- fresh multi-task reserve；
- raw / authentic / event-permuted / ledger-blind / target-native ceiling；
- paired success、reward/steps、negative-transfer 和 failure gates；
- source round-robin 与 target-native neural grounding 保持冻结；
- 先做较小的 powered cohort，再扩到 DiscoveryWorld 和 TIRBench。

## 复核命令

```bash
export PYTHONPATH=src:.
PY=/fs/gamma-projects/vlm-robot/conda/envs/vlm_benchmarks/bin/python

$PY -m pytest -q \
  tests/test_webshop_candidate_failclosed_v3.py \
  tests/test_phase2_webshop_utility_v1.py \
  tests/test_webshop_search_automaton_v16.py \
  tests/test_webshop_coverage_transfer_v14.py \
  tests/test_webshop_v14_safeguards.py \
  tests/test_webshop_unique_goal_server_v14.py

AUDIT_OUT=$(mktemp /tmp/phase2_webshop_v4_audit.XXXXXX.json)
$PY scripts/audit_phase2_webshop_utility_v4.py --output "$AUDIT_OUT"
cmp "$AUDIT_OUT" docs/results/phase2_webshop_utility_v4_audit.json
```
