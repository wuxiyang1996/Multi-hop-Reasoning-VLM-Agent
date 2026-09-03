# WebShop V14 repair and natural-video cost boundary

日期：2026-08-14

## Bottom line

WebShop 值得继续，但旧的 32-task replication 不能再当作 32 个独立目标：本地 server 实际只有
13 个 human goals，session bridge 用 task ID 对 `len(goals)` 取模。V13 `114–145` 与 replication
`146–177` 各自都只是同一组 13 个 goal semantics 的 2–3 次重复。任务级结果仍是历史观察，但
independent-goal confirmatory claim 被撤回。

这个基础设施问题已经修到可运行状态，而且没有消耗 provider calls：

- 启用 WebShop 自带的 synthetic goal generator，不改 vendor 搜索、页面、动作、reward 或 executor；
- 把 seed 提前到随机 product-price generation 之前，消除跨 restart 的 price-threshold 漂移；
- 冻结 24 个 development + 32 个 formal goals，共 56 个不同 instruction semantics 和 56 个不同 ASIN；
- 56 个 ASIN 与旧 human-goal pool 的 13 个 ASIN 完全隔离；
- live server 全量检查 `56/56` goal hashes 一致，formal outcome 未打开；
- 加入 target-native constraint set coverage：只有观测到 state change 的 prerequisite 才算 verified。

自然视频则相反：当前应该停新 API spend。V37 已用同一个 Gemini model、同一组 24 frames，在
201 questions / 28 disjoint videos 上完成正式 matched evaluation，花费约 `$11.63`。Typed proof 仅比
direct 多 1 题，authentic Sokoban source gate 反而少 1 题，而 inverted source 多 2 题。缺的不是更多帧，
而是有实际 answer headroom 的 target evidence operation，以及 source 相对 destructive controls 的独特价值。

## WebShop 旧结果的 independence correction

| split | task-level result | unique semantic goals | semantic-cluster sign test |
|---|---:|---:|---:|
| V13 IDs 114–145 | 18/32 vs 11/32；7W/0L；`p=.015625` | 13 | 5 positive / 0 negative；`p=.0625` |
| replication IDs 146–177 | 14/32 vs 9/32；6W/1L；`p=.125` | 13 | 4 positive / 1 negative；`p=.375` |
| combined, descriptive only | — | 同一 13 clusters | 6 positive / 1 negative；`p=.125` |

两段 ID 的 13 个 unique instruction strings 完全重合。不同 `initial_state_hash` 不能修复该问题，因为
canonical URL 和 prompt 包含 task-specific ID；它们可以不同，但目标语义仍相同。机器审计为
[`results/webshop_semantic_independence_v14_audit.json`](results/webshop_semantic_independence_v14_audit.json)。

因此准确状态是：

```text
WEBSHOP_SEMANTICALLY_INDEPENDENT_REPLICATION_NOT_ESTABLISHED
```

不能再写“WebShop 已在 32 个独立 fresh goals 上确认”。可以写“在 13 个 local human-goal families 的
重复 episodes 上存在正向 task-level signal，但 cluster-aware evidence 不显著”。

## V14 runnable repair

Vendor data 不是只有 13 个可用目标。审计得到：

| pool | products | option-specific goals |
|---|---:|---:|
| installed human subset | 13 | 13 |
| native synthetic generator | 415 eligible products | 6,910 |

新的 adapter `webshop_unique_goal_server_v14.py` 只把 vendor 已有的 `human_goals=False` 路径暴露出来。
第一次 56-goal preflight 发现 4 个 hash 漂移，原因是 upstream 在 product price 随机采样后才 seed；V14
把 seed 提前后重新冻结，全量 preflight 通过：

```text
WEBSHOP_SYNTHETIC_SERVER_V14_PREFLIGHT_PASSED
live goals checked = 56
unique semantics = 56
unique ASINs = 56
hash mismatches = 0
```

Formal manifest 是
[`../configs/webshop_synthetic_unique_v14_frozen.json`](../configs/webshop_synthetic_unique_v14_frozen.json)，
live receipt 是
[`results/webshop_synthetic_server_v14_preflight.json`](results/webshop_synthetic_server_v14_preflight.json)。

### Multiplicity-safe readiness

旧 selector 用两个 booleans 表示 ready state：是否看到 checked constraint、是否看到 unchecked
constraint。它不能表达“size 与 color 都必须完成”。在唯一 regression `webshop.171` 中：

```text
required = {10.5, black}
verified by state-changing actions = {black}
missing = {10.5}
old action = Buy Now
reward = 0.75
V14 commit_authorized = false
```

`ConstraintCoverage` 现在以 goal-overlap signature 建 target-native set ledger；no-op radio click 不计入
verified，successful paired-label state change 才计入。这个修复保持 source 的 `POSITION/COMMIT` 为二元，
但 target grounding 不再丢失 prerequisite multiplicity。

### V14 development execution：target bridge 已修复，source-specific increment 未通过

最初 synthetic development smoke 暴露了两个 live blocker：BrowserGym 的直接 radio click 是 no-op，旧的
goal-token overlap 又把 `3x-large` 错误扩成多个普通 `large` prerequisites。V14 live runner 现在执行：

1. 从冻结 synthetic manifest 读取 typed `goal_options`，以 exact normalized value 建 constraint identity；
2. 把配对 `LabelText` action 确定性加入所有 matched conditions 的候选池，但不改变 target-only rank 0；
3. 只有 action 后 observation hash 改变才把 constraint 标为 verified；
4. 商品缺少任一 required option 时 fail closed、返回搜索页，并跳过已经被证伪的 ASIN；
5. coverage ready 后才允许 source program 决定 `COMMIT` 或 `POSITION`。

最终机制在两个不同 synthetic ASIN development goals 上完成六条件 matched probe：

| condition | strict success | mean steps | source decision |
|---|---:|---:|---:|
| target-only | 0/2 | 12 | — |
| target-native coverage-only | 2/2 | 8 | — |
| authentic Sokoban effect + coverage | 2/2 | 8 | 2 COMMIT |
| commit-availability control + coverage | 2/2 | 8 | 2 COMMIT |
| inverted effect + coverage | 0/2 | 12 | 5 POSITION |
| position-prior + coverage | 0/2 | 12 | 5 POSITION |

全部 receipts 完整、initial states matched、coverage conditions 的 unsafe commits 为 0。`webshop.1` 的精确
`black` 与 `3x-large` LabelText actions 都造成 state change，随后 authentic source 执行 COMMIT 并得到
reward 1；`webshop.4` 先拒绝两个缺失 options 的商品，再打开未试 ASIN、完成 options 和购买，同样得到
reward 1。

这个结果验证了 **target-native symbolic bridge 的 live success-rate effect**，但没有通过 source-specific
transfer gate：authentic 与 coverage-only 完全持平，也与更简单的 commit-availability control 完全持平。
它只超过 inverted/position controls。准确状态是：

```text
TARGET_NATIVE_BRIDGE_VALIDATED
SOURCE_SPECIFIC_WEBSHOP_TRANSFER_NOT_VALIDATED
FORMAL_RESERVE_REMAINS_SEALED
```

因此不能把 0/2 → 2/2 的恢复归因于 Sokoban knowledge；当前增益来自 target-native exact grounding、
intervention verification 和 infeasible-product backtracking。机器记录为
[`results/webshop_coverage_transfer_v14_development.json`](results/webshop_coverage_transfer_v14_development.json)。
到 V5 为止，去除 copied cache 后的 unique provider cost 约 `$0.0874`。

### 被保留的早期 smoke 记录

新的 synthetic development split 最初做了两组完整 diagnostic smoke，然后因明确 futility signal
主动停止，没有继续消耗第 3/4 个 smoke goal：

| complete goal | target-only | authentic | authentic source decisions |
|---|---:|---:|---:|
| webshop.0 | 0 | 0 | 0 |
| webshop.1 | .7143 | .7143 | 0 |

两个 goal 的六条件组完整、12 receipts 无 failure；30 个 cached provider calls 约 `$0.0400`。第三个 goal
只开始 target-only，因停止而不进入任何比较或 summary。这个结果不是效应估计，而是 development
preflight failure：旧 target MLP 在真实新语义上没有打开 source applicability。`webshop.1` 的直接 radio
click 连续 no-op，最后在 `black` 与 `3x-large` 都未 verified 时购买，得到 5/7 reward；V14 coverage gate
会拒绝这次 commit。

机器记录是
[`results/webshop_synthetic_v14_development_smoke_v1.json`](results/webshop_synthetic_v14_development_smoke_v1.json)。
因此还没有在 synthetic split 上证明 success-rate 提升，也不能把 coverage fix 本身当作 source transfer。
下一次只先修 development path，并至少比较：

1. `target_only`；
2. `target_native_coverage_only`；
3. `authentic_sokoban_effect + target_native_coverage`；
4. inverted/permuted source + 同一 coverage；
5. target-written isomorphic effect guard + 同一 coverage。

这段早期结果只作为 blocker diagnosis，不与最终机制结果合并。下一步也不是扩大 task 数或降低 source
threshold，而是在 development 中先找到 `positive expected commit effect` 与 `mere commit availability`
预测不同的 target states，并用无 outcome leakage 的 target-native grounding 区分它们。只有 authentic
超过 coverage-only 和 commit-availability control，才允许打开 32-goal formal reserve。Primary inference
unit 是 unique ASIN goal，不再是任意 task ID。

本地复现命令：

```bash
PYTHONPATH=src:. /fs/gamma-projects/vlm-robot/conda/envs/vlm_benchmarks/bin/python \
  scripts/run_webshop_unique_goal_server_v14.py

PYTHONPATH=src:. python scripts/verify_webshop_synthetic_server_v14.py
```

## 视频：正式结果已经回答“缺什么”

V37 不是小样本 smoke：201 个从未进入 V36 adaptation 的 questions，28 个 video clusters，STAR 136 题、
NExT-QA 65 题。Direct 与 proof 使用同一个 `google/gemini-3.1-pro-preview` 和完全相同的 24 frames。

| condition | correct | versus matched direct |
|---|---:|---:|
| matched direct | 150/201 = 74.63% | — |
| raw typed target proof | 151/201 = 75.12% | 8W/7L；`+0.50pp`；`p=1` |
| authentic source CATE | 149/201 = 74.13% | 3W/4L；`-0.50pp`；`p=1` |
| inverted source | 152/201 = 75.62% | 5W/3L；`+1.00pp` |
| same-rate marginal | 151/201 = 75.12% | 5W/4L；`+0.50pp` |

13 个正式 gates 失败。V27 还已经测试过 active frame navigation：STAR active source 为 48/128，
matched active direct 为 60/128，7W/19L，`-9.375pp`，`p=.02896`。因此增加帧或让 pixel-change tool
继续搜索不是合理修复。

另一方面，generic grounding 已不是主要 blocker：V30–V33 的 natural candidate grounding 覆盖
35/36，covered accuracy 33/35 = 94.3%。真正缺的是：

- **target operator headroom**：如果 raw proof 本身只有 `+1/201`，source selector 没有稳定的好动作可选；
- **source specificity**：authentic 必须严格超过 inverted、marginal、binding 与 target-written controls；
- **结构同构性**：自然视频更需要 event dependency / causal ancestor / counterfactual mutation，而不是
  generic `COMMIT → VERIFY`；
- **prospective applicability**：必须在看当前 outcome 前识别哪些 target states 会从该 evidence operation 获益。

CLEVRER 仍是有效的 positive boundary：511/720 vs 489/720，27W/5L，`p=.000113`。它说明显式 event
graph 上 game→video neural-symbolic mechanism 可成立；它不能外推到 raw natural video。

机器审计为
[`results/natural_video_cost_boundary_v38_audit.json`](results/natural_video_cost_boundary_v38_audit.json)。
当前资源决定：

```text
new natural-video provider calls = STOP
increase frame count = DO NOT DO
Video-Holmes = DEPRIORITIZE
CLEVRER structured-video result = RETAIN
```

如必须继续自然视频，先只用现有 V37 receipts 做 leave-video-out error audit。只有新的、预注册的 target
evidence operator 在旧数据上显示稳定 intrinsic headroom，并且 source features 能区分 proof wins 与
destructions，才值得采集新视频结果。

## V15 source-only triage gate（2026-08-15）

在继续 WebShop 前，我们先要求 source 独立识别三分支控制：`INFEASIBLE -> BACKTRACK`、
`FEASIBLE_AND_UNTRIED -> EXPLORE`、`READY_AND_POSITIVE_EFFECT -> COMMIT -> VERIFY`。审计没有读取
WebShop outcome，也没有 provider call。

结果只有第三个分支通过。Sokoban fresh confirmation 的 96 个 labels 全部属于 COMMIT/POSITION，
REPLAN/BACKTRACK 为 0；真实 GymV 的 32 个 matched SWITCH/PERSIST cells 中，qualification 与 held-out
的 common-continuation estimand 都没有 SWITCH win。因而：

```text
SOURCE_TRIAGE_GATE_V15_FAILED_CLOSED
WebShop V15 target adapter = NOT AUTHORIZED
WebShop formal reserve = SEALED
```

详细结果见 [`SOURCE_TRIAGE_GATE_V15_RESULTS.md`](SOURCE_TRIAGE_GATE_V15_RESULTS.md) 与
[`results/source_triage_gate_v15.json`](results/source_triage_gate_v15.json)。下一步必须先在真实 source
states 上做 matched `BACKTRACK / EXPLORE_UNTRIED / COMMIT` forks；不能把 V14 target coverage
heuristic 反向包装成 source skill。

## Authoritative files

- WebShop semantic audit: `scripts/audit_webshop_semantic_independence_v14.py`
- WebShop reserve guard: `src/motif_transfer/webshop_semantic_reserve.py`
- WebShop multiplicity guard: `src/motif_transfer/webshop_constraint_coverage_v14.py`
- WebShop V14 live controller: `src/motif_transfer/webshop_coverage_transfer_v14.py`
- WebShop V14 development runner: `scripts/run_webshop_coverage_transfer_v14.py`
- Synthetic server adapter: `src/motif_transfer/webshop_unique_goal_server_v14.py`
- Outcome-blind freezer: `scripts/freeze_webshop_synthetic_reserve_v14.py`
- Live fail-closed verifier: `scripts/verify_webshop_synthetic_server_v14.py`
- V15 source-only gate: `scripts/audit_source_triage_gate_v15.py`
- Video cost audit: `scripts/audit_natural_video_cost_boundary_v38.py`
