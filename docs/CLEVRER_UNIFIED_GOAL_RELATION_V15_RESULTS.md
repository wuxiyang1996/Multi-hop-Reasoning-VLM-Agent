# CLEVRER V15：统一 source-induced goal-relation transfer

## 结论

CLEVRER synthetic event-graph extension 已经接入与 Phase 8 相同的 fail-closed
neural-symbolic authority chain，并在事先冻结、此前未读取的 360-question reserve 上通过
预注册 gate：

| Condition | Exact correct | Accuracy | Recovery actions |
|---|---:|---:|---:|
| Neural-only explicit relation | 236/360 | 65.56% | 0 |
| **Authentic source-induced** | **252/360** | **70.00%** | 25 |
| Strong target-base recovery | 251/360 | 69.72% | 28 |
| Generic error scaffold | 245/360 | 68.06% | 39 |
| Source-permuted uplift | 240/360 | 66.67% | 57 |
| Shuffled proof binding | 243/360 | 67.50% | 29 |
| Inverted source effect | 201/360 | 55.83% | 70 |
| Target-native representation ceiling | 273/360 | 75.83% | 37 |

Matched comparisons：

- authentic vs neural-only：`16W/0L/344T`，exact two-sided `p=0.0000305`；
- authentic vs source-permuted：`16W/4L/340T`，`p=0.01182`；
- authentic vs shuffled proof binding：`10W/1L/349T`，`p=0.01172`；
- authentic vs strong target-base：`5W/4L/351T`，`p=1.0`。

因此可以支持：在这个固定的 CLEVRER synthetic event-graph setup 中，source-induced
program 相对 neural-only 提高 whole-question exact success rate，并且正确 source/effect
binding 显著优于 source-permuted 与 shuffled-binding controls。

不能支持：source provenance 必然优于同类 target-only learner。Strong target-base 已达到
`251/360`，与 authentic 的 `252/360` 统计不可区分；generic scaffold 的差异也不显著。

## 迁移的东西

V15 不再使用旧 V14 代码中固定写出的 `COMMIT/POSITION/VERIFY` controller。Source 是后来从
Sokoban `(state, action, effect, next_state)` 自动归纳、并在 fresh source episodes 上确认的匿名
program：

```text
UPDATE ENTITY_GOAL_RELATION / RELATION_COVERAGE
  -- observed positive relation delta -->
UPDATE ENTITY_GOAL_RELATION / RELATION_COVERAGE  [ONE_OR_MORE]

terminal: ENTITY_GOAL_RELATION coverage == 1
otherwise: abstain
```

该 artifact 明确记录 `named_controller_template_used=false`、`target_data_read=false`。CLEVRER
侧只替换 target-native grounding：frozen neural proof grounder 把 paired event-graph executor
receipts ground 成 relation-coverage uplift。Unified selector 只决定是否授权这个 anonymous
program；只有 CLEVRER executor 可以输出 `trajectory` representation switch。若 grounding
不满足 positive-delta guard，harness fail closed，保留 explicit-relation baseline。

正式 360 题中只授权 25 次，且恰好只有这 25 次调用 target executor。所有 authorization、state、
proof receipt、source envelope 与文件 lineage 都是 content-addressed；runtime receipt 不含 gold
answer 或 official functional program。

## 实验顺序与无泄漏边界

1. 旧 V14 formal 的 720 题被明确降为 V15 consumed development；它不产生新的 confirmation。
2. Development adapter 逐题复现旧 decision，并通过 source-only、template-free、controls、authority
   和 cost gates。
3. 冻结 config 前扫描 V14 reserve 的 360 个 question IDs；`runs/` 与 `docs/results/` 中 exposure
   matches 为 0，且 reserve 与 development/formal IDs 完全不交叠。
4. 固定 grounder、threshold `0.2`、conditions、lineage 和 failure gates后，才打开 reserve。
5. 独立 auditor 重算 condition counts、paired wins/losses、exact p-values、reserve identity 和所有
   authority gates。

这仍然是一次 adaptation-based prospective replication，不是 zero-shot ontology induction。

## 成本

- 外部 provider calls：`0`；
- OpenRouter/Gemini/Claude cost：`$0.00`；
- 只读取本地 CLEVRER official executor 与已下载的 paired PropNet prediction files；
- formal 使用 706 个唯一的本地 prediction files，没有上传视频、帧或 prompt。

这也是为什么本轮优先做 CLEVRER，而没有恢复 STAR/NExT-QA/Video-Holmes spend。

## 研究边界

V15 验证的是 **game-induced symbolic relation program → structured synthetic-video event-graph
recovery**。它尚未验证：

- raw-pixel target-native tracking/event grounding；
- active `TEST → Bayesian world-particle update → replan` 多步 video belief MDP；
- STAR、NExT-QA、Video-Holmes 或任意自然视频；
- authentic 显著优于 strongest target-only learner；
- source provenance 对一个 extensionally identical target-written controller 是必要的。

因此论文中合适的表述是“prospective structured-video extension with source-binding evidence”，
而不是“general video MDP transfer solved”。

## 证据与复现

- Development audit：
  [`results/clevrer_unified_goal_relation_v15_development.json`](results/clevrer_unified_goal_relation_v15_development.json)
- Independent compact audit：
  [`results/clevrer_unified_goal_relation_v15_summary.json`](results/clevrer_unified_goal_relation_v15_summary.json)
- Frozen reserve config：
  [`../configs/clevrer_unified_goal_relation_v15_reserve.json`](../configs/clevrer_unified_goal_relation_v15_reserve.json)
- Portable formal artifact：
  `artifacts/video_event_graph_v15/formal_report.json.gz`

关键 hashes：

```text
frozen config file    3736d9f6c306753151d9570dde57da45700b9422089853d39ef348e0e844ddc4
raw formal report     bc788b73808be87b7f8b51428e6c208b0733ebfdb9f77dab3607805951414491
portable gzip         fbdd2a54f54e0118b549cfcaf276efef867e7e5dfbe1a1d62c813afc9b2b33b3
compact audit file    62d0ea4a0cdebb6d2a3365e30bb2cf4c11c64d6497b6572252fa743e90c5b905
```

```bash
PYTHONPATH=src:. python scripts/audit_clevrer_unified_goal_relation_v15_development.py
PYTHONPATH=src:. python scripts/freeze_clevrer_unified_goal_relation_v15.py
PYTHONPATH=src:. python scripts/run_clevrer_unified_goal_relation_v15.py \
  --config configs/clevrer_unified_goal_relation_v15_reserve.json \
  --output runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json
PYTHONPATH=src:. python scripts/audit_clevrer_unified_goal_relation_v15.py
```

注意：reserve 已经消费；不要再次把它当作 fresh confirmation。
