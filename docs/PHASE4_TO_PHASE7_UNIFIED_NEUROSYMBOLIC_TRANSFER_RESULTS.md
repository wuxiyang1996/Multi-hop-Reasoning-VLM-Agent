# Phase 4–7：统一 neural-symbolic transfer harness

> **Phase 8 更新（2026-08-17）：** 本文保留 Phase 7 当时的历史结论。旧的
> Minigrid PutNear→ALFWorld route 仍然正确 abstain；新的 source-only
> goal-acquisition route 已通过 24-task independent formal replication，并在同一个
> unified harness 下把注册的正向 route 扩展为 4/4。见
> [`PHASE8_FOUR_DOMAIN_UNIFIED_NEUROSYMBOLIC_TRANSFER_RESULTS.md`](PHASE8_FOUR_DOMAIN_UNIFIED_NEUROSYMBOLIC_TRANSFER_RESULTS.md)。

## 最终结论

截至 Phase 7，非视频路线已经形成一个可运行、fail-closed 的统一 harness：

> source 端只从 `(state, action, effect, next_state)` intervention receipts 归纳 content-addressed symbolic program；target-native neural grounder 只提供当前状态的 typed binding；匿名 structural type checker 选择相容的 source program；冻结的 paired evidence 决定这条 route 是否值得使用；只有 target-native executor 能输出本域动作。

它支持三条已有正式成功率证据的正向 route：

| Target | Source-induced | Neural-only | Source-permuted | 结论 |
|---|---:|---:|---:|---|
| WebShop V21 | **23/32** | 7/32 | 7/32 | validated；16W/0L/16T，exact `p=3.05e-5` |
| DiscoveryWorld Easy | **12/12** | 3/12 | 3/12 | validated；9W/0L/3T，exact `p=0.00390625` |
| TIR single-image maze | **12/18** | 6/18 | 6/18 | validated；6W/0L/12T，exact `p=0.03125` |
| ALFWorld multiplicity | 22/72 | 20/72 | 22/72 | **未验证**；utility/authenticity 不足，harness abstain |

因此这里不是“四个 target 都成功”。准确结论是：**三条 route 有正式正向证据；第四条 route 被同一个 harness 正确拒绝。** 视频 benchmark 按当前决定不在 Phase 4–7 范围内。

机器可读结果：

- [`results/webshop_structural_v21_formal_compact.json`](results/webshop_structural_v21_formal_compact.json)
- [`results/phase5_unified_applicability_v1_audit.json`](results/phase5_unified_applicability_v1_audit.json)
- [`results/phase6_source_specific_applicability_v1.json`](results/phase6_source_specific_applicability_v1.json)
- [`results/phase7_unified_neurosymbolic_harness_v1.json`](results/phase7_unified_neurosymbolic_harness_v1.json)

## Phase 4：WebShop fresh formal transfer

V21 在冻结的 32 个 fresh synthetic goals 上一次性运行五个 matched conditions：`neural-only`、`source-induced`、`terminal-permuted`、`generic scaffold` 和 `target-native ceiling`。

结果：

- strict success：source-induced `23/32`，neural/permuted/generic 均为 `7/32`，ceiling `23/32`；
- source 对 neural 和 permuted 都是 `16W/0L/16T`；
- mean reward：source `0.7525`，neural `0.5781`；reward pair 为 `18W/3L/11T`；
- source 与 ceiling 的 32 条 trajectory 全部一致；
- 261 次 source-authorized decision，0 次 unsafe authorization；
- 16/16 preregistered gates 通过。

这不是把 Sokoban 动作搬进 WebShop。迁移的是 recurrent relational transition program；WebShop 的 low-sample grounder 和 executor 仍然使用本域 search/click/option relation。

## Phase 5：冻结 applicability 与 utility gate

Phase 5 在不 reset 新任务、不读取新 outcome 的条件下冻结并审计 route：

- WebShop、DiscoveryWorld Easy、TIR：结构相容，而且过去 matched evidence 的 Beta posterior 单侧 95% 下界同时支持 `source > neural` 和 `source > permuted`，因此允许；
- DiscoveryWorld Normal / Space Sick：target interface 与已验证 grounder contract 不同，因此拒绝；
- 8 个 execution-untouched ALFWorld multiplicity tasks：结构相容，但历史证据只有 `2W/0L/70T` 对 neural、`1W/1L/70T` 对 permuted，因此以 `DIRECTIONAL_UTILITY_NOT_CALIBRATED` 拒绝。

审计结果为 `PHASE5_FROZEN_SELECTIVE_ROUTING_VALIDATED`，4 个 probe 被允许、12 个被拒绝，8/8 gates 通过。这个阶段证明的是 **outcome-blind routing**，不声称未打开的新任务已经成功。

## Phase 6：source-specific，而不是 source identity

Source catalog 扩展到 10 个自动归纳 program：

- MiniGrid：DoorKey、PutNear、UnlockPickup 三个不同 finite operator sequences；
- Sokoban：一个 recurrent relational program；
- arcade：Tetris、Candy Crush、Columns、Streets of Rage 2、Thunder Force III、Strider 六个 sparse temporal functions。

匿名 structural contract 只比较：

1. IR kind；
2. ordered operator signatures：`operation / predicate_family / arity / value_kind`；
3. recurrence；
4. terminal predicate families。

它不接收 game name、source identity、target action 或当前 target outcome。四个 target interface 都唯一选中了已有正式 evidence 对应的 program：DiscoveryWorld/ALFWorld 选择 PutNear 的 `ADD ENTITY_SLOT → REMOVE ENTITY_SLOT`；TIR/WebShop 选择 Sokoban 的 recurrent `UPDATE CONTROL_STATE` 加 `ENTITY_GOAL_RELATION` terminal predicate。把这两个 program 对调后，4/4 target interface 都 fail closed。

六个 arcade program 对四个 structural target 的误匹配为 `0`。同时保留旧 V4 negative result：qualified arcade functions 在 source reserve 上 authentic 和 source-permuted 都是 `22/27`，所以 `SOURCE_SPECIFIC_DOMAIN_FUNCTIONS_FAILED`，没有把它们升级为 target route。这揭示了一个重要 bitter lesson：只用高度相关的 H1/H4/H8/persistence value features，会学出不同参数，却未必产生 source-specific 行为。

Phase 6 的准确 status 是 `PHASE6_SELECTIVE_SOURCE_SPECIFIC_APPLICABILITY_VALIDATED`：**structural programs 的 source-specific applicability 成立；arcade temporal-function family 仍然 abstain。**

## Phase 7：统一的四段 authority chain

统一 runtime 的执行顺序是：

```text
source transition tuples
  → source-only inducer
  → typed program + abstention rule
  → anonymous structural contract
  → target-native grounding requirement + current-state binding
  → unique structural type match
  → frozen utility/authenticity posterior gates
  → execution authorization (no action field)
  → target-native executor (only action authority)
```

Phase 7 将 10 个 source program 全部包装为统一 induction envelope，并对四条 route 做 end-to-end authority canary：

- 3 条 calibrated route 被授权并到达 target-native executor；
- ALFWorld 结构匹配，但在 utility gate 被拒绝，executor 调用为 0；
- selector authorization 没有 action field；
- program、grounder、executor、state 与 evidence 都由 SHA-256 绑定；
- requirement 与 applicability receipt 不一致、source/route program 不一致、hash drift、当前 outcome 暴露或非 native action 都 fail closed；
- 9/9 Phase 7 gates 通过，status 为 `PHASE7_UNIFIED_NEUROSYMBOLIC_HARNESS_VALIDATED`。

Phase 7 canary 验证 composition 和 authority boundary；它复用已有正式 success evidence，没有再次打开 formal target tasks，因此不会制造额外 success-rate claim。

## 代码入口

- source/target 匿名 type checker：`src/motif_transfer/structural_ir_applicability.py`
- 冻结 utility/authenticity runtime：`src/motif_transfer/unified_transfer_runtime.py`
- Phase 7 composition：`src/motif_transfer/unified_neurosymbolic_harness.py`
- Phase 5 freeze/audit：`scripts/freeze_phase5_unified_applicability_v1.py`、`scripts/audit_phase5_unified_applicability_v1.py`
- Phase 6 audit：`scripts/audit_phase6_source_specific_applicability_v1.py`
- Phase 7 audit：`scripts/audit_phase7_unified_neurosymbolic_harness_v1.py`

复现核心审计：

```bash
PYTHONPATH=src:. python scripts/audit_phase6_source_specific_applicability_v1.py
PYTHONPATH=src:. python scripts/audit_phase7_unified_neurosymbolic_harness_v1.py
PYTHONPATH=src:. pytest -q \
  tests/test_structural_ir_applicability.py \
  tests/test_unified_transfer_runtime.py \
  tests/test_unified_neurosymbolic_harness.py \
  tests/test_online_transfer_utility.py
```

## Phase 7 后仍缺什么

1. **ALFWorld 正向 replication**：需要只用 development tasks 改进 target-native grounding/decision variables，然后冻结新的 formal reserve；在 evidence 变强前，现有 harness 应继续 abstain。
2. **Phase 5 unopened probes 的 prospective outcome test**：当前只完成冻结的 applicability audit；若执行，必须保持 threshold、source artifacts 和 route evidence 不变。
3. **Arcade source IR redesign**：需要从 intervention tuples 归纳非共线的 causal predicates，例如 delayed gain、irreversibility、hazard、recoverability，而不是继续扩大同一组累积 value horizons。
4. **独立 replication**：WebShop、DiscoveryWorld 与 TIR 的当前结果需要不同 seeds/model 或独立实现复现，特别是 TIR 的 18-task 边界显著性。
5. **Video**：本阶段明确跳过；若以后恢复，应先建立 target-native event graph 与 intervention semantics，不能直接复用 maze executor。

## 最重要的研究边界

现在成立的不是“game skill 普遍迁移到任何领域”，也不是“一个 canonical controller 横跨所有 domain”。成立的是：

> 在 source-only intervention induction、匿名结构类型检查、target-native neural grounding、source-permuted 对照和校准 abstention 同时存在时，某些结构程序可以跨语义领域提高成功率；不满足证据条件的 route 可以被同一系统可靠拒绝。
