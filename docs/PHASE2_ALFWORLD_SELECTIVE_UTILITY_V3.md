# Phase 2：Game → ALFWorld selective neural-symbolic transfer

## 结论

截至 2026-08-16，fresh V3 confirmatory run 状态为：

> **PHASE2_ALFWORLD_SELECTIVE_CAUSAL_UTILITY_VALIDATED — 13/13 preregistered gates passed.**

在 ALFWorld `valid_seen` 中所有剩余的 75 个未执行任务上：

| Condition | Success | Mean reward | Mean steps | Failures |
|---|---:|---:|---:|---:|
| Raw target-only neural policy | 37/75 (49.3%) | 0.493 | 45.05 | 0 |
| **Selective game-derived symbolic controller + target neural grounding** | **57/75 (76.0%)** | **0.760** | **32.97** | **0** |
| Event-binding permuted | 37/75 (49.3%) | 0.493 | 45.05 | 0 |
| Ledger-blind | 13/75 (17.3%) | 0.173 | 61.59 | 0 |
| Selective target-native isomorphic ceiling | 57/75 (76.0%) | 0.760 | 32.97 | 0 |

Matched authentic vs raw：

```text
wins / losses / ties       = 21 / 1 / 53
exact two-sided p          = 0.000010967254638671875
absolute success gain      = +20/75 = +26.67 percentage points
discordant loss rate       = 1/22 = 4.55% (frozen maximum 25%)
mean-step improvement      = 12.08 fewer steps
```

## Selective transfer 为什么必要

V2 的 fresh 32-task experiment 得到 25/32 vs 19/32，但只有 10W/4L，`p=0.1796`，没有通过强验证。四个 losses 中三个属于 `pick_two_obj_and_place`。

这不是按游戏或 outcome seed 调参，而是明确的 symbolic type mismatch：当前 controller 的 state 是一个 active candidate 加 attempt ledger，不能表示两个目标对象的 multiplicity state。因此 V3 在 outcome 前冻结：

```text
target task arity == 1  -> ENABLE_SOURCE
target task arity == 2  -> ABSTAIN_TO_MATCHED_RAW_TARGET
```

V3 没有挑选新任务：历史 outcome、V1 reset cohort 和 V2 development cohort 共占 65 个任务；数据集剩余 75 个全部进入 formal。61 个单对象任务启用 source，14 个双对象任务 abstain。独立审计确认这 14 个任务在所有 transfer/control 条件下与 raw 的 action trace 和 outcome 精确相同。

## Neural-symbolic mechanism

六个 Phase-1 游戏 lineage 各自提供独立冻结 artifact，但共享由 intervention evidence 支持的 typed policy：

```text
UNBOUND  -> EXPLORE_UNTRIED
REFUTED  -> BACKTRACK_REPLAN
VERIFIED -> COMMIT_VERIFY
```

ALFWorld target-native neural grounder 负责把当前 goal、observation、native actions 和 history 映射到 target candidates、workflow options 与 applicability。Source 不携带 ALFWorld object token、game file、native action、candidate order 或 target outcome。

Authentic 实际执行 3,172 个 source decisions，admitted actions 为：

```text
EXPLORE_UNTRIED  = 1,699
BACKTRACK_REPLAN = 1,419
COMMIT_VERIFY    = 50
unsafe commits   = 0
```

## Controls 的含义

- Event-permuted 保留 selector、source artifact 和 neural grounder，但破坏 event binding，精确退回 raw 的 37/75。
- Ledger-blind 保留局部 rule，但每步丢失跨步 attempt state，只得到 13/75。
- Target-native ceiling 在 75/75 tasks 上与 authentic 的 outcome 和 action trace 精确一致，证明 source artifact 正确实例化同构 target mechanism；它不证明 source 超过 target-written 同构算法。

## Bitter lessons

1. **V1 preflight failure**：ALFWorld batch environment 重排冻结 task order，只有 1/32 positional matches。V1 只 reset、0 action/outcome，并被保留为失败证据。
2. **V2 complete but negative**：改为 one-task environment 后完整执行 160 receipts；25/32 vs 19/32，但 4 losses 和不足的 paired power 使 5/17 gates 失败。
3. **V3 typed applicability**：不改变 controller，不删除困难任务；只把无法由单-candidate state 表达的 task arity 标为 out-of-scope，并对全部剩余 reserve 前瞻确认。

## Claim boundary

可以写：

> Six independently game-qualified artifacts instantiated a shared selective neural-symbolic search controller that causally improved success on the complete fresh 75-task ALFWorld reserve under matched neural grounding and destructive controls.

不能写：

- 六个游戏分别有 powered target effect；统计功效属于 aggregate shared policy。
- Source 迁移了 ALFWorld semantics；target-native neural grounding 仍然必需。
- 当前 controller 解决双对象 multiplicity；它对这类任务明确 abstain。
- Source 超过 target-native 同构算法；两者完全匹配。
- 该 success-rate result 已经自动扩展到 DiscoveryWorld 或 TIRBench。

## Evidence

```text
manifest  0612c38d5195122e00303de56ec1cd02c6b71a00190f9c9026408712b5abf960
report    981600e4d0f03aa77802045e1aa461c8a032fc1aee333c8d8d1af0fabf24a8d5
audit     1fb75a064f1db0cb538205e64e2ef62cd11aa614664f7e32122081c4a2ffc0fd
receipts  b39f9b6ea78a5139f177c73ca51e396406e441830abd2310271cd84e589653b9
```

- Manifest: `configs/phase2_alfworld_utility_v3/manifest.json`
- Formal report: `runs/phase2_alfworld_utility_v3/report.json`
- 375 receipts: `runs/phase2_alfworld_utility_v3/receipts/`
- Reset-only preflight: `docs/results/phase2_alfworld_utility_v3_preflight.json`
- Independent audit: `docs/results/phase2_alfworld_utility_v3_audit.json`

## Reproduction

```bash
export PYTHONPATH=src:.
PY=/fs/gamma-projects/vlm-robot/conda/envs/alfworld/bin/python

AUDIT_OUT=$(mktemp /tmp/phase2_alfworld_v3_audit.XXXXXX.json)
$PY scripts/audit_phase2_alfworld_utility_v3.py --output "$AUDIT_OUT"
cmp "$AUDIT_OUT" docs/results/phase2_alfworld_utility_v3_audit.json
```
