# Phase 9：source-program heterogeneity 与跨域效用审计

## 结论

这轮补上了旧 Phase 6 最弱的一环：Phase 6 只证明匿名 structural type checker 能选到
正确 program；Phase 9 进一步把这个 outcome-blind 预测逐条绑定到四个已经冻结的 matched
formal reports，并重算 authentic 与 destructive source control 的最终成功率。

结果是：11 个 source-only induced programs 中，四个目标接口匿名选中了 3 个不同的
program body；wrong-family program 在 4/4 route 上都 fail closed。四个被选 program 的
authentic binding 均严格优于各自 source-permuted/effect-permuted control：

| Target route | Selected source program | Authentic | Source control | Paired |
|---|---|---:|---:|---:|
| WebShop V21 | Sokoban recurrent relation | 23/32 | 7/32 | 16W/0L，`p=3.05e-5` |
| DiscoveryWorld Easy | MiniGrid PutNear `ADD→REMOVE` | 12/12 | 3/12 | 9W/0L，`p=.003906` |
| TIR maze | Sokoban recurrent relation | 12/18 | 6/18 | 6W/0L，`p=.03125` |
| ALFWorld multiplicity | Sokoban goal acquisition + relation | 20/24 | 13/24 | 7W/0L，`p=.015625` |

四条 route 描述性合计为 38 wins / 0 losses，但没有报告一个 pooled iid p-value：不同
domain 的 task 与 intervention 不可交换，推断单位仍然是各 route 自己的 paired test。

机器可读结果见
[`results/phase9_source_program_heterogeneity_v1.json`](results/phase9_source_program_heterogeneity_v1.json)。

## 为什么这比 source identity 更强

Selector 只读取匿名 contract：IR kind、ordered typed operators、recurrence 和 terminal
predicate family。它不读取 game name、source identity、target action 或当前 target
outcome。三个被选择的 program 分别是：

1. MiniGrid PutNear：`ADD ENTITY_SLOT → REMOVE ENTITY_SLOT`；
2. Sokoban relation：recurrent `UPDATE CONTROL_STATE/POSITION`，terminal 为
   `ENTITY_GOAL_RELATION`；
3. Sokoban goal acquisition：`UPDATE ENTITY_RELATION → UPDATE CONTROL_STATE →
   UPDATE POSITIVE_EFFECT_BINDING → UPDATE ENTITY_GOAL_RELATION`。

这三个 body 不同。对每个 target 强制换成另一个 IR family 时，type checker 都 abstain；
在正式 target execution 内破坏正确 effect/terminal binding 时，四个 matched controls 又都
回落到 baseline。这把“结构适用性预测”和“结构正确绑定产生效用”连在了一起。

旧 arcade family 的负结果没有被覆盖：其 authentic/source-permuted 在 held-out source 上仍为
22/27 vs 22/27，所以没有一个 arcade temporal function 被晋升为这四条 structural route。

## 仍不能声称什么

Phase 9 是对 prospective formal runs 的 retrospective cross-report audit，不是一个新的
prospective target experiment。它支持 registered interfaces 上的 content-specific program
selection 与 source-binding utility，但仍不证明 source provenance 相对 extensionally
isomorphic target-written program 是必要的，也不证明任意 game 可以迁移到任意 target。

## DiscoveryWorld Normal 的 stop decision

Normal 没有被偷偷补成 positive result。新的 development-only V25 qualification audit 重算了
V22/V24：Proteomics Normal acquisition 已经达到 2/2 commit coverage 和 2/2 target success，
但唯一 matched final-commit fork 中 target-native comparator 为 1/1、authentic 为 0/1；binder
只表示 `same_location`，而任务需要 `adjacent`。开发任务数 2、matched forks 1，也低于新 formal
所需的最小证据量。

因此 fresh Normal reserve 保持关闭。下一次合法实验必须改到更早、确有 target-policy headroom
的 measurement/tool-acquisition decision，在至少 8 个 development tasks、6 个 matched forks 上
学会 target-native neural grounding 和 `adjacent` relation 后再冻结。机器 stop receipt 见
[`results/discoveryworld_normal_v25_qualification_stop.json`](results/discoveryworld_normal_v25_qualification_stop.json)。

## 复现

```bash
PYTHONPATH=src:. python scripts/audit_phase9_source_program_heterogeneity_v1.py
PYTHONPATH=src:. python scripts/audit_discoveryworld_normal_v25_qualification.py
PYTHONPATH=src:. pytest -q \
  tests/test_phase9_source_program_heterogeneity_v1.py \
  tests/test_discoveryworld_normal_v25_qualification.py \
  tests/test_structural_ir_applicability.py
```
