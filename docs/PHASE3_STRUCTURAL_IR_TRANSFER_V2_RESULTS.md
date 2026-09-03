# Phase 3 V2：source-only structural IR 的跨域迁移

> 后续 Phase 4–7 的 WebShop formal、冻结 applicability、10-source catalog 与统一执行 harness 见 [`PHASE4_TO_PHASE7_UNIFIED_NEUROSYMBOLIC_TRANSFER_RESULTS.md`](PHASE4_TO_PHASE7_UNIFIED_NEUROSYMBOLIC_TRANSFER_RESULTS.md)。

## 结论

当前可以支持的最强、但仍然有限的结论是：

> Agent 可以只从 source game 的 `(state, action, effect, next_state)` intervention tuples 自动归纳匿名 typed operators、transition、terminal predicate 与 abstention rule；在不复制 source controller、只更换 target-native neural grounder 的条件下，这些结构在 untouched DiscoveryWorld 和 TIR maze 上都提高了最终成功率，并通过 source-permuted、generic、alpha-renaming 与 target-native ceiling 对照。

这里的 **same IR** 不是“所有领域运行一个 canonical controller”。共享的是：

1. `anonymous-structural-operator-type-v1` 类型系统；
2. source-only induction 与 content-addressed artifact；
3. typed effect executor、fail-closed abstention 和 matched-control protocol；
4. target 只能学习本域实体、关系、动作和 effect grounding。

每个 target 的 domain function 可以不同，而且必须从 target development data 或 target-native interface 获得。机器可读摘要见 [`results/phase3_structural_ir_transfer_v2_summary.json`](results/phase3_structural_ir_transfer_v2_summary.json)。

## 为什么这次不是高层 heuristic

MiniGrid source learner 不接收 `EXPLORE / BACKTRACK / COMMIT` 模板。它比较前后 symbolic state，输出 alpha-invariant graph edits，例如：

| Source task | 自动归纳的 operator sequence |
|---|---|
| DoorKey | `ADD ENTITY_SLOT → UPDATE ENTITY_ATTRIBUTE` |
| PutNear | `ADD ENTITY_SLOT → REMOVE ENTITY_SLOT` |
| UnlockPickup | `ADD → UPDATE → REMOVE → ADD` |

在 fresh source reserve 上，12/12 success paths 支持 authentic program；32/32 operator occurrences 正确绑定，shuffled-effect binding 为 0/32；source-permuted program 只支持 4/12。三个 program body 不同，blind program selection 为 12/12，因此 source identity 本身不是 selector feature。

Sokoban 的 relational route 进一步从 source tuples 自动学出：

- operator：`UPDATE CONTROL_STATE POSITION`；
- transition：同一 typed effect 的 `ONE_OR_MORE` self-loop；
- terminal predicate：`ENTITY_GOAL_RELATION coverage == 1`；
- abstention：恰好一个 grounded candidate 被接受，否则 abstain。

该 program 在 72 个新 source episodes 上 72/72 唯一选中真实 path；authentic next-state binding 496/496，shuffled-next-state control 0/496。artifact 不含 source coordinates、action tokens 或手写 `EXPLORE/BACKTRACK/COMMIT` body。

## DiscoveryWorld：第一个 prospective 正向目标

DiscoveryWorld 自己从 development receipts 学出包含 target-specific observation relation 的 domain function；它不是 source body 的副本。target-native MLP 只读取当前 native state/action features，formal outcome 不作为 inference input。

12 个 fresh matched tasks 的结果：

| Arm | Success |
|---|---:|
| neural-only | 3/12 |
| source-induced | **12/12** |
| source-permuted | 3/12 |
| generic scaffold | 0/12 |
| target-native ceiling | 12/12 |

Source 相对 neural 为 9 wins / 0 losses / 3 ties，双侧 exact sign-test `p=0.00390625`；相对 source-permuted 也是 9/0/3。11 个任务出现 authentic/permuted behavior contrast，所有冻结 gate 通过。

## ALFWorld：安全的小收益，但 source-specific gate 失败

ALFWorld development-only grounder 的 operator F1 为 1.0，entity-binding F1 为 0.902，destination-binding F1 为 0.939；shuffled-effect macro F1 只有 0.0125。它自动学出 target multiplicity sequence `ADD → REMOVE → ADD → REMOVE`，并唯一匹配 PutNear 的 `ADD → REMOVE` 子图重复两次。

然而 72-task frozen extension 的最终结果是：

| Arm | Success |
|---|---:|
| neural-only | 20/72 |
| source-induced | **22/72** |
| source-permuted | **22/72** |
| generic scaffold | 0/72 |
| target-native ceiling | 50/72 |

Source 对 neural 是 2 wins / 0 losses，但 `p=0.5`；对 source-permuted 是 1 win / 1 loss。它是安全的 `+2.8pp`，却没有 source identity effect。因此 **ALFWorld 尚未验证**，不能拿这 22/72 当作 cross-domain transfer 成功。

## TIR maze：第二个 prospective 正向目标

旧 TIR maze 结果不能直接使用，因为旧 source topology rules 是手写的。V3 先完成上述 Sokoban source-only induction，再审计所有历史 TIR configs/receipts；120 个 single-image maze 中 93 个已分配，剩余 27 个在看 prompt、image、answer 或 outcome 前按 hash 冻结为 9 qualification + 18 formal。

目标端只更换 neural grounding：OpenRouter `openai/gpt-4.1-mini` 在不看 gold 的条件下绑定 `R/L/U/D` native relations 与 start/goal visual entities；pixel graph executor 将它们实例化为 typed position effects。source program 不输出答案 token。

Qualification 一次通过：

| Arm | Success |
|---|---:|
| neural-only | 5/9 |
| source-induced | **8/9** |
| source-permuted | 5/9 |
| generic scaffold | 4/9 |
| target-native ceiling | 8/9 |

随后未修改 code、threshold 或 split，直接解锁 18 个 formal untouched tasks：

| Arm | Success |
|---|---:|
| neural-only | 6/18 |
| source-induced | **12/18** |
| alpha-renamed source | 12/18 |
| source-relation-permuted | 6/18 |
| generic scaffold | 4/18 |
| target-native ceiling | 12/18 |

Source 对 neural 和 source-permuted 都是 6 wins / 0 losses / 12 ties，双侧 exact `p=0.03125`；对 generic 是 8/0/10。没有 negative transfer；alpha-renaming 完全不改变答案；source-induced 与不带 source lineage 的 target-native isomorphic ceiling 完全一致。

## 这证明了什么，没证明什么

已支持：

1. source-only state-delta / relational program induction；
2. held-out source 上 authentic effect 明显优于 shuffled effect；
3. 不同 game interventions 产生不同 program，并预测不同 target applicability；
4. shared typed IR + target-native neural grounding 可以在两个 prospective target 上提高 final success；
5. TIR 中 authentic source structure 的行为和成功率都严格优于 source-permuted control。

尚未支持：

1. 任意游戏技能都能迁移到任意 agent domain；
2. ALFWorld 的 source-specific success transfer；
3. source provenance 是解决 TIR maze 的必要条件——target-native isomorphic ceiling 与 source 相同；
4. TIR maze 的关系同构会自然扩展到 video understanding；
5. 旧的 `24/24 mechanism cells` 自动等价于 24 个 powered success-rate gains。

## 下一步

最有价值的后续不是再手写 controller，而是：

1. 在未消费的 ALFWorld/WebShop reserve 上复用冻结 source artifacts，只重训 target-native grounding；
2. 为 ALFWorld 加入 target-native deadline、irreversibility 和 binding-risk 变量，再从 development-only tasks 冻结；
3. 扩大 source family，预先预测哪个 source program 适用于哪个 target structural family；
4. 对 TIR relational result 做独立 seed/model replication，避免把 18-task `p=0.03125` 解释得过宽；
5. 视频任务先学习 target-native event graph 与 intervention effects，再测试同一 IR，不直接搬 maze executor。
