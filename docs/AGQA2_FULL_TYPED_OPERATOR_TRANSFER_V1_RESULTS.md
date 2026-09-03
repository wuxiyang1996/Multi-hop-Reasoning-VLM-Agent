# AGQA 2.0 full typed-operator neural-symbolic transfer V1

## 结论

AGQA 2.0 不再只是早期的单一 temporal skill selective result。V1 已把官方 functional-program
DSL 编译为一个 target-independent typed operator VM，并在一个预注册、fresh-question reserve 上通过
六臂 formal gate：**source-induced 2320/2400（96.67%）**，对 generic scaffold 为
**374W/4L，McNemar exact `p=2.75e-105`**。所有 formal gates 均通过。

准确的 claim 是：游戏 intervention rollouts 归纳出的 symbolic capability/composition structure，配合
AGQA-native question-to-program compiler 和官方 STSG executor，可以跨域提高回答正确率。这里没有迁移
游戏像素、native actions 或 target perception weights。

## 数据边界

- reserve：2400 个此前未消费的问题，来自 600 个视频，每视频 4 题；历史 task IDs 全部排除；
- freeze 时未读取 reserve answer、functional program 或 question-specific `sg_grounding`；
- compiler 仅在 AGQA train/development functional programs 上训练；
- formal evaluator 才一次性 join official answer/program；
- **视频与 official STSG 并非 unseen**：历史实验已经覆盖全部 1814 个 test videos。因此 claim boundary 是
  `FRESH_QUESTIONS_AND_OUTCOMES;VIDEOS_AND_OFFICIAL_STSG_REUSED`，不是 video-disjoint generalization。

## 系统组成

1. source-only induction 从 Sokoban relation-binding transitions 与六个游戏的 source function programs
   归纳 15 个 authorized operators 和 32 个 composition edges；
2. target-independent VM 实现 `EXISTS/CARDINALITY/FILTER_EQ/UNIQUE/FIRST/LAST/ARGMAX/COMPARE/AND/XOR/NOT/CHOOSE/...`，
   类型或唯一性不满足时 fail closed；
3. target-native Flan-T5-small compiler 将 AGQA question 编译为完整官方 DSL program；
4. target-native executor 在共享 official STSG 上解释 program；symbolic abstention 时回退到冻结 Qwen3.5-9B actor；
5. Qwen baseline 使用相同 official STSG 来源的 compact facts，thinking disabled，不使用 program、source artifact 或 answer。

## Qualification

| Gate | Result |
|---|---:|
| AGQA development operator/composition coverage | 155,050/155,050 |
| held-out compiler program exact | 154,671/155,050 = 99.756% |
| held-out compiler syntax/source admission | 154,852/155,050 = 99.872% |
| executor development coverage | 1812/2000 = 90.60% |
| executor development conditional accuracy | 1805/1812 = 99.614% |
| fresh compiler admission before outcomes | 2397/2400 = 99.875% |

Development audit 覆盖 Query、Verify、Choose、Compare、Logic、Exists、First、Last 八类 root functions。
Executor 的 abstention 主要来自非唯一集合，而不是删除题型或 regex 不支持；formal harness 用 neural fallback
保证 2400/2400 都有最终答案。

## Six-arm formal results

| Arm | Correct | Accuracy | Symbolic committed |
|---|---:|---:|---:|
| neural-only | 1945/2400 | 81.04% | 0 |
| source-permuted | 1945/2400 | 81.04% | 0 |
| generic scaffold | 1950/2400 | 81.25% | 99 |
| **source-induced** | **2320/2400** | **96.67%** | **2201** |
| target-written-isomorphic | 2320/2400 | 96.67% | 2201 |
| oracle-program execution ceiling | 2322/2400 | 96.75% | 2212 |

Paired comparisons:

- source vs neural-only：379W/4L，`p=9.05e-107`；
- source vs source-permuted：379W/4L，`p=9.05e-107`；
- source vs generic scaffold：374W/4L，`p=2.75e-105`；
- source vs target-written-isomorphic：0W/0L。

五个 structural families 均在 reserve 中出现。source-induced accuracy 分别为：choose 77.99%、compare
98.54%、logic 95.28%、query 98.67%、verify 98.47%。Choose 是剩余主要弱点。

## “Source structure，还是任何 target controller？”

结果给出清晰但有限的回答：

- source-permuted 与 generic scaffold 都无法复现增益，说明不是任意 symbolic tokens 或一个空 VM 即可；
- target-written-isomorphic 与 source-induced 完全相同，说明**结构内容**而非 source identity 产生效果；
- 因此 novelty 不能写成“只有游戏来源才可能得到该 controller”。应写成：source intervention induction 自动恢复了
  一个达到人工同构 controller ceiling 的可执行结构，而无需在 target outcome 上手写/调节 controller。

六个 source 也不是共享一个换名字的 program。source-only lineage audit 得到六个不同 fingerprints：Candy
Crush、Columns、Thunder Force III、Tetris 产生四个不同且合格的 temporal functions；Streets of Rage 2
和 Strider 因 held-out applicability gate 失败而 abstain。四个合格 source 在独立 calibration 上为
22/30，shuffled-effect control 为 2/30，并全部可追溯进最终 AGQA algebra。

## 不能声称

- 不能称为 full official AGQA benchmark run；formal 是预注册的 2400-question sample；
- 不能称为 raw-video SOTA；本实验使用 official STSG 来隔离 grounding；
- 不能称为 video-disjoint；
- 不能说六个游戏都正向贡献，两个 source 正确 abstain；
- 不能说 target-native engineering 为零。Question compiler 与 STSG executor 是允许且必要的 domain-specific
  grounding/execution；跨域部分是 typed IR、operator content、composition 和 abstention contract。

## Authoritative artifacts

- `runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json`
- `runs/agqa2_full_operator_transfer_v1/source_specificity_audit.json`
- `runs/agqa2_full_operator_transfer_v1/compiler_flan_t5_small_v1b/heldout_evaluation.json`
- `runs/agqa2_full_operator_transfer_v1/stsg_executor_development_2000_v3.json`
- `runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/manifest.json`
- `runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/preregistration.json`
- `runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/formal_six_arm_result.json`

Formal report SHA-256：`a7fb4cb5e5d9c001f3e892ed645ab266cd72d9b6d010d5f70188dba1d6386e41`。
