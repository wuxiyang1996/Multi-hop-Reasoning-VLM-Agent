# VisualToolBench 官方评测与游戏 Motif 迁移协议

## 结论

VisualToolBench 不缺官方 evaluator。官方仓库为
[`xi1ngang/VisualToolBench`](https://github.com/xi1ngang/VisualToolBench)，本实验固定到
commit `d4f200a0a44790349667bed09334ac88623074b2`。旧项目只实现了 gold-string
diagnostic；它不能替代官方 rubric judge。

当前 VTB 不能直接作为完整 benchmark 主结果，原因有三项：

1. v1 冻结样本 `row:865` 是 2-turn，但旧 runner 只执行 turn 0；
2. 官方环境提供六类工具，旧 runner 只有局部图像工具；
3. 官方 reference 对 `row:865` 使用 `google_search → python_image_processing`，旧 runner
   缺前者，因此其空答案不能解释成 motif 或模型能力失败。

因此 v2 先限定为 **VTB single-turn subset**，不能写成完整 VisualToolBench。冻结 manifest 为
[`vtb_single_turn_manifest_v2.json`](../configs/vtb_single_turn_manifest_v2.json)：603 个 single-turn
任务，one-shot adaptation 为 `row:105`，held-out smoke 为 `row:55,row:319`。选择只使用
`turncase/num_turns` 和 ID hash；未读取 prompt、gold、rubric、reference tool trajectory 或模型结果。

2026-07-22 的机械 preflight 已固定并检出六个官方工具，tool-contract SHA-256 为
`8e97e537c4be22733d911d5aa85e84330c04b7c4a816e29fbb1d466f89e9c16f`。安装缺失 Python 依赖后，
官方 inference 可加载；但 `keys.py` 和环境均缺 `SERP_API_KEY`、`OPENWEATHER_API_KEY`，所以
`paper_faithful_full_tool_ready=false`。报告见
[`vtb_official_runtime_audit_v2.json`](results/vtb_official_runtime_audit_v2.json)。
再次直接解析 `/fs/gamma-projects/vlm-robot/keys.py` 后，现有变量只有 B2、OpenAI 和 OpenRouter；
两个外部工具 key 确实不在文件中。OpenAI key 还是 regional key，Harness/judge 必须固定请求
`https://us.api.openai.com/v1`，默认 hostname 会返回 `incorrect_hostname`。

还需注意，官方参数名虽为 `max_tool_calls=20`，固定 commit 的实现实际每轮模型响应只把 counter
加一；单轮若返回多个函数调用，真实 function-call 数可超过 counter。我们不静默修改官方代码，
而是同时记录 `official_round_counter` 与 `executed_function_calls`，并用 tool-contract hash 保证六条件
使用完全相同行为。

## 官方 evaluator

官方 judge 对每个 atomic rubric 单独调用一次 LLM。输入为 question、gold answer、单个 rubric
description 和 model answer，输出 `Met/Not Met` 与解释。

```text
ARS = sum(weight_j * 1[Met_j]) / sum(weight_j)
APR pass = all(1[Met_j] for every rubric with weight_j >= 4)
```

必须按 `weight >= 4` 判断 critical。不能读取数据的 `critical=yes` 代替：本地 7777 个 rubrics
中，两种定义有 2357 处不一致。论文主实验 judge 是 `o4-mini`；正式 matched conditions 必须
共用同一个固定 judge、prompt、temperature 和 retry policy。

实现位于：

- `src/motif_transfer/vtb_evaluator.py`：APR/ARS、turn/rubric 完整性和 judge JSON 校验；
- `scripts/run_vtb_official_judge.py`：逐 rubric judge、prompt/response hash、usage receipt；
- `tests/test_vtb_evaluator.py`：critical rule、multi-turn aggregation 和 fail-closed tests。

## 如何验证 game mega-skill / motif 的可迁移性

研究对象不是“给 target prompt 多一段游戏文字”，而是 game-trained policy 中由 source receipt
支持的 reasoning motif，经过 one-shot 初始化后是否能在 target live evidence 下减少适配代价或提高
官方结果。

六个条件固定为：

```text
target_only
generic_reasoning
authentic_game_source
renamed_game_source
shuffled_game_source
other_game_source
```

每个 held-out item 必须匹配：初始图像 hash、Decision/Harness/judge model hash、官方工具 schema、
固定 commit 的 20-round cap、temperature 和输出上限。Motif Agent 不能产生目标工具参数或最终答案；它只能提出
advisory、prediction、verification、replan、abstain 或 source-off。Decision Agent 始终独占官方
target tool call 和最终答案。

一个 target adaptation example 只初始化 provisional binding。test 时每次 advisory 都必须引用
frozen source receipt、当前 target 的真实 tool/transition receipt，以及一个下一步可判定的 expected
transition 或 termination test。引用不存在、证据不足或 binding 被反驳时，Harness 必须 source-off
并继续 matched target-only，不能改写 source skill 文本来迁就 target。

主要 estimand 是 authentic 相对 generic、shuffled 和 other-game 的 APR/ARS 差，而不是只比较
target-only。`renamed_game_source` 是 alpha-renaming **不变性检验**，不是应被击败的 destructive
control：保留 receipt 和图拓扑，只替换游戏特定符号；若它失效而 authentic 有效，更像词汇/身份依赖，
不能支持结构 backbone。正式实验须预注册 equivalence margin。若 authentic 只超过 target-only，
但不超过等长 generic/control context，不能归因于游戏 reasoning backbone。负迁移同样是正式结果：
还要报告 tool calls、invalid calls、成本、fallback step，以及 Harness 是否及时 source-off。

机械验证器位于 `src/motif_transfer/transfer_matrix.py`，会拒绝缺条件、初始资产/model/tool/budget
不匹配的矩阵，并区分 `NO_LIVE_SUPPORT`、`STRUCTURE_TRANSFER_PILOT`、
`ALPHA_RENAMED_NO_LIVE_SUPPORT`、`LEXICAL_OR_IDENTITY_DEPENDENT_PILOT`、
`NEGATIVE_TRANSFER_PILOT` 和
`GENERIC_OR_CONTROL_EXPLAINS_EFFECT`。

在线 runner 已实现为 `scripts/run_vtb_interposed_single_turn.py`：每轮严格执行

```text
Decision Agent target-native proposal
→ deterministic schema/authority validation
→ Motif/Harness review (no tool name/arguments fields)
→ official tool execution
→ hash-bound live receipt
→ Motif/Harness verification
→ continue / replan / source-off
```

相同 Decision 请求使用跨条件共享的 persistent exact-request cache。temperature=0 仍不能保证
OpenRouter completion 相同；没有 cache 的旧 smoke 已实测首 proposal 漂移，因此正式 matched run
强制要求 `--decision-cache`。Harness 若输出目标工具字段、引用不存在的 receipt、或 source treatment
缺少 `SOURCE_SUPPORTED` qualification，runner 会立即拒绝。

`scripts/compile_vtb_treatments.py` 只接受两个独立的 `SOURCE_SUPPORTED` bundle，并机械生成
authentic、alpha-renamed、shuffled、other-game 和 matched-generic 五个 treatment。它不读取 target
prompt/gold，不做 source→target 语义映射。other-game 也必须独立通过 source gate，不能拿任意失败
candidate 填充对照。

## 当前诚实状态

当前没有任何 motif 达到 `SOURCE_SUPPORTED`。因此配置
[`vtb_motif_transfer_v2.json`](../configs/vtb_motif_transfer_v2.json) 的 confirmatory source gate 是
`BLOCKED_NO_SOURCE_SUPPORTED_MOTIF`。现在可以运行的 authentic 只能叫
`UNQUALIFIED_SOURCE_DIAGNOSTIC`，不能称为迁移成功或失败。

这条 gate 是必要的：如果 source motif 自己没有通过 fresh no-human-hints、blind recurrence 和
multi-horizon value controls，那么 target 上的任何提升都无法归因于从游戏学到的 reasoning backbone。

当前 readiness receipt 为
[`vtb_transfer_readiness_v2.json`](results/vtb_transfer_readiness_v2.json)：`BLOCKED`，阻塞项恰为
“官方 full-tool key 不完整”、“没有冻结的 SOURCE_SUPPORTED authentic motif”和“没有独立
SOURCE_SUPPORTED other-game control”。这不是说实验不可证伪；而是 confirmatory treatment 还未被
授权。六条件 compiler、在线 interposition、官方 scorer 和身份检查均已可执行。

## 已运行的官方 adaptation 诊断

只在冻结前声明的 adaptation `row:105` 上使用 `--allow-degraded-adaptation` 跑了官方 tool loop，
没有触碰 held-out gold/rubric。Qwen3.5-35B-A3B 执行 20/20 function calls：前三次因虚构
`image_clue` 变量失败，之后生成 17 个图像变换，但反复旋转/裁剪且没有最终回答，最终以
`OFFICIAL_CAP_EXHAUSTED` 终止。完整 receipt 为
[`vtb_adaptation_official_target_only_v2.json`](results/vtb_adaptation_official_target_only_v2.json)。

这条轨迹给出了一个可在线判定的 target failure pattern（失败后重复同类动作且无信息增益），但它
**不能**用来人工选择 source motif。冻结 source candidate 后，Harness treatment 能否更早 replan/stop，
必须由六条件 matched run 决定；若 generic 或 shuffled 同样改善，就不属于游戏 motif transfer。

首次 3-round 比较没有共享 Decision cache，temperature=0 下首 proposal 仍漂移；该批只用于发现
common-randomness bug，不进入结果。修复后重新完成 matched mechanism smoke：target-only 与 generic
condition 使用同一图像、官方 commit、tool contract、初始动态 schema、Decision model、budget 和可见
asset path。所有 Harness user payload 机械 padding 到 exact 6000 `o200k_base` tokens。exact-request
cache 保证 generic 首轮 review 的 proposal 与 target-only 首轮 proposal 完全相同；发生 replan 后才允许
Decision 轨迹分叉。结果为：

```text
target-only: 3 executed calls, 3 successful transforms, 0 final answer
generic:     1 replan, 2 executed calls, 0 successful transforms, 0 final answer
```

generic verifier 对两次执行均返回 `REFUTED`。因此当前结论是
`GENERIC_INTERPOSITION_HARMED_TOOL_EXECUTION_WITHOUT_TASK_SUCCESS`：generic review 的 replan 把一个
原本可执行的 proposal 改成两个失败 proposal。这证明 online interposition、negative-effect capture、
receipt、token matching 和 common-randomness 机制工作；不是 task success，更不是 game motif transfer。
机器摘要见
[`vtb_interposition_matched_smoke_summary_v2.json`](results/vtb_interposition_matched_smoke_summary_v2.json)。

剩余执行顺序为：

1. 补齐 SerpAPI/OpenWeather capability；在此之前 held-out runner 继续 fail-closed；
2. 六游戏 source gate 通过后冻结一个 authentic motif 和一个独立 other-game motif，编译五个 treatment；
3. 使用已采集的 adaptation trace 初始化 provisional binding；
4. 在 `row:55,row:319` 运行六条件 matched smoke，并用官方 judge 评分；
5. smoke 只用于查 bug；扩量、alpha-equivalence margin 和统计阈值必须在读取更多 held-out outcome 前预注册。
