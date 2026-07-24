# 四域七 Cell 可迁移性诊断

## 结论先行

本轮诊断矩阵固定为四个 target domain、七个 cell：

| Domain | Cell |
|---|---|
| Visual reasoning | VisualToolBench、TIR-Bench |
| Video reasoning | Video-Holmes |
| Browser | MiniWoB、WebShop |
| Embodied text | ALFWorld valid-seen、valid-unseen |

SIV-Bench 已移除，不进入 adaptation、binding、test 或汇总指标。因此 Video 当前只有一个 benchmark，不能写成多 benchmark video evidence。

资产审计表明，问题不是普遍缺数据。VTB、TIR、Video-Holmes、MiniWoB、WebShop 和 ALFWorld 数据均存在；七个 cell 的一样本/测试 manifest 现已冻结。新建的 target-only runner 均 fail-closed，不调用旧 Harness 的 silent stub fallback。机器可读结果见 [`results/target_feasibility_audit_v1.json`](results/target_feasibility_audit_v1.json)。

当前其余六个 cell 已具备真实 executor、冻结 split 与 official outcome。VTB 官方 evaluator 已确认公开，但尚未接入 clean runner；工程可运行不等于迁移成立：所有现有 source 候选仍是 `GENERIC_ONLY` 或未通过 source gate。

## 已核实资产

| Cell | 数据与 official outcome | 当前执行状态 |
|---|---|---|
| VisualToolBench | 1204 rows、1917 turns、7777 rubrics；7.3 GB parquet 的 SHA-256 与官方 LFS 对象一致。DuckDB 可稳定读取 | 官方 `xi1ngang/VisualToolBench@d4f200a` 已发布 single/multi-turn judge；clean runner 尚未接入。冻结 `row:865` 实为 2-turn，旧 runner 只运行 turn 0；官方 reference 需要 `google_search → python_image_processing`，而当前 runner 只有局部图像工具。因此还存在 multi-turn、官方六工具集合与 typed-coordinate 三类执行差异 |
| TIR-Bench | 1215 rows、1255 image refs、0 missing；MCQ gold 可直接评分 | 两个冻结 smoke 均真实运行：一个无答案，另一个答错且声称不存在的 `verify_claim`；必须机械校验 region/receipt 引用 |
| Video-Holmes | train/test 共 3388 questions、503 unique videos；503/503 视频存在 | 冻结 smoke 解码 6050 帧并产生 34 个真实 tool receipts，但 6 轮全部用于扫描而无最终答案；需保留 finalization budget |
| MiniWoB | 125 task dirs、250 episodes；125 条 episode 有 official reward | 已安装固定 commit 的 MiniWoB 页面资产；冻结 `unicode-test` live smoke 一步 `click('13')`，official reward=1.0 |
| WebShop | 6 个历史 rollout roots、50 unique tasks、至少 519 条本地 episode metadata；full 1k-product vendor tree 和 bridge 均存在 | 已恢复 full BM25 server；冻结 `webshop.32` live smoke 4 步完成，official reward=0.6667。进程已在 smoke 后关闭；stub reward 不得用于实验 |
| ALFWorld valid-seen | data、真实 admissible actions、official `won` evaluator、adaptation receipt 和 frozen manifest 均存在 | matched target-only/authentic 动作逐步相同；source review 在 step 0 因无 receipt citation 被拒绝，`NO_VALID_SOURCE_INTERVENTION` |
| ALFWorld valid-unseen | data、adaptation receipt、frozen manifest 和 official outcome 均存在 | binding induction 不稳定而 fail-closed；matched 两组完全相同，`BINDING_REJECTED_SAFE_FALLBACK` |

## 第一轮真实 smoke 诊断

真实执行说明当前最先要修的不是 source→target 语义，而是跨域通用的 Harness 可信执行协议：

1. **Capability preflight**：任务需要 OCR/caption/video decode 时，缺能力必须在 episode 前标记，不消耗模型轮次后才发现。
2. **Typed tool calls**：所有参数按工具自身 JSON schema 验证和无歧义 coercion；模型不能自由改写 detector region。
3. **Receipt-only evidence**：最终 schema 只能引用 runtime 已签发的 tool/transition receipt ID；不存在的 `verify_claim` 必须使结果 invalid/abstain。
4. **Budget state machine**：interaction budget 与 finalization budget 分开，最后一轮禁止继续扩张搜索，只允许基于已有 receipt 提交或 abstain。
5. **Liveness receipts**：每轮调用、timeout 和中间 receipt 立即落盘，避免数分钟无 heartbeat。

这些约束只依赖环境声明的工具 schema、真实调用记录和预算，不包含游戏/ALFWorld/视觉的手写 ontology，因此可作为通用 Harness，而不是 target heuristic。

逐样本 receipt hash、真实 tool-event 数、validation error、Browser official reward 和 ALFWorld
matched 结果汇总在 [`results/target_smoke_summary_v1.json`](results/target_smoke_summary_v1.json)。
这些仍是 target-only/基础设施 smoke，不是 source motif transfer 结果。

## Source treatment 审计

不能直接把旧 mega-skill 文件塞进 target prompt：

- `megaskills_all_stages/mega_skills.jsonl` 有 20 families、180 members；严格限制到六个 Phase-1 游戏后只剩 79 members。
- `reasoning_aligned_mega_skills.json` 有 59 rows，其中至少 20 个 target-domain members。原文件作为 source treatment 会泄漏 target 数据，已禁止。
- mega-skill 的 authority 固定为 `LINEAGE_RETRIEVAL_ONLY`。自然语言 family、name、description 和压缩 plan 都不是可执行真值。
- 当前 source qualification 尚无 `SOURCE_SUPPORTED` motif。因此 target 实验中的 authentic game source 必须标记为 `UNQUALIFIED_SOURCE_DIAGNOSTIC`，不能提前称为 reasoning-backbone transfer。

允许进入诊断的 source 信息只有：

1. 严格六游戏 allowlist 内的 member lineage；
2. 真实 source transition/replay receipt 所支持的 graph topology；
3. frozen source candidate 及其 renamed、shuffled、other-game controls。

不允许 target skill bank、target member、手写 source→target action mapping、embedding alignment 或 predicate ontology。

## 冻结实验协议

### 1. 一样本边界

每个 cell 预先生成一个 manifest：

- adaptation pool 与 test pool 先按 benchmark 官方 split 分离；
- adaptation example 只通过 sample ID 的 SHA-256 顺序机械选择，不能看题型、答案、reward 或 source candidate；
- 一个 example 只用于初始化 provisional binding；
- binding artifact 在 test 前冻结；test 期间只能用 live official transition 更新 version space，不能改写 motif 文本；
- binding 不稳定、无共同支持动作或 verifier 不可用时，立即 source-off 并回退 matched target-only。

### 2. 两个 Agent 的权限

Decision Agent 独占 target-native action proposal 和最终选择。Motif/Harness Agent 只能：

- 从 frozen game graph 与一个 adaptation trace 提出 binding hypothesis；
- review 已选 action；
- 引用 live transition 给出 supported/refuted/unknown；
- 建议 replan、abstain 或 source-off。

Harness Agent 不能输出 target action，不能访问 test gold，不能修改 official evaluator。

Target-native tool/action schema 不是人工 semantic mapping。TIR 的图像工具、Video-Holmes 的 frame tools、BrowserGym high-level actions和 ALFWorld admissible commands 由环境本身定义；禁止的是把 source 名称手写成这些 target actions。

### 3. 六个 matched conditions

每个 test item 使用相同 Decision Agent、temperature、seed、上下文上限、action/tool budget 和 wall-clock cap：

1. `target_only`
2. `generic_reasoning`
3. `authentic_game_source`
4. `renamed_game_source`
5. `shuffled_game_source`
6. `other_game_source`

必须同时报告 official task score、steps/tool calls、token/cost、source replan、abstain、fallback step 和 invalid-action rate。主要 estimand 是：

```text
authentic_game_source - generic_reasoning
authentic_game_source - renamed/shuffled/other_game_source
```

只超过 target-only 不够，因为提升可能来自更多文字或更多推理预算。

## 实施顺序

1. 先冻结七个 adaptation/test manifests，以及只含六游戏 lineage 的 source bundle。
2. 把 ALFWorld valid-seen 补成与 valid-unseen 相同的真实 vertical slice。
3. 在 clean repo 实现 TIR/VTB 的 target-native tool loop；任何媒体、executor 或 official evaluator 缺失必须 `NOT_RUNNABLE`，禁止 fallback。
4. 实现 Video-Holmes real frame/tool loop。不能复用 deterministic `harness/video_executor.py` 作为结果路径。
5. 配置 MiniWoB HTML runtime，启动 WebShop server；Browser helper crash 必须终止 cell，不能降级为 echo executor。
6. 每个 cell 先跑 1 adaptation + 2 held-out items × 6 conditions 的 matched smoke。只有 initial-state/media hash、budget 和 evaluator 全部配对后才扩量。

## 停止规则

出现以下任一情况，该 cell 停止并标记，不补假结果：

- `STUB_FALLBACK_BLOCKED`：执行器是 echo/deterministic/synthetic；
- `CONTAMINATED`：source bundle 出现 target member 或 target-derived skill；
- `BINDING_UNSTABLE`：重复 induction 没有稳定交集；
- `NO_LIVE_SUPPORT`：source advisory 未被任何 target transition 支持；
- `NEGATIVE_TRANSFER`：相同 budget 下 authentic 显著降低 official score 或增加 steps/cost；
- `GENERIC_ONLY`：authentic 不优于 renamed/shuffled/generic controls。

若七个 cell 都没有 authentic 相对 controls 的增量，则停止“显式 transferable reasoning backbone”主张。仍可保留的贡献是：一个能在线检测 source motif 无用/有害并安全回退的 selective transfer harness，以及对远域负迁移边界的系统诊断。
