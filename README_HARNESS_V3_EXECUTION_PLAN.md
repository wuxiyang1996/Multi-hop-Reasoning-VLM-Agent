# Harness v3：从可观察游戏轨迹到可检验跨域控制先验

本文是 `README_NON_HEURISTIC_REASONING_BACKBONE.md` 的执行计划。目标不是把游戏
动作翻译成 ALFWorld 动作，而是检验游戏中 evidence-qualified 的多步控制候选，在
同一条 target one-shot 证据约束下，是否比 target-only 候选带来额外泛化。

## 1. 不变的研究边界

Harness 只预定义可判定接口：receipt、hash、Agent closed schema、环境
`reset/step/admissible/evaluator` 和 proof obligations。以下内容禁止进入 verdict：

- 人工 source→target semantic mapping；
- 手写 predicate ontology；
- embedding/text clustering、阈值和 top-k score；
- 固定长度 skill segmentation；
- majority vote、LLM judge 或 confidence；
- 用 held-out reward 更新 candidate/artifact。

35B/9B 都容易 hallucinate，因此模型永远只是 proposer/Actor：伪造 ID、operator、
action 或 receipt 都由 closed-schema 和 exact native evidence 拒绝。模型解释不构成证据。

## 2. 端到端数据流

```text
frozen full source episode
  → exact transition-ID evidence queries
  → independent Agent control hypotheses
  → reference validation + content-hash set union
  → fresh-env seeded replay-to-fork interventions（支持时）
  → independent target binding Agents
  → one fixed successful target-native demo admission
  → immutable v3 candidate-set artifact
  → per-candidate native exact-action sets
  → exact set intersection
  → one Actor call restricted to the common set：execute；empty set 则 abstain
  → held-out official evaluator（read-only）
```

这里的 candidate set 是 budget-limited Agent-proposed set，不是完整 version space。
除非未来定义有限 grammar 并穷举，否则不能声称“所有与证据一致的程序”。

## 3. 已实现的机械层

- `skill_agents/evidence_query.py`：Agent 只能按 immutable transition ID 请求证据，
  response 自身有 hash；没有 semantic retrieval。
- `skill_agents/control_hypotheses.py`：验证 program/receipt/node/edge 引用与 observed
  path 连续性；只做 hash dedup + set union，不排序或投票。
- `harness/reasoning_event_log.py`：hash-chained rollout event log，统一记录 proposal
  set、admissibility、decision、step、delta 和 official stop。
- `harness/replay_fork.py`：同 seed 重放 prefix、验证 fork state hash、再执行原生
  admissible alternative；不支持或不一致时输出结构化 gap。
- `env_wrappers/subprocess_env.py`：Gym-V seed/options 已通过 RPC 传给 worker；
  Orak seed 明确 `RESET_SEED_UNSUPPORTED`，不再静默退化为 unseeded reset。
- `harness/multistep_binding.py`：独立 v3 artifact；一个 source node 可绑定一个非空、
  有序 target transition span；只接受 target demo 中真实出现的 operator/type/linear
  order，不从单条成功轨迹证明 branch/loop/retry。
- `harness/candidate_set_runtime.py`：保留所有 qualified candidates，机械求它们当前
  cursor 支持的原生 exact-action set 交集，并只在交集上调用一次 Actor。交集为空即
  abstain；endpoint failure 是 error，不伪装成 abstention。该规则避免把同构候选的
  重复随机采样误当成结构分歧。

查看代码层 readiness（不是实验 performance）：

```bash
python scripts/report_multistep_harness_readiness.py \
  --output runs/harness_v3/readiness.json
```

## 4. source replay 的真实限制

旧 299 条 episode 没有 seed receipt、environment/config/ROM/code fingerprint，也没有
Agent proposal/decision events。旧生成器中的 `42 + episode_index` 只能标为
`PROVENANCE_HYPOTHESIS`；只有 fresh environment 按该 seed 重放并逐步对齐 observable
state、native action set、action、reward 和 done 后，才能升级为 replay receipt。

当前环境支持边界：

- GamingAgent 的 2048/Candy/Tetris：可做 fresh-env replay probe；Tetris 每个分支必须
  新建环境，不能依赖同实例 repeated reset；
- Gym-V：seed RPC 已修，但仍需记录 env/ROM/config fingerprint；
- Orak/Mario：当前 `initial_obs()` 路径忽略 seed，必须输出 gap；
- 当前只能可靠声称 `observable_state_sha256`；没有 hidden-state snapshot 时不能声称
  完整 simulator-state causality。

2026-07-20 的最小真实 probe 已在 `vlm_benchmarks` 环境通过：2048 使用两个 fresh
env、seed 42 重放 prefix `up`，fork state 的 observable receipt 完全一致，随后执行
原生 alternative `right` 得到一条 `INTERVENTION_OBSERVED` receipt。它只证明这一
seed/prefix/alternative 的可观察干预被执行；没有升级任何 guard、branch、boundary 或
transfer claim。receipt 位于
`runs/harness_v3_20260720/source_replay_fork_receipt.json`。

## 5. 下一批真实运行，按 proof dependency 排序

### P0：新 source rollout instrumentation

production episode runner 已有显式 opt-in，记录 seed、environment fingerprint、完整
text observation、structured state、skill candidate receipts、raw Agent response、parsed
decision、policy transform、executed action、native admissibility/delta 和 stop。它严格区分
`AGENT`、`FALLBACK` 与 `POLICY_POSTPROCESSOR`；后两者不能作为 Agent reasoning evidence。
当前 Actor 协议本身不产生 `REPLAN/ABSTAIN`，因此记录为 unsupported，而不是伪造
`CONTINUE`。2048 没有 official win evaluator，只记录 native reward/termination。

2026-07-21 已用 Qwen3.5-35B、无 LoRA、seeds 42–45 运行 4×12-step 真实 smoke：

- 4/4 episode event protocol 完整，合计 444 events；
- 48/48 decisions 为 `AGENT` origin；0 parser fallback、0 policy override；
- 31,998 prompt tokens、2,905 completion tokens，reported cost `$0.0092031075`；
- 仅生成四个 batch 文件，合计约 292 KB，没有 per-frame 小文件。

artifact 位于 `runs/source_evidence_2048_35b_seed42_45/`。旧 299 条 source episodes 与
`emnlp2026`、`emnlp2026-2` 中对应的五个游戏逐内容 hash 一致，因此没有重复下载。

### P1：deterministic replay pilot

先选一个 GamingAgent 环境和一个 Gym-V 环境。每个 fork 都重新创建 env、相同 seed
重放 prefix、验证 fork receipt，再从该时刻 native actions 中运行 alternative。不得
用语义挑 alternative；资源不足时可分批穷举，但必须记录未执行项。

报告 exact replay rate、unsupported、bootstrap failure 和 mismatch；后三者都是 gap，
不能写成“skill failed”。

同一 smoke 已在每个 episode 的 step-1 fork 上，对所有三个非原始 native actions 分别
创建 fresh env 并重放：12/12 为 `INTERVENTION_OBSERVED`，0 replay mismatch。加上此前
单条 probe，当前共有 13 条 2048 observable intervention receipts。它们只验证对应
seed/prefix/action 的可观察分叉，不证明 guard、branch 或 transferable semantics。

### P2：真实 source hypothesis Agents

`proposer_a`、`proposer_b`、`skeptic` 独立通过 content-addressed query 读取证据并提出
node/span/control edge。Harness 保留所有结构合法的非等价候选。没有 intervention
receipt 的 guard/branch/retry/termination 永远保持 `AGENT_HYPOTHESIS`。

首个真实 smoke 已对完整 6-step Sokoban trace 运行 Qwen3.5-35B：3 个角色中 1 个明确
abstain，2 个产生结构合法但互相不同的 hypothesis，0 endpoint failure；总计 30,370
prompt tokens、601 completion tokens、OpenRouter reported cost `$0.0054214575`。其中
`BRANCH`/`LOOP` 文字仍是未验证 Agent claim，没有 intervention receipt，因此没有任何
control claim 被升级。完整 raw replies、usage、proposal receipts 和 hypothesis hashes 在
`runs/harness_v3_20260720/source_hypotheses_35b.json`。

第二个真实 source-program run 使用上述 4 条 instrumented 2048 episodes 和 12 条 fork
receipts。每条候选必须把完整 12-step observed path 无遗漏、无重叠、按顺序分区为至少
两个 nodes；这是结构完整性约束，不是 skill segmentation heuristic。4 episodes × 3 roles
共 12 次 Qwen3.5-35B calls，0 endpoint failure，历史 reported cost `$0.01450221`：

- 6 个完整多步候选通过，四条 episodes 分别保留 `1/3/1/1` 个；
- 4 个单-node 输出被 `MULTIPLE_NODES` 拒绝；
- 2 个把 step-1 receipt 挂到后期 edge 的输出被
  `INTERVENTION_RECEIPTS_EDGE_ATTACHED` 拒绝；
- 2 个通过候选带有位置合法的 fork citations；其余 4 个只依赖 observed path；
- 所有 6 个仍是 `AGENT_HYPOTHESIS`，0 semantic control claims verified。

原始 12 replies 在 `source_program_hypotheses_35b.json`；收紧 attachment 后的离线重验证
结果在 `source_program_hypotheses_35b_v2.json`，重验证 API calls 为 0。两者位于
`runs/source_evidence_2048_35b_seed42_45/`。

这批数据已足够进入下一步 matched target-binding smoke：让相同预算的 binding Agents
分别看到完整 source candidate set 或空 source evidence。仍不得按 node 数、reward、
receipt 数或自然语言解释挑选“最佳”source program。

### P3：paired target binding

用完全相同的 model、roles、prompt budget、candidate budget 和固定 target demo 生成：

1. `SOURCE_HYPOTHESIS` candidates；
2. `TARGET_NATIVE_SAME_DEMO` candidates（source evidence 字段为空）。

禁止直接复制 expert demo 作为 target oracle policy。两组唯一允许差异是 source
evidence 是否存在。v3 admission 命令：

```bash
python scripts/admit_alfworld_multistep.py \
  --demo artifacts/admission_demos/alfworld/pick_and_place/train_seed42_v3_shot0.json \
  --proposals runs/harness_v3/source_or_target_proposals.json \
  --output-root runs/harness_v3/admission
```

2026-07-21 使用新 2048 instrumented source programs 重跑时发现并修复三项 fail-open/
实验设计错误：旧 loader 会把嵌套的 6 个候选静默读成 0；35B 会混淆 source/target
数字索引；target binding 允许只覆盖 demo 子集。最终协议使用独立 `target_tN` identity，
并要求两组都完整、无重复地分区固定 demo。相同 Qwen3.5-35B、3 roles 下，两组均为
3/3 valid、0 endpoint error。

随后审计发现 node/span 只被冻结但没有送进 runtime Actor，旧 P4 因而不能检验 source
贡献。`node_binding_version=3` 现在把每个 source node 的 observed transitions 和 incident
Agent-hypothesis edges 原样冻结；admission 检查 exact identity，runtime 把所有活动候选
同时送给 Actor，不排序、不投票。target-only 的 source conditioning 严格为空。

### P4：small end-to-end pilot

在开发 episode 上验证 exact action-set intersection、single Actor call、cursor advance、
native transition receipt、abstain/error 分类和 artifact immutability。任何 candidate
不能因 invalid/异议被运行时静默删除。

修复后的 source-conditioned 版本已完成 8 个成对 development episodes；8/8 initial
observation hash 相同，source treatment 在 39/39 source Actor calls 中存在，在 target-only
中为 0/35：

| 条件 | success | mean steps | abstain | error | Actor calls | API cost |
|---|---:|---:|---:|---:|---:|---:|
| source conditioned | 2/8 | 4.875 | 6 | 0 | 39 | $0.02737905 |
| target-only | 2/8 | 4.375 | 6 | 0 | 35 | $0.00278768 |

source abstain 原因是 3 次 `CANDIDATE_PROGRAM_FINISHED`、3 次
`NO_COMMON_EXACT_COMMAND`；target-only 是 1 次 program finished、5 次 no-common。
两组各有 2 次 official success；其中一对仅 source 成功、一对仅 target-only 成功，净
增益为 `0`。source prompt tokens 为 161,898，target-only 为 14,424，成本约 9.8 倍，
没有性能增益，不能声称 transfer。

原始结果和机械汇总位于 `runs/harness_v3_matched_binding_20260721/`：
`source_conditioned_runtime_seed47_dev8.json`、
`target_only_conditioned_runtime_seed47_dev8.json` 和
`conditioned_paired_dev8_report.json`。旧 `source_span_dev8` 结果只保留为 bug audit，不能
作为 transfer comparison。

### P5：先补 online target Harness，再新建 v3 2×4 preregistration

当前状态：**BLOCKED BY NO POSITIVE DEVELOPMENT SIGNAL**，不是系统错误。不要修改既有
v2 spec，也不要现在占用 2×4 L40S。source programs、fork receipts 和 runtime treatment
均已真实接通，但当前 2048 receipts 没有带来增益且显著增加成本。下一步应预先选择另一
source domain/program family，或设计通用的 Agent receipt-use protocol；不得根据上述
ALFWorld 失败 episode 手调 source→target mapping。

在扩大实验前，target Harness 必须从当前的 frozen-program executor 补成 online
proposal–execution–verification loop。one-shot demonstration 的作用不是永久确定
source→target 对应关系，而是启动一个暂定实例化：

> 用一个 adaptation example 启动在线实例化，在后续真实交互中持续验证其是否可用。

2026-07-21 已实现最小 episode-local online rebinding vertical slice：source abstention 或
连续无 observable delta 请求有界 `REBIND`；独立 Agent 必须用 closed JSON 覆盖全部当前
candidate，复制精确 candidate/hypothesis/node identity，只引用当前 native action 编号，
并在执行前声明 closed target-native evidence queries。Admission 检查 scope、identity、
candidate coverage、action-set intersection 和 evidence contract，把通过项保留为
`AGENT_HYPOTHESIS` receipt。Controller 和 runtime 只接受 admission registry 中的 receipt。

Rebind action 每次只执行一步；Harness 随后机械检查 `ACTION_ACCEPTED`、observation/action-set
delta、executed-action membership、native reward 或 official success。满足 contract 只表示
该暂定 binding 没有被所声明的 target-native evidence 反驳，不证明语义等价、任务进展或
positive transfer。所有 transition、binding、verification 和 state-machine logs 都有可重新
加载验证的 hash receipt。

Target-only fallback 也已改为 shadow runtime：只有实际 source/rebind command 同时符合
fallback artifact 当前 exact native contract 时才推进 cursor。若在线 rebind 改变了轨迹，
系统输出 `TARGET_ONLY_FALLBACK_PREFIX_INCOMPATIBLE` 并 fail closed，而不是按 step count
伪造 target-only 对齐。

同 seed 47、相同四个 ALFWorld train episodes、6-step cap、Qwen3.5-35B 的开发矩阵如下：

| 条件 | success | mean steps | abstain | total tokens | reported cost |
|---|---:|---:|---:|---:|---:|
| B：target-only | 1/4 | 3.75 | 3 | 8,119 | `$0.00150018` |
| N：naive source | 1/4 | 4.50 | 2 | 77,966 | `$0.01413726` |
| H：source + online Harness | 1/4 | 5.50 | 2 | 102,854 | `$0.01958119` |
| R：rotated conditioning + online Harness | 1/4 | 5.50 | 2 | 102,843 | `$0.01864268` |

四组 initial observation hashes 全部配对。H 中 2 次 rebind calls 均通过 admission、真实执行，
且 2/2 预声明 contracts 被 transition 满足；但 H、N、R 对 success 都没有增益，H 与 R 的
success vector 完全相同。R 只在 frozen candidate identities 之间机械轮换 untrusted
source-conditioning payload；它不随机化 admitted target program topology。26 条 control
receipts、2 条 binding receipts 和 2 条 verification receipts 均已从磁盘重新验证。

该矩阵仍不是 causal negative-transfer result：只有 4 pairs，而且实际 Agent calls/tokens
没有 matched；当前只能报告 source/Harness 增加 steps、tokens 和 cost，却没有开发集 success
gain。汇总位于
`runs/harness_v3_online_rebind_20260721/online_transfer_matrix_dev4.json`，明确保持
`authorizes_large_scale_2x4=false`。

仍未实现：`DEFER/DECOMPOSE/RECOMPOSE`、在线 target-only 重新绑定，以及正式六个 source
games 的同等级 instrumented reasoning/replay receipts。因此 readiness 仍为 `PARTIAL`。

正式六个 source games 的旧数据已完成第一层 source-only 编译：固定使用每个游戏的
`episode_000`，没有按 ALFWorld 结果或 skill 名称筛选；Thunder Force III、Streets of
Rage 2、Strider、Columns、Tetris、Candy Crush 共生成 6 个完整 episode TracePrograms、
409 个 observed transition receipts，6/6 exact source-file replay 通过、0 failure。结果在
`runs/formal_six_source_observed_20260721/`。这些旧 episodes 没有完整 proposal/replan/abstain
event protocol、seeded fresh-env replay 和 intervention receipts，因此 artifact 明确写为
`reasoning_claim=none_observational_trace_only`、`official_success_claim=false`；它们不能直接
升级成 evidence-qualified reasoning programs。下一 source 数据步骤必须是对正式六游戏做
fresh instrumented rollout，而不是让 Agent 给旧 409 transitions 补写不存在的 reasoning。

Fresh Gym-V rollout 还可能包含 `POLICY_POSTPROCESSOR/NO_VALID_AGENT_EXECUTION` fallback。
这些 transition 必须留在完整 reasoning event log 中，但不得进入 source reasoning program。
编译器只机械保留长度至少为 2 的 maximal contiguous Agent-origin spans；fallback 是固定切点，
不依据 reward、动作名称或 target 语义选边界。完整纯 Agent episode 保持原 artifact identity，
被切分的 span 记录 parent episode、原始 step 区间和被排除的 fallback 数量。很短或仅含菜单
动作的合法 span 仍可能没有迁移价值；Harness 必须让 target 证据暴露这一点，不能在 source
阶段以“看起来有用”为由换一个 span。

重新通过冻结的开发门槛后，再新增四个配对条件：

```text
base_source_multistep_harness
base_target_demo_harness
game_sft_source_multistep_harness
game_sft_target_demo_harness
```

核心 source 增益比较是前两项；后两项检验 Game SFT 与 Harness 的交互，因为 Game SFT
本身已含 source 信息。正式 2×4 前冻结 demo/proposal/artifact/model/adapter hash、task
list、seed、最大步数和 error policy。

## 6. Online target Harness：验证、失败恢复与安全回退

### 6.1 每一步都必须在线验证

source program 只提供一个待检验的 control proposal，不预先获得 target 语义。每个 target
step 执行以下闭环：

```text
observation + native admissible actions + active source candidates
  → Agent 提出 binding、下一 native action 和可检查的 expected evidence
  → Harness 做 closed-schema、identity、admissibility 和 receipt 检查
  → 只执行一个 native action
  → 记录 next observation、native delta、reward 和 official outcome
  → CONTINUE / REBIND / DEFER / DECOMPOSE / NEED_MORE_EVIDENCE /
    MARK_GAP / DISABLE / ABSTAIN / TARGET_ONLY
```

Harness 只能依据 target 环境返回的原生 action acceptance、observation/delta、reward、done
和 official evaluator 更新状态。Agent 可以提出自然语言解释或新的 binding，但解释本身
不是证据。系统不得内置 `COLLECT→TAKE`、`NAVIGATE→GOTO` 或任何相似名称映射。

### 6.2 “没有帮助”不能被压缩成一个 failure bit

一次失败至少要保留以下互斥诊断；无法机械区分时记录 `UNRESOLVED`，不能让 Agent 猜测
结果直接成为 verdict：

| 在线结果 | 含义 | 下一动作 |
|---|---|---|
| `FAILED_BINDING` | 当前实体、参数或 span 实例化被 transition 反驳 | 撤销当前 binding，允许 Agent `REBIND` |
| `NOT_APPLICABLE_NOW` | 当前 observation 不支持执行，但没有反驳 program | `DEFER`，等待新状态 |
| `GRANULARITY_MISMATCH` | 完整 program 不成立，但已执行边界中可能存在可复用子结构 | Agent 提出 `DECOMPOSE/RECOMPOSE`；Harness 重新验证 |
| `NEED_MORE_EVIDENCE` | 一次 adaptation demonstration 不能识别可用绑定 | 停止作强结论，在新的 adaptation example 上再检验 |
| `CAPABILITY_GAP` | 当前候选集不能提出可验证进展 | 写入结构化 gap，转 `TARGET_ONLY` |
| `NEGATIVE_TRANSFER` | source conditioning 持续增加无效动作、steps 或 cost | 本 episode `DISABLE` source 并安全回退 |
| `UNSUPPORTED` | 预注册证据预算内，候选被目标 transition 明确反驳 | 保留失败 receipts，不再执行该实例化 |

`DECOMPOSE/RECOMPOSE` 也不是手写 skill segmentation：Agent 只能引用 source program 中
已有 node、edge 和真实 transition 边界提出候选，Harness 检查引用、连续性和新的 target
receipt。失败的是 binding、适用状态、粒度还是整个 source program，必须分别记录；不能
因为一次错误对象选择就永久删除 source skill。

### 6.3 结构化 capability gap

当 source 没有帮助时，Harness 仍应产出可审计结果，而不是强迫迁移：

```json
{
  "target_context_receipt": "sha256:...",
  "attempted_source_program_ids": ["..."],
  "failed_binding_receipts": ["..."],
  "unresolved_agent_hypothesis": "...",
  "verdict": "CAPABILITY_GAP",
  "resolution": "TARGET_ONLY"
}
```

`unresolved_agent_hypothesis` 可以描述 Agent 认为缺少的能力，但不能自动升级成新的 predicate
或 skill。只有新的 adaptation transition、合法引用和 replay/official receipts 才能支持
新的候选程序。

### 6.4 one-shot、更多示例与 test 隔离

one-shot 的主要报告点是：单个 adaptation example 是否足以启动一个后来被在线证据支持
的实例化。若返回 `NEED_MORE_EVIDENCE`，应在预先划分的 adaptation set 上测量
`0/1/2/4-shot` curve，区分：

- `ONE_SHOT_SUFFICIENT`：一个 example 后即可获得在线支持；
- `FEW_SHOT_REQUIRED`：程序可能可迁移，但 one-shot 不足；
- `UNSUPPORTED_WITHIN_BUDGET`：增加 adaptation evidence 仍无帮助。

adaptation episode 可以更新非参数 receipt/status memory；进入 held-out test 前必须冻结
program library、adaptation receipts、模型、预算和 verifier。test episode 内仍可根据本
episode 的 live transition `verify/replan/abstain`，但 test outcome 不得写回供后续 test
episode 使用。

### 6.5 negative-transfer avoidance 是一等指标

source 无帮助时，合理目标不是伪造正迁移，而是尽快停止错误条件化并接近 target-only。
必须区分 runtime 可观察事实与实验级因果结论。单个 live transition 只能产生
`CONTRADICTED_BINDING`、`NO_OBSERVABLE_DELTA`、`BUDGET_EXHAUSTED` 或
`SOURCE_DISABLED` 等操作性状态；因为 runtime 看不到“不使用 source 时会怎样”，它不能
把这些状态直接标成 `NEGATIVE_TRANSFER`。后者只能由预注册、matched target-only 对照在
实验结束后计算：

```text
delta_source = performance(source condition) - performance(target-only)
```

最小实验矩阵为：

```text
B = target-only
N = naive source injection（无在线验证/回退）
H = verified source + online Harness
R = renamed/randomized source + 同一 online Harness
```

若 `N < B` 且 `H > N`，说明 Harness 缓解了 naive source 的负迁移；若 harmful strata 上
`H ≈ B` 且 useful strata 上 `H > B`，才支持 selective transfer。`R` 用于排除更多上下文、
token 或调用次数本身带来的效果。四组必须使用相同 model、episode、seed、action budget、
Agent call budget 和 token cap；超预算视为预注册 failure，不能事后补调用。

除了 success，还必须成对报告：

- 首个被反驳 receipt 前后的无效 action 数；
- 从 source-conditioned 切换到 target-only 的 steps、tokens 和 wall time；
- `REBIND/DEFER/NEED_MORE_EVIDENCE/MARK_GAP/DISABLE` 次数；
- source 引入的额外 cost 与最终 success delta；
- erroneous source 下相对 target-only 的 regret。

当指标方向统一为“越大越好”时，可以报告恢复比例：

```text
mitigation_rate = (performance(H) - performance(N))
                  / (performance(B) - performance(N))
```

仅在 `performance(B) > performance(N)` 时定义，并同时报告原始分子、分母和置信区间，
不得只报比例。若 Harness 对所有 source 一开始就 abstain，它只是退化为 target-only；因此
还必须报告 source adoption/coverage，以及 useful/harmful strata 上分别接受和禁用的行为。

必须保留 `target-only`、`verified source`、`renamed source` 和 `randomized source` 对照。
若 source 有用时被采用、无用时能在有界代价内回退，这可以支持 selective transfer 与
negative-transfer avoidance；若 source 始终没有增益，只能报告 transfer signal 未建立，
不能把成功回退本身写成 game skill transfer。

## 7. 何时可以声称 skill transfer

必须同时满足：source evidence 与 hypothesis status 表述准确；source/target-only 使用
matched Agent budget；admission/runtime 无 semantic heuristic；held-out 前冻结 artifact
且 evaluation 无写回；`source + same demo` 在预注册 held-out 上优于
`target-only + same demo`，并同时报告 success、steps、abstain、error 和 cost。

若只提升 Game-SFT 条件，结论是 source training 与 Harness 的交互；若 target-only
相同或更好，则不能声称 game reasoning transfer。若高 abstention 提高正确率但降低
coverage，也必须同时报告，不能只报 conditional success。

## 8. 2026-07-21 最小可证伪闭环实现

本轮实现修复了旧 runtime 中“任意 observation/action-set 变化即可继续”的漏洞。新的
Harness 规则如下：

1. 每个 source-guided action 在执行前必须为每个活动 candidate 分别冻结 closed-schema
   evidence contract。Agent 负责 binding hypothesis、native action selection 和必要时的在线
   rebind；contract 本身由 Harness 从已冻结的 one-shot transition receipt 确定性编译，Agent
   无权增加、删除或排序 evidence predicates。
2. `COMMAND_WAS_ADMISSIBLE` 只表示命令来自执行前的 native admissible list，不再表述为
   环境“接受”动作，也不单独构成 informative evidence。
3. contract 通过只能记为 `NOT_REFUTED_LOCALLY`，不能写成 skill verified、task progress 或
   semantic equivalence。
4. 每个 candidate 独立维护 `ACTIVE/DEFERRED/REFUTED/FINISHED`、cursor 和 refutation
   receipt。一次真实 target transition 可让满足自己 contract 的候选推进，同时让不满足的
   候选在 cursor 不变的情况下被反驳；不能再用集合级布尔值让所有候选一起通过或失败。
5. source 被禁用后，从当前 live observation 和 native actions 启动无 source-conditioning 的
   target-only Actor；不再要求实际轨迹与 one-shot demo prefix 相同。
6. evaluation 输出分开记录 `official_success`、`max_native_step_reward` 和
   `cumulative_native_reward`；最大单步 reward 不再冒充 `official_score`。

Source control 也已前移到 binding Agent 第一次看到 source 之前：

| treatment | binding 前输入 |
|---|---|
| `empty` | 不提供 source graph，作为 target-only reference |
| `correct` | 协议中的 designated source artifact；只表示预注册主 treatment，不声称语义正确 |
| `wrong` | 另一个预注册的跨游戏 source artifact；只表示 cross-game control，不声称已知语义错误 |
| `renamed` | 机械替换 hypothesis/node identity，保留 topology 和 evidence payload |

每个 transformation 都产生 content-addressed receipt；admission 将 treatment 和 receipt hash
写入冻结 artifact。正式 pilot 执行时应启用
`--require-binding-source-control`，拒绝没有上述 provenance 的旧 artifact。`rotate` runtime
control 仅保留为旧的局部 ablation，不再代替 pre-binding `wrong/renamed` control；只有一个
active context 时必须记录 `CONTROL_UNAVAILABLE_SINGLE_ACTIVE_CONTEXT`，不能静默 identity
rotation。

特别地，六个游戏相对 ALFWorld 都是匿名 source treatments。由于本方法禁止手工
`COLLECT→TAKE` 映射、embedding 相似度和 target reward 选源，当前没有证据可以事先把某个
游戏叫作真正的 “correct source”。代码中的 `correct/wrong` 是稳定的 protocol 枚举；论文和
结果表必须分别写成 **designated source** 与 **cross-game source control**。designated source
按预注册、与内容无关的游戏 ID 顺序选定，held-out 结果不得反向改变这个选择。

在下面记录的单对 smoke 之前，代码闭环只有单元测试。因此其启动门槛是：

- 不能用旧 B/N/H/R 数字验证本轮修改；
- `authorizes_large_scale_2x4` 仍为 `false`；
- 下一次只运行小型 E/S/W/R pilot，并分别运行 Harness off/on 所需 contrast；
- 必须报告 realized calls/tokens。预算不一致时只能报告 development diagnostic，不能作
  source-causal claim。

Harness-off source 条件必须使用 `--shadow-source-control`：它执行相同的 pre-action
receipt contract compilation 和机械 verification，但不让 contract verdict 改变 native
action、cursor 或 fallback。Harness-on 使用 `--online-source-control`。两者互斥。编译过程
不调用模型，因此不会把额外一次 35B 调用误认为 Harness 效果；由真实 rebind/fallback 引起
的 realized call 差异仍需单独报告，不能隐藏。

### 8.1 Fresh six-game evidence 与第一次 E/S/W/R smoke

2026-07-21 已用六个 best checkpoints 各跑 2 个 episode × 12 steps（seed base 242）：

- 共 12 episodes、1,332 reasoning events、166 个 fresh-env replay fork，0 protocol/replay
  mismatch；
- 144 个 execution decisions 中 137 个为 Agent-origin，7 个为
  `POLICY_POSTPROCESSOR/NO_VALID_AGENT_EXECUTION`；后者按上述固定切段规则排除；
- 每游戏按字典序固定取第一个可编译 program，18 次 35B role calls 得到 17 个 qualified
  hypotheses；1 个 Candy 输出被 1,400-token cap 截断并按格式失败拒绝，没有重试或择优；
- 六游戏 exact-union artifact 为
  `016a9c354b6cb40c74d5ac33cfd43ca6ec6039d09d06d61ecca0fa54096d425a`。

随后用同一条冻结 ALFWorld train demo、seed 47、1 episode、8-step cap 跑第一次机制
smoke。协议枚举 `correct` 机械指定字典序第一的 Candy artifact，论文标签为 designated；
`wrong` 使用 Columns，论文标签为 cross-game control。四种 binding 输入均得到 3/3
qualified candidates，仍不表示语义正确。

第一次运行还暴露出旧的 action-contract completion cap 为 256 tokens，无法容纳三个
candidate 的独立 contract JSON；所有输出被截断。该运行保留为 fail-closed bug receipt，
不能进入结果汇总。cap 提高到固定的 768 后重跑，closed schema 可完整解析；正式 smoke
汇总位于 `runs/alfworld_v3_eswr_smoke_20260721/summary_v2.json`：

- 7/7 条件 target instance identity 与注册上限一致，0 error；
- 所有条件 0/1 success，因此没有 positive-source ordering 或 Harness success gain；
- empty/designated-shadow/cross-game-shadow/renamed-shadow 均在 3 steps 后因 exact-action
  consensus 消失而 abstain；
- 三个 enforce 条件在在线 contract 反驳所有活动 source candidates 后 fail closed 到 live
  target-only，并执行到 8-step cap，仍未成功；
- realized calls/tokens 因真实 fallback 路径不同而不 matched，不能作 source-causal 解释；
- `authorizes_large_scale_2x4=false`。

本轮 35B 费用：fresh source induction 约 `$0.03051`，binding proposals 约 `$0.01516`，
有效 v2 evaluator 约 `$0.04138`。加上用于发现 256-token bug 的旧 evaluator runs，本轮实际
OpenRouter 总费用约 `$0.13114`。这只是开发诊断，不是 held-out transfer result。

该 smoke 还暴露出下一项必须先修的设计问题：35B 在 `look` 上为全部 candidate 提议了
`POSITIVE_NATIVE_REWARD`，真实 transition 没有正 reward，于是 enforce 安全地反驳了全部
candidate。fail-closed 行为本身正确，但一个未引用既有 receipt 的 Agent 预测不应拥有任意
制造 false rejection 的权力。下一版 contract 必须引用该 candidate/node 在 one-shot demo
中的具体 target transition receipt，并由 Harness 从该 receipt 机械导出允许检查的 delta
signature；没有 citation 或 live 状态超出 citation scope 时只能 `INCONCLUSIVE/NEED_MORE_EVIDENCE`，
不能 `REFUTED`。这仍不引入 `COLLECT→TAKE`、predicate ontology 或手工 source→target
语义，只限制 untrusted Agent 的否决权。在完成该修复并通过新的小型 paired smoke 前，不得
运行大规模 2×4。

### 8.2 Receipt-grounded contract 修复与第二次 smoke

上述问题已修复。admission artifact 现在冻结 demo 中每个 target transition 的
content-addressed receipt，并机械记录该 transition 实际支持的 observable signature：命令是否
在 native admissible set、observation/action-set 是否变化、动作是否消失、native reward 是否
为正以及 official success。每个 active candidate/node 必须精确引用其 target transition index、
receipt hash 和 signature；artifact loader 会重新计算并校验这些字段。

最初的 grounded smoke 仍让 35B 逐项复制 Harness 已知的 receipt/signature。模型内容复制正确，
但 2/3 次输出被 768-token cap 截断；designated enforce 因而在第 0 步把格式失败记为
`INCONCLUSIVE` 并回退。这个结果说明复制 call 没有提供新信息，只增加 truncation、幻觉、成本
和 latency。当前 runtime 已改为
`exact_one_shot_transition_receipt_signature_v1` 确定性 compiler：

- Agent 仍提出跨域 binding、选择 target-native action，并在无共同动作时提出在线 rebind；
- Harness 独占 evidence provenance 与 contract compilation；
- 缺少或篡改 receipt/signature 时为 `INCONCLUSIVE`，不会反驳 candidate；
- live transition 才能产生 candidate-level `NOT_REFUTED_LOCALLY/REFUTED` receipt；
- 每步 action-contract Agent calls 从 1 降为 0。

随后用相同 ALFWorld train demo、seed 47、1 episode、8-step cap 重跑完整七条件 matched
development smoke，汇总位于
`runs/alfworld_v3_eswr_smoke_20260721/summary_compiled.json`：

- 7/7 target identities 与 registered caps matched，0 errors；
- 三个 shadow 条件各确定性编译 3 个 contract，三个 enforce 条件各编译 4 个；合计 21 次
  compilation、0 次 action-contract Agent call；
- designated enforce 的前三个 source actions 均通过精确 receipt signature；无共同动作后，在线
  rebind 被 admission 接纳并真实执行 1 步，之后再无共同动作才进入 live target-only；
- 所有七条件仍为 0/1 success，positive-source ordering=false；enforce 路径因真实
  rebind/fallback 产生更多 action calls 和 tokens，realized budgets 不 matched；
- `authorizes_large_scale_2x4=false`，不得把本次机制 smoke 写成 positive transfer。

重要限制：one-shot receipt 只定义一次局部、可证伪的行为兼容性检查。某个 delta 在单次 demo
中出现，并不能证明它是该 skill 的语义必要条件；通过也不能证明 task progress 或 source skill
价值。未来若重复 target evidence 显示同一 binding 的 observable effect 不稳定，应保留多个可行
contract 或请求更多 adaptation examples，而不是手工挑选 predicate。该问题与多样本 evidence
aggregation/decompose-recompose 一起，仍是扩大实验前的研究项。

### 8.3 Native target Actor 校准与 matched negative-transfer diagnostic

旧 `empty-off` 不是充分的 target-only baseline：它仍沿 same-demo candidate consensus 执行，
3 步后没有共同动作便 abstain。当前 evaluator 对 `condition=target_only` 从 reset 起直接调用
native Actor，不读取 source conditioning，也不把 admissible actions 与 demo/source program
求交；source program 结束或被禁用后也从当前 live state 进入同一 native Actor，而不是提前结束。

为了避免引入反重复、搜索顺序或对象映射 heuristic，Actor 只增加两类环境/Agent 原生信息：

1. wrapper 机械提取 ALFWorld 明确给出的 `Your task is to:` 行，不解析对象语义；
2. 每步向 Agent 提供完整 target-native action/observation/reward/outcome history。Agent 返回严格
   `{state_summary,next_subgoal,action_number}`；前两项是不受信任的计划摘要，Harness 只接纳仍在
   当前 native list 内的精确 action number，随后由真实 environment transition 验证。

曾尝试使用 Qwen 35B 的 OpenRouter hidden reasoning。256、1024 和 1280 completion caps 均被
reasoning tokens 吃满而没有 final content，Harness 因格式缺失正确 abstain。当前默认使用
non-thinking closed-plan proposal（512-token cap）；这比继续扩大 hidden token budget 更可审计，
也保留 Agent 规划而非手写 policy。

同一 train seed 47 的 10-episode、30-step native calibration 位于
`runs/alfworld_native_calibration_20260721/target_only_seed47_ep10_s30_closed_plan.json`：

- official success `7/10`，0 errors，0 abstentions；
- 成功 episode 的平均步数 `10.43`，全部 episode 平均 `16.3`；
- 163 Actor calls，200,459 total tokens，OpenRouter reported cost `$0.04139`；
- 三个失败 episode 到达 30-step cap，其中仍可见 location/examine 循环。因此这里只证明
  target Actor viability，不是 held-out baseline result。

随后统一 empty、designated-shadow 和 designated-enforce 的 closed-plan schema、完整 history、
512-token cap，并对同一 episode 做单对 diagnostic：

| condition | official success | steps | reported cost | 观察 |
|---|---:|---:|---:|---|
| native target-only | 1/1 | 8 | `$0.00185` | 直接完成 butterknife→drawer |
| designated shadow | 1/1 | 19 | `$0.01213` | source 先引导操作 spatula，之后恢复 |
| designated enforce | 1/1 | 11 | `$0.00710` | 3 步后回退，减少但未消除伤害 |

这是可观测的单对负迁移：Harness-on 相比 shadow 减少 8 步，但仍比 target-only 多 3 步。
不能把三者都成功写成 positive transfer，也不能用这一对估计 effect size。下一 source gate 是
实现多 adaptation examples 的 receipt version space：保留所有仍与证据一致的 binding/effect
候选，只有全部候选都不兼容才 refute，否则继续或请求额外 example。在完成该机制和小型
E/S/W/R paired pilot 前，`authorizes_large_scale_2x4=false`。

### 8.4 两样本 receipt version space：真实负结果与设计边界

已实现 content-addressed 多样本 version space。它不做文本 clustering、embedding similarity、
投票或 reward ranking；每个 version 是 Agent 提出的精确
`source hypothesis/node → target operator + argument types` schema，Harness 只对冻结 admission
artifact 做集合运算。adaptation set 未收齐时，非空交集为 `PROVISIONAL`，空交集为
`NEED_MORE_EVIDENCE`；预注册样本收齐后，非空交集为 `READY`，空交集只声明
`NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET`，不外推成 source skill 全局无效。运行时遇到
未在 adaptation receipts 出现过的 effect signature 时返回 `NEED_MORE_EVIDENCE`，不会把新证据
自动当作反证。

真实检查使用两个独立 ALFWorld train success demonstrations：seed42 的 admission artifact
保留 2 个精确 schema，seed48 由同一 frozen 六游戏 source artifact、相同 35B 模型和 3 个固定
Agent roles 新生成 proposal，3/3 可解析且 3/3 admission 通过，去重后保留 2 个 schema。冻结的
两样本 version space 位于
`runs/alfworld_native_calibration_20260721/designated_version_space_2of2.json`：

- expected/observed examples 为 `2/2`；
- 总计 4 个不同完整 schema；
- 精确共同 schema 为 0；
- verdict 为 `NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET`。

这不是 API 或 admission failure。两个成功实例需要不同长度/位置的 navigation scaffolding；当前
candidate identity 又要求完整目标 action partition 与 source node 分配完全相同，因此没有足够
证据证明一个固定的 episode-level binding 可以复用。Harness 按 fail-closed 规则在 episode 开始
前禁用 source，而不是任选一个候选。seed47 的真实 fallback smoke 重复两次：第一次 native
target Actor 在 8 步完成；加入逐 episode gate 审计字段后重跑，Actor 在 30-step cap 内搜索多个
cabinet/drawer 仍未完成。因此当前观测为 `1/2` success，而不是稳定性能结论。两次均为 0 rebind、
0 contract call、0 version verification，且 source 从 reset 前即被 gate off；这里只验证安全回退
路径确实被执行，也再次说明 35B Actor 有明显的 run-to-run 不稳定性，不是 positive transfer。
最终可审计重跑位于
`runs/alfworld_native_calibration_20260721/designated_version_space_2of2_enforce_ep1.json`，其中
`fallback_mode=NATIVE_TARGET_ONLY_FROM_RESET_VERSION_SPACE_GATE`。

该结果还限定了下一版设计：不能简单删除多余 `GOTO`、忽略 action 顺序或人工把它标成
“contextual navigation”，因为这些都会暗中加入 ontology/heuristic。若要从完整程序推进到
decompose-recompose，Agent 必须显式提出带条件的 node-local target policy（包括何时开始、何时
结束、允许的局部 action set 与 source-edge continuation），每项条件都引用真实 target receipts；
Harness 再机械检查完整 transition coverage、node 间顺序/边约束和 live action-set 一致性。
在这种 evidence-carrying conditional program 尚未实现前，现有数据只能支持上述 bounded
`NOT_APPLICABLE` 结论，不能启动大规模 2×4 实验。

### 8.5 Conditional node-local program：实现、Agent repair 与负 admission

已实现第一版 evidence-carrying conditional node program，而没有人工把 seed42 的额外 `GOTO`
移入 gap。新协议由 untrusted Agent 对多个冻结 adaptation examples 联合提出 ordered
segmentation；每个 segment 只能是：

- `SOURCE_NODE`：精确引用一个已注册 source node；
- `TARGET_NATIVE_GAP`：显式保留当前无法由 source node 共同解释的 target transitions。

Harness admission 机械检查：所有 target transitions 按原顺序恰好覆盖一次；每个 source node 按
source graph 顺序恰好出现一次且非空；同一 source node 在所有 examples 中必须产生完全相同的
target operator/argument-type 序列；连续 source nodes 必须存在已注册 edge，且 edge 携带合法
intervention receipts。每个 admitted node 的 entry/step/exit evidence 均引用真实 target transition
receipt。Gap 可以跨实例变化，但不能删除、重排或伪装 transition。实现位于
`harness/conditional_node_program.py`。

runtime 只在所有 retained candidates 对当前 node step 给出相同 native signature，且该 signature
在 live admissible actions 中出现时开放 source-conditioned action；node 尚未开始且 entry signature
不可用时返回 `TARGET_NATIVE_GAP_REQUIRED`，由 native target Actor 继续交互并随后重检；node 已
开始后 signature 消失或 live effect 未出现在任何冻结 witness 中时返回
`NEED_MORE_EVIDENCE`，不会猜测 continuation。source node 执行会生成 content-addressed transition
receipt。当前真实 admission 没有 READY artifact，因此没有把这一 source 分支接入大规模 evaluator；
synthetic vertical-slice tests 已覆盖 source→gap→source 恢复路径和未覆盖 transition 的拒绝。

真实两样本 35B development 流程还加入两个非语义修复：proposal 前只展示所有连续 edge 均有
intervention receipts 的 graphs；rejected proposal 可以在 source identity 固定的条件下接收
Harness failure receipt 做有界 repair。Failure receipt 机械包含 expected/observed node order、
transition coverage 和逐 demo observed node schemas，不告诉 Agent 应把哪一步移到 gap。

在 seed42/seed48 上，21 个可审计 35B calls 共使用 434,199 tokens，OpenRouter reported cost
约 `$0.07518`；另有一次在写 artifact 前暴露 dataclass serialization bug 的调用，不计入上述可审计
统计。开发过程中 Agent 曾提出显式 gap，并选中 `eb689…` source graph，但反复保留不一致的 N0
schema，或遗漏 source node/生成空 segment。最终冻结 artifact 位于
`runs/alfworld_native_calibration_20260721/conditional_nodes_two_example_admission_final.json`：

- `n_proposed=3`，`n_admitted=0`，`n_rejected=3`；
- status 为 `NEED_MORE_AGENT_PROPOSALS`；
- artifact hash 为 `6c1f4ee6ddfeda5438fde993976c3f313eea440f8787862ce815ada47049622e`。

因此当前正确行为是不开启 source execution，也不运行 2×4。这个 verdict 表示“当前固定 35B
proposal/repair budget 未恢复出可接纳程序”，不是 source skill 全局无效。下一项实验应预注册更强
proposal Agent 或 source-graph enumeration budget，并与同预算 target-only proposal control 匹配；
不得由人手写那个看似明显的 gap 后宣称 transfer 成功。
