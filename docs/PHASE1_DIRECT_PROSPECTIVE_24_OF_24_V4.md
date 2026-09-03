# Phase 1：六游戏到四领域的 direct neural-symbolic transfer

## 结论

截至 2026-08-15，新的 fail-closed composite audit 为：

- **24/24 direct prospective mechanism cells validated**；
- WebShop、ALFWorld、DiscoveryWorld、TIRBench 各 **6/6**；
- 精确覆盖 `6 source games × 4 target domains`，没有缺 cell；
- 24 个 execution fingerprint 全部唯一；
- DiscoveryWorld V1/V2/V3 的 consumed outcomes **0 个计入**；
- historical target outcomes **0 个计入**。

这已经把此前的“24/24 program-equivalent mechanism cells，0/24 new joint executions”升级为：

> 六个独立合格的 game lineages，分别在四个 target domain 的 frozen reserve 上完成了 24 次新的 source-lineage × target joint mechanism execution。

机器可复核结果见 [`results/phase1_direct_prospective_24_of_24_v4.json`](results/phase1_direct_prospective_24_of_24_v4.json)，状态页见 [`PHASE1_DIRECT_PROSPECTIVE_24_OF_24_V4.html`](PHASE1_DIRECT_PROSPECTIVE_24_OF_24_V4.html)。

## Claim boundary

这里成立的是 **operational mechanism transfer**，不是下面两个更强命题：

1. 不是 24 个有统计功效的 success-rate improvement estimates；
2. 不是“每个 source game 都提高每个 target task 的最终 reward”。

每个 cell 必须在 frozen target execution 中实际产生 source-conditioned online route，并通过 identity、hash、freshness、target-native grounding、matched-state、runtime 与 nontrivial-route gates。因此它强于静态 artifact compatibility 或历史 outcome relineage；但一个 target task / cell 仍不足以估计平均提升和置信区间。

## 24 个 direct cells

| Target domain | Direct cells | Authentic source decisions | Frozen evidence | Protocol |
|---|---:|---:|---|---|
| WebShop | 6/6 | 66 | V1 untouched reserve | CORE_V1 |
| ALFWorld | 6/6 | 324 | V1 untouched reserve | CORE_V1 |
| DiscoveryWorld | 6/6 | 31 | V4 seeds 39–44, one target process/task | DISCOVERYWORLD_V4 |
| TIRBench | 6/6 | 3,000 | V1 untouched reserve | CORE_V1 |
| **Total** | **24/24** | **3,421** | no historical target outcomes counted | composite V4 audit |

六个 source lineages 是 Tetris、Candy Crush、Columns、Streets of Rage 2、Thunder Force III、Strider。每一行 target 都包含这六个 source，且每个 `source artifact × target task × authentic route receipt` 指纹只出现一次。

## 为什么这是 neural-symbolic transfer

跨域迁移的不是 source-native action sequence，也不是高层 skill label，而是 intervention-grounded search automaton：

```text
NO_ACTIVE_CANDIDATE_AND_UNTRIED_REMAIN -> EXPLORE_UNTRIED
ACTIVE_CANDIDATE_REFUTED               -> BACKTRACK_REPLAN
ACTIVE_CANDIDATE_VERIFIED              -> COMMIT_VERIFY
```

执行链为：

```text
independent game intervention evidence
  -> common typed event / attempt ledger
  -> source symbolic route
  -> target-native neural candidate grounding
  -> symbolic applicability and evidence guard
  -> target-native executor
  -> observed effect updates the ledger
```

神经部分负责在当前 target state 里提出 object、candidate 和 native action；symbolic 部分只传递 domain-invariant 的 explore/backtrack/commit structure，并对 applicability、effect witness 和状态更新做 fail-closed 检查。source-native token、candidate identity、candidate order 与 raw trajectory 均不进入 target artifact。

六个 direct source artifact 的 hash 各不相同；每个只绑定一个独立 game formal lineage，但共享相同的 canonical policy，并可追溯到同一个六源 consensus parent：

```text
parent_consensus_artifact_sha256 = 36bec02285d4b2aca98f0a938447085545b24aff8edd23e2bc7485260da27f58
canonical_policy_sha256          = 02d3adc83616688ae0b51b152bae3ce8ab468f0fc3a99ac0f27b7d399e696fdf
direct source artifact hashes    = 585e5b29…, 6f80cf4d…, 39c51940…,
                                   0811312f…, 78ccd564…, 9993c7b0…
status                           = SOURCE_SEARCH_AUTOMATON_FROZEN
target_authorized                = true
```

## Prospective isolation

核心三域在任何 target execution 前冻结于 V1 manifest：

```text
configs/phase1_direct_prospective_v1/manifest.json
manifest_sha256 = 7a0d600bc3c9c3ec5ff4c7bb2384e5740cbbb5b6aba8918962faa57b5468f5a1
```

DiscoveryWorld 的最终六条新 reserve 在任何 seed 39–44 reset、provider call 或 outcome 前冻结于 V4 manifest：

```text
configs/phase1_direct_prospective_v4/discoveryworld_manifest.json
manifest_sha256 = a37256458e4ac07da5b2ae761395c70ac0841436998165745ca918bc5241b2f1
```

V4 preparation receipt 验证：

- 六个 task 各自 `target_process_count = 1`；
- 六个 task 都在预声明的首次 `PUT`/`DROP` 前形成 eligible fork；
- eligibility 没有读取 outcome fields；
- target summary、fork receipt 与每个 episode 都由 hash 绑定；
- preparation receipt SHA-256 为 `e7804ab99e807f81a317df417b0ab64ee0a6eef26ffeb0e02e2795f469d74c47`。

## DiscoveryWorld 的失败与修复

最终结果没有删除或重写此前失败。V1–V3 全部作为 consumed development 保留，但不进入 24/24：

1. **V1：transport/schema 不稳定。** Qwen matched grounder 出现 malformed/truncated JSON；并行脚本还暴露过同一路径写入竞争。
2. **V2：错误地合并 neural roles。** GPT grounder 本身稳定，但让同一模型承担长程 target acquisition 后重复生成 invalid `PICKUP`，fresh target episode 未完成。
3. **V3：syntactic validity 不等于 applicability。** Qwen acquisition + GPT grounding 得到 4/6；两个失败 cell 的 neural candidate set 没有 admissible `POSITION`，旧 selector fallback 到没有 effect witness 的 `COMMIT`，harness 正确 fail closed。
4. **V4：补齐 neural candidate-set applicability contract。** `COMMIT` 必须有精确 positive-effect witness；neural grounder 必须按冻结 schema 把 `POSITION` 标注为可逆操作，其 native action 同时通过 target-native API parser/precondition。若旧 selector 将选择 unwitnessed `COMMIT`，只能在同一个 neural candidate set 中改选最高 information-gain 的 parser-valid `POSITION`；不能创建 target action。若两类都不存在，则在冻结的三次 attempt budget 内拒绝并重采样。

V4 修复先在已 consumed 的 V3 seed 35 精确 failure fork 上诊断通过，再冻结全新的 seeds 39–44；没有在读取 V4 outcome 后修改 runtime、降低 gate 或更换 task。

## 每个 cell 的硬 gates

所有 24 个正式 receipt 均满足：

- runtime complete；
- 一个 fresh target 只归属一个 cell；
- historical outcome 未复用；
- matched conditions 完整且 initial target state 一致；
- 使用 target-native neural grounding；
- source route 在线发生并被 admission rule 接受；
- 每个 routed decision 都绑定 source artifact hash；
- 至少触发一个 nontrivial symbolic action；
- report 与 cell receipt 的 self-hash 有效；
- target/source identity 与 frozen manifest 完全一致。

最终 audit：

```text
status       = DIRECT_PROSPECTIVE_24_OF_24_VALIDATED
audit_sha256 = 9df87079672378370ae2db3a2acdcd817a220cb07d0465fe0a9c68bfb518bba7
```

## 仍然缺什么

下一阶段不应再把“多跑几个 source labels”当作首要任务，而应测 **transfer utility**：

- 每个 source-target pair 扩展到多个 untouched target tasks/seeds；
- 预冻结 primary endpoint，例如 success、steps、token/API cost；
- 与 target-native-only、symbolic-control、wrong-source/control policy 做 matched comparison；
- 报告平均 effect、bootstrap confidence interval、negative-transfer rate 与 abstention calibration；
- 检查 source identity 是否带来可预测的差异，而不是六个 lineage 只共享同一个 controller hash。

因此当前可以准确写：**24/24 direct prospective neural-symbolic mechanism transfer validated**。只有完成上述多任务对照，才可以写“transfer reliably improves target success rate”。

## 复核命令

```bash
export PYTHONPATH=src:.

pytest -q \
  tests/test_direct_prospective_matrix_v1.py \
  tests/test_discoveryworld_applicability_grounder_v4.py

python scripts/audit_phase1_direct_prospective_v4.py
```
