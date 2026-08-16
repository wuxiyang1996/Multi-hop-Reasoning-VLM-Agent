# Phase 2：DiscoveryWorld selective neural-symbolic utility V2

## 结论

在固定的 36 个 Proteomics Easy seeds45–80 上，选择性 Phase-2 结果为：

> **`PHASE2_DISCOVERYWORLD_SELECTIVE_CAUSAL_UTILITY_VALIDATED`**

| Condition | Success | Rate |
|---|---:|---:|
| target-native neural myopic | 12/36 | 33.33% |
| authentic source effect guard + target grounding | **27/36** | **75.00%** |
| commit-availability / ledger-blind control | 6/36 | 16.67% |
| inverted-effect / wrong-controller control | 3/36 | 8.33% |
| always-position control | 3/36 | 8.33% |

Primary paired comparison 为 **17W / 2L / 17T**，exact two-sided sign-test
`p = 0.000728607177734375`，提升 **15/36 = 41.67 percentage points**。discordant
negative-transfer rate 为 `2/19 = 10.53%`，低于 frozen 25% 上限。

V2 使用六个独立 game lineages，各绑定 6 个 target tasks，共产生 114 个 admitted online source
routes：86 `EXPLORE_UNTRIED`、28 `COMMIT_VERIFY`。source action 不是 DiscoveryWorld action；
object UUID、空间关系、DROP/PUT、movement 与 effect witness 全部来自 target-native grounding 和
官方环境。

独立 raw audit 重算 36 cell hashes、35 matched results、source routes、success counts 与 paired
statistics，**15/15 gates passed**：

```text
V2 manifest sha256 = 36e9fbb35836bd1e4a0d51fa3a305b598534c617b6d53a74671f8bbb2c779afa
report sha256      = 9ff0c6cb0eb9e31c11c793695f12e46457df244980f41f96c6362fa560cc9a4b
audit sha256       = bd365fc127fb3e01015742edba14baa5702f08ad97f6526e768f9f74e752cd65
compact sha256     = 93d68605417be1b2e7366d430b68592c1d8aaa2db08fb3a54e35163467714e70
```

## V1 为什么失败，以及 V2 为什么仍然合法

V1 在任何 matched arm 前完成 36 个 frozen target acquisitions，然后 outcome-blind fork selector
只找到 35 个 first-DROP/PUT forks。`proteomics.easy.seed70` 在 96 步内没有任何预声明 commit
action：

```text
V1 status       = PHASE2_DISCOVERYWORLD_V1_COVERAGE_FAILED_NO_MATCHED_ARMS_RUN
eligible        = 35/36
ineligible      = proteomics.easy.seed70
reason          = NO_PREDECLARED_COMMIT_ACTION
matched arms run= 0
```

V1 没有被改成成功。V2 在任何 matched outcome 出现前单独 commit/push，固定：

- 35 个 outcome-blind eligible forks 全部执行五个 matched arms；
- 唯一 inapplicable task 必须 fail closed；
- abstention 对所有 arms 精确继承同一个 recorded target-only outcome，不允许创建 fork/action；
- primary endpoint、paired sign-test、negative-transfer bound 与 three-control superiority 不变。

因此 V2 的准确结论是 **conditional-on-applicability causal utility + calibrated abstention**，不是
36/36 fork coverage。inapplicable seed70 作为 tie 留在 36-task denominator 中，没有删除或更换。

## 与 recorded target-only continuation 的 secondary comparison

预注册 primary 是 authentic vs matched `target_native_myopic`。独立 audit 另外计算了未作为
primary gate 的 secondary comparator：原 target-only trajectory 从同一 fork 继续执行。

| Comparison | Authentic | Recorded target-only | Paired |
|---|---:|---:|---:|
| secondary, not preregistered primary | 27/36 | 18/36 | 11W / 2L / 23T, `p=.0224609375` |

它表明增益不只来自一个特别弱的 myopic selector；但因该 comparison 不是 V2 primary gate，必须
标为 supporting evidence，而不能替换主结论。

## Neural-symbolic boundary

```text
source-qualified game lineage
  -> intervention-grounded EXPLORE / COMMIT routing
target Qwen acquisition
  -> scientific state and first native commit proposal
target GPT binder / candidate grounder
  -> hypothesis object, target UUID, relation, native candidates
symbolic applicability + exact target predicates
  -> positive-effect witness or reversible POSITION
target-native spatial realizer + official environment
  -> native action and observed outcome
```

Matched conditions共享同一个 target fork、binder、candidate bundle schema、native realizer、8-step
horizon 与 evaluator。差异只在 controller：authentic 要求 positive-effect witness；ledger-blind 只看
commit availability；wrong control 反转 effect binding；position control 拒绝 commit。

## Source-lineage breakdown（描述性，不做 per-source claim）

| Source lineage | Authentic | Raw | W/L |
|---|---:|---:|---:|
| Tetris | 6/6 | 3/6 | 3/0 |
| Candy Crush | 3/6 | 2/6 | 1/0 |
| Columns | 6/6 | 2/6 | 4/0 |
| Streets of Rage 2 | 4/6 | 1/6 | 4/1 |
| Thunder Force III | 3/6 | 2/6 | 2/1 |
| Strider | 5/6 | 2/6 | 3/0 |

每个 lineage 只有 6 tasks，而且六 artifact 共享同一个 canonical search policy；该表只验证 identity
coverage 与没有单一 source 独占总体效果，不是六个独立 powered effects。

## Bitter lessons 与限制

1. **Coverage 不能假定。** 历史 Proteomics forks 近乎全覆盖，但 fresh seed70 仍没有 native commit。
   正确处理是 abstain，不是延长 horizon 或换 seed。
2. **Target acquisition 仍是瓶颈。** 36 个 target-only episodes 共 2,761 步，其中 1,009 步
   （36.5%）使用 schema-failure feed fallback。native action invalid count 为 0，但 neural policy 的
   long-horizon reliability 很弱。
3. **成立范围很窄。** 结果只覆盖 Proteomics Easy first-commit recovery，不覆盖 Space Sick、Normal
   difficulty 或整个 DiscoveryWorld。
4. **存在真实 negative transfer。** seed48 与 seed73 是两次 strict losses；选择器只把总体
   discordant loss rate 控制在 10.5%，不是宣称 zero negative transfer。
5. **不是 source semantics 搬运。** source-native objects/actions/coordinates 都未进入 target；迁移的是
   intervention-grounded controller，target grounding 仍不可替代。

## Operational disclosure

最初 6-worker 和 12-worker target-acquisition schedulers 因 wall-clock 优化被停止，随后在相同
manifest、model、prompt、seed、horizon、deterministic environment 与 per-task memoized cache 下以
24 workers 重放前缀。没有换 task 或读取 matched outcome；但因此不主张“每个 target 只 reset
一次”。最终 36 episodes 均完整、所有 V2 matched children exit 0。

## Evidence 与复核

- V1 coverage failure：`docs/results/phase2_discoveryworld_utility_v1_coverage_failure.json`
- V2 compact：`docs/results/phase2_discoveryworld_utility_v2_compact.json`
- V2 independent audit：`docs/results/phase2_discoveryworld_utility_v2_audit.json`
- V2 report：`runs/phase2_discoveryworld_utility_v2/report.json`

```bash
export PYTHONPATH=src:.
python scripts/verify_phase2_discoveryworld_utility_v2.py
python scripts/audit_phase2_discoveryworld_utility_v2.py
```
