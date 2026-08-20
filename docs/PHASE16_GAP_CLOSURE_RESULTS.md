# Phase 16：Phase 14–15 acquisition gaps closure

## 结论

Phase 14–15 明确列出的四个实验缺口均已执行：

1. 完整 source candidate-fork primitive cost 已由冻结 source plan 精确重建；
2. target-only zero-trajectory LLM synthesis/retrieval baseline 已冻结并运行；
3. 第三个 algebraic program family 已从 fresh official-Tetris intervention tuples 自动归纳；
4. 独立 source acquisition reserve 在 qualification 通过后才打开，并通过。

综合机器报告状态为：

```text
DECLARED_PHASE14_15_GAPS_CLOSED_WITH_BOUNDARIES
```

最重要的科学结论没有变，但现在更精确：

> 执行收益来自正确的 symbolic program content，而不是 source provenance。Source interventions
> 是取得完整、fail-closed program 的一种经验证 acquisition route；它不是取得该内容的唯一逻辑方式。

## 1. 完整 source fork cost

V25 只报告了 ALFWorld source primary K=4 的成功路径，共 27 primitive transitions。冻结 source plan
和下游 compact receipts 都绑定到同一个 pre-compression primitive dataset SHA：

```text
bd18a1a84b3a0faf8c37d5e986d7e01e224ae98eb98515fd7da976cd293e3265
```

因此可以调用原 source simulator generator，精确重建被 compact receipt 丢弃的失败 forks，而不是估算：

| Cost item | Count |
|---|---:|
| source snapshot episodes | 4 |
| candidate fork resets | 16 |
| successful forks | 4 |
| failed forks | 12 |
| all executed primitive transitions | **108** |
| successful-path transitions | 27 |
| failed-fork transitions | 81 |

Target K=1 的单条完整 ALFWorld 成功轨迹为 15–39 transitions，median 25。现在两边至少都能用
“executed primitive environment transition”这个句法单位报告；但 source simulator 与 ALFWorld 的难度、
reset 成本和经济成本仍不等价。因此不能声称 source 更 sample-efficient。相反，这次完整核算显示主 source
acquisition 的实际执行量是 108 transitions，而不是旧摘要里的 27。

## 2. Zero-trajectory target LLM baseline

V29 在调用前冻结：

- model：`openai/gpt-4.1-mini` via OpenRouter；
- targets：ALFWorld、DiscoveryWorld、TIR rotation；
- 每个 target 4 calls，共 12 calls；
- 输入只包含 target interface description 和共享 IR grammar；
- 完整 target trajectories、target outcomes、source receipts/programs、gold answer 全部为 0；
- strict evaluator 没有在看结果后放宽。

运行成本：4,474 prompt tokens、1,167 completion tokens、总计 5,641 tokens，provider reported cost
为 **$0.0036568**。

| Target | Exact safe program | Correct family | Required constraints | Correct terminal | Fail-closed contract |
|---|---:|---:|---:|---:|---:|
| ALFWorld | 0/4 | 4/4 | 4/4 | 4/4 | 0/4 |
| DiscoveryWorld | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| TIR rotation | 0/4 | 4/4 | 4/4 | 0/4 | 0/4 |

这不是简单的“LLM 全错”。它凭 target domain prior 找到了 ALFWorld 与 TIR 的 family 和核心 constraint，
但加入了 unsupported operators、遗漏 abstention，并在 DiscoveryWorld 上选成错误的 recurrent family。
因此：

- target prior 可以取得 partial structural retrieval；
- 在冻结预算下没有取得 exact safe executable program；
- 该负结果只适用于此 model/prompt/budget，不能排除其他 LLM 或人类。

代码中已有 source-independent target-written oracle ceiling：ALFWorld 的 action/state/effect traces 与
source-induced 完全一致，TIR target-written 与 source semantics 都是 19/21。它证明 source provenance
不是执行必要条件，但不是被试人类 acquisition-time 实验；仓库没有伪造 human timing 数据。

## 3. 第三个 program family：cyclic identity recovery

旧 Tetris compiler 直接写出 `r^-1`，不满足 source-only rule induction。V28 新 learner 不接收 action name
或 inverse formula；它只读取 official Tetris 环境执行得到的匿名 source forks：

```text
(state element, probe effect, recovery effect, next element, returned-to-identity)
```

固定 hypothesis class 包含：

- compose probe/recovery to identity；
- copy probe effect；
- recovery is identity；
- fixed generator；
- fixed predecessor。

只有当一个关系精确分离全部 successful/failed forks 时才输出 program；否则 abstain。学到的 program 是：

```text
COMPOSE(PROBE_EFFECT, RECOVERY_EFFECT) == IDENTITY
```

它没有直接输出 `-r mod n`；target-native grounder/executor 必须寻找满足 identity constraint 的 native action，
零个或多个 binding 都 abstain。

## 4. Fresh source qualification 与 reserve

96 个新 seeds 在采集前被冻结为 48/24/24：development、qualification、reserve。只保留 official dynamics
中 observed order=4 的 episodes。

| Split | Retained episodes | Candidate forks | Primitive transitions | Result |
|---|---:|---:|---:|---|
| development | 39 | 156 | 567 | K=3 首次唯一归纳 |
| qualification | 23 | 92 | 315 | 92/92 classified，0 false positive |
| reserve | 18 | 72 | 282 | 72/72 classified，0 false positive |

Reserve 在 qualification 的 11/11 gates 通过后才打开。Terminal-label permutation 与 recovery-effect
binding permutation 都 fail closed。全部 source artifacts 均为：

- `raw_source_action_tokens_exported=false`；
- `target_data_read=false`。

新 induced relation 与旧 TIR executor 使用的 `compose inverse element` 在 execution semantics 上相同。
既有 TIR fresh utility 为 source semantics 19/21、raw 4/21、15W/0L，三种 destructive controls 均 0；
target-written isomorphic 同为 19/21。

这里必须保留时间边界：TIR target reserve 早于 V28 inducer，并使用 extensionally identical 旧 artifact。
所以这是 fresh source acquisition + semantic bridge + 既有 prospective target utility context，不是为 V28
重新打开的新 target prospective reserve。

## 最终因果解释

当前证据支持：

```text
source interventions
  → exact typed/fail-closed program acquisition
  → target-native neural grounding
  → program-content execution gain
```

同时也支持：

```text
target-written isomorphic content
  → identical execution
```

因此不能再写成 “source provenance causes success”。准确表述为：

> Receipt-grounded source interventions can acquire exact transferable program content without a complete
> target trajectory. The content—not its provenance—causes the execution benefit. Target priors can recover
> partial structure, while exact fail-closed synthesis was not achieved by the frozen zero-trajectory LLM baseline.

## 仍然存在的 claim boundaries

以下是外推边界，不是本轮未执行的 protocol gap：

- 没有招募人类做 timing/sample-efficiency study；target-written code 只是 oracle ceiling；
- source/target primitive counts 可同名计数，但环境语义与经济成本不可直接等价；
- V28 没有新的 untouched TIR target population；已有 75 个 rotation items 已被旧 V1/V2 splits 分配；
- 三个 program families 不代表 arbitrary-domain universality。

## 复现

```bash
python scripts/analyze_tetris_cyclic_source_induction_v28.py
python scripts/analyze_phase16_gap_closure_v30.py

python -m pytest -q \
  tests/test_cyclic_identity_induction.py \
  tests/test_source_fork_cost.py \
  tests/test_target_schema_synthesis.py \
  tests/test_analyze_phase16_gap_closure_v30.py
```

机器报告：

- [`results/tetris_cyclic_source_induction_v28.json`](results/tetris_cyclic_source_induction_v28.json)；
- [`results/phase16_gap_closure_v30.json`](results/phase16_gap_closure_v30.json)；
- `runs/target_schema_synthesis_v29/report.json`。
