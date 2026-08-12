# Controlled neural-symbolic cross-domain transfer V3

## Decision

在预冻结的 claim boundary 内，**跨域 neural-symbolic transfer 已验证**。正式状态为：

```text
CONTROLLED_NEURAL_SYMBOLIC_CROSS_DOMAIN_TRANSFER_VALIDATED
```

这个结论严格限定为：synthetic intervention-rich hidden-rule games → semantic-disjoint
synthetic diagnosis domains。它不等于真实 arcade → ALFWorld/WebShop transfer 已成立。

## Transfer object 与 neural grounding

迁移对象不是 skill 文本，也不是 source action token，而是 source matched interventions 导出的
逐状态 symbolic control/value structure：

```text
belief / uncertainty / budget state
    -> compare TEST(i) against COMMIT(h)
    -> SELECT or ABSTAIN
```

source 使用 48 个 hidden-rule game domains、24 states/domain，共 9,268 个 matched
state-action-value examples。target 的 raw action tokens 为 `order_assay_* / diagnose_syndrome_*`，
与 source `pull_rune_* / unlock_vault_*` 交集为空。

每个 target domain 独立训练一个 32-unit tanh MLP grounder。输入只有 target-local anonymous
hypothesis/test one-hots；label 只有每个 cell 的 48 个 binary calibration outcomes。隐藏环境
likelihood 不进入 policy 或训练 label，只在运行结束后用于 grounder MSE audit。target task
success 不用于训练 source program 或 grounder。

## Frozen protocol

开发只使用 284xx/285xx target domains。开发复现通过后，正式 config 才绑定其 report SHA，
并冻结此前未运行的：

- qualification：29100–29115；
- held-out：29200–29215；
- 每个 domain 128 个 matched episode seeds；
- 四个 conditions：target-only、authentic、within-state shuffled、source-marginal。

正式运行每个 split/condition 含 2,048 episodes。所有 conditions 共 16,384 episode receipts。
gate 要求在两个 splits 上，authentic 的 paired `net_return` 和 `success` 都分别超过三类 controls，
mean delta > 0.005 且 paired-bootstrap 95% CI 下界 > 0；共 12 个预声明比较。另有 5 个
representation/grounding invariants。

## Formal results

### Authentic versus target-only

| split | target success | authentic success | paired success delta [95% CI] | target net | authentic net | paired net delta [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| qualification | 70.02% | 72.02% | **+2.00pp [0.15, 3.81]** | 0.6153 | 0.6541 | **+0.0388 [0.0201, 0.0563]** |
| held-out | 72.12% | 74.07% | **+1.95pp [0.05, 3.96]** | 0.6354 | 0.6743 | **+0.0388 [0.0191, 0.0584]** |

Authentic 也降低平均 test 数：qualification `3.3945→2.6440`，held-out
`3.4297→2.6572`。success gate 单独通过，所以净收益不是只靠少做 tests 得到。

### Non-trivial controls

| split | control | success delta [95% CI] | net-return delta [95% CI] |
|---|---|---:|---:|
| qualification | within-state shuffled | +26.12pp [23.53, 28.61] | +0.2951 [0.2699, 0.3207] |
| qualification | source marginal | +5.57pp [3.12, 7.96] | +0.0896 [0.0655, 0.1139] |
| held-out | within-state shuffled | +27.54pp [24.85, 30.13] | +0.3090 [0.2816, 0.3356] |
| held-out | source marginal | +3.81pp [1.61, 6.15] | +0.0717 [0.0490, 0.0946] |

这排除了“任意 source data 都有用”和“只迁移 TEST/COMMIT 边际频率”两种解释。authentic
source value MSE 为 `0.00376`，优于 marginal `0.02438` 与 shuffled `0.06443`。

target neural grounder 的 post-hoc MSE 为 qualification `0.00285`、held-out `0.00339`，均通过
预设 `≤0.01` gate。零 raw-token overlap、MLP grounder kind、至少 9,000 source examples、
authentic source MSE 优于 controls 等 5 个 invariants 全部通过。

## What is now established

可支持：

1. intervention-grounded、state-dependent `TEST/COMMIT` control/value structure 可以跨语义表面迁移；
2. target-native neural grounding 可以把该 frozen source structure 落到全新的 target actions；
3. 增益同时体现在 success rate 与 cost-sensitive net return；
4. 增益依赖正确的 source state-action-value correspondence，而非高层标签或边际频率。

仍不可支持：

1. 旧 arcade rollouts 已产生同样稳定的 source control law；
2. real-game → ALFWorld 的 V24 neural selector 有效；
3. WebShop/video/VLM target 已验证；
4. 这种机制对任意 domain 都成立。

## Artifacts and reproduction

- Frozen development config：`configs/controlled_neural_symbolic_transfer_v3_development.json`
- Development report：`runs/controlled_neural_symbolic_transfer_v3_development/report.json`
- Frozen formal config：`configs/controlled_neural_symbolic_transfer_v3_formal.json`
- Formal report：`runs/controlled_neural_symbolic_transfer_v3_formal/report.json`
- Formal episode rows：`runs/controlled_neural_symbolic_transfer_v3_formal/episode_rows.json`

Formal config content hash：
`4aa52752f157c30e1ddf1fe52cf70cb43b5b3f9e32ad353deedfc3ea939e6d53`。

Formal report file SHA-256：
`f5eccd6776b5dc6c1e780f91ada93f4dfcc062e242d7d269201e23c2bc7d6c69`。

```bash
PYTHONPATH=src:. python scripts/run_controlled_exploration_transfer_v1.py \
  --config configs/controlled_neural_symbolic_transfer_v3_formal.json \
  --output-dir runs/controlled_neural_symbolic_transfer_v3_formal_reproduction
```
