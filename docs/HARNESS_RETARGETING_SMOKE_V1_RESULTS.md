# Permission-Bounded Harness Retargeting Smoke V1

日期：2026-08-11

## Outcome

新的 permission-bounded runner 在 controlled cross-semantic target 上通过全部预注册的
mechanism gates：

| Condition | Success | Mean steps | Harness |
|---|---:|---:|---|
| raw target-only | 0/12 | 7.0 | bypass |
| null skill + same Harness | 0/12 | 7.0 | frozen hash identical |
| shuffled source skill + same Harness | 0/12 | 7.0 | frozen hash identical |
| authentic source skill + same Harness | **12/12** | **4.5** | frozen hash identical |
| target oracle + same Harness | 12/12 | 4.5 | frozen hash identical |

Authentic condition 的 option intervention rate 为 `0.7778`，intervention 后 expected
observable-effect applicability 为 `1.0`；相对 null 有 12 个 paired rescues、0 regressions。
Evaluator verdict 为 `MECHANISM_SUPPORTED`。

机器可读 summary 与每个 episode 的 receipt-chain hash 见
[`results/harness_retargeting_smoke_v1_summary.json`](results/harness_retargeting_smoke_v1_summary.json)。

## What was actually tested

Source 是一个 controlled puzzle-game workflow。它在 frozen source qualification 中超过
shuffled/marginal controls，并携带 24 个 intervention receipt IDs。Skill 不是自然语言 memo，
而是六条 typed rules，覆盖：

```text
SEARCH → ACQUIRE → optional TRANSFORM → PLACE → VERIFY
```

每条 rule 包含 requires/forbids、expected add/remove effect、priority 和 failure-specific
recovery。Target 的 surface 是 incident response：例如 `scan telemetry`、`capture packet`、
`decode packet`、`submit incident`、`audit closure`。Source artifact 不包含这些 native tokens。

Target Harness 把 observation ground 为 canonical facts，并将 native actions 分组到 options；
source skill 只选择 option，Harness 只能在外部已选 option 内按 target-native score 选择 action。
Harness 类没有 `choose_option()`。

Shuffled control 保留 rule IDs、guards、effects、priority、source lineage 和 qualification hash，
只对五个 option labels 做 fixed-point-free permutation。Evaluator 进一步要求 authentic 与
shuffled 的 program-structure hash 相同，且 shuffled parent hash 精确指向 authentic artifact。

## How the old failure modes are blocked

| Prior failure | V1 guard |
|---|---|
| Behavior change 被误报为 value | verdict 使用 paired official success；intervention/applicability 分开报告 |
| Harness 偷带 target policy | API 没有 option selection；realization 只能返回指定 option 中的 native member |
| Control 使用不同 Harness | 全部 core conditions 必须在所有 pairs 上共享一个 Harness hash |
| Shuffled control 被弱化 | guards/effects structure hash 必须相同，option mapping 必须是显式 derangement |
| Outcome 泄漏 | Harness observation 递归拒绝 `reward`、`official_reward`、`official_score`、`official_success` |
| Artifact/debug run 漂移 | skill、Harness、fallback、environment、budget 与 receipt chains 全部 hash/identity checked |
| 缺 condition 或拼接旧结果 | 每个 pair 必须恰好包含五个 frozen conditions，否则 `INVALID_EXPERIMENT` |
| Skill 无法 grounding 时 soft skip | 非 ABSTAIN 的 invalid/ungrounded option 直接 reject，不计入 positive report |

## Interpretation boundary

这个结果证明的是：

- permission separation 可以在代码中执行，而不只是 prompt 约定；
- same-Harness null/shuffled/authentic/oracle attribution matrix 可运行；
- 对结构同构、surface 不同的 workflow，source symbolic option program 可以控制 target-native
  action grounding 并提高 official success；
- evaluator 能 fail closed 于 hash、identity、control lineage 和 outcome receipt tampering。

这个结果不证明：

- 已经从 Sokoban、Thunder 或其他真实 game rollout 中抽取出该 skill；
- controlled source qualification 等价于真实 source intervention evidence；
- target ontology 是自动发现的；canonical predicate/option vocabulary 在 smoke 中是预定义的；
- 相对强 target policy 仍有增益。当前 null/raw fallback 被故意冻结为弱 policy，因此
  `12/12 vs 0/12` 是机制单测，不是 effect-size estimate；
- ALFWorld/WebShop 上得到新的 confirmatory improvement。

因此正确结论是：**Harness idea 现在有一个可执行、可审计、不会靠缺 control 自动 PASS 的
reference implementation；real-game transfer claim 仍必须通过下一阶段数据门。**

## Next real-data gate

不要直接消费旧 ALFWorld held-out。下一次真实实验应：

1. 从 Sokoban/Thunder/Candy 的新 matched intervention forks 中选择一个 typed option program；
2. 在 source qualification/held-out 上先证明 authentic value 超过 shuffled 与 marginal；
3. 冻结 source artifact 和 program-structure hash；
4. 用 disjoint target adaptation data 训练 observation/action/effect grounder；
5. 将这些 adapter 接入本 runner，同一轮执行五条件 paired matrix；
6. 只有 authentic 超过 null/shuffled，且强 target baseline 不退化，才升级为 real-game transfer。

已有 V4 synthetic→ALFWorld `19/24 vs 14/24` 可以作为 target adapter 的开发参考，但其 held-out
已经 consumed，且 ontology 部分手工指定，不能拿来充当本协议的新 confirmatory run。
