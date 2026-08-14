# DiscoveryWorld neural-symbolic transfer: V16 qualification

## 结论

V16 得到了一个可信但很小的正向信号，尚不能宣称 cross-domain transfer 已验证。

冻结的 Sokoban source program 是：当 `DIRECT_PROGRESS_AVAILABLE` 或
`ASSIGNMENT_IMPROVEMENT_AVAILABLE` 时才执行不可逆 `COMMIT`，否则执行可逆
`POSITION` 并重新计算 effect predicates。DiscoveryWorld 只提供 target-native neural
binding；当前 observation 中的 exact relation、inventory、action schema 和 containment
决定符号 predicate 的真假。

Seed1 qualification 在打开前冻结了代码、prompt、两条任务、fork rule 和 gates。Fork
选择只找 target-only trajectory 中第一个预声明的 `DROP`/`PUT` proposal，不读取
action success、terminal、evaluation 或 scorecard。Space Sick 与 Proteomics 都 eligible，
policy/audit fork hashes 和 selection receipts 全部匹配。

## 结果

| Condition | Space Sick | Proteomics | Successes |
|---|---:|---:|---:|
| recorded target-only suffix | 1 | 0 | 1/2 |
| target-native myopic | 1 | 0 | 1/2 |
| authentic Sokoban effect program | 1 | 1 | 2/2 |
| commit-availability-only | 1 | 0 | 1/2 |
| inverted-effect | runtime error | 0 | incomplete |
| always-position | runtime error | 0 | incomplete |

Proteomics 是关键 matched case。Myopic 与 availability-only 在 effect predicate 不成立时
先 `DROP`，6 步内失败；authentic 先执行 `TELEPORT_TO_OBJECT 43006`，观察到目标物体
位于 agent 正东一格后形成 `DIRECT_PROGRESS_AVAILABLE`，再 `DROP 17557`，2 步成功。

Space Sick 是 non-degradation case。Binder 将污染蘑菇、jar 和 `PUT` 绑定为 target-native
symbols；exact native preconditions 形成 `ASSIGNMENT_IMPROVEMENT_AVAILABLE`。Recorded、
myopic、authentic 和 availability-only 都一步成功。

## 为什么整体仍判失败

Primary paired signal 通过：authentic 为 2/2，myopic 为 1/2，且没有 negative transfer。
但预注册协议要求所有 arms 无 runtime error。Space Sick 的 inverted/always-position 在
第 4 个 recovery decision 只剩一个合法候选，而冻结 parser 要求至少两个，因此两臂
中止。zero-runtime gate 失败，也使完整的 source-control superiority 无法评估。

此外 N=2 只适合作为 pilot。不能用 V10 的有利采样替代 V11 的 tie，也不能在看到
seed1 后修 parser 再把同一 split 称为 qualification。

## Bitter lessons

1. 神经模块适合绑定 task-native entity/action，不应重复猜 exact symbolic relation。
2. 方向视角必须由固定 relation algebra 反转；containment 必须是独立的 `inside` 类型。
3. `COMMIT_AVAILABLE` 不等于 `DIRECT_PROGRESS_AVAILABLE`；Proteomics 的差异来自这一点。
4. Candidate bundle 应逐候选验证，但在 controls 中只剩一个合法候选时，不能把整个
   arm 视为 schema failure；下一版必须显式记录 degenerate choice set 并继续。
5. Archaeology 的公开任务文本同时写 right/west，属于 specification ambiguity，不能按
   outcome 反向选择方向后纳入确认性证据。
6. Reactor seed0 在 32 步内没有到达 final commit interface，当前方法不能解决其上游
   navigation/exploration failure。

## 下一步

V17 应把 seed1 明确降为 adaptation data，仅修复 one-valid-candidate 的 harness 行为，
重新冻结后在从未打开的 Space Sick / Proteomics Easy seeds2-4（6 instances）上运行。
只有 formal success、no-negative-transfer、source-control 和 zero-runtime gates 全部通过，
才能把结论提升为“game-to-DiscoveryWorld transfer validated”。

机器可读结果见 `docs/results/discoveryworld_v16_qualification_summary.json`。
