# Active Video/TIR neural-symbolic preflight results

这些实验只消费 adaptation data，用于判断 target evidence 是否足以支撑同一个
state-dependent TEST/COMMIT transfer protocol。它们不是 formal transfer results。

## Video-Holmes

V1 的机械 preflight 曾通过：8 个 adaptation samples 全部完成，增加 visual evidence 会让
4/8 predictions 改变，raw 与 leave-one-video-out calibrated accuracy 都是 `3/8`。但从
`k0` 到 `k4` accuracy 始终为 `.375`，所有 source conditions 也都是 `.375`；也就是说
“prediction changed”没有证明 evidence 有正向信息价值。

V2 因此新增 positive-evidence-response 与 source action-contrast gates。结果：

- raw/calibrated accuracy：`k0=.50`、best later prefix 仍为 `.50`；
- authentic 与 target 在 2/8 samples 上产生 action contrast；
- authentic 和 target accuracy 都为 `.50`；
- positive-evidence-response gate 失败。

正式状态为 `ADAPTATION_PREFLIGHT_FAIL`，没有消费 test transfer outcomes。失败原因是
target crop/evidence stream 没有展示可靠的 accuracy gain；在此基础上比较 source
exploration policy 没有可识别性。

## TIR

16 个 adaptation samples 全部完成。overview accuracy 为 `.25`，best crop prefix 为
`.3125`，有 `+.0625` evidence response；但 authentic 与 target 的 action contrast 为
`0/16`。target、authentic、shuffled 和 marginal conditions 的 accuracy 全为 `.3125`，
tests 全为 4。

正式状态同样为 `ADAPTATION_PREFLIGHT_FAIL`，没有消费 qualification/held-out IDs。这里
target evidence 有一点响应，但 source policy 与 target policy 没有行为差异，因此不能用
outcome tie 声称 transfer。

## Lesson

active perception target 必须同时满足两个 identification 条件：

1. target-native evidence acquisition 对 belief/outcome 有稳定正响应；
2. authentic source structure 实际改变 intervention timing/choice。

只满足其中一个都不应打开 formal evaluation。WebShop 最终成功，恰好是因为真实
constraint-click/commit interventions 同时提供了 causal evidence response 和
source-specific action contrast。

Authoritative adaptation artifacts：

- compact committed summary：
  `docs/results/active_video_tir_neurosymbolic_preflight_summary.json`
- `runs/active_video_neural_symbolic_transfer_v1/adaptation/adaptation_report.json`
- `runs/active_video_neural_symbolic_transfer_v2/adaptation/adaptation_report.json`
- `runs/active_tir_neural_symbolic_transfer_v1/adaptation/adaptation_report.json`
