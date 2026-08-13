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

## 2026-08 wrapper parameterized-intervention update

本轮没有复制一个简化 tool schema，而是直接导入
`Multi-hop-Reasoning-VLM-Agent/visual_reasoning_wrapper` 的 registry、question router 和
typed tools。TIR 使用 `zoom_region/read_text_region/describe_region`；Video-Holmes 使用新增的
`inspect_multimodal_window(start_sec,end_sec,n)`，将同一窗口的 visual frames 与
question-independent audio-event grounding 对齐。source model 始终只看到 9 个 token-free
causal features，不看到问题文本、答案、坐标、时间戳、tool 名或音频内容。

### TIR wrapper V2：adaptation preflight 通过

16 个已消费 adaptation tasks 上，修复了三个协议错误：overview/native coordinate mapping、
provider 的最小 crop 限制，以及 source game `test_cost` 与 TIR accuracy-only objective 的不一致。
belief calibration 和 candidate effect prediction 使用 nested leave-one-task-out；held-out task
从两个 target heads 中都被完全排除。

- baseline：`6/16 = .375`
- target-only：`6/16 = .375`
- authentic source + target-native grounding：`7/16 = .4375`
- shuffled source：`6/16 = .375`
- source marginal：`6/16 = .375`
- oracle candidate：`9/16 = .5625`

所有预注册 adaptation gates 通过，包括 evidence headroom、action contrast、相对 baseline、
target-only 与两个 source controls 的 strict accuracy superiority。这个结果支持
`intervention-grounded symbolic TEST/COMMIT structure + target-native neural grounding` 的
adaptation-level 可行性；它仍不是 qualification/held-out transfer claim。

权威报告：
`runs/active_tir_wrapper_neurosymbolic_v2_final_reanalysis/adaptation_reanalysis_report.json`。

### Video-Holmes wrapper V4--V11：evidence 已修复，transfer 仍未通过

V4 证明纯 visual window 没有 headroom：baseline 和 oracle 都是 `3/8`。日志显示关键错误需要
sound lure、cry/scream、dialogue 或跨段 narrative evidence。wrapper 因此新增 provider-neutral
audio analyzer callback；实验 runner 使用 `gpt-audio` 对所选窗口做 question-independent
audio-event description，并用 6 段低带宽 audio scout 辅助定位。GPT-5.4 target evidence judge
则修复了部分“把 diegetic supernatural evidence 强行解释为现实错觉”的错误。

V5 在原困难 8-task pool 上把 oracle 从 `3/8` 提高到 `5/8`，说明多模态 intervention
确实产生了 target evidence headroom；但 authentic 仍只有 `3/8`。随后冻结 20 个不同 video ID、
按 7 个官方 question types 分层的额外 train-adaptation tasks：

- V7（新增 20）：baseline `16/20`，oracle `17/20`，authentic `17/20`，target-only `17/20`，
  shuffled `15/20`，marginal `14/20`。source 优于 controls，但不严格优于 target-only。
- V8（合并 28，semantic MLP nested cross-fit）：baseline `19/28`，authentic `17/28`，
  target-only `20/28`。高维 semantic descriptor 在少量异质 task 上过拟合。
- V9（outcome-blind neural applicability）：authentic `19/28`，target-only `20/28`，
  shuffled `17/28`，marginal `16/28`。source structure 有 control contrast，但没有 success-rate gain。
- V10（27-task leave-one-out target residual）：authentic/controls/target-only 都是 `20/28`；
  target residual 提升结果但淹没了 source prior，不能作为 transfer evidence。
- V11（每个 held-out 只允许 SHA256 选出的 4-shot residual）：authentic、shuffled、marginal
  都是 `18/28`，target-only `20/28`；few-shot 下同样未证明 source prior 的独特效用。

因此 Video-Holmes 当前正式状态仍是 `ADAPTATION_PREFLIGHT_FAIL`。已排除的 blocker 包括
loader、wrapper dispatch、provider endpoint、视觉带宽和缺失音频；剩余 blocker 是 source
active-identification prior 与 narrative video evidence reliability/value 的结构错配。继续搜索
residual K、seed、threshold 或 MLP size 会成为结果驱动调参，不应打开 formal test evaluation。

权威报告：

- `runs/active_video_wrapper_neurosymbolic_v5_audio_scout/adaptation/adaptation_report.json`
- `runs/active_video_wrapper_neurosymbolic_v7_expanded/adaptation/adaptation_report.json`
- `runs/active_video_wrapper_neurosymbolic_v9_outcome_blind_applicability/adaptation_reanalysis_report.json`
- `runs/active_video_wrapper_neurosymbolic_v10_source_target_residual/adaptation_reanalysis_report.json`
- `runs/active_video_wrapper_neurosymbolic_v11_four_shot_residual/adaptation_reanalysis_report.json`

### Updated lesson

游戏里学到的不是一个可以直接搬到所有 target 的“多看几次”规则。可迁移接口至少需要：

1. target-native typed intervention 真能改变可用证据；
2. target-native neural grounder 能预测 intervention applicability/reliability；
3. source symbolic value prior 在相同 target support 下严格优于 shuffled/marginal source controls；
4. combined policy 严格优于一个合理的 target-only policy，而不只是优于弱 baseline。

TIR 在 adaptation 上满足四项；Video-Holmes 目前只稳定满足前两项。
