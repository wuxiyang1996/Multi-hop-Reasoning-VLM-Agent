# AGQA 2.0 active neural-symbolic transfer：V3–V8 结果

## 结论

AGQA 2.0 已接入统一的 game-source neural-symbolic harness，但截至 V8，**不能声称
game→AGQA success-rate transfer 已验证**。成立的是机制与安全性：source-induced typed IR
确实控制 operand 数量、是否 recurrent、最大 rescan 次数和 source-type applicability；错误
source permutation 9/9 abstain，target-written equivalent 9/9 dynamics match。没有成立的是
target-native neural grounding 的跨视频泛化。

最终 V8 fresh reserve 使用 12 个从未做过 V3–V7 model call 的 Charades videos，先在完全不读
direct/gold/program/scene graph 的条件下按 typed evidence 选择每类 3 个，再读取 9 个 gold。
结果是 3 个 typed decisions、0/3 正确、0 win/2 loss。全部预注册 gate 没有通过，因此
unified harness 授权数为 0，最终保持 matched direct 的 4/9，而不是执行有害的 source program。

## 实现的 neural-symbolic boundary

运行时数据流为：

1. public question 经显式 operand grammar 变成 typed query；无法从问题本身获得 operands 的
   open-ended question 直接 abstain，不读取 official functional program。
2. source-only IR 只决定 acquisition dynamics：relation 和 event-order 可 recurrent rescan，
   duration 是 non-recurrent dual-operand scan。
3. visual grounder 每次只看到一个 operand、时间戳和 RGB frames；看不到完整问题、另一 operand、
   answer、program、scene graph 或 source identity。
4. recurrent primary/rescan 必须一致；OBSERVED/UNOBSERVED 冲突降为 PARTIAL，两次完整扫描都
   UNOBSERVED 才允许 closed-world `no`。
5. target-native typed executor 只消费审计 receipt。qualification 之前 source executor 始终关闭；
   gate 任一失败后，统一 harness 回退到同帧 direct baseline。

QUERY_OBJECT 另加入了本地 OpenCV Zoo YOLOX corroboration。它只看 VLM 定位的 frames；在
development 的 MXATD 上，VLM `pan` 与 COCO `bowl` 独立一致后映射到 AGQA 宽类 `dish`，产生
了一个真实的 direct→typed win。这个映射和模型哈希都进入 grounder semantic lineage。

## 逐版结果

| 版本 | split | 结果 | typed / direct | 关键发现 |
|---|---|---|---:|---|
| V3 | development 9 | qualified | 5 / 4，1W/0L | 本地 object corroboration 产生正增益 |
| V3 | reserve | runtime incomplete | 未评测 | inflection validator 与 provider null envelope |
| V4 | reserve | runtime incomplete | 未评测 | open-ended duration question 没有 public operands |
| V5 | reserve 9 | not qualified | 3 / 2，1W/0L | 只有 5 次 decisive，3/5 正确 |
| V6 | reserve 9 | not qualified | 5 / 6，1W/2L | lexicographic rescan chooser 导致 negative transfer |
| V8 | development 12→9 | qualified | 9 / 6，3W/0L | outcome-used adaptation；不具 confirmatory 意义 |
| V8 | reserve 12→9 | **not qualified** | 2 / 4，0W/2L | evidence rank 不能预测真正 correctness |

V8 reserve 的完整哈希摘要在
[`agqa2_active_grounding_v8_reserve_result.json`](results/agqa2_active_grounding_v8_reserve_result.json)。
V3/V4 的 runtime failures、V5/V6 的完整负结果也分别保存在 `docs/results/`，没有被覆盖或
重新命名为成功。

## Bitter lessons

1. **route/type correctness 不是 semantic grounding correctness。** V8 仍是 9/9 route
   correct，但 `shoe or window` 这类 object-choice surface form 被 parser 当作 EXISTS，导致
   typed executor 输出 `yes`，而正确答案空间是 object noun。
2. **高 confidence 和大 temporal margin 都不可靠。** V8 evidence selector 选出的三个
   decisions 全错；Gemini 对 `holding paper`、`tidying floor` 等状态持续时间明显过延伸。
3. **更多 rescan 不能自动带来更好证据。** primary/rescan 冲突如果被强行择优会产生 loss；
   改成 conjunction 能消除旧 development loss，却显著降低 coverage。
4. **不能用不断换 fresh seed 来制造成功。** V5、V6、V8 三个完整 reserve 均失败；继续迭代
   reserve 会成为 optional stopping，而不是验证。
5. **真正的 blocker 是 target grounder，不是 symbolic IR。** source-permuted 与
   target-written controls 始终通过，但 IR 没有可靠的 target-native predicates/durations 可执行。

## 下一步的诚实门槛

在再次做 confirmatory reserve 前，需要先在独立 development data 上训练或校准一个真正的
AGQA-native event grounder：显式区分 EXISTS/CHOOSE answer space；对 action predicate 做
primary/rescan agreement；用 frame-level temporal boundary supervision 代替自由文本区间；并在
不看 answer 的 held-out grounding labels 上证明 confidence 能预测 correctness。完成这些之前，
AGQA 保持 fail-closed，不进入“跨域 transfer validated”统计；已验证的 TIR/WebShop/ALFWorld
结果不受影响。
