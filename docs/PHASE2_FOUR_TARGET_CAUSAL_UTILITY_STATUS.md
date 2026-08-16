# Phase 2：四 target neural-symbolic causal utility status

## 当前结论

截至 2026-08-16，四个 target route 都有 source-derived symbolic controller 相对 matched
target-native neural baseline 的 causal utility evidence：

| Target route | N | Authentic | Matched raw | Paired W/L/T | Exact p | Status |
|---|---:|---:|---:|---:|---:|---|
| WebShop constrained search | 32 | 19 | 9 | 10/0/22 | .001953125 | validated |
| ALFWorld selective search | 75 | 57 | 37 | 21/1/53 | 1.0967e-5 | validated |
| DiscoveryWorld Proteomics Easy | 36 | 27 | 12 | 17/2/17 | .000728607 | validated, 1 abstention |
| TIR single-image maze | 48 | 41 | 19 | 22/0/26 | 4.7684e-7 | validated |

合计是 191 target tasks、authentic 144 successes、matched raw 77 successes、70W/3L/118T。
这个 aggregate 只作描述，不计算 pooled p-value：四个实验的 task、controller、eligibility 与
evaluator 不同，不能把 post-hoc pooled statistic 当作新的预注册 hypothesis test。

## 这比 24/24 mechanism transfer 多证明了什么

Phase 1 的 `6 source games × 4 targets = 24/24` 证明每个 source lineage 都能在 fresh target
execution 中实际驱动 online symbolic route。Phase 2 进一步加入：

- 多 target tasks，而不是一个 task/cell；
- matched neural-only baseline；
- wrong/permuted、ledger-blind 或其他 symbolic controls；
- paired success endpoint 和 exact test；
- negative-transfer 与 abstention calibration；
- immutable manifest、cell receipts 和 independent audit。

因此现在可以说：

> 在四条已限定 target routes 上，正确的 intervention-grounded symbolic structure 与 target-native
> neural grounding 组合，显著提高 matched target success；错误或缺失的 symbolic structure 不能
> 解释该增益。

## 不能说什么

- 不是 WebShop、ALFWorld、DiscoveryWorld、TIRBench 全 benchmark 的普遍结论；每个结果都只对应
  文档里的具体 route/split/interface。
- 不是六个 game 各自都有 powered independent effect。WebShop、ALFWorld、DiscoveryWorld 平衡绑定
  六 lineage，但它们共享 canonical controller；TIR utility 使用独立 Sokoban topology artifact。
- 不是 source provenance 唯一性。WebShop/ALFWorld 的 target-authored isomorphic ceiling 可精确匹配
  authentic；证明的是 structure utility，不是只有从某个 game 才能得到该结构。
- 不是 zero negative transfer。ALFWorld 有 1 次、DiscoveryWorld 有 2 次 strict losses；选择性
  applicability 把总体风险控制在冻结阈值内。
- 不是 neural grounding 已解决。DiscoveryWorld acquisition 的 schema fallback 尤其严重，下一步
  应提升 target-native acquisition/grounding，而不是继续堆 source labels。

## Evidence map

| Target | Primary document | Independent audit status |
|---|---|---|
| WebShop | `PHASE2_WEBSHOP_CAUSAL_UTILITY_V4.md` | 17/17 gates |
| ALFWorld | `PHASE2_ALFWORLD_SELECTIVE_UTILITY_V3.md` | 15/15 gates |
| DiscoveryWorld | `PHASE2_DISCOVERYWORLD_SELECTIVE_UTILITY_V2.md` | 15/15 gates |
| TIRBench | `PHASE2_TIRBENCH_CAUSAL_UTILITY_V1.md` | 14/14 gates |

## 下一步

优先级不再是增加更多 source game labels，而是：

1. 在完全 untouched target reserves 上复现实验，尤其 DiscoveryWorld；
2. 把 applicability evaluator 单独做 calibration（coverage、precision、selective risk）；
3. 改善 DiscoveryWorld target-native acquisition，降低 36.5% schema-fallback steps；
4. 对六 source lineages 做真正 source-specific heterogeneity study，而不是共享 controller relineage；
5. 扩 target interface：DiscoveryWorld Normal/其他 scenario、TIR 非 maze、ALFWorld multiplicity；
6. 预注册 target-authored isomorphic controller，区分 structure utility 与 source-provenance necessity。
