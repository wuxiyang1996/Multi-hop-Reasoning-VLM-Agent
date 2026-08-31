# Source-only Qwen3.5-9B Harness: Fresh Six-Benchmark Model Substitution

## Paper-ready result

A source-only Qwen3.5-9B LoRA made **2,246/2,246 exact symbolic route decisions** over **1,346 tasks** from six target benchmarks. Under locked replay, all content-addressed target-native receipts were available and success-critical action equivalence was **100.0%** over **1,864 decisions**, with zero divergence episodes.
The evaluated catalog presentations and opaque aliases were frozen after the earlier diagnostic but before the V3 source-only weight update. They are disjoint from the consumed diagnostic presentations; the underlying pre-outcome target task identities and native traces are intentionally the same, so this is not presented as a new target-task sample.

The six benchmarks cover five semantic domains; CLEVRER and AGQA2 are two benchmarks in the same video-understanding domain.

| Benchmark | Semantic domain | Tasks | 9B routes | Neural-only | Source-induced | Δ correct | Δ pp | Action eq. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WebShop | web interaction | 32 | 32 | 7/32 (21.9%) | 23/32 (71.9%) | +16 | +50.0 | 100.0% |
| ALFWorld | embodied text interaction | 24 | 24 | 13/24 (54.2%) | 20/24 (83.3%) | +7 | +29.2 | 100.0% |
| DiscoveryWorld | scientific discovery | 12 | 12 | 3/12 (25.0%) | 12/12 (100.0%) | +9 | +75.0 | 100.0% |
| TIRBench | visual reasoning | 18 | 18 | 6/18 (33.3%) | 12/18 (66.7%) | +6 | +33.3 | 100.0% |
| CLEVRER | video understanding | 360 | 360 | 236/360 (65.6%) | 252/360 (70.0%) | +16 | +4.4 | 100.0% |
| AGQA2 | video understanding | 900 | 1800 | 249/900 (27.7%) | 290/900 (32.2%) | +41 | +4.6 | 100.0% |

Descriptive micro-average: 514/1346 (38.2%) to 609/1346 (45.2%), +95 correct (+7.1 pp). This pooled number is not the primary inferential statistic because the benchmarks are heterogeneous.

## What was trained

Qwen3.5-9B was continued for 800 steps on 10,594 source-only examples (seed 20260901). No target prompt, completion, action, success label, formal outcome, or video example was used for a weight update. The target Decision Agents, neural grounders, utility/verifier modules, composers, and native executors remained frozen.
The V3 update used balanced, source-only anonymous catalog permutation and alias closure for all seven source-induced programs; it did not train on diagnostic errors or target examples.

## Claim boundary

After a consumed diagnostic, a fresh anonymous catalog-presentation reserve was frozen before a symmetric source-only permutation-closure update. The Qwen3.5-9B LoRA exactly replaces every frozen symbolic route decision and preserves success-critical native actions on the content-addressed traces of six previously validated benchmarks. The official success outcomes are inherited, not fresh live 9B reruns; the fresh presentation is not a new target-task sample; target-native grounding/execution remains domain-specific, and CLEVRER plus AGQA2 constitute one semantic video domain.

This result supports locked model substitution on existing formal traces. It does not relabel those outcomes as a fresh end-to-end live 9B evaluation, does not claim that target-native grounding is domain-agnostic, and does not place raw video perception inside the 9B text/IR controller.

## Reproducibility artifacts

- `protocol`: `/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean/runs/harness_controller_qwen35_9b_mixed_v3_protocol/protocol.json` (`e536e61c0948fce33af47e382a8da6499e04e23f688c0e82c83210b95fe2531d`)
- `training_receipt`: `/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean/runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901/training_receipt.json` (`61446db7bd91839805839b9746062dd627917c92c23cbf38e15e5c8038d8ba6b`)
- `source_qualification`: `/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean/runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901/source_mixed_qualification.json` (`0a4bb7bad367623dfb3b42fab543ff8f920959d379fc6e73f820d740247fedd8`)
- `route_report`: `/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean/runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901/six_benchmark_route_report.json` (`ceb38032d4bda283c2e84237bfac0d8cde227a211fde6f258026017fe8cf656b`)
- `action_equivalence`: `/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean/runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901/six_benchmark_action_equivalence.json` (`0ee8092e69ce53beb13e802bc7da129017d925cca9570664c4ba443f24561615`)
