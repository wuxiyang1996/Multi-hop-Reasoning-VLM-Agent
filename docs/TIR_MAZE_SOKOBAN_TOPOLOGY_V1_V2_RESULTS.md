# Sokoban topology skill to TIR maze transfer

## Outcome

The frozen fresh formal split validates a narrow but non-trivial game-to-visual-reasoning transfer:

| Condition | Correct | Rate | Change versus raw |
|---|---:|---:|---:|
| Raw target-only neural answer | 14/24 | 58.3% | reference |
| Authentic Sokoban topology + target grounding | 23/24 | 95.8% | 9 wins, 0 losses |
| Alpha-renamed authentic | 23/24 | 95.8% | identical to authentic |
| Direction-permuted source control | 14/24 | 58.3% | 24/24 source abstentions |
| Endpoint-only target control | 21/24 | 87.5% | authentic wins 2, loses 0 |
| Path-length marginal control | 7/24 | 29.2% | authentic wins 16, loses 0 |

Authentic versus raw has exact two-sided sign-test `p=0.00390625`. All seven frozen gates passed, including zero negative-transfer cases, alpha-renaming invariance, non-trivial action changes, valid neural bindings, and superiority over every specified control.

## What transferred

The source skill was discovered from real Sokoban transition receipts and freshly confirmed on disjoint source seeds. It contains only this anonymous symbolic procedure:

1. bind an anonymous topology;
2. execute a candidate edge while the prefix remains valid;
3. refute a candidate when an edge is blocked;
4. verify that exactly one candidate reaches the goal;
5. commit that candidate, otherwise abstain.

It does not contain TIR coordinates, color names, direction deltas, answer labels, correct answers, or path lengths. On 62 eligible fresh Sokoban examples, the canonical executor scored 100%, while direction-permuted, phase-reversed, and sequence-length controls each scored 0%.

## Neural-symbolic boundary

The neural component is target-native. GPT-4.1-mini sees the instruction and image, but not the answer choices or gold label, and binds the meanings of `R/L/U/D` plus start/goal colors. Target-native pixel processing constructs the graph. The source symbolic executor receives only anonymous nodes, edges, candidates, and the bound goal predicate, then executes the source-qualified procedure.

This separation matters: moving Sokoban action names directly into TIR would be lexical transfer, while an all-neural target solver would not test the source program. Here, the target grounder supplies perception and interface binding; the source program supplies the reusable intervention-grounded computation.

## Evidence protocol

- Consumed development: 9 tasks, raw 7/9, authentic 9/9.
- Fresh qualification: 12 outcome-blind selected tasks, raw 5/12, authentic 8/12; all qualification gates passed before formal access.
- Fresh formal: 24 outcome-blind selected tasks, raw 14/24, authentic 23/24.
- Full run receipts remain under `runs/`; compact tracked evidence is in `docs/results/tir_maze_topology_v2_summary.json` and `docs/results/sokoban_topology_skill_v1_compact_receipt.json`.

## Claim boundary and negative result

This result establishes transfer only for TIR single-image maze action-sequence tasks. It does not establish transfer to RefCOCO, rotation, visual search, or all of TIR.

The earlier four-family TIR V7 qualification remains a failed result: every condition scored 4/12, so its 24-example formal split stayed locked. The failure showed that a generic `POSITION -> VERIFY -> COMMIT` label mapping is insufficient when the target lacks native measurement operators for ratios, comparisons, rotation, and search. The topology result narrows the transferable principle instead of hiding that failure.

## Reproduction

```bash
PYTHONPATH=src python scripts/run_sokoban_topology_skill_v1.py
PYTHONPATH=src python scripts/run_tir_maze_topology_v2.py --config configs/tir_maze_topology_v2_frozen.json --stage qualification
PYTHONPATH=src python scripts/run_tir_maze_topology_v2.py --config configs/tir_maze_topology_v2_frozen.json --stage heldout
PYTHONPATH=src pytest -q tests/test_tir_maze_topology.py tests/test_tir_sokoban_effect_harness.py
```
