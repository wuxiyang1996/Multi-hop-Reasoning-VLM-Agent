# CLEVRER + AGQA2 shared-grounding transfer protocol V1

## Outcome

The two video routes now fit the same fail-closed measurement contract:

```text
video / official event graph
    -> one target-native grounding pass
    -> one content-addressed shared grounding receipt
    -> matched controller arms
    -> target-native executor
    -> evaluator opens gold only after every prediction freezes
```

The implementation supports two claims that must never be pooled:

1. `ORACLE_EVENT_GRAPH -> CONDITIONAL_SKILL_TRANSFER` reads an official target
   scene/event graph, but rejects functional programs, answers, and target
   outcomes.  It measures controller transfer conditional on correct grounding.
2. `MODEL_TOOL_EVENT_GRAPH -> END_TO_END_VIDEO_TRANSFER` uses the repository's
   real `sample_frames`, `detect_scene_changes`, and `compare_frames` wrapper
   tools.  The tool plan runs once before any controller arm and its resulting
   evidence receipt is reused verbatim by every arm.

The target adapters remain the existing CLEVRER and AGQA unified-neurosymbolic
adapters.  No second video-specific harness was introduced.  CLEVRER gained an
optional shared-state binding; AGQA already accepted a content-addressed target
state.

The official primary-track backends are now real rather than schema-only:

- CLEVRER uses the official validation simulator annotations (object
  properties, 128-frame motion trajectories, and collisions), archive SHA256
  `d89e74714dff7a1117fd113b54af99fbbc51583ebd742c788c484d708eb426d2`.
- AGQA2 joins its frozen public task manifest to the official supporting AGQA
  STSG release, SHA256
  `9273ec6055df1c4730b9ea971994ae6f552a14620711273da81eab9d11d85061`.
  This version join is disclosed explicitly.

`official_video_event_graph.py` normalizes both sources into acyclic ID graphs.
The AGQA pickle is loaded by a builtin-only unpickler that rejects class/global
reconstruction. Neither adapter receives a QA row, answer, functional program,
or program-derived `sg_grounding`.

## Scientific boundary

The primary novelty is cross-domain skill/controller transfer, not video
perception.  Therefore a model or tool may perform target-native grounding.
Grounding cannot decide which experimental arm receives better evidence, and
it cannot return an answer or target functional program.

The required matched arms are:

- `neural_only`;
- `source_induced`;
- `source_permuted`;
- `generic_scaffold`;
- optional `target_native_ceiling`.

A future source-transfer claim requires source-induced to beat neural-only,
source-permuted, **and** generic scaffold under the same grounding receipt.

## Consumed-report compatibility result

The V1 audit replays the already consumed CLEVRER V15 and AGQA V62 reports.  It
does not call a provider, reopen a video, or add fresh formal evidence.

| Benchmark | Tasks | Neural-only | Source-induced | Source-permuted | Generic | Source vs generic |
|---|---:|---:|---:|---:|---:|---:|
| CLEVRER | 360 | 236 | 252 | 240 | 245 | 13W / 6L, two-sided `p=.1671` |
| AGQA2 | 900 | 249 | 290 | 249* | 290* | 0W / 0L / 900T |

`*` AGQA V62 did not run these two controls prospectively.  In this diagnostic,
source-permuted is the mandated direct fallback; generic is the target-written
rule “use the typed candidate whenever one exists.”  These are consumed-data
diagnostics, not new formal control results.

The audit exposes an important limitation:

- AGQA's `290/900 vs 249/900` gain is real relative to the frozen direct actor,
  but its final acceptance policy is extensionally identical to the simple
  target-written generic rule on all 900 rows.  V62 therefore validates the
  usefulness of structured target grounding/adoption, not source-specific
  controller necessity.
- CLEVRER beats the source-permuted controller (`16W/4L`, `p=.0118`) but does
  not significantly beat the generic scaffold in the current 360 tasks.

The stronger paper claim remains gated.  The next fresh experiment must make
the source program control a consequential operation such as evidence
acquisition, rescan, verification, or abstention, then compare it against a
matched target-written generic controller with the same tool/frame budget.

## Real oracle-grounding audit

`official_video_event_graph_v1_audit.json` validates the actual frozen cohorts,
not toy schemas:

| Benchmark | Frozen tasks | Unique videos/scenes | Official graph available | Runtime answer/program read |
|---|---:|---:|---:|---:|
| CLEVRER | 360 | 353 | 360/360 | 0 |
| AGQA2 | 900 | 300 | 900/900 | 0 |

Every task has a unique content-addressed oracle receipt, explicitly discloses
official graph access, and has zero model/tool/provider budget. All five
boundary gates pass. This is deliberately **not** a new QA score.

There is also an identifiability result. Replacing uncertain video grounding
with a complete event graph removes the perception error that the current
video intervention often repairs. On AGQA V62, source-induced final adoption
is identical to the target-written generic adoption on all 900 rows. Thus
`290/900` cannot be relabeled source-specific transfer merely because oracle
graphs are now available. Oracle and model/tool results must remain separate,
and a prospective source-controlled operation must diverge from generic before
we claim source-specific video transfer.

The retrospective source-controlled tiebreak audit finds divergence on the
consumed V62 cohort (`16W/2L`, exact two-sided `p=.0013123`) but only `3W/0L`
on qualification and `0W/0L` on development. This is mechanism evidence, not
fresh validation. Rebinding old predictions to oracle receipts would mix
incompatible groundings and is forbidden.

## Grounding-error repair results

The first concrete oracle-query MDP is now implemented rather than only
specified. It hides the complete AGQA STSG behind budgeted typed tools
(`LOCATE_ACTION`, `QUERY_RELATION_IN_WINDOW`) and uses the existing
public-question compiler. All runtime predictions are durable before the
separate evaluator file opens.

On the consumed V62 diagnostic cohort:

| AGQA2 condition | Correct / 900 | Accuracy |
|---|---:|---:|
| frozen neural-only | 249 | 27.7% |
| old model-grounded typed route | 290 | 32.2% |
| localized oracle + direct fallback | 468 | 52.0% |
| unlocalized generic oracle + direct fallback | 493 | 54.8% |
| source-localized + generic + direct fallback | 562 | 62.4% |
| temporal-permuted + generic + direct fallback | 497 | 55.2% |

The compiler applies to 418/900 questions. The oracle executor commits on
251, and all 251 are correct. It therefore recovers 219 errors without losing
any previously correct direct answer. This diagnoses the old low result as a
grounding/candidate-coverage failure: the model route produced only 165 typed
candidates, while the answer-blind official executor finds 251.

The preregistered fresh V2 qualification uses 300 new videos (900 questions),
all disjoint from V62. Selection depends only on public task/video/question
hashes. Results are 393/900 compiler-applicable and 228/228 correct committed
queries. All five frozen qualification gates pass. This is fresh evidence that
the grounding/executor repair generalizes; it is not yet a matched source
transfer result because a new direct actor was not run.

The subsequent V3 transfer qualification was frozen before selecting another
300 videos/900 questions and excludes every V62 and V2 video. It evaluates
query-policy coverage without a direct actor, so abstention counts as wrong:

| Fresh AGQA2 V3 policy | Correct / 900 | Coverage | Conditional accuracy |
|---|---:|---:|---:|
| unlocalized generic scaffold | 322 | 35.8% | 100.0% |
| temporal-permuted + generic fallback | 331 | 37.2% | 98.8% |
| source-localized + generic fallback | 386 | 42.9% | 100.0% |
| target-written isomorphic controller | 386 | 42.9% | 100.0% |

The source policy beats generic by `64W/0L` and the temporal permutation by
`55W/0L`. Its localized stage commits 246/246 correctly, all runtime arms use
the same official STSG backend and a frozen maximum of five tool calls, and no
answer, functional program, or program-derived grounding is read before
predictions freeze. This validates that the source-acquired temporal structure
adds useful target query coverage. It does **not** show that no human could
write the same target controller: the explicitly disclosed isomorphic
target-written arm is equal by construction. The contribution is learning and
reusing that structure without target program labels, not algorithmic
uniqueness.

CLEVRER has the same diagnosis. Converting official simulator annotations to
the native symbolic executor gives 120/120 on the explanatory reserve, versus
101/120 for neural-only and 112/120 for the old two-representation ceiling.
The adapter refuses predictive and counterfactual questions because the
factual annotation does not contain authorized future/counterfactual
rollouts. Those families remain on the dynamics-model track.

The earlier identifiability failure is therefore narrowed rather than hidden.
An unlocalized generic query is no longer extensionally identical and loses to
the source-localized acquisition policy on untouched V3. A target-written
*isomorphic* policy remains identical, as it must. This is the appropriate
claim boundary for neural-symbolic skill transfer with target-native tools.

## Implemented files

- `src/motif_transfer/video_transfer_measurement.py`: shared receipt, resource
  budget, matched-arm freezer, evaluator, oracle/model claim separation.
- `src/motif_transfer/visual_wrapper_bridge.py`: evidence-only wrapper
  dispatcher for shared video grounding.
- `src/motif_transfer/source_controlled_grounding.py`: domain-blind interpreter
  of the source transition/terminal/abstention guards.  Its anonymous
  `APPLY_TYPED_TRANSITION` authorization is bound by separate CLEVRER and AGQA
  adapters to native grounding tools; the target adapter cannot change the
  source verdict.
- `src/motif_transfer/official_video_event_graph.py`: safe official graph
  loaders, canonical answer/program-blind states, and oracle receipts.
- `src/motif_transfer/agqa_oracle_query_mdp.py`: hidden-STSG typed tools,
  deterministic temporal relation executor, budgets, and receipts.
- `src/motif_transfer/clevrer_oracle_query_mdp.py`: factual simulator adapter
  with an explicit explanatory-only authority boundary.
- `scripts/audit_two_video_shared_grounding_v1.py`: two-benchmark consumed
  compatibility replay.
- `scripts/audit_official_video_event_graph_v1.py`: real 1,260-task official
  graph availability and authority-boundary audit.
- `scripts/audit_agqa2_oracle_query_mdp_v1.py`: consumed matched grounding
  diagnosis; `scripts/evaluate_agqa2_oracle_query_fresh_v2.py`: fresh
  300-video qualification.
- `scripts/freeze_agqa2_oracle_query_transfer_v3.py` and
  `scripts/evaluate_agqa2_oracle_query_transfer_v3.py`: untouched matched
  source/generic/permuted qualification.
- `scripts/audit_clevrer_oracle_explanatory_v1.py`: official factual-graph
  explanatory execution.
- `configs/two_video_shared_grounding_transfer_v1_development.json`: future
  fresh-evaluation gates.
- `docs/results/video_shared_grounding_v1_compatibility.json`: current audit.
- `docs/results/official_video_event_graph_v1_audit.json`: real oracle audit.
- `docs/results/agqa2_oracle_query_mdp_v1_consumed.json` and
  `docs/results/agqa2_oracle_query_fresh_v2.json`: consumed and fresh AGQA
  oracle-query reports.
- `docs/results/agqa2_oracle_query_transfer_v3.json`: preregistered fresh
  source-structure transfer qualification.
- `docs/results/clevrer_oracle_explanatory_v1_consumed.json`: CLEVRER factual
  grounding diagnosis.

## Verification

The suite now also covers safe pickle loading, cycle removal, official CLEVRER
zip loading, graph tamper detection, and oracle receipt construction. Tests
include answer/program leakage, outcome exposure,
resource-cap violations, mismatched grounding receipts, both benchmark
adapters, and real wrapper execution on a synthetic video transition.

The fresh V3 result demonstrates that the source policy's localized tool plan
differs consequentially from generic and temporally permuted controls and
improves correct evidence coverage under the matched budget. Raw-pixel video
grounding and predictive/counterfactual CLEVRER dynamics remain separate open
tracks; neither is needed for the conditional controller-transfer claim.

## AGQA official-test bridge and broad executor

The repository already contained frozen, answer-blind V61 direct predictions
from `qwen/qwen3-vl-32b-instruct` on 1,769 operator-unfiltered questions from
590 official-test videos. We joined those immutable predictions to the
official **test** STSG and froze every matched arm before reopening the
historically consumed V61 evaluator. This gives a useful retrospective bridge,
not a new untouched benchmark result:

| V61 official-test sample, official-STSG track | Correct / 1,769 | Accuracy |
|---|---:|---:|
| frozen Qwen3-VL-32B actor | 490 | 27.7% |
| qualified broad generic stack | 1,063 | 60.1% |
| temporal-permuted + broad generic | 1,075 | 60.8% |
| source-localized + broad generic | **1,155** | **65.3%** |
| target-written isomorphic controller | 1,155 | 65.3% |

Source-induced beats the same generic stack by `92W/0L`, and beats the frozen
actor by `665W/0L` over 434 positive and zero negative video clusters. The
union of the localized and broad public grammars parses 1,228/1,769 rows.

The broad executor supports the existing public-plan IR for relation,
ordering, and duration families, but authorization is learned only on consumed
train development reserves. With a frozen minimum of 20 commits and 95%
conditional accuracy, only `QUERY_OBJECT` qualifies (`63/63`). `EXISTS` is
correct on only `102/220` and is rejected; low-support temporal/duration arms
also abstain. This train-only gate removes all 26 official-test losses produced
by an unqualified broad executor.

A subsequent preregistered V5 qualification uses another 300 train videos/900
questions, disjoint from V62, V2, and V3:

| Fresh V5 policy | Correct / 900 | Conditional accuracy |
|---|---:|---:|
| qualified broad generic | 334 | 100.0% |
| temporal-permuted + broad generic | 340 | 98.6% |
| source-localized + broad generic | **382** | **100.0%** |

Source beats generic by `48W/0L` and permutation by `42W/0L`; every frozen
gate passes. Thus the broad target-native stack preserves the fresh
source-structure result.

The `65.3%` official-test-sample number must not be labeled AGQA SOTA. It uses
official oracle STSG, covers only 1,769 sampled questions, and the identities
were historically consumed by V61. Published raw-video systems train on the
AGQA target distribution and report the full official test protocol. A future
apples-to-apples result requires a public competitive raw-video checkpoint as
the frozen actor and the complete official evaluator; the harness delta, not
the oracle absolute score, is the intended comparison.

## 2026-08-31: frozen Qwen3-VL-235B off-the-shelf grounding attempt

We replaced the AGQA visual semantic backend with the frozen non-thinking
`qwen/qwen3-vl-235b-a22b-instruct` endpoint and retained the source-induced
typed controller, target-written-equivalent control, source-permuted
abstention, and answer/program/scene-graph runtime barriers.  No target
grounder was trained.

The 36-row consumed-development run cost $0.0722.  A development-fitted
selective rule improved direct 24/36 to 25/36 with 1 win and 0 losses.  It was
then frozen before a 30-video, one-question-per-video reserve whose videos had
not appeared in prior runtime receipts.  The fresh result failed: direct was
17/30 and source-selective was 16/30 (2 wins, 3 losses).  Public-question route
accuracy was only 7/30.  A subsequent consumed-development wrapper-window
skeptical verifier cost $0.0027 and produced 0 wins and 1 loss at the frozen
0.9 threshold.

Therefore this attempt is a negative result, not evidence that AGQA transfer
works.  Stronger off-the-shelf perception and focused frame tools did not solve
program-type applicability or correlated visual false positives.  The complete
machine-readable record is `docs/results/agqa2_qwen235_grounder_v65_summary.json`.
Any future attempt must qualify a target-native program-type router on AGQA
train/development and evaluate on a new untouched unit; V65 cannot be reused as
formal evidence.

## 2026-09-01: final Qwen3-VL-235B AGQA2 verdict

We completed the allowed train/development qualification path, froze the
question-only program router, and evaluated new video-disjoint formal units.
The source-induced IR, source-permuted control, target-written isomorphic
control, and target-native visual grounder were held fixed. Runtime did not
read answers, functional programs, scene graphs, or source identity.

| Evaluation | Videos | Neural-only | Source-induced | W / L | Delta | Verdict |
|---|---:|---:|---:|---:|---:|---|
| V73 untouched formal | 240 | 149 (62.1%) | 154 (64.2%) | 19 / 14 | +2.08 pp | failed preregistered minimum-wins gate |
| V74 final disjoint replication | 266 | 172 (64.7%) | 168 (63.2%) | 12 / 16 | -1.50 pp | failed |

V74 used the V73-derived, preregistered structural applicability rule
`SINGLE_EVENT_BINDING_AND_NO_CONFLICT_TIEBREAK`. It authorized 151/266 rows;
the one-sided exact binomial p-value was `0.8275`. Route accuracy,
source-permuted abstention, and target-written-equivalent matching were all
100%, so this is not a routing, control-arm, API, or infrastructure failure.
It is a scientific negative: off-the-shelf visual grounding errors remain
correlated with the symbolic overrides, and the small V73 gain did not
replicate.

Accordingly, AGQA2 supports the implementation-level claim that the same
source-induced typed IR can be routed, grounded, executed, and audited in a
natural-video domain. It does **not** validate a robust AGQA2 success-rate
gain. AGQA2 must be reported as a preregistered negative replication and
limitation, not counted among passed target domains. The V74 failure policy is
`NO_FURTHER_AGQA_ADAPTATION`; all evaluated cohorts are consumed and may not be
used for another tuning/evaluation cycle.

The authoritative compact record is
`docs/results/agqa2_qwen235_source_transfer_v65_v74_final_summary.json`; the
immutable detailed reports are
`runs/agqa2_source_executor_formal_v14/report.json` and
`runs/agqa2_source_binding_formal_v15/evaluation_report.json`. This verdict
does not alter the frozen results of WebShop, ALFWorld, DiscoveryWorld, or
TIRBench.

### Fresh-evidence inventory after V74

An answer-blind inventory audit found 1,814 videos in the official AGQA2 test
split and zero videos disjoint from all prior AGQA runtime receipts. All 1,807
test videos containing a parser-compatible `EXISTS` question had also appeared
in prior runtime. The train formal partition retains 96 untouched videos, but
the preregistered V74 failure policy explicitly withholds them from further
AGQA adaptation or evaluation. Consequently, another confirmatory,
video-disjoint AGQA2 run cannot be formed from the available official test
data. `scripts/audit_agqa2_untouched_inventory_v16.py` reproduces this
inventory without reading answers, functional programs, or scene graphs and
without making provider calls.

### Post-V74 independent-grounder diagnostic

After the V74 endpoint, we ran a separately disclosed exploratory test of the
hypothesis that the negative result was specific to the Qwen235 grounder. The
source IR, symbolic executor, controls, and target question router were not
changed; only the off-the-shelf grounder/actor was replaced by
`google/gemini-3.1-pro-preview`. The experiment used already-consumed
development videos and therefore cannot alter the V74 formal verdict.

The clean 16-video V75C pilot improved from 6/16 direct to 11/16 typed
(`6W/1L`). Before reading a pilot-disjoint 48-video qualification, we froze
requirements of at least 12 wins, at most 4 losses, net gain at least 8, and a
one-sided exact paired p-value at most 0.05. V76 produced:

| Arm | Correct / 48 | Accuracy |
|---|---:|---:|
| Gemini neural-only | 23 | 47.9% |
| source-induced typed execution | 32 | 66.7% |

The paired result was `15W/6L`, net `+9`, `+18.75 pp`, one-sided exact
`p=0.0392`. Route accuracy and both controls were 100%. Nevertheless, six
losses exceed the frozen maximum of four, so V76 is formally **failed** and
the remaining 96-video exploratory reserve was not opened. The result is
useful evidence that a stronger independent grounder improves average transfer
utility, while also confirming that negative-transfer calibration is not yet
safe enough for an end-to-end AGQA success claim.

Two pre-result transport attempts are retained as aborts: V75 produced no
complete runtime receipts, and V75B produced 14/16 before outcome access. V75C
reran all 16 from scratch with a frozen sufficient JSON token budget, so no
mixed-transport receipts entered either reported evaluation. The compact
record is `docs/results/agqa2_gemini_grounder_v75_v76_summary.json`.
