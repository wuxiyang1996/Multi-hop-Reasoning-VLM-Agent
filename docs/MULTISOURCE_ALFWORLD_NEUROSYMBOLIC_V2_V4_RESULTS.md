# Multi-source neural-symbolic transfer to ALFWorld: V2–V4 results

Date: 2026-08-10

## Outcome

The frozen V4 held-out experiment passed every pre-registered gate. A source-trained
hierarchical option-value controller, combined with target-native neural grounding,
solved 19/24 ALFWorld held-out tasks (79.2%). The neural-only and source-marginal
conditions each solved 14/24 (58.3%), the within-state source-value shuffle solved
12/24 (50.0%), and the phase-permuted source program solved 2/24 (8.3%).

This supports a narrow positive claim:

> Hierarchical option structure and option values learned over structurally varied
> source workflows can transfer to real ALFWorld tasks when the source controller
> selects only the symbolic option and target-native neural models ground actions
> inside that option.

It does **not** yet establish real-game-to-arbitrary-domain transfer. The source
workflows are controlled synthetic domains with eight semantic surfaces, including
game, software, science, forensics, network, logistics, medicine, and maintenance.
The target is real ALFWorld, but target adaptation data and a partially specified
target action ontology are used.

## What is transferred

The source side contains stochastic workflows of three to six stages over:

`SEARCH → ACQUIRE → TRANSFORM → PLACE → VERIFY`

including optional stages and repeated acquire/place cycles. Across 256 source train
domains and 64 source evaluation domains, an ensemble learns finite-budget option
values from exact dynamic-programming targets. The transferred representation has
23 token-free features: option and required-phase indicators, precondition and
completion estimates, goal binding, workflow progress, remaining budget, repeated
action/no-op risk, urgency, failure cost, and interactions. No ALFWorld action token
is present in the source model.

The target side learns independent neural heads from ALFWorld train receipts:

- option applicability;
- goal-object binding;
- intervention completion/effect;
- a phase-masked neural-only action policy.

At test time the successful V3/V4 factorization is strict:

1. The source model may select an abstract option.
2. The target-native neural grounder selects the concrete ALFWorld action *within*
   that option.
3. The source model cannot reorder concrete actions inside an option.

This factorization is the main result. V2 allowed source values to reorder concrete
actions and produced negative transfer.

## Frozen target adaptation pool

The outcome-blind pool was frozen before collection by hashing ALFWorld train path
names, stratified across six task families:

- 36 adaptation-train tasks: six per family;
- 12 adaptation-validation tasks: two per family;
- 48 total; no selection used task contents or rollout outcomes.

The first collection exposed an environment-wrapper bug: TextWorld deterministically
shuffles registered games at reset, while the wrapper originally attached input-order
task IDs. That receipt was quarantined as
`adaptation_expert_receipts.invalid_order_labels.json`. The wrapper was changed to
read `extra.gamefile` on every reset, verify membership in the frozen set, and recover
the actual task ID and train/validation partition. The corrected receipt contains 48
unique task IDs, with 34/36 train and 12/12 validation expert successes.

## Frozen pre-evaluation gates

### Target-native grounder

| Metric | Observed | Minimum |
|---|---:|---:|
| Goal-object binding recall@3 | 1.000 | 0.75 |
| Macro required-option AUC | 0.975 | 0.70 |
| Intervention completion balanced accuracy | 0.900 | 0.65 |
| Neural-only policy AUC (diagnostic) | 0.922 | — |

Completion validation contained 266 capped, episode-balanced transitions with a
30.8% positive rate. Binding recall was evaluated on 103 validation states where a
goal-object action was available.

### Source structure/value gate

The source artifact contains 40,960 train and 10,240 held-out-source value examples.

| Source condition | Held-out-source value MSE | Authentic relative improvement |
|---|---:|---:|
| Authentic | 0.03768 | — |
| Within-state value shuffle | 0.05122 | 26.4% |
| Phase-permuted program | 0.05663 | 33.5% |
| Source marginal | 0.06867 | 45.1% |

The pre-registered source gate required at least 10% relative MSE improvement over
every control. Earlier source generation leaked the correct phase through completion
and binding. That version was blocked at 6.5% improvement over phase permutation.
The accepted generator makes affordances vary independently of symbolic phase, so
the model must use the workflow relation instead of a correlated neural proxy.

## Development results

### V2: action-level source arbitration failed

On the 8-task qualification split:

| Condition | Success | Mean steps | Changed option |
|---|---:|---:|---:|
| Neural hierarchy baseline (then named target-only) | 8/8 | 21.0 | 0% |
| Authentic action-level source | 7/8 | 28.8 | 0.87% |
| Shuffled source | 5/8 | 37.5 | 24.3% |
| Source marginal | 8/8 | 21.0 | 0% |
| Phase-permuted source | 0/8 | 70.0 | 43.8% |

The V2 candidate correctly failed. It changed 44.3% of concrete actions but almost
never changed the option. On the failed Egg task, it repeatedly selected
`examine egg 2` (21 times) because the completion head predicted approximately 0.99,
although the observation only restated that the egg was clean. This showed that an
action-level source controller can amplify target-grounder miscalibration.

It also showed that the baseline already contained the full transferred phase
monitor, making the source comparison redundant.

### V3: post-qualification factorization diagnostic

V3 moved to option-only source control and used a phase-masked neural-only baseline.
Because this design followed inspection of V2 qualification logs, V3 qualification
is diagnostic only:

| Condition | Success | Mean steps | Changed option |
|---|---:|---:|---:|
| Authentic source + target neural | 7/8 | 28.4 | 22.5% |
| Neural-only | 6/8 | 39.0 | 0% |
| Source marginal | 6/8 | 39.0 | 0% |
| Shuffled source | 5/8 | 42.6 | 32.6% |
| Phase-permuted source | 1/8 | 62.5 | 47.0% |

V3 motivated an exact freeze for V4; it is not counted as final evidence.

## V4 one-shot held-out result

Before any held-out environment reset, V4 froze and hashed the V3 artifact, V3 config,
qualification report, runner, seeds, 70-step cap, conditions, and gates. The runner
verified these hashes before allowing the held-out split. All five conditions executed
the same 24 tasks in the same verified order.

| Condition | Success | Success rate | Mean capped steps | Source admission | Changed option |
|---|---:|---:|---:|---:|---:|
| **Authentic source + target neural** | **19/24** | **79.2%** | **30.7** | 54.3% | 19.4% |
| Neural-only | 14/24 | 58.3% | 43.7 | 0% | 0% |
| Source marginal | 14/24 | 58.3% | 43.7 | 0% | 0% |
| Within-state value shuffle | 12/24 | 50.0% | 45.2 | 57.8% | 36.6% |
| Phase-permuted program | 2/24 | 8.3% | 66.0 | 62.1% | 43.1% |

All pre-registered V4 gates passed:

- authentic success count strictly exceeded every control;
- authentic changed-option rate was 19.4%, above the 8% non-triviality floor;
- authentic mean capped steps were below neural-only (30.7 versus 43.7).

Paired success comparisons provide useful but small-sample context:

- authentic vs neural-only: 5 rescues, 0 regressions, 19 ties; exact one-sided
  sign/McNemar p=0.03125, two-sided p=0.0625;
- authentic vs shuffled: 8 wins, 1 loss, 15 ties; one-sided p=0.0195,
  two-sided p=0.0391;
- authentic vs phase-permuted: 17 wins, 0 losses, 7 ties; two-sided p≈1.53e-5.

Among the 14 tasks solved by both authentic and neural-only, authentic required a
mean 18.4 steps versus 24.9, a paired reduction of 6.6 steps. Mean steps in the main
table count failures at the frozen 70-step cap.

### Task-family breakdown

| Task family | Tasks | Authentic | Neural-only |
|---|---:|---:|---:|
| Pick and place simple | 4 | 4 | 4 |
| Look under light | 4 | 4 | 4 |
| Clean then place | 4 | 4 | 3 |
| Heat then place | 4 | 4 | 2 |
| Cool then place | 3 | 2 | 1 |
| Pick two and place | 5 | 1 | 0 |

The authentic controller rescued five neural-only failures: Bread/cool/place,
Knife/clean/place, Mug/heat/place, Apple/heat/place, and one two-SoapBar task. Its five
remaining failures are four two-object tasks and one Lettuce/cool/place task.

## Interpretation

The controls rule out three easy explanations:

1. **Extra intervention alone:** shuffled source values intervene more often but score
   12/24, below neural-only.
2. **A generic source prior:** the marginal model abstains and exactly reproduces
   neural-only at 14/24.
3. **Only target neural grounding matters:** permuting source phase semantics collapses
   performance to 2/24 despite using the same target neural heads.

The positive evidence is therefore specifically for the combination of authentic
source option structure/value and target-native grounding. The result also explains
why earlier high-level skill libraries did not transfer: reusing a prose skill or
letting source values directly choose target actions is too unconstrained. The useful
unit is a small option with explicit preconditions and completion semantics, while
the target domain owns entity/action grounding.

## Limitations and next experiment

- Source domains are synthetic controlled workflows, not logged trajectories from a
  real large game environment.
- ALFWorld target adaptation is not zero-shot: 36 train tasks supervise neural heads,
  and 12 frozen train tasks validate them.
- The option ontology and parts of goal/action parsing are specified, not fully
  induced neurally.
- The final set has only 24 tasks; authentic vs neural-only is compelling in direction
  but the two-sided paired p-value is 0.0625.
- Two-object loops remain weak (1/5), indicating the source state needs an explicit
  object-count/loop invariant and target receipts need completion supervision per
  object instance.
- The 24 held-out tasks are now consumed and must not be used for further model or
  threshold selection.

The next credible test should freeze a V5 on new development data, extract the same
option contracts from trajectories in a real source such as Sokoban/WebShop or a
larger game suite, learn rather than hand-map option symbols, and evaluate on a fresh
target split or a second target environment. Multi-object loop structure should be
added before that freeze, not tuned on this held-out report.

## Artifacts

- Outcome-blind pool: `configs/alfworld_v2_outcome_blind_pool.json`
- Corrected target receipts:
  `runs/multisource_alfworld_neurosymbolic_v2/adaptation_expert_receipts.json`
- V3 frozen artifact:
  `runs/multisource_alfworld_neurosymbolic_v3_diagnostic/frozen_candidate_artifact.json`
- V3 diagnostic report:
  `runs/multisource_alfworld_neurosymbolic_v3_diagnostic/qualification_report.json`
- V4 frozen contract: `configs/multisource_alfworld_neurosymbolic_v4_frozen.json`
- V4 final report:
  `runs/multisource_alfworld_neurosymbolic_v4_frozen/heldout_report.json`
- V4 final report SHA-256:
  `cd50129490f34fdb4e3da19dbde76630af63f8f86d5e59b9456281ab236738a8`

