# Procedural-game to ALFWorld neural-symbolic transfer V1

Four-domain web overview: [`NEUROSYMBOLIC_TRANSFER_FOUR_DOMAIN_STATUS.html`](NEUROSYMBOLIC_TRANSFER_FOUR_DOMAIN_STATUS.html).

## Result

The frozen 24-task `ALFWorld valid_unseen` evaluation supports the narrow transfer claim:

> Matched intervention returns learned in a controlled suite of finite-horizon procedural games can improve an ALFWorld-native neural-symbolic agent when the transferred object is a typed workflow value function, rather than source action tokens.

| Condition | Success | Mean steps | Changed option rate |
|---|---:|---:|---:|
| Target-only neural grounder | 14/24 | 44.125 | 0% |
| Authentic procedural-game value + target grounder | **22/24** | **24.417** | 17.75% |
| Shuffled source value + target grounder | 10/24 | 54.000 | 33.72% |
| Source marginal + target grounder | 14/24 | 44.125 | 0% |
| Phase-permuted source value + target grounder | 3/24 | 65.125 | 38.90% |

Against target-only, authentic transfer produced 9 wins, 1 loss, and 14 ties. The exact two-sided paired sign-test value is `p=0.021484375`. It strictly beat every frozen control, changed the selected symbolic option on 17.75% of decisions, and reduced mean episode length by 44.67%.

The one regression was the only two-object placement task. Authentic transfer succeeded on 22/23 single-object tasks but failed on the 1/1 two-object task, so the original event-count workflow state was not sufficient for multiplicity.

## Identity-aware multiplicity extension

The follow-up V1 extension replaced the ambiguous number of `PLACE` events with
an ALFWorld-native set of distinct object identities currently bound to the goal
receptacle. Removing a completed object is represented as a negative effect. The
source value model still selects only an abstract option; the frozen target
neural grounder retains native action authority.

On six prospectively locked `pick_two_obj_and_place` tasks, authentic transfer
succeeded on **4/6**, versus **1/6** for target-only, **0/6** for shuffled source,
**1/6** for the source marginal, and **3/6** for the phase-permuted control. The
paired authentic-versus-target result was 3 wins, 0 losses, and 3 ties
(`p=0.25`; small-sample sign test). It changed the selected hierarchical option
on 4/6 tasks, reduced mean steps from 69.67 to 46.00, and the audit verified that
every reported count equals the cardinality of the distinct-ID set.

The raw legacy runner reports `FINAL_HELDOUT_FAILED` because it also retained a
singleton-era gate requiring an 8% option-change rate averaged over every
environment step; the observed rate was 2.17%. Before opening these six tasks,
the multiplicity protocol explicitly made the task-level intervention gate
(at least 4/6 tasks) authoritative and retained the old status for disclosure.
All eight frozen multiplicity gates pass in
`docs/results/alfworld_multiplicity_v1_formal_summary.json`. This validates the
narrow count-two repair; it does not establish arbitrary cardinalities.

## What transfers

The source is a finite-horizon interactive MDP with a typed automaton:

```text
SEARCH -> ACQUIRE -> [TRANSFORM] -> PLACE -> [VERIFY]
```

For each frozen source state, the harness forks the same state and random stream across all five actions, records the first intervention effect, and follows the exact source-native optimal continuation. A neural value ensemble learns from the resulting option/effect features and returns. Native source action tokens are independently alpha-renamed in every game and never enter the transferred representation.

At the target, ALFWorld-native neural heads infer goal bindings, action feasibility, and effect likelihood from the current text observation and admissible commands. The source value model scores typed symbolic options; the target grounder and executor retain all target-action authority.

This factorization is the neural-symbolic transfer:

```text
source game interventions
  -> typed symbolic state/action/effect features
  -> learned option value
  -> ALFWorld-native neural grounding
  -> admissible ALFWorld action
  -> observed effect and workflow-state update
```

## Source and development gates

Four source surfaces were used for fitting (`grid_quest`, `factory_quest`, `dungeon_quest`, and `space_quest`). Two disjoint surfaces (`island_quest` and `circuit_quest`) were held out for the source gate. The held-out source evaluation contains 24 domains and 46,080 matched intervention receipts.

| Source model | Held-out value MSE | Authentic relative improvement |
|---|---:|---:|
| Authentic | **0.027365** | — |
| Phase-permuted | 0.045968 | 40.47% |
| Shuffled | 0.039969 | 31.53% |
| Marginal | 0.052710 | 48.08% |

The consumed eight-task ALFWorld development split gave 8/8 for authentic transfer, versus 6/8 target-only, 3/8 shuffled, and 1/8 phase-permuted. All thresholds and the exact 24 final identities were then frozen before reading the inherited held-out reserve.

## Claim boundary

This is a positive result for a deliberately controlled procedural-game suite. It is not evidence for:

- Sokoban-only to ALFWorld transfer; the earlier binary `POSITION/COMMIT` route remains negative;
- arbitrary-game or raw-trajectory transfer;
- zero-shot ALFWorld grounding; the target-native neural grounder uses an ALFWorld adaptation split;
- unsupervised ontology discovery; the five-option typed ontology and target symbolic parser are designed components;
- a universal skill shared by all four registered target domains.

The nontrivial part is that source games are interactive stochastic MDPs, action labels do not align across source domains, all values derive from matched interventions, and authentic structure must beat shuffled, marginal, and phase-permuted controls on an unread target reserve. The main remaining scientific weakness is the human-specified cross-domain option ontology.

## Operational disclosure

The first final invocation completed target-only and authentic conditions plus two shuffled tasks, but its foreground transport was interrupted before it wrote a report. No artifact, task identity, threshold, seed, policy, or other scientific configuration was changed. The exact frozen deterministic run was replayed from the beginning and produced the report summarized here. Therefore this evidence is a frozen deterministic replay, not an uninterrupted one-shot execution.

Candidate rebuilding is version-bound: the artifact was frozen with the repository's system Python 3.13/NumPy environment and reproduces exactly there. The ALFWorld Python 3.9 environment has an older NumPy RNG implementation and is used only for target execution.

## Evidence and reproduction

Compact hash-bound evidence is in `docs/results/procedural_game_alfworld_v1_summary.json`. Full reports are local under:

- `runs/procedural_game_alfworld_v1_development/`
- `runs/procedural_game_alfworld_v1_frozen/`

Rebuild the compact receipt and run the focused tests with:

```bash
PYTHONPATH=src:. python scripts/summarize_procedural_game_alfworld_v1.py

PYTHONPATH=src:. pytest -q \
  tests/test_procedural_workflow_game.py \
  tests/test_train_procedural_game_alfworld_candidate.py \
  tests/test_summarize_procedural_game_alfworld_v1.py \
  tests/test_procedural_game_alfworld_freeze.py
```
