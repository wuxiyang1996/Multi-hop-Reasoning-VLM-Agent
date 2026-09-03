# STAR Interaction target-native grounding factorial V27

## Outcome

The wrapper grounding protocol ran correctly, but it did **not** improve
neural-symbolic transfer.  The development gate is rejected and no fresh target
videos should be opened for this mechanism.

V27 used 128 already-consumed STAR Interaction questions from 64 videos.  Every
condition used `openai/gpt-4.1-mini` through OpenRouter.  The frozen 2x2 design
crossed:

1. evidence view: uniform 24 frames versus 12 wrapper-selected adjacent
   BEFORE/AFTER pairs (24 frame slots); and
2. controller: target-native direct commitment versus the real Sokoban
   `REFUTED -> REPLAN` executor driven by a five-step typed target proof.

The active view executed `detect_scene_changes` once and `compare_frames` twelve
times per question.  It had no silent stub or schema fallback.  Direct and proof
branches within a view received byte-identical panels.  Gold and official STAR
structures were attached only after all four neural branches, both source
executions, and all destructive controls were immutable.

| Condition | Correct | Accuracy |
|---|---:|---:|
| uniform direct | 60/128 | 46.88% |
| uniform typed proof | 62/128 | 48.44% |
| uniform source executor | 62/128 | 48.44% |
| active direct | 60/128 | 46.88% |
| active typed proof | 51/128 | 39.84% |
| active source executor | 48/128 | 37.50% |
| active binding control | 60/128 | 46.88% |
| active topology control | 58/128 | 45.31% |

The decisive paired comparison was active source versus matched active direct:

- 7 wins, 19 losses, 102 ties;
- -12 net wins / -9.38 percentage points;
- exact two-sided McNemar p = 0.02896;
- video-cluster bootstrap 95% interval = [-17.97, -0.78] percentage points.

The active-by-source difference in differences was -10.94 percentage points.
All positive transfer gates except the direction-agnostic significance threshold
failed.  The formal status is `NOT_QUALIFIED`.

## What the grounding tools did and did not do

The tools were real and observable in every receipt.  They changed the evidence
layout, but did not add semantic event grounding:

```text
24 uniform proxy frames
  -> detect_scene_changes
  -> retain 4 outcome-blind uniform anchors + 8 largest adjacent changes
  -> compare_frames on 12 retained edges
  -> 12 labeled BEFORE/AFTER pairs
```

This view used 15--20 unique frames per question (mean 16.89) because neighboring
transition pairs repeat endpoints.  Nevertheless, active direct and uniform direct
both scored 60/128, with 13 wins and 13 losses.  Therefore the aggregate failure
cannot be described as a general loss of visual-answering capability alone.

The failure was specific to the structured proof/controller interaction:

- uniform direct/proof disagreed on 29 questions; active direct/proof disagreed on
  44;
- uniform source fired 24 replans; active source fired 35;
- active replans produced 7 corrections, 19 destructions, and 9 wrong-to-wrong
  changes;
- the binding derangement fired zero active replans and preserved the 60/128 direct
  baseline, so authentic symbolic binding was causally involved in the damage;
- active raw proof was already below direct (51 versus 60), and the source executor
  reduced it by another three answers (48 versus 51).

The loss cases were not separable by a simple confidence rule.  Mean alternative
`ANSWER_ENTAILMENT` confidence was approximately 0.93 for both corrective and
destructive replans.  Post-hoc thresholding would therefore be both weak and an
invalid outcome-tuned repair.

## Bitter lesson

`detect_scene_changes + compare_frames` is a useful target-native **navigation**
protocol, but pixel-change-selected pairs are not an event predicate grounder.
Wrapping those pairs in a long candidate-factorized prompt does not make the
resulting claims intervention-grounded.  The model confidently asserted wrong
object/action bindings (for example, replacing a correctly observed refrigerator
opening with a cabinet-opening claim), and the source executor treated those
claims as hard counterevidence.

Do not proceed by adding a confidence threshold, increasing replan margin, or
selecting only STAR question types where the post-hoc delta looks better.  The next
mechanism must keep the full-clip direct commitment and use a separate,
candidate-conditional target observer that returns independently checkable event
receipts.  A valid receipt must bind:

```text
(candidate, entity, requested predicate, localized time interval,
 before observation, after observation, status, frame hashes)
```

The generic-active control must receive the same extra frame and neural-call budget.
The source automaton may replan only when an independently grounded receipt refutes
the commitment and supports one alternative; missing binding or missing temporal
localization must produce `UNKNOWN -> ABSTAIN`.

For short STAR clips, 24 uniform frames already appear sufficient, so another
scene-change layout is low priority.  Candidate-conditional localization is more
appropriately tested next on long Video-Holmes clips, where native temporal
navigation can acquire genuinely new frames rather than rearrange an already dense
short-clip overview.

## Reproducibility boundary

- frozen config: `configs/star_interaction_grounding_factorial_v27_development.json`
  (`sha256 7d2550dba454076ee65ed09d6c8b90ec4bfde25715fbe425fe89c274f85dbc95`);
- full receipts: `runs/star_interaction_grounding_factorial_v27_development/receipts.json`
  (`sha256 4439e45fea0cb40da21f94b2f7ac809991621e757a9f0f7b11b036d0f3098d1a`);
- formal report: `runs/star_interaction_grounding_factorial_v27_development/formal_report.json`
  (`sha256 530c89890af477e236360189354aa40abe6841fc9b1976ca19ff3cc1d20192e2`);
- compact committed summary:
  `docs/results/star_interaction_grounding_factorial_v27_summary.json`.

The run made 512 matched neural calls, recorded no transport failure, and cost
approximately USD 0.856.  No fresh confirmation video was read or executed.
