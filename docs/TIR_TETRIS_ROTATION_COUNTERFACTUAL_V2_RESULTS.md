# Tetris → TIR rotation: counterfactual neural-symbolic transfer V2

## Result

The fresh formal gate passed on TIR `rotation_game`:

| Condition | Qualification | Formal |
|---|---:|---:|
| Raw target-only VLM | 1/8 | 4/21 |
| Authentic Tetris inverse + target grounding | **8/8** | **19/21** |
| Alpha-renamed authentic | 8/8 | 19/21 |
| Target-written isomorphic | 8/8 | 19/21 |
| Opposite-group control | 0/8 | 0/21 |
| Binding-rotation control | 0/8 | 0/21 |
| Half-turn marginal | 0/8 | 0/21 |

Formal authentic versus raw produced 15 wins, 0 losses, and 6 ties
(`p=0.00006103515625`, exact two-sided sign test). It changed 16/21 target
answers. All nine frozen gates passed.

## Why V1 failed and V2 worked

V1 asked a VLM to estimate an absolute counterclockwise angle from one rotated
image. Its fresh qualification failed badly: authentic scored 0/12 while raw
scored 4/12. The symbolic cyclic inverse was correct, but the neural variable
binding was not.

V2 executes every target-native candidate rotation on the image and constructs
a deterministically shuffled anonymous panel. The neural grounder sees only
`P0...P5` images and selects the intervention that restores physical uprightness;
numeric angles, A--F slots, and gold are not in its request. The symbolic
executor then binds the anonymous identity witness to the target-native group
action:

```text
Tetris interventions establish: recovery = inverse group element
  → generate target-native candidate group interventions
  → neural model verifies which successor is upright identity
  → symbolic binding maps anonymous panel to TIR action
```

On the 24 already-consumed V1 development/qualification images, this redesign
scored 24/24 versus raw 5/24. Only then were eight of the 29 unopened IDs assigned
to V2 qualification and 21 to formal by a salted ID hash. Formal remained locked
until qualification passed.

## Claim boundary

This validates a non-maze TIR transfer mechanism and shows that target-native
counterfactual neural grounding can repair an otherwise valid symbolic transfer.
It does not prove that the program must come from Tetris: the target-written
isomorphic condition is extensionally identical and scores exactly the same.
The result is specific to `rotation_game`, and the neural model evaluates
rendered target interventions rather than regressing an absolute angle zero-shot.

The compact hash-bound evidence is
`docs/results/tir_rotation_counterfactual_v2_summary.json`; full qualification,
formal, and per-sample receipts remain under
`runs/tir_rotation_counterfactual_v2_fresh/`.
