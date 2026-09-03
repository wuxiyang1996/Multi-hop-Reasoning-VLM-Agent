# Portable two-video audit artifact

`agqa2_full_train_broad_powered_v4_audit.tar.gz` contains the seven immutable
JSON inputs needed to recompute the AGQA2 portion of the two-video result
bundle. It does not contain raw videos, model checkpoints, API credentials, or
provider caches.

- Archive size: 5,113,399 bytes
- SHA-256: `ff5929bb549e9bffc778818acf7cf7d3acd4e0bcb3aee7f828ffcc8fc648aed5`
- Source cohort: 179 videos / 1,790 tasks from official balanced-train
- Purpose: portable audit of frozen predictions, gates, receipt hashes, costs,
  and aggregate statistics; not raw-provider-call reproduction

`source_induction_audit_inputs_v1.tar.gz` contains the frozen source-only
rollouts/configuration needed by the source-algebra and Harness tests.

- Archive size: 2,514,408 bytes
- SHA-256: `27176a4833de1a4b0ca9b599d95fbc07e9ea3e9c2f07b1506a0cdf45c2af3720`
- Purpose: reproduce held-out source induction, shuffled controls, and the
  frozen AGQA capability interface without depending on an ignored `runs/`
  directory

Run from a fresh checkout:

```bash
bash scripts/reproduce_two_video_transfer_v2.sh "$PWD"
```

The script checks the archive hash, extracts it into a temporary directory,
recomputes the evidence bundle exactly, runs the targeted unit tests, and
removes only that temporary directory.
