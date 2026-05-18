# Legacy training notes

This directory preserves training plans that have been **superseded by
newer, live designs** but kept for design provenance and ablation
reference. Nothing here should drive a fresh experiment — read the
"Superseded by" pointer to find the current source of truth.

| File | Superseded by | Why moved |
|---|---|---|
| `coevo-3phase-cross-game-ood-transfer-plan.md` (2026-05-03 PM) | `frontier_data/PLAN_GAME_SPLIT_AND_NO_SFT_GRPO.md` §1 + `trainer/coevolution/config.py:{PHASE1_DEFAULT_GAMES,PHASE2_HOLDOUT_GAMES}` | The 2026-05-03 PM roster paired in-genre P1 sources with P2 targets by hand (TF3, AlteredBeast, Columns, DynamiteHeaddy, candy_crush, tetris in Phase 1). The 2026-05-12 mega-skill pipeline ran an exhaustive search over all 1-per-genre assignments to maximise cross-phase mega-skill transfer links (50 vs 43, +16%), which swapped `{AlteredBeast, DynamiteHeaddy}` for `{SoR2, Strider}` in Phase 1 and moved the displaced games to Phase 2. The §11.2 "Option C sequential curriculum with bank+LoRA carry-over" mechanism is still live in `scripts/run_phase1_curriculum.sh`, but it now reads its game roster from `config.py` instead of hard-coding it. The `harness_enabled=False` choice in `config.py` (mega-skill ICL needs cross-domain admit) also means the F2/F2′ task-axis veto discussed in the legacy plan's §11 is no longer the active gate — see `harness/README.md §22.5` for the current "RAG retrieves, harness informs, LLM picks" pipeline.

## Re-measuring the live split

```bash
python frontier_data/scripts/coverage_audit.py
```

dumps the three live headline numbers (H1 Phase-1 family coverage,
H2 Phase-2 transfer-link total, H3 Phase-3 cross-domain coverage)
against the current `mega_skill_clusters.json`. Run this after any
roster change in `config.py` — if the cluster file is stale (`WARN:
roster drift`), re-run `frontier_data/scripts/run_full_pipeline.sh`
stages 4-5.
