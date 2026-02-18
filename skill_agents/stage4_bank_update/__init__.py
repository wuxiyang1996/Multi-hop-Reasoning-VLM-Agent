"""
Stage 4 — Skill Bank Update: Split / Merge / Refine + fast re-decode.

Maintains a high-quality Skill Bank by applying three update operations:

  **SPLIT**   one skill contains multiple modes → split into child skills
  **MERGE**   two skills are near-duplicates → merge into one canonical skill
  **REFINE**  contracts are too strong / weak → drop fragile or add discriminative literals

Also updates:
  - Duration model p(ℓ|k)
  - Indices (effect inverted index, MinHash/LSH, optional ANN)

Entry point: :func:`run_stage4.run_stage4`.

All public symbols are importable from the sub-modules directly::

    from skill_agents.stage4_bank_update.config import Stage4Config
    from skill_agents.stage4_bank_update.run_stage4 import run_stage4
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency with skill_bank ↔ stage3_mvp."""
    _lazy_map = {
        "Stage4Config": "skill_agents.stage4_bank_update.config",
        "run_stage4": "skill_agents.stage4_bank_update.run_stage4",
        "Stage4Result": "skill_agents.stage4_bank_update.run_stage4",
        "BankDiffReport": "skill_agents.stage4_bank_update.schemas",
        "RedecodeRequest": "skill_agents.stage4_bank_update.schemas",
        "SkillProfile": "skill_agents.stage4_bank_update.schemas",
    }
    if name in _lazy_map:
        import importlib
        mod = importlib.import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Stage4Config",
    "run_stage4",
    "Stage4Result",
    "BankDiffReport",
    "RedecodeRequest",
    "SkillProfile",
]
