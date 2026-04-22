"""Backbone model invariant test.

Pins the project-wide backbone to GPT-4o (current phase) and verifies
every key surface defaults to it. The 8B / 32B / 72B Qwen tracks are
deferred and must not appear as a *runtime default* anywhere we control.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.models import (
    BACKBONE_JUDGE_MODEL,
    BACKBONE_MODEL,
    BACKBONE_TEACHER_MODEL,
    DEFERRED_MODELS,
    assert_default_is_gpt4o,
    is_deferred,
)


class TestBackboneModelDefaults:
    def test_backbone_is_gpt4o(self) -> None:
        assert BACKBONE_MODEL == "gpt-4o", BACKBONE_MODEL
        assert_default_is_gpt4o()

    def test_teacher_defaults_to_backbone(self) -> None:
        # In the current phase teacher == backbone (both gpt-4o).
        assert BACKBONE_TEACHER_MODEL == BACKBONE_MODEL

    def test_judge_defaults_to_backbone(self) -> None:
        assert BACKBONE_JUDGE_MODEL == BACKBONE_MODEL

    def test_deferred_models_are_not_default(self) -> None:
        assert BACKBONE_MODEL not in DEFERRED_MODELS
        assert BACKBONE_TEACHER_MODEL not in DEFERRED_MODELS
        assert BACKBONE_JUDGE_MODEL not in DEFERRED_MODELS
        assert is_deferred("Qwen/Qwen3-8B")
        assert is_deferred("Qwen/Qwen2.5-72B")
        assert not is_deferred("gpt-4o")


class TestOrchestratorConfigUsesBackbone:
    def test_teacher_config_default(self) -> None:
        from orchestrator.config import TeacherConfig

        assert TeacherConfig().model_name == BACKBONE_MODEL

    def test_judge_config_default(self) -> None:
        from orchestrator.config import JudgeConfig

        assert JudgeConfig().model_name == BACKBONE_MODEL

    def test_orchestrator_config_default(self) -> None:
        from orchestrator.config import OrchestratorConfig

        cfg = OrchestratorConfig()
        assert cfg.backbone_model == BACKBONE_MODEL
        assert cfg.teacher.model_name == BACKBONE_MODEL
        assert cfg.judge.model_name == BACKBONE_MODEL


class TestCrafterUsesBackbone:
    def test_crafter_service_teacher_default(self, tmp_path) -> None:
        from crafter import SkillCrafterService
        from orchestrator import ArtifactStore
        from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
        from skill_bank.stores import StoreName

        repo = SkillRepository(
            draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
            candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "candidate")),
            active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
            archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "archive")),
        )
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)
        # The crafter holds the canonical teacher model on its private slot.
        assert crafter._teacher == BACKBONE_MODEL  # noqa: SLF001


class TestDecisionAgentDefaults:
    """Ensure live decision-agent surfaces also default to GPT-4o.

    The legacy `decision_agents` package has optional heavy deps
    (`google.genai`, `vllm`, etc.) so we check the source text rather
    than the imported module — the check stays meaningful in lean
    test environments and remains tied to the literal that ships.
    """

    def test_actor_agent_default(self) -> None:
        path = os.path.join(_ROOT, "decision_agents", "actor_agent.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'DEFAULT_MODEL: str = "gpt-4o"' in src, (
            "decision_agents.actor_agent.DEFAULT_MODEL is not gpt-4o; "
            "see common/models.py for the project-wide backbone."
        )
        assert 'DEFAULT_MODEL: str = "gpt-4o-mini"' not in src

    def test_agent_helper_default(self) -> None:
        path = os.path.join(_ROOT, "decision_agents", "agent_helper.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'DEFAULT_LLM_MODEL: str = "gpt-4o"' in src
        assert 'DEFAULT_LLM_MODEL: str = "gpt-4o-mini"' not in src

    def test_vlm_decision_agent_default(self) -> None:
        path = os.path.join(_ROOT, "decision_agents", "agent.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'DEFAULT_MODEL = "gpt-4o"' in src
        assert 'DEFAULT_MODEL = "gpt-4o-mini"' not in src


class TestApiFuncRoutingDefault:
    """The central LLM router must default to gpt-4o when called with
    `model=None` (already true historically; this guards regression)."""

    def test_ask_model_defaults_to_gpt4o(self) -> None:
        path = os.path.join(_ROOT, "API_func.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'model = "gpt-4o"' in src, (
            "API_func.ask_model lost its gpt-4o default."
        )


class TestEnvOverridePath:
    """Confirm the `VLM_AGENT_BACKBONE_MODEL` override path is honored at
    import time without breaking when unset."""

    def test_override_present_when_env_set(self, monkeypatch) -> None:
        monkeypatch.setenv("VLM_AGENT_BACKBONE_MODEL", "gpt-4o-2024-11-20")
        # Re-import in a fresh module namespace.
        import importlib

        import common.models as m

        importlib.reload(m)
        try:
            assert m.BACKBONE_MODEL == "gpt-4o-2024-11-20"
        finally:
            # Restore default for downstream tests.
            monkeypatch.delenv("VLM_AGENT_BACKBONE_MODEL", raising=False)
            importlib.reload(m)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
