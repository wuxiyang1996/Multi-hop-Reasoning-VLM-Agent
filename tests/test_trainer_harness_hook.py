"""Unit tests for ``trainer.coevolution._harness_hook.SkillHarnessHook``.

Covers:

* ``filter_candidates`` admits known PROVISIONAL skills with the
  ``gymv`` adapter and vetoes status / domain / task / adapter
  mismatches.
* The ``RejectedSkillSink`` accumulates the per-step vetoes and
  ``flush_to_lifecycle`` lands them on
  ``SkillRecord.false_binding_patterns`` via
  ``SkillLifecycleManager.record_false_binding_pattern``.
* ``validate_choice`` returns the structured ``ok / veto`` verdict and
  degrades gracefully (admit) for skills the cache doesn't know.
* The hook is a no-op pass-through when the candidate list is empty
  or when every candidate is unknown to the cache.
* ``SkillHarnessHook.for_game`` hydrates a per-game bank.jsonl into
  the cache and is robust to missing or malformed lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.enums import SkillSourceType, SkillStatus, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.adapters.gymv_adapter import GymvAdapter
from harness.eligibility import RejectedSkill
from harness.rejected_skill_sink import RejectedSkillSink
from skill_bank.lifecycle import SkillLifecycleManager
from skill_bank.repository import SkillRepository
from skill_bank.stores import SkillStore, StoreName

from trainer.coevolution._harness_hook import (
    HarnessStepStats,
    SkillHarnessHook,
    _hydrate_records_from_bank,
    _state_for_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_record(
    *,
    skill_id: str,
    name: str = "skill",
    skill_type: SkillType = SkillType.ACTION,
    domain: str = "gymv",
    feasible_tasks=None,
    status: SkillStatus = SkillStatus.PROVISIONAL,
    protocol=None,
    contract: SkillContract = None,
) -> SkillRecord:
    rec = SkillRecord.new(
        name=name,
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=[domain],
        feasible_tasks=list(feasible_tasks or []),
        protocol=list(protocol or [{"action": "EXEC", "payload": {}, "notes": "step"}]),
        contract=contract or SkillContract(),
    )
    object.__setattr__(rec, "skill_id", skill_id)
    object.__setattr__(rec, "status", status)
    return rec


def _state(domain: str = "gymv", task: str = "twenty_forty_eight") -> StateSchema:
    return StateSchema(domain=domain, task=task, evidence=[])


def _cand(skill_id: str, **extra) -> dict:
    """Cold-start RAG candidate dict shape."""
    base = {
        "skill_id": skill_id,
        "skill_name": skill_id,
        "execution_hint": "",
        "protocol": {},
        "confidence": 0.5,
        "relevance": 0.5,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# _state_for_step / state_for_step
# ---------------------------------------------------------------------------


def test_state_for_step_minimal():
    s = _state_for_step(
        game="tetris",
        summary_state="empty=10",
        intention="[CLEAR] tetris",
    )
    assert s.domain == "gymv"
    assert s.task == "tetris"
    assert s.extra["summary_state"].startswith("empty=")
    assert s.extra["intention"].startswith("[CLEAR]")


def test_state_for_step_truncates_long_summary():
    long = "x" * 5000
    s = _state_for_step(game="g", summary_state=long, intention="i")
    assert len(s.extra["summary_state"]) == 1024


def test_hook_state_for_step_uses_hook_domain():
    h = SkillHarnessHook(domain="gymv", records={})
    s = h.state_for_step(game="tetris")
    assert s.domain == "gymv"
    assert s.task == "tetris"


# ---------------------------------------------------------------------------
# filter_candidates — eligibility filter
# ---------------------------------------------------------------------------


def test_filter_admits_runnable_gymv_skill():
    rec = _mk_record(skill_id="s1", status=SkillStatus.PROVISIONAL)
    h = SkillHarnessHook(records={"s1": rec})

    state = _state()
    cands = [_cand("s1")]
    out, diag = h.filter_candidates(cands, state)

    assert len(out) == 1
    assert out[0]["skill_id"] == "s1"
    assert out[0]["_harness_eligible"]["adapter_name"] == "gymv"
    assert diag["n_in"] == 1
    assert diag["n_admitted"] == 1
    assert diag["n_rejected"] == 0
    assert diag["eligible_ids"] == ["s1"]


def test_filter_vetos_draft_status():
    """Skill with status=DRAFT is rejected by F1."""
    rec = _mk_record(skill_id="s_draft")
    object.__setattr__(rec, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"s_draft": rec})

    out, diag = h.filter_candidates([_cand("s_draft")], _state())

    assert out == []
    assert diag["n_admitted"] == 0
    assert diag["n_rejected"] == 1
    assert diag["rejected"][0]["veto"] == "status_not_runnable"
    # Sink absorbed the rejection.
    assert len(h.sink) == 1


def test_filter_vetos_domain_mismatch():
    """Skill whose feasible_domains doesn't include the state domain."""
    rec = _mk_record(skill_id="s_browser", domain="browser")
    h = SkillHarnessHook(records={"s_browser": rec})

    state = _state(domain="gymv")
    out, diag = h.filter_candidates([_cand("s_browser")], state)

    assert out == []
    assert diag["rejected"][0]["veto"] in ("domain_mismatch", "no_adapter")


def test_filter_vetos_task_mismatch():
    """Skill with feasible_tasks=[X] is rejected when state.task != X (F2′)."""
    rec = _mk_record(
        skill_id="s_task",
        feasible_tasks=["tetris"],
    )
    h = SkillHarnessHook(records={"s_task": rec})

    out, diag = h.filter_candidates([_cand("s_task")], _state(task="candy_crush"))
    assert out == []
    assert diag["rejected"][0]["veto"] == "task_mismatch"

    # Same skill on the matching task is admitted.
    out2, diag2 = h.filter_candidates([_cand("s_task")], _state(task="tetris"))
    assert len(out2) == 1
    assert diag2["task_match_distribution"].get("same_task", 0) >= 1


def test_filter_passes_through_unknown_skill_ids():
    h = SkillHarnessHook(records={})
    out, diag = h.filter_candidates([_cand("unknown")], _state())
    # Unknown skills bypass the harness — degrade gracefully.
    assert len(out) == 1
    assert out[0]["skill_id"] == "unknown"
    assert "_harness_eligible" not in out[0]
    assert diag["n_unknown"] == 1


def test_filter_empty_candidates_is_noop():
    h = SkillHarnessHook(records={})
    out, diag = h.filter_candidates([], _state())
    assert out == []
    assert diag["n_in"] == 0


# ---------------------------------------------------------------------------
# validate_choice
# ---------------------------------------------------------------------------


def test_validate_choice_admits_runnable_skill():
    rec = _mk_record(skill_id="s1")
    h = SkillHarnessHook(records={"s1": rec})

    ok, d = h.validate_choice("s1", _state())
    assert ok is True
    assert d["status"] == "ok"
    assert d["skill_id"] == "s1"


def test_validate_choice_vetos_when_adapter_missing():
    """A skill whose domain has no registered adapter is vetoed by validate_invocation."""
    rec = _mk_record(skill_id="s_browser", domain="browser")
    # No browser adapter registered in the default hook.
    h = SkillHarnessHook(records={"s_browser": rec})

    state = StateSchema(domain="browser", task="t", evidence=[])
    ok, d = h.validate_choice("s_browser", state)
    assert ok is False
    assert any("missing_adapter" in s for s in d.get("veto_reasons", []))


def test_validate_choice_unknown_skill_admits():
    h = SkillHarnessHook(records={})
    ok, d = h.validate_choice("not_in_cache", _state())
    assert ok is True
    assert d["status"] == "skill_not_in_cache"


def test_validate_choice_no_skill_id_admits():
    h = SkillHarnessHook(records={})
    ok, d = h.validate_choice(None, _state())
    assert ok is True
    assert d["status"] == "no_skill_id_supplied"


# ---------------------------------------------------------------------------
# Sink → lifecycle drainage (Day-9c integration)
# ---------------------------------------------------------------------------


def test_flush_to_lifecycle_writes_false_binding_patterns(tmp_path: Path):
    """End-to-end: filter rejects a skill, flush_to_lifecycle lands the
    veto on `SkillRecord.false_binding_patterns`."""
    # Build a real lifecycle so the write actually round-trips.
    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "cand")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "arch")),
    )
    lifecycle = SkillLifecycleManager(repo)

    rec_draft = _mk_record(skill_id="s_draft")
    object.__setattr__(rec_draft, "status", SkillStatus.DRAFT)
    # Seed the lifecycle with the same skill_id as the hook will see,
    # so flush_to_lifecycle finds it.
    seed = _mk_record(skill_id="s_draft")
    object.__setattr__(seed, "status", SkillStatus.DRAFT)
    lifecycle.ingest_draft(seed)
    lifecycle.transition(
        "s_draft",
        to_status=SkillStatus.CANDIDATE,
        rationale="seed",
    )

    h = SkillHarnessHook(records={"s_draft": rec_draft})
    h.filter_candidates([_cand("s_draft")], _state())
    assert len(h.sink) == 1

    report = h.flush_to_lifecycle(lifecycle)
    assert report.n_patterns_written == 1
    assert report.n_skills_touched == 1

    after = lifecycle.get("s_draft")
    assert after is not None
    assert len(after.false_binding_patterns) == 1
    fbp = after.false_binding_patterns[0]
    assert fbp["veto"] == "status_not_runnable"
    assert fbp["domain"] == "gymv"
    assert fbp["task"] == "twenty_forty_eight"


def test_flush_skips_skill_ids_not_in_lifecycle(tmp_path: Path):
    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "cand")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "arch")),
    )
    lifecycle = SkillLifecycleManager(repo)

    rec_draft = _mk_record(skill_id="s_orphan")
    object.__setattr__(rec_draft, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"s_orphan": rec_draft})
    h.filter_candidates([_cand("s_orphan")], _state())

    report = h.flush_to_lifecycle(lifecycle)
    assert report.n_patterns_written == 0
    assert "s_orphan" in report.skipped_unknown_skill_ids


# ---------------------------------------------------------------------------
# _hydrate_records_from_bank
# ---------------------------------------------------------------------------


def test_hydrate_records_from_bank_promotes_to_provisional(tmp_path: Path):
    """Skills hydrated from a legacy bank.jsonl come out as PROVISIONAL
    so the eligibility filter admits them."""
    bank_path = tmp_path / "skill_bank.jsonl"
    payload = {
        "skill": {
            "skill_id": "EXEC/CLEAR",
            "name": "exec_clear",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {
                "steps": ["clear the bottom row"],
                "preconditions": [],
                "success_criteria": [],
                "abort_criteria": [],
            },
            "contract": {"eff_add": [], "eff_del": []},
        }
    }
    bank_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    out = _hydrate_records_from_bank(bank_path, default_domain="gymv")
    assert len(out) == 1
    rec = next(iter(out.values()))
    assert rec.status == SkillStatus.PROVISIONAL
    assert "gymv" in rec.feasible_domains


def test_hydrate_records_handles_missing_bank(tmp_path: Path):
    out = _hydrate_records_from_bank(
        tmp_path / "nope.jsonl", default_domain="gymv",
    )
    assert out == {}


def test_hydrate_records_skips_malformed_lines(tmp_path: Path):
    bank_path = tmp_path / "skill_bank.jsonl"
    bank_path.write_text("not-json\n", encoding="utf-8")
    out = _hydrate_records_from_bank(bank_path, default_domain="gymv")
    assert out == {}


# ---------------------------------------------------------------------------
# for_game
# ---------------------------------------------------------------------------


def test_for_game_with_missing_bank_returns_empty_hook(tmp_path: Path):
    h = SkillHarnessHook.for_game(
        game="tetris", bank_path=tmp_path / "absent.jsonl",
    )
    assert h.n_records() == 0
    # Filtering with an empty cache is a graceful no-op.
    out, diag = h.filter_candidates([_cand("anything")], _state())
    assert len(out) == 1
    assert diag["n_unknown"] == 1


def test_for_game_with_real_bank_admits_candidates(tmp_path: Path):
    bank = tmp_path / "skill_bank.jsonl"
    payload = {
        "skill": {
            "skill_id": "STRAT/STACK",
            "name": "stack",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["stack tiles"]},
            "contract": {"eff_add": [], "eff_del": []},
        }
    }
    bank.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    h = SkillHarnessHook.for_game(game="twenty_forty_eight", bank_path=bank)
    assert h.n_records() == 1

    sid = next(iter(h._records))
    out, diag = h.filter_candidates([_cand(sid)], _state(task="twenty_forty_eight"))
    assert len(out) == 1
    assert diag["n_admitted"] == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_aggregate_across_calls():
    rec_ok = _mk_record(skill_id="ok")
    rec_bad = _mk_record(skill_id="bad")
    object.__setattr__(rec_bad, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"ok": rec_ok, "bad": rec_bad})

    h.filter_candidates([_cand("ok"), _cand("bad")], _state())
    h.validate_choice("ok", _state())
    h.validate_choice("bad", _state())  # bad is DRAFT — but validate_invocation
                                          # checks adapter/binding/preconditions,
                                          # not status, so admits
    s = h.stats.to_json()
    assert s["n_candidates_in"] == 2
    assert s["n_candidates_admitted"] == 1
    assert s["n_candidates_rejected"] == 1
    assert "status_not_runnable" in s["veto_class_distribution"]
    assert s["n_validate_ok"] >= 1
