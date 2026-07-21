"""Unit tests for ``trainer.coevolution._crafter_hook``.

Covers:

* helpers: ``corpus_for_game`` mapping; ``_synthesize_failures`` for both
  trainer signals (OUTCOME_FAILURE, NO_SKILL_BOUND); ``_to_offline_row``
  shape for each ``BankMutationProposal`` subclass; bank-seeding from
  legacy JSONL.
* end-to-end: ``run_crafter_step`` over a synthetic ``EpisodeResult``
  fixture against a per-game legacy ``skill_bank.jsonl`` written from
  the real Airstriker bank. Confirms the JSONL output is parseable by
  ``decide_promotion_gpt54.py::_OfflineProposal.from_json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

from data_structure.extensions.bank_mutation_proposal import (
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.skill_record import SkillContract
from trainer.coevolution._crafter_hook import (
    ALL_FIVE_DOMAINS,
    DEFAULT_OUTCOME_FAILURE_THRESHOLD,
    _record_from_bank_entry,
    _seed_repo_from_legacy_jsonl,
    _synthesize_failures,
    _to_offline_row,
    corpus_for_domain_or_game,
    corpus_for_game,
    run_crafter_step,
)


# ---------------------------------------------------------------------------
# Lightweight EpisodeResult fixture (avoids importing the heavy
# trainer.coevolution.episode_runner module which pulls in env_wrappers).
# ---------------------------------------------------------------------------


@dataclass
class FakeEpisodeResult:
    game: str
    episode_id: str
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    experiences: List[Dict[str, Any]] = field(default_factory=list)


def _exp(step: int, *, reward: float = 0.0, skill_id: Any = None) -> Dict[str, Any]:
    return {
        "step": step,
        "state": f"s{step}",
        "action": "0",
        "reward": reward,
        "raw_env_reward": reward,
        "next_state": f"s{step + 1}",
        "done": False,
        "intention": "",
        "summary_state": "",
        "skill_id": skill_id,
    }


# ---------------------------------------------------------------------------
# corpus_for_game
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game,expected", [
    ("Temporal_Airstriker-v0", "gym_v"),
    ("Temporal_Strider-v0", "gym_v"),
    ("Temporal_MortalKombatII-v0", "gym_v"),
    ("tetris", "env_wrappers"),
    ("twenty_forty_eight", "env_wrappers"),
    ("candy_crush", "env_wrappers"),
    ("super_mario", "env_wrappers"),
    ("Avalon", "env_wrappers"),  # unknown is not gym_v unless prefixed
])
def test_corpus_for_game(game, expected):
    assert corpus_for_game(game) == expected


# ---------------------------------------------------------------------------
# _synthesize_failures
# ---------------------------------------------------------------------------


def test_synthesize_outcome_failure_fires_when_total_reward_zero():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=3, total_reward=0.0,
        experiences=[_exp(0), _exp(1), _exp(2)],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=DEFAULT_OUTCOME_FAILURE_THRESHOLD,
        max_failures=8, bank_was_available=False,
    )
    assert len(out) == 1
    t = out[0]
    assert t.failure_class == "INVARIANT_VIOLATION"
    assert t.failed_step_index == 2
    assert t.extra["synthesis_signal"] == "OUTCOME_FAILURE"
    assert t.extra["total_reward"] == 0.0
    assert t.skill_id == ""        # nothing was bound


def test_synthesize_outcome_failure_picks_most_frequent_bound_skill():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=4, total_reward=0.0,
        experiences=[
            _exp(0, skill_id="ATTACK"),
            _exp(1, skill_id="ATTACK"),
            _exp(2, skill_id="DEFEND"),
            _exp(3, skill_id="ATTACK"),
        ],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=True,
    )
    outcome = next(o for o in out if o.extra["synthesis_signal"] == "OUTCOME_FAILURE")
    assert outcome.skill_id == "ATTACK"


def test_synthesize_skips_outcome_failure_above_threshold():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=2, total_reward=5.0,
        experiences=[_exp(0), _exp(1)],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=False,
    )
    assert out == []


def test_synthesize_no_skill_bound_only_when_bank_available():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=3, total_reward=10.0,
        experiences=[
            _exp(0, skill_id="ATTACK"),
            _exp(1, skill_id=None),
            _exp(2, skill_id=""),
        ],
    )
    # bank not available: no NO_SKILL_BOUND at all
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=False,
    )
    assert out == []

    # bank available: 2 NO_SKILL_BOUND traces (steps 1, 2)
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=True,
    )
    assert len(out) == 2
    for t in out:
        assert t.failure_class == "MISSING_ADAPTER"
        assert t.extra["synthesis_signal"] == "NO_SKILL_BOUND"


def test_synthesize_caps_at_max_failures():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=20, total_reward=0.0,                 # 1 OUTCOME_FAILURE
        experiences=[_exp(i, skill_id=None) for i in range(20)],   # 20 NO_SKILL_BOUND
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=5, bank_was_available=True,
    )
    assert len(out) == 5
    # OUTCOME_FAILURE comes first (severity ordering).
    assert out[0].extra["synthesis_signal"] == "OUTCOME_FAILURE"


def test_synthesize_records_truncated_flag():
    ep = FakeEpisodeResult(
        game="tetris", episode_id="ep1", steps=10,
        total_reward=-1.0, truncated=True,
        experiences=[_exp(i) for i in range(10)],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=False,
    )
    assert len(out) == 1
    assert out[0].extra["truncated"] is True
    assert "(truncated)" in out[0].abort_reason


# ---------------------------------------------------------------------------
# _to_offline_row — every BankMutationProposal subclass
# ---------------------------------------------------------------------------


def test_to_offline_row_patch():
    p = PatchProposal(
        proposal_id="prop-1", rationale="test patch",
        target_domains=list(ALL_FIVE_DOMAINS),
        base_skill_id="COMMIT__ATTACK",
        recovery_strategy="precondition",
        patched_contract=SkillContract(expected_evidence_roles=["COMMIT"]),
        seed_failure_ids=["fail-a", "fail-b"],
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_id"] == "prop-1"
    assert row["proposal_kind"] == "patch"
    assert row["proposer"] == "reflector"
    assert row["target_skill_id"] == "COMMIT__ATTACK"
    assert row["patch_kind"] == "precondition"
    assert row["evidence_role"] == "COMMIT"
    assert row["seed_failure_ids"] == ["fail-a", "fail-b"]
    assert sorted(row["adapter_plan"].keys()) == sorted(ALL_FIVE_DOMAINS)
    assert row["adapter_plan"]["gymv"]["strategy"] == "reuse"
    assert row["adapter_plan"]["browser"]["strategy"] == "synthesize_from_slot_ontology"


def test_to_offline_row_retire():
    p = RetireProposal(
        proposal_id="prop-2", rationale="evidence-starved",
        target_domains=list(ALL_FIVE_DOMAINS),
        target_skill_id="STALE__SKILL", reason="evidence-starved",
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "retire"
    assert row["proposer"] == "reflector"
    assert row["target_skill_id"] == "STALE__SKILL"
    assert row["retire_reason"] == "evidence-starved"


def test_to_offline_row_compose():
    p = ComposeProposal(
        proposal_id="prop-3", rationale="hot pair",
        target_domains=list(ALL_FIVE_DOMAINS),
        component_skill_ids=["A", "B", "C"],
        contract=SkillContract(expected_evidence_roles=["GATHER"]),
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "compose"
    assert row["proposer"] == "composer"
    assert row["components"] == ["A", "B", "C"]
    assert row["compose_op"] == "sequence"
    assert row["evidence_role"] == "GATHER"


def test_to_offline_row_generalize():
    p = GeneralizeProposal(
        proposal_id="prop-4", rationale="single-domain mature",
        target_domains=list(ALL_FIVE_DOMAINS),
        base_skill_id="MATURE__SKILL",
        source_domain="gymv", target_domain="browser",
        slot_remap={"tile": "node"},
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "transfer"
    assert row["proposer"] == "generalizer"
    assert row["source_skill_id"] == "MATURE__SKILL"
    assert row["source_domain"] == "gymv"
    assert row["new_adapter_per_target"] == {"browser": True}
    assert row["slot_remap_per_target"] == {"browser": {"tile": "node"}}


def test_to_offline_row_hypothesis():
    p = HypothesisProposal(
        proposal_id="prop-5", rationale="gap",
        target_domains=list(ALL_FIVE_DOMAINS),
        name="hypothesis-skill",
        contract=SkillContract(expected_evidence_roles=["REASON"]),
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "hypothesize"
    assert row["proposer"] == "hypothesizer"
    assert row["new_skill_name"] == "hypothesis-skill"
    assert row["evidence_role"] == "REASON"


# ---------------------------------------------------------------------------
# Bank seeding
# ---------------------------------------------------------------------------


def test_record_from_bank_entry_returns_none_for_malformed():
    assert _record_from_bank_entry({}, "gymv") is None
    assert _record_from_bank_entry({"skill": "string"}, "gymv") is None
    assert _record_from_bank_entry({"skill": {"name": "no-id"}}, "gymv") is None


def test_record_from_bank_entry_round_trip(tmp_path: Path):
    entry = {
        "skill": {
            "skill_id": "COMMIT/ATTACK",
            "name": "Attack",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {
                "preconditions": ["enemy_visible"],
                "steps": ["press fire", "wait"],
                "success_criteria": ["volley_complete"],
                "abort_criteria": ["player_dead"],
            },
            "contract": {
                "eff_add": ["volley_complete"],
                "eff_del": [],
            },
        },
        "report": {},
    }
    rec = _record_from_bank_entry(entry, "gymv")
    assert rec is not None
    # `/` was rewritten to `__` for filesystem safety.
    assert rec.skill_id == "COMMIT__ATTACK"
    assert rec.feasible_domains == ["gymv"]
    assert len(rec.protocol) == 2
    # NL strings were lifted into typed hops.
    assert rec.protocol[0]["notes"] == "press fire"
    assert rec.contract.preconditions == ["enemy_visible"]
    assert rec.contract.effects_add == ["volley_complete"]
    assert rec.contract.expected_evidence_roles == ["COMMIT"]


def test_seed_repo_from_legacy_jsonl_seeds_only_valid_lines(tmp_path: Path):
    from skill_bank.lifecycle import SkillLifecycleManager
    from skill_bank.repository import SkillRepository
    from skill_bank.stores import SkillStore, StoreName

    bank_path = tmp_path / "skill_bank.jsonl"
    valid_a = {
        "skill": {
            "skill_id": "A",
            "name": "A",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["a step"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }
    valid_b = {
        "skill": {
            "skill_id": "B",
            "name": "B",
            "evidence_role": "GATHER",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["b step"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }
    with bank_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(valid_a) + "\n")
        f.write("not-json\n")
        f.write("\n")
        f.write(json.dumps(valid_b) + "\n")
        f.write(json.dumps({"skill": {}}) + "\n")            # malformed: no skill_id

    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)

    n = _seed_repo_from_legacy_jsonl(
        lifecycle=lifecycle, bank_path=bank_path, default_domain="gymv",
    )
    assert n == 2
    assert {r.skill_id for r in repo.candidates()} == {"A", "B"}


def test_seed_repo_returns_zero_for_missing_file(tmp_path: Path):
    from skill_bank.lifecycle import SkillLifecycleManager
    from skill_bank.repository import SkillRepository
    from skill_bank.stores import SkillStore, StoreName

    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)
    n = _seed_repo_from_legacy_jsonl(
        lifecycle=lifecycle, bank_path=tmp_path / "nope.jsonl",
        default_domain="gymv",
    )
    assert n == 0


# ---------------------------------------------------------------------------
# End-to-end run_crafter_step
# ---------------------------------------------------------------------------


def _real_airstriker_bank() -> Path:
    """Locate the live cold-start Airstriker bank if present."""
    root = (
        Path(__file__).resolve().parent.parent
        / "labeling" / "skill_bank_out"
    )
    if not root.is_dir():
        return Path()
    pair = root / "run_20260430_030637" / "gym_v" / "Temporal_Airstriker-v0"
    bank_path = pair / "skill_bank.jsonl"
    return bank_path if bank_path.is_file() else Path()


def test_run_crafter_step_no_signal_short_circuits(tmp_path: Path):
    """Healthy episodes (positive reward, all skills bound) produce zero
    proposals and an empty proposals.jsonl file."""
    bank_path = tmp_path / "bank" / "skill_bank.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({
        "skill": {
            "skill_id": "S1",
            "name": "S1",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["a"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }) + "\n", encoding="utf-8")

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=2, total_reward=10.0,
        experiences=[_exp(0, skill_id="S1"), _exp(1, skill_id="S1")],
    )

    report = run_crafter_step(
        step=0,
        run_dir=tmp_path,
        rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
    )
    assert report.n_episodes_reflected == 1
    assert report.n_failure_traces == 0
    assert report.n_proposals == 0
    out = (tmp_path / "crafter_proposals_out" / "step_0000"
           / "gym_v" / "Temporal_Airstriker-v0" / "proposals.jsonl")
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == ""


def test_run_crafter_step_outcome_failure_emits_proposals(tmp_path: Path):
    """A failed episode against a real seeded bank should produce ≥1
    PatchProposal that round-trips through the offline schema parser."""
    real_bank = _real_airstriker_bank()
    if not real_bank.is_file():
        pytest.skip("no real Airstriker bank on disk")

    # Copy bank into tmp (test stays read-only against the input).
    bank_path = tmp_path / "bank" / "Temporal_Airstriker-v0" / "skill_bank.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(real_bank.read_text(encoding="utf-8"), encoding="utf-8")

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep-real",
        steps=4, total_reward=0.0,                       # OUTCOME_FAILURE
        experiences=[
            _exp(0, skill_id="COMMIT__ATTACK"),
            _exp(1, skill_id="COMMIT__ATTACK"),
            _exp(2, skill_id=None),                      # NO_SKILL_BOUND
            _exp(3, skill_id="COMMIT__ATTACK"),
        ],
    )

    report = run_crafter_step(
        step=1,
        run_dir=tmp_path,
        rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
        # Lane-(b) opt-in: this test asserts a PatchProposal lands;
        # under the lane-(a) default (T1.3a) the dispatch falls through
        # to the Hypothesizer instead. Both paths are exercised: see
        # test_run_crafter_step_lane_a_default_routes_to_hypothesizer.
        enable_protocol_patching=True,
    )
    assert report.n_episodes_reflected == 1
    assert report.n_failure_traces >= 1                  # at least OUTCOME_FAILURE
    assert "Temporal_Airstriker-v0" in report.proposals_per_game
    out = report.proposals_jsonl_paths["Temporal_Airstriker-v0"]
    assert out.is_file()

    # Validate every emitted row parses through the consumer schema.
    from labeling_supplement.decide_promotion_gpt54 import _OfflineProposal
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == report.n_proposals
    for row in rows:
        op = _OfflineProposal.from_json(row)
        assert op.proposal_id
        assert op.proposal_kind in {"patch", "retire", "compose", "transfer", "hypothesize"}
        assert op.target_domains == list(ALL_FIVE_DOMAINS)


def test_run_crafter_step_writes_step_summary(tmp_path: Path):
    bank_path = tmp_path / "bank" / "skill_bank.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({
        "skill": {
            "skill_id": "S1", "name": "S1",
            "evidence_role": "COMMIT", "applicable_domains": ["gymv"],
            "protocol": {"steps": ["a"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }) + "\n", encoding="utf-8")

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep-s",
        steps=1, total_reward=0.0, experiences=[_exp(0)],
    )
    run_crafter_step(
        step=42, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=False,
    )
    summary_path = (
        tmp_path / "crafter_proposals_out" / "step_0042" / "_step_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["step"] == 42
    assert summary["n_games"] == 1
    assert summary["n_episodes_reflected"] == 1
    assert "params" in summary


def test_run_crafter_step_skips_unknown_games_gracefully(tmp_path: Path):
    """Episodes for a game not in legacy_bank_paths are skipped, not crashed."""
    ep = FakeEpisodeResult(
        game="unknown_game", episode_id="ep-x",
        steps=1, total_reward=-1.0, experiences=[_exp(0)],
    )
    report = run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={},                         # no banks
        bank_was_available=False,
    )
    assert report.n_proposals == 0
    assert report.n_episodes_reflected == 0


def test_run_crafter_step_skips_sentinel_results(tmp_path: Path):
    sentinel = FakeEpisodeResult(game="__SENTINEL__", episode_id="__SENTINEL__")
    bank_path = tmp_path / "skill_bank.jsonl"
    bank_path.write_text("", encoding="utf-8")

    report = run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[sentinel],
        legacy_bank_paths={"__SENTINEL__": bank_path},
        bank_was_available=False,
    )
    assert report.n_episodes_reflected == 0
    assert report.n_proposals == 0


# ---------------------------------------------------------------------------
# Regression: Path 3 (internal LLM hooks) and Path 4 (per-domain failure
# synthesisers) wiring. These two kwargs were added 2026-05 to let
# CoEvolutionConfig opt the production trainer into the same LLM-backed
# Repairer/Hypothesizer/Diagnoser the offline reflect script uses, and to
# route VR / video / browser / ALFWorld episodes to per-domain synthesisers
# instead of the gymv fallback. The tests below pin three properties:
#
# 1. ``install_internal_llm_hooks=False`` is byte-identical to never having
#    added the kwarg (defaults preserve legacy behaviour).
# 2. ``install_internal_llm_hooks=True`` actually invokes
#    ``crafter._llm_runtime.install_llm_hooks`` once per game (patched via
#    monkeypatch — we never want to reach the network during unit tests).
# 3. ``episode_domain_per_game={game: "visual_reasoning"}`` causes
#    ``_synthesize_failures`` to dispatch to
#    ``labeling_supplement._failure_synth.get_synthesizer("visual_reasoning")``
#    for an episode carrying ``raw_sample`` (no gymv shape required).
# ---------------------------------------------------------------------------


@dataclass
class FakeVREpisodeResult(FakeEpisodeResult):
    raw_sample: Dict[str, Any] = field(default_factory=dict)


def _make_minimal_bank(tmp_path: Path) -> Path:
    """Tiny non-empty bank — content doesn't matter; we just need
    ``_seed_repo_from_legacy_jsonl`` to leave the SkillCrafterService
    with at least one CANDIDATE so the Path 3 install path runs."""
    bank_path = tmp_path / "bank" / "Temporal_Airstriker-v0" / "skill_bank.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({
        "skill": {
            "skill_id": "S_TEST",
            "name": "S_TEST",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["a"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }) + "\n", encoding="utf-8")
    return bank_path


def test_run_crafter_step_path3_default_off_does_not_call_installer(
    tmp_path: Path, monkeypatch
):
    """When ``install_internal_llm_hooks`` is the default ``False``, the
    LLM-runtime installer must not be imported / called — even when there
    are failure proposals to mint. Guards against accidental opt-in."""
    bank_path = _make_minimal_bank(tmp_path)

    calls: List[str] = []

    def _spy(*args, **kwargs):
        calls.append("install_called")
        return {"installed": True, "model": "spy", "hooks": []}

    monkeypatch.setattr(
        "crafter._llm_runtime.install_llm_hooks", _spy, raising=False,
    )

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep-default",
        steps=2, total_reward=0.0,                       # OUTCOME_FAILURE
        experiences=[
            _exp(0, skill_id="S_TEST"),
            _exp(1, skill_id="S_TEST"),
        ],
    )
    run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
        # NB: install_internal_llm_hooks omitted — defaults to False.
    )
    assert calls == [], (
        "install_llm_hooks must not be called when "
        "install_internal_llm_hooks is left at its default False; "
        f"got {calls}"
    )


def test_run_crafter_step_path3_install_invokes_runtime(
    tmp_path: Path, monkeypatch
):
    """When ``install_internal_llm_hooks=True`` the hook MUST call
    ``crafter._llm_runtime.install_llm_hooks`` — once per game with a
    failed episode — forwarding the model id and the three enable flags.

    We monkeypatch the installer so the test never touches the network.
    """
    bank_path = _make_minimal_bank(tmp_path)

    captured: List[Dict[str, Any]] = []

    def _spy(service, *, model, audit_sink, enable_diagnoser,
             enable_repairer, enable_hypothesizer):
        captured.append({
            "model": model,
            "enable_diagnoser": enable_diagnoser,
            "enable_repairer": enable_repairer,
            "enable_hypothesizer": enable_hypothesizer,
            "audit_sink_callable": callable(audit_sink),
            "service_class": type(service).__name__,
        })
        return {
            "installed": True, "model": model,
            "hooks": ["repairer"] if enable_repairer else [],
        }

    monkeypatch.setattr(
        "crafter._llm_runtime.install_llm_hooks", _spy, raising=False,
    )

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep-llm-on",
        steps=2, total_reward=0.0,
        experiences=[
            _exp(0, skill_id="S_TEST"),
            _exp(1, skill_id="S_TEST"),
        ],
    )
    run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
        install_internal_llm_hooks=True,
        internal_llm_model="gpt-5.4",
        internal_llm_enable_repairer=True,
        internal_llm_enable_hypothesizer=False,
        internal_llm_enable_diagnoser=False,
    )
    assert len(captured) == 1, f"expected one install call, got {captured}"
    rec = captured[0]
    assert rec["model"] == "gpt-5.4"
    assert rec["enable_repairer"] is True
    assert rec["enable_hypothesizer"] is False
    assert rec["enable_diagnoser"] is False
    assert rec["audit_sink_callable"] is True
    assert rec["service_class"] == "SkillCrafterService"


def test_run_crafter_step_path3_install_failure_is_soft(
    tmp_path: Path, monkeypatch
):
    """If the LLM-runtime installer raises, the hook must swallow
    the error, log a warning, and let the deterministic Crafter path
    proceed. Otherwise a single bad config could brick a training step."""
    bank_path = _make_minimal_bank(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated installer failure")

    monkeypatch.setattr(
        "crafter._llm_runtime.install_llm_hooks", _boom, raising=False,
    )

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep-llm-boom",
        steps=2, total_reward=0.0,
        experiences=[
            _exp(0, skill_id="S_TEST"),
            _exp(1, skill_id="S_TEST"),
        ],
    )
    report = run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
        install_internal_llm_hooks=True,
        internal_llm_model="anything",
    )
    assert report.n_episodes_reflected == 1, (
        "Path 3 installer failure must not skip the episode — the "
        "deterministic chain should still process it."
    )


def test_run_crafter_step_path4_episode_domain_routes_to_synthesiser(
    tmp_path: Path, monkeypatch
):
    """When ``episode_domain_per_game`` maps a game to a transfer-target
    domain (e.g. visual_reasoning) AND the episode carries a ``raw_sample``
    dict, ``_synthesize_failures`` MUST dispatch to the per-domain
    synthesiser registered in ``labeling_supplement._failure_synth``
    (instead of the gymv heuristic)."""
    bank_path = _make_minimal_bank(tmp_path)

    seen: List[Dict[str, Any]] = []

    def _fake_synth(sample, *, domain, sample_id, max_failures):
        seen.append({
            "domain": domain, "sample_id": sample_id,
            "max_failures": max_failures, "sample_keys": sorted(sample.keys()),
        })
        # Return one trivially-shaped FailureTrace so the rest of the
        # pipeline has at least one row to work with. Only the fields
        # actually defined on the FailureTrace dataclass — extra
        # keyword args would TypeError and (because
        # ``_synthesize_failures`` wraps the call in try/except) would
        # be silently swallowed, masking the real wiring bug.
        from data_structure.extensions.failure_trace import FailureTrace
        return [FailureTrace(
            skill_id="vr_skill",
            skill_episode_id=f"{sample_id}#vr0",
            domain=domain,
            failed_step_index=0,
            failure_class="WRONG_ANSWER",
            abort_reason="fake",
        )]

    monkeypatch.setattr(
        "labeling_supplement._failure_synth.get_synthesizer",
        lambda domain: _fake_synth,
    )

    ep = FakeVREpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="vr-ep-1",
        steps=1, total_reward=0.0,
        raw_sample={"task_id": "vtb_42", "predicted": "x", "expected": "y"},
    )

    report = run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
        episode_domain_per_game={"Temporal_Airstriker-v0": "visual_reasoning"},
    )
    assert len(seen) == 1, f"expected one synth dispatch, got {seen}"
    assert seen[0]["domain"] == "visual_reasoning"
    assert seen[0]["sample_id"] == "vr-ep-1"
    assert "task_id" in seen[0]["sample_keys"]
    # The fake synth emitted 1 FailureTrace, so the report must reflect it.
    assert report.n_failure_traces >= 1


def test_run_crafter_step_path4_no_raw_sample_skips_silently(
    tmp_path: Path, monkeypatch
):
    """When ``episode_domain_per_game`` resolves a transfer-target domain
    BUT the episode lacks ``raw_sample``, ``_synthesize_failures`` MUST
    NOT fall through to the gymv path (would produce a domain-mislabelled
    failure) — it should skip and emit zero failures for that episode."""
    bank_path = _make_minimal_bank(tmp_path)

    called: List[str] = []
    monkeypatch.setattr(
        "labeling_supplement._failure_synth.get_synthesizer",
        lambda domain: (called.append(domain) or (lambda **kw: [])),
    )

    ep = FakeEpisodeResult(                                   # NO raw_sample
        game="Temporal_Airstriker-v0", episode_id="ep-no-raw",
        steps=1, total_reward=0.0,
        experiences=[_exp(0, skill_id="S_TEST")],
    )
    report = run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
        episode_domain_per_game={"Temporal_Airstriker-v0": "visual_reasoning"},
    )
    assert called == [], (
        "transfer-target dispatch must NOT call the synthesiser when "
        "raw_sample is missing — the gymv heuristic does not transfer."
    )
    assert report.n_failure_traces == 0


# ---------------------------------------------------------------------------
# Regression: corpus_for_domain_or_game (Path 4 corpus routing)
#
# The promotion gate's ``_discover_pairs`` walks one bucket per
# ``CORPORA`` entry; before 2026-05 the trainer only emitted ``gym_v``
# and ``env_wrappers``, and any Path-4 transfer-target proposal was
# silently mis-folded into ``env_wrappers/<game>/`` (which then forced
# the LLM judge to evaluate a VR-derived patch as if it were a 2048
# patch — see _smoke_attr_v2/_attribution_summary.md "cross-domain
# mismatch" note).  ``corpus_for_domain_or_game`` fixes this seam by
# routing transfer-target proposals to the new ``visual_reasoning``
# bucket.  Tests below pin both the legacy fall-through and the new
# transfer-target dispatch.
# ---------------------------------------------------------------------------


def test_corpus_for_domain_or_game_legacy_paths_unchanged():
    """For domain ∈ {gymv, "", None} the dispatch must match
    ``corpus_for_game`` exactly — no behaviour change for any caller
    that hasn't opted into Path 4."""
    cases = [
        ("Temporal_Airstriker-v0", "gym_v"),
        ("Temporal_DynamiteHeaddy-v0", "gym_v"),
        ("twenty_forty_eight", "env_wrappers"),
        ("super_mario", "env_wrappers"),
    ]
    for game, expected in cases:
        assert corpus_for_game(game) == expected
        for legacy_domain in ("gymv", "", None):
            assert corpus_for_domain_or_game(legacy_domain, game) == expected, (
                f"domain={legacy_domain!r} game={game} should fall through "
                f"to {expected!r} but got "
                f"{corpus_for_domain_or_game(legacy_domain, game)!r}"
            )


def test_corpus_for_domain_or_game_transfer_targets_route_to_vr():
    """All four transfer-target domains must collapse onto the single
    ``visual_reasoning`` corpus bucket regardless of game name (the
    game string remains the per-source segment INSIDE the bucket)."""
    transfer_domains = ("visual_reasoning", "video", "browser", "alfworld")
    games = (
        "Temporal_Airstriker-v0",
        "twenty_forty_eight",
        "vtb_easy",
        "siv_bench_holdout",
    )
    for d in transfer_domains:
        for g in games:
            corpus = corpus_for_domain_or_game(d, g)
            assert corpus == "visual_reasoning", (
                f"domain={d!r} game={g!r} should map to "
                f"'visual_reasoning' but got {corpus!r}"
            )


def test_corpus_for_domain_or_game_writes_under_vr_bucket(tmp_path: Path):
    """End-to-end: when run_crafter_step's resolved domain is a transfer
    target AND the synthesiser emits at least one failure, the on-disk
    proposals.jsonl must land under
    ``<run_dir>/crafter_proposals_out/step_<n>/visual_reasoning/<game>/``,
    NOT ``env_wrappers/<game>/``.  This guards against a regression
    in ``_write_proposals_jsonl`` where the corpus path was computed
    from the game name alone."""
    bank_path = _make_minimal_bank(tmp_path)

    # Use the Path-4 dispatch hook to force a non-empty failure trace
    # without depending on the on-disk failure-synth registry.
    import pytest as _pytest  # noqa: F401  (avoid F841 lint on unused symbol path)

    from data_structure.extensions.failure_trace import FailureTrace as _FT

    def _fake_synth(sample, *, domain, sample_id, max_failures):
        return [_FT(
            skill_id="vr_skill", skill_episode_id=f"{sample_id}#vr0",
            domain=domain, failed_step_index=0,
            failure_class="WRONG_ANSWER", abort_reason="test",
        )]

    import labeling_supplement._failure_synth as _fs
    orig = getattr(_fs, "get_synthesizer", None)

    try:
        _fs.get_synthesizer = lambda d: _fake_synth                # type: ignore[attr-defined]
        ep = FakeVREpisodeResult(
            game="Temporal_Airstriker-v0", episode_id="vr-ep-corpus",
            steps=1, total_reward=0.0,
            raw_sample={"task_id": "vtb_99"},
        )
        report = run_crafter_step(
            step=7, run_dir=tmp_path, rollout_results=[ep],
            legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
            bank_was_available=True,
            episode_domain_per_game={
                "Temporal_Airstriker-v0": "visual_reasoning",
            },
            # Force failure-driven dispatch threshold so even the rule
            # path will mint *something* off our fake trace.
            hypothesize_min_recurrences=1,
        )
    finally:
        if orig is not None:
            _fs.get_synthesizer = orig                              # type: ignore[attr-defined]

    assert report.n_failure_traces >= 1
    expected = (
        tmp_path / "crafter_proposals_out" / "step_0007"
        / "visual_reasoning" / "Temporal_Airstriker-v0"
        / "proposals.jsonl"
    )
    assert expected.is_file(), (
        f"expected proposals.jsonl under VR bucket at {expected}, but "
        f"got tree:\n{sorted((tmp_path / 'crafter_proposals_out').rglob('proposals.jsonl'))}"
    )
    legacy = (
        tmp_path / "crafter_proposals_out" / "step_0007"
        / "env_wrappers" / "Temporal_Airstriker-v0"
        / "proposals.jsonl"
    )
    assert not legacy.exists(), (
        "Path-4 episode must NOT also write to the legacy env_wrappers "
        "bucket — that would be exactly the 'cross-domain fold' bug "
        f"corpus_for_domain_or_game was added to fix.  Found: {legacy}"
    )
