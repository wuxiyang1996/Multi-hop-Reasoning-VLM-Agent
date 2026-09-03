from dataclasses import replace

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.retargeting_harness import (
    EffectReceipt,
    ExperimentVerdict,
    FrozenSkillArtifact,
    FrozenSkillExecution,
    GroundingDraft,
    NativeActionCandidate,
    OptionGrounding,
    PermissionBoundTargetHarness,
    RetargetingCondition,
    RetargetingReject,
    SkillArtifactKind,
    SkillDecisionVerdict,
    SourceQualification,
    SymbolicOptionRule,
    TargetHarnessArtifact,
    TargetObservation,
    TargetStep,
    evaluate_retargeting_experiment,
)
from scripts.run_harness_retargeting_smoke import (
    IncidentResponseEnvironment,
    authentic_source_skill,
    build_harness,
    run_smoke,
)


SMOKE_CONFIG = {
    "maximum_steps": 7,
    "minimum_authentic_intervention_rate": 0.4,
    "episodes": [
        {"episode_id": f"test-{index}", "requires_decode": bool(index % 2)}
        for index in range(6)
    ],
}


def test_authentic_artifact_is_source_gated_and_hash_bound():
    artifact = authentic_source_skill()
    artifact.assert_valid()
    assert artifact.qualification is not None
    assert artifact.qualification.passed
    with pytest.raises(RetargetingReject, match="hash mismatch"):
        replace(artifact, source_domain="post-freeze-edit").assert_valid()

    failed_gate = SourceQualification.create(
        authentic_value=0.1,
        shuffled_value=0.2,
        marginal_value=0.3,
        receipt_ids=(stable_hash("fork-a"), stable_hash("fork-b")),
    )
    with pytest.raises(RetargetingReject, match="failed the source value gate"):
        FrozenSkillArtifact.create(
            skill_id="failed-source",
            artifact_kind=SkillArtifactKind.AUTHENTIC_SOURCE,
            source_domain="source",
            option_vocabulary=("A", "B"),
            predicate_vocabulary=("start", "done"),
            role_vocabulary=("subject",),
            rules=(
                SymbolicOptionRule("a", "A", ("start",), expected_add=("done",)),
                SymbolicOptionRule("b", "B", ("done",), expected_remove=("done",)),
            ),
            termination_facts=("done",),
            source_lineage=(stable_hash("source-fork"),),
            qualification=failed_gate,
        )


def test_shuffled_control_preserves_guards_effects_and_lineage_but_deranges_options():
    authentic = authentic_source_skill()
    shuffled = authentic.shuffled_control()
    assert shuffled.artifact_kind == SkillArtifactKind.SHUFFLED_CONTROL
    assert shuffled.parent_artifact_hash == authentic.artifact_hash
    assert shuffled.artifact_hash != authentic.artifact_hash
    assert shuffled.program_structure_hash == authentic.program_structure_hash
    assert shuffled.source_lineage == authentic.source_lineage
    permutation = dict(shuffled.control_option_permutation)
    assert set(permutation) == set(authentic.option_vocabulary)
    assert set(permutation.values()) == set(authentic.option_vocabulary)
    assert all(source != target for source, target in permutation.items())
    for source_rule, control_rule in zip(authentic.rules, shuffled.rules, strict=True):
        assert control_rule.option != source_rule.option
        assert control_rule.requires == source_rule.requires
        assert control_rule.forbids == source_rule.forbids
        assert control_rule.expected_add == source_rule.expected_add
        assert control_rule.expected_remove == source_rule.expected_remove


def test_harness_cannot_choose_an_option_and_realizes_only_inside_external_option():
    environment = IncidentResponseEnvironment({
        "episode": {"episode_id": "episode", "requires_decode": True},
    })
    observation = environment.reset("episode")
    harness = build_harness()
    assert not hasattr(harness, "choose_option")
    frame = harness.ground(observation)
    decision = harness.realize("SEARCH", frame)
    assert decision.selected_option == "SEARCH"
    assert decision.action == "scan telemetry"
    assert set(decision.candidate_actions) == {"scan telemetry", "reboot display"}
    assert decision.validate()


def test_harness_rejects_cross_option_duplicate_native_action():
    class BadGrounder:
        grounder_hash = stable_hash("bad-grounder")

        def ground(self, observation):
            candidate = NativeActionCandidate("native", 1.0)
            return GroundingDraft(
                facts=frozenset({"start"}),
                role_bindings=(("subject", "x"),),
                option_groundings=(
                    OptionGrounding("A", (candidate,)),
                    OptionGrounding("B", (candidate,)),
                ),
            )

    grounder = BadGrounder()
    artifact = TargetHarnessArtifact.create(
        harness_id="bad",
        target_domain="target",
        option_vocabulary=("A", "B"),
        predicate_vocabulary=("start",),
        role_vocabulary=("subject",),
        adaptation_receipt_ids=(stable_hash("adaptation"),),
        grounder_hash=grounder.grounder_hash,
        realizer_hash=stable_hash("realizer"),
        verifier_hash=stable_hash("verifier"),
    )
    harness = PermissionBoundTargetHarness(artifact, grounder)
    observation = TargetObservation("obs", {"surface": "x"}, ("native",))
    with pytest.raises(RetargetingReject, match="multiple options"):
        harness.ground(observation)


def test_official_outcome_is_not_part_of_the_harness_observation_contract():
    observation_fields = set(TargetObservation.__dataclass_fields__)
    assert not {"reward", "score", "official_score", "official_success"}.intersection(
        observation_fields
    )
    assert {"reward", "official_score", "official_success"}.issubset(
        TargetStep.__dataclass_fields__
    )
    harness = build_harness()
    leaking = TargetObservation(
        "leak",
        {
            "telemetry": "unlocated",
            "nested_evaluator": {"official_success": True},
        },
        (),
    )
    with pytest.raises(RetargetingReject, match="forbidden official-outcome"):
        harness.ground(leaking)


def test_refuted_expected_effect_uses_declared_recovery_option():
    artifact = authentic_source_skill()
    harness = build_harness()
    environment = IncidentResponseEnvironment({
        "episode": {"episode_id": "episode", "requires_decode": False},
    })
    frame = harness.ground(environment.reset("episode"))
    execution = FrozenSkillExecution(artifact)
    first = execution.select(frame, None)
    assert first.verdict == SkillDecisionVerdict.SELECT
    assert first.option == "SEARCH"
    effect_payload = {
        "before_frame_hash": frame.frame_hash,
        "after_frame_hash": frame.frame_hash,
        "decision_hash": stable_hash("no-op-decision"),
        "added_facts": [],
        "removed_facts": [],
        "harness_hash": harness.artifact.artifact_hash,
    }
    no_effect = EffectReceipt(
        before_frame_hash=frame.frame_hash,
        after_frame_hash=frame.frame_hash,
        decision_hash=effect_payload["decision_hash"],
        added_facts=(),
        removed_facts=(),
        harness_hash=harness.artifact.artifact_hash,
        receipt_hash=stable_hash(effect_payload),
    )
    recovery = execution.select(frame, no_effect)
    assert recovery.verdict == SkillDecisionVerdict.SELECT
    assert recovery.option == "SEARCH"
    assert recovery.reason == "FAILURE_SPECIFIC_RECOVERY"


def test_controlled_cross_semantic_smoke_passes_all_attribution_gates():
    report, artifacts = run_smoke(SMOKE_CONFIG)
    assert report.verdict == ExperimentVerdict.MECHANISM_SUPPORTED
    summaries = report.metrics["summaries"]
    assert summaries["authentic_source_skill"]["successes"] == 6
    assert summaries["target_oracle_skill"]["successes"] == 6
    assert summaries["null_skill_same_harness"]["successes"] == 0
    assert summaries["shuffled_source_skill"]["successes"] == 0
    assert report.metrics["authentic_effect_applicability"] == 1.0
    assert report.metrics["authentic_vs_null_paired_regressions"] == 0
    assert artifacts["authentic_skill_hash"] != artifacts["shuffled_skill_hash"]


def test_evaluator_fails_closed_on_missing_condition_or_harness_mismatch():
    report, _ = run_smoke(SMOKE_CONFIG)
    missing = report.outcomes[:-1]
    invalid = evaluate_retargeting_experiment(
        missing, minimum_authentic_intervention_rate=0.4,
    )
    assert invalid.verdict == ExperimentVerdict.INVALID_EXPERIMENT
    assert "required condition" in invalid.reason

    changed = list(report.outcomes)
    index = next(
        index for index, row in enumerate(changed)
        if row.condition == RetargetingCondition.NULL_SKILL_SAME_HARNESS
    )
    changed[index] = replace(changed[index], harness_hash=stable_hash("other-harness"))
    invalid = evaluate_retargeting_experiment(
        changed, minimum_authentic_intervention_rate=0.4,
    )
    assert invalid.verdict == ExperimentVerdict.INVALID_EXPERIMENT
    assert "one frozen Harness" in invalid.reason


def test_evaluator_calls_authentic_regression_negative_transfer():
    report, _ = run_smoke(SMOKE_CONFIG)

    def rewrite_official_outcome(row, success):
        receipts = list(row.receipts)
        final = replace(
            receipts[-1],
            official_success_after=success,
            official_score_after=float(success),
            receipt_hash="",
        )
        final = replace(
            final,
            receipt_hash=stable_hash({
                key: value for key, value in final.__dict__.items()
                if key != "receipt_hash"
            }),
        )
        receipts[-1] = final
        return replace(
            row,
            official_success=success,
            official_score=float(success),
            receipts=tuple(receipts),
        )

    changed = [
        rewrite_official_outcome(row, False)
        if row.condition == RetargetingCondition.AUTHENTIC_SOURCE_SKILL
        else rewrite_official_outcome(row, True)
        if row.condition == RetargetingCondition.NULL_SKILL_SAME_HARNESS
        else row
        for row in report.outcomes
    ]
    negative = evaluate_retargeting_experiment(
        changed, minimum_authentic_intervention_rate=0.4,
    )
    assert negative.verdict == ExperimentVerdict.NEGATIVE_TRANSFER
