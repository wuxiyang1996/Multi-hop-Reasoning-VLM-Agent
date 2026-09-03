from motif_transfer.agqa_qwen235_selective_authorizer import authorize_source_override


def _runtime(*, route="RELATION_RECURRENT", comparison="EXISTS", decision="yes", tiebreak=False):
    return {
        "query_plan": {"obligation_kind": route, "comparison": comparison},
        "target_native_execution": {"decision": decision},
        "direct_response": "no",
        "grounding_receipt": {
            "canonicalizations": ["RECURRENT_DOUBLE_SCAN_CONFIRMED_OBSERVED"]
        },
        "operand_runs": {"A": {"tiebreak_triggered": tiebreak}},
    }


def test_authorizes_stable_recurrent_relation_observation():
    result = authorize_source_override(_runtime())
    assert result["authorized"] is True
    assert result["prediction"] == "yes"


def test_abstains_after_conflict_tiebreak():
    result = authorize_source_override(_runtime(tiebreak=True))
    assert result["authorized"] is False
    assert result["prediction"] == "no"
    assert "CONFLICT_TIEBREAK_USED" in result["reasons"]


def test_abstains_on_duration_route():
    result = authorize_source_override(
        _runtime(route="TEMPORAL_SINGLE_NONRECURRENT", comparison="VERIFY_A_LONGER")
    )
    assert result["authorized"] is False


def test_abstains_on_negative_observation():
    assert authorize_source_override(_runtime(decision="no"))["authorized"] is False
